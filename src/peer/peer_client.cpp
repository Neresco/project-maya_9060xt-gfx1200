#include "peer_client.hpp"

#include "peer_compute.hpp"
#include "peer_read.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace strata::peer {

PeerClient::PeerClient(int window, size_t chunk, int timeout_ms, const std::string& device)
    : window_(window), timeout_ms_(timeout_ms), chunk_(chunk), device_(device) {}

PeerClient::~PeerClient() {
    close();
}

void PeerClient::close() {
    up_ = false;
    reader_.reset();
    dst_.clear();       // (the destination MRs are deregistered while the PD they were registered on still lives:
                        //  ibv_dereg_mr after dev_.close() is a call on a context that is gone, and it segfaults)
    qp_.destroy();
    dev_.close();
    map_.clear();
    ctl_.close();
}

ibv_pd* PeerClient::pd() const { return dev_.pd(); }

bool PeerClient::add_dst(void* buf, size_t bytes, std::string& err) {
    auto mr = std::make_unique<Mr>();
    if (!mr->reg(dev_.pd(), buf, bytes, IBV_ACCESS_LOCAL_WRITE, err)) return false;
    dst_.push_back(std::move(mr));
    return true;
}

void PeerClient::clear_dst() {
    dst_.clear();
}

bool PeerClient::connect(const std::string& host, int port, const std::string& pack_dir,
                         const std::vector<std::string>& shard_names, const std::vector<uint64_t>& shard_sizes,
                         const std::vector<Row>& rows, std::string& err) {
    close();
    pack_dir_ = pack_dir;
    if (!ctl_.connect(host, port, timeout_ms_, err)) return false;
    rows_ = rows;
    return handshake(host, port, shard_names, shard_sizes, rows, err) && (up_ = true);
}

bool PeerClient::handshake(const std::string& host, int port, const std::vector<std::string>& shard_names,
                           const std::vector<uint64_t>& sizes, const std::vector<Row>& rows,
                           std::string& err) {
    if (!dev_.open(device_, err)) return false;
    // hello
    Json hello;
    hello.add("type", "hello").add("proto", 1).add("window", window_)
         .add("chunk_mb", (long long) (chunk_ >> 20)).add("timeout_ms", timeout_ms_);
    if (!ctl_.send(hello.str(), err)) return false;
    const JsonView welcome(ctl_.recv(timeout_ms_, err));
    if (!welcome.ok() || welcome.type() != "welcome") {
        err = "no welcome: " + err;
        return false;
    }
    if (welcome.num("proto") != 1) {
        err = "the daemon speaks protocol " + welcome.str("proto") + ", this client speaks 1";
        return false;
    }
    // what the peer can hold at all, and what its compute worker is (if it has one): the tier caps the rows it asks
    // to be kept resident by the first and gates a compute order on the second
    capacity_ = (uint64_t) std::max<long long>(0, welcome.num("cap", 0));
    vram_ = (uint64_t) std::max<long long>(0, welcome.num("vram", 0));
    compute_caps_ = welcome.str("compute");
    // manifest: names + sizes plus the fingerprint of this side's own copy - the daemon computes
    // the same string over its copy and refuses a mismatch (a half-copied pack would otherwise
    // serve wrong bytes silently)
    std::string fp = pack_fingerprint(pack_dir_, shard_names, sizes, err);
    if (fp.empty()) return false;
    Json man;
    man.add("type", "manifest").add("shards", (long long) shard_names.size()).add("fp", fp);
    for (size_t i = 0; i < shard_names.size(); ++i) {
        man.add("shard" + std::to_string(i), shard_names[i]);
        man.add("size" + std::to_string(i), (long long) sizes[i]);
    }
    if (!ctl_.send(man.str(), err)) return false;
    const JsonView fp_ack(ctl_.recv(timeout_ms_, err));
    if (!fp_ack.ok() || fp_ack.type() != "fingerprint") {
        err = "the daemon refused the manifest: " + (fp_ack.ok() ? fp_ack.str("msg") : err);
        return false;
    }
    // the row table: one raw binary frame after a one-line header (16 bytes a row: u16 shard,
    // u16 pad, u64 offset, u32 len - the daemon's Row, laid out identically).  The frame rides the
    // socket hex-encoded: arbitrary bytes would carry newlines and break the line framing.
    std::vector<char> blob(rows.size() * 16);
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint16_t s = rows[i].shard;
        const uint64_t off = rows[i].off;
        const uint32_t len = rows[i].len;
        std::memcpy(blob.data() + i * 16, &s, 2);
        std::memcpy(blob.data() + i * 16 + 2, &s, 2);      // the pad keeps the offsets aligned
        std::memcpy(blob.data() + i * 16 + 4, &off, 8);
        std::memcpy(blob.data() + i * 16 + 12, &len, 4);
    }
    static const char* hex = "0123456789abcdef";
    std::string framed;
    framed.reserve(blob.size() * 2 + 5);
    for (char c : blob) {
        framed += hex[(unsigned char) c >> 4];
        framed += hex[(unsigned char) c & 15];
    }
    // one JSON line carries the whole table (a bare "hex:"-prefixed line would not parse as JSON)
    const Json rows_line = Json().add("type", "rows").add("count", (long long) rows.size())
                                .add("hex", framed);
    if (!ctl_.send(rows_line.str(), err)) return false;
    const JsonView rows_ack(ctl_.recv(timeout_ms_, err));
    if (!rows_ack.ok() || rows_ack.type() != "rows") {
        err = "the daemon refused the row table: " + (rows_ack.ok() ? rows_ack.str("msg") : err);
        return false;
    }
    // the QP: create it here, hand {qpn, psn, gid} over the ctl channel (llama.cpp's ggml-rpc
    // recipe), and arm it from the daemon's numbers - both sides then talk RC over the link
    if (!qp_.create(dev_, window_ * 2 + 8, 8, err)) return false;
    if (!ctl_.send(Json().add("type", "qp-up").add("qpn", (long long) qp_.qpn())
                       .add("psn", (long long) qp_.psn()).add("gid", qp_.gid()).str(), err)) {
        return false;
    }
    const JsonView ready(ctl_.recv(timeout_ms_, err));
    if (!ready.ok() || ready.type() != "ready") {
        err = "the daemon never came ready: " + (ready.ok() ? ready.type() : err);
        return false;
    }
    if (!qp_.arm(dev_, (uint32_t) ready.num("qpn"), (uint32_t) ready.num("psn"),
                 ready.str("gid"), err)) {
        return false;
    }
    // the chunked-read engine rides the fresh QP: one shared CQ (the device's), the window and
    // chunk from the hello (the daemon sized its QP against the same window)
    reader_ = std::make_unique<Reader>(qp_.qp_ptr(), dev_.cq(), window_, chunk_);
    say("session up: " + std::to_string(rows.size()) + " rows in the table, window " +
        std::to_string(window_));
    return true;
}

bool PeerClient::hint(const std::vector<long long>& rows, std::string& err, int timeout_ms) {
    if (!up_ || rows.empty()) return true;
    Json h;
    h.add("type", "hint").add("n", (long long) rows.size());
    h.add("rows", rows);
    if (!ctl_.send(h.str(), err)) {
        up_ = false;
        return false;
    }
    // the first hint makes the daemon allocate and register a fresh arena (GiBs of pinning), so
    // the reply can legitimately outrun the read timeout - hints are rare, reads are the hot path.
    // `timeout_ms` overrides that: the resident hint at hello fills GiBs of the peer's own disk, which is minutes
    int budget = timeout_ms_ > 30000 ? timeout_ms_ : 30000;
    if (timeout_ms > 0) budget = timeout_ms;
    const JsonView map(ctl_.recv(budget, err));
    if (!map.ok()) {
        say("hint reply lost: up=" + std::to_string(up_) + " closed=" +
            std::to_string(ctl_.closed()) + " after " + std::to_string(budget) + " ms: " + err);
        up_ = false;
        return false;
    }
    say("hint reply: " + std::to_string(map.nums("i").size()) + " entries");
    if (map.type() == "error") {
        err = "the daemon refused the hint: " + map.str("msg");
        return false;
    }
    if (map.type() != "map") {
        err = "expected a map line, got " + map.type();
        up_ = false;
        return false;
    }
    for (long long dropped : map.nums("dropped")) map_.erase(dropped);
    // the entries: the daemon answers with four parallel integer arrays - i, addr, len, rkey
    const auto idx = map.nums("i");
    const auto addr = map.nums("addr");
    const auto len = map.nums("len");
    const auto rkey = map.nums("rkey");
    if (idx.size() != addr.size() || idx.size() != len.size() || idx.size() != rkey.size()) {
        err = "a map line with mismatched arrays";
        up_ = false;
        return false;
    }
    for (size_t k = 0; k < idx.size(); ++k) {
        map_[idx[k]] = Cached{0, (uint64_t) addr[k], (uint32_t) len[k], (uint32_t) rkey[k]};
    }
    return true;
}

bool PeerClient::read_row(long long ri, Mr& dst, uint64_t dst_off, std::string& err) {
    const auto it = map_.find(ri);
    if (it == map_.end()) return false;                  // unadvertised: the caller's disk fallback
    if (!up_) return false;
    const Cached& c = it->second;
    const bool ok = reader_->read(c.addr, c.rkey, dst.mr_ptr(), dst_off, c.len, timeout_ms_, err);
    if (!ok) {
        up_ = false;             // a failed read leaves the QP's state unknown: reconnect, not retry
        err = "row " + std::to_string(ri) + " failed: " + err;
    }
    return ok;
}

bool PeerClient::read_rows(const std::vector<RowRead>& reqs, std::string& err) {
    if (!up_ || reqs.empty()) return false;
    // resolve every row's advertisement first: one that was dropped (an evicted arena) fails the
    // batch before a byte moves, so the caller's disk fallback sees an untouched buffer
    std::vector<ReadReq> rr;
    rr.reserve(reqs.size());
    for (const RowRead& q : reqs) {
        const auto it = map_.find(q.ri);
        if (it == map_.end()) {
            err = "row " + std::to_string(q.ri) + " is no longer advertised";
            return false;
        }
        const Cached& c = it->second;
        if (q.rel > (uint64_t) c.len || (uint64_t) q.bytes > (uint64_t) c.len - q.rel) {
            err = "row " + std::to_string(q.ri) + "'s sub-range [" + std::to_string(q.rel) + ", +" +
                  std::to_string(q.bytes) + ") is not within its " + std::to_string(c.len) + " bytes";
            return false;   // (a table mismatch, never a torn read: nothing posted)
        }
        const uint64_t n = q.bytes != 0 ? (uint64_t) q.bytes : (uint64_t) c.len - q.rel;
        rr.push_back(ReadReq{c.addr + q.rel, c.rkey, q.dst->mr_ptr(), q.dst_off, (size_t) n});
    }
    if (!reader_->read_many(rr, timeout_ms_, err)) {
        up_ = false;             // a failed read leaves the QP's state unknown: reconnect, not retry
        err = "peer batch read failed: " + err;
        return false;
    }
    return true;
}

bool PeerClient::compute(const std::vector<ComputeReq>& jobs, std::vector<char>& ok,
                         std::vector<std::vector<float>>& y, std::string& err) {
    ok.assign(jobs.size(), 0);
    y.assign(jobs.size(), {});
    if (jobs.empty()) return true;
    if (!up_) {
        err = "the peer session is not up";
        return false;
    }
    // the layer's geometry travels once a batch (it is the model's, not one job's): a batch of two
    // geometries is a caller's bug, not something the daemon should be asked to sort out
    const int n_embd = jobs[0].n_embd, n_ff = jobs[0].n_ff;
    bool any_w = false;      // (the weights are all-or-nothing a batch: a job without one carries 1s)
    for (const ComputeReq& j : jobs) {
        if (j.n_embd != n_embd || j.n_ff != n_ff) {
            err = "a compute batch of two geometries is not one layer";
            return false;
        }
        if (j.xq == nullptr || j.xd == nullptr || j.n_rows <= 0 || n_embd % 32 != 0) {
            err = "a compute job without its activation (or of a width not whole 32s) is not one";
            return false;
        }
        if (j.w != nullptr) any_w = true;
    }
    // the header: the geometry, the rows a job, and each job's metadata - every byte of the activation and
    // of the sums rides the body after it (a body is 20 KB in and 41 KB out where hex was 61 KB, and neither
    // side pays a Json parse or a hex decode a buffer)
    Json req;
    req.add("type", "compute").add("n", (long long) jobs.size()).add("n_embd", n_embd).add("n_ff", n_ff);
    std::vector<long long> rows;
    rows.reserve(jobs.size());
    for (const ComputeReq& j : jobs) rows.push_back(j.n_rows);
    req.add("rows", rows);
    if (any_w) req.add("has_w", 1);
    for (size_t k = 0; k < jobs.size(); ++k) {
        const ComputeReq& j = jobs[k];
        const std::string p = "j" + std::to_string(k);
        req.add(p + "_gate", j.gate).add(p + "_up", j.up).add(p + "_down", j.down);
        req.add(p + "_tg", j.tg).add(p + "_td", j.td);
        // (the limit is a float: the scanner keeps a number as its text, so it goes out as one)
        char lim[32];
        std::snprintf(lim, sizeof lim, "%.9g", (double) j.limit);
        req.add_raw(p + "_lim", lim);
    }
    // the body, in the request's own job order: xq (rows x n_embd int8), xd (rows x n_embd/32 f32), and -
    // when the batch carries weights - w (rows f32).  One buffer, one send: a write a buffer is a syscall a
    // buffer and a packet of its own with TCP_NODELAY on.
    std::string body;
    body.reserve(jobs.size() * (size_t) (n_embd + n_embd / 32 * 4 + 4));
    for (const ComputeReq& j : jobs) {
        body.append((const char*) j.xq, (size_t) j.n_rows * (size_t) n_embd);
        body.append((const char*) j.xd, (size_t) j.n_rows * (size_t) (n_embd / 32) * sizeof(float));
        if (!any_w) continue;
        // (a job without a routing weight is the expert's own output: 1.0f a row, so the body's layout
        // stays one shape whatever the batch carries)
        for (int t = 0; t < j.n_rows; ++t) {
            const float w = j.w != nullptr ? j.w[t] : 1.0f;
            body.append((const char*) &w, sizeof w);
        }
    }
    if (!ctl_.send(req.str(), err) || !ctl_.send_body(body.data(), body.size(), err)) {
        up_ = false;
        return false;
    }
    // one reply for the batch, bounded by the session's timeout: a hung compute never stalls a decode
    const JsonView ans(ctl_.recv(timeout_ms_, err));
    if (!ans.ok()) {
        up_ = false;
        err = "no computed reply: " + err;
        return false;
    }
    if (ans.type() == "error") {
        err = "the daemon refused the compute batch: " + ans.str("msg");   // (the session stays up)
        return false;
    }
    if (ans.type() != "computed") {
        up_ = false;
        err = "expected a computed line, got " + ans.type();
        return false;
    }
    const auto idx = ans.nums("i"), done = ans.nums("ok");
    if (idx.size() != done.size()) {
        up_ = false;
        err = "a computed line with mismatched arrays";
        return false;
    }
    // the answer's own arrays first: one that does not line up against the batch is a protocol error,
    // and the body cannot be read against it - so it is checked before a byte of it is taken
    size_t served = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        const long long k = idx[i];
        if (k < 0 || (size_t) k >= jobs.size()) {
            up_ = false;
            err = "a computed line named job " + std::to_string(k) + ", the batch has " +
                  std::to_string(jobs.size());
            return false;
        }
        if (done[i] != 0) served += (size_t) jobs[(size_t) k].n_rows * (size_t) n_embd * sizeof(float);
    }
    if (ans.num("body", -1) != (long long) served) {
        up_ = false;
        err = "a computed body of " + ans.str("body", "no") + " bytes, the sums take " +
              std::to_string(served);
        return false;
    }
    // the sums, in the answer's own order: one body read, straight into the caller's buffers
    for (size_t i = 0; i < idx.size(); ++i) {
        if (done[i] == 0) continue;                 // (the caller computes this one itself)
        const long long k = idx[i];
        const size_t n = (size_t) jobs[(size_t) k].n_rows * (size_t) n_embd;
        y[(size_t) k].resize(n);
        if (!ctl_.recv_body(y[(size_t) k].data(), n * sizeof(float), err)) {
            up_ = false;
            err = "job " + std::to_string(k) + "'s sums did not arrive: " + err;
            return false;
        }
        ok[(size_t) k] = 1;
    }
    return true;
}

bool PeerClient::keepalive(std::string& err) {
    if (!up_) return false;
    if (!ctl_.send(Json().add("type", "ping").str(), err)) {
        up_ = false;
        return false;
    }
    const JsonView pong(ctl_.recv(timeout_ms_, err));
    if (!pong.ok() || pong.type() != "pong") {
        up_ = false;
        err = "no pong: " + (pong.ok() ? pong.type() : err);
        return false;
    }
    return true;
}

}   // namespace strata::peer
