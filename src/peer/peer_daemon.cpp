// strata-peer: the daemon side of the peer expert tier.  It serves one initiator at a time (the
// engine's session): verifies the initiator's pack fingerprint against its own copy, takes the row
// table, brings the RDMA link up (its QP at RTS is all a READ responder needs - no per-row
// software), faults hinted rows from its own pack copy into a budget-capped pinned slab and
// advertises them.  The initiator pulls bytes with RDMA READ; the daemon never touches them.
//
// The daemon is deliberately engine-free (no GPU runtime, no strata_core): it runs beside an engine
// on the peer machine under that machine's own RAM rules, and its slab budget is cgroup-aware for
// exactly the reason the engine's RAM tier is (a container's /proc/meminfo is the host's).
//
// Usage: strata-peer --pack <dir> [--bind <ip>] [--port 9447] [--ram-gb <n>] [--arena-mb <n>]
//                     [--device <ibv device>] [--vram <gpu>:<gb>]... [--compute <worker .so>]
//                     [--compute-options <str>]
//                     (-x: a budget of 0 probes a quarter of what is free)
//
// --vram is the opt-in VRAM tier (peer_vram.cpp): one slab per spec, held in that card's VRAM and
// registered as a dmabuf MR, so hinted rows land there instead of in the pinned RAM slab.  Off by
// default, repeatable, and start-up-fatal when it cannot be honoured - a peer whose card is busy with
// ComfyUI should say so on the spot, not fail a fill an hour later.
//
// --compute is the opt-in compute tier (peer_compute.cpp, docs/PEER_COMPUTE.md): a worker shared object
// the daemon loads by path and hands compute orders to, so the peer computes the experts it holds instead
// of only serving their bytes.  Absent: storage only, exactly as before.  Given and not loadable: a start-up
// error, the same contract as --vram (a worker that cannot start must not surface at the third layer of a
// decode).  The daemon still links no GPU runtime: the worker is the only place one could ever live.
//
// --compute-options is the string the worker itself reads (its open()'s argument): a HIP worker takes
// "gpu:<ordinal>" to name the peer card it should compute on, and without it picks the machine's first one.
// The daemon never interprets it; it only compares the card the worker reports back with the ordinal of a
// --vram slab before it hands that slab's rows over (peer_compute_abi.h).
#include "peer_cache.hpp"
#include "peer_compute.hpp"
#include "peer_ctl.hpp"
#include "peer_rdma.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace strata::peer {

namespace {

/// The compute workers' caps lines as ONE welcome field: " | " between them, so a daemon with two cards' workers
/// names both (the client keeps the field as the tier's own description and asks nothing of its shape).
std::string compute_caps(const std::vector<ComputeWorker*>& w) {
    std::string s;
    for (ComputeWorker* x : w) {
        if (x == nullptr) continue;
        if (!s.empty()) s += " | ";
        s += x->caps();
    }
    return s;
}

/// One session's serving state (a borrowed cache and a borrowed cm listener per accepted
/// connection - one initiator at a time, so the listener's child id serves this session alone).
struct Session {
    Ctl ctl;
    Device* dev = nullptr;          // the daemon-wide device: its PD carries both the QPs and
                                    // the cache's arena MRs (one PD, or the RDMA reads break)
    Qp qp;                          // this session's RC QP (armed from the client's qp-up)
    PeerCache* cache;
    std::vector<ComputeWorker*> workers;   // the daemon's --compute workers, one per flag (empty: storage only)
    int w_rr = 0;                          // (the worker a RAM-slab job goes to next: any of them takes one)
    std::vector<Row> rows;
    std::string pack_dir;
    uint64_t clock_stamp = 0;       // the session's LRU stamps (one initiator at a time)

    bool serve(std::string& err);
    bool handshake(std::string& err);
    int hello_window = 48;          // the client's window (identifies this session in the log)
    bool on_hint(const JsonView& hint, std::string& err);
    bool on_compute(const JsonView& line, std::string& err);
};

bool Session::handshake(std::string& err) {
    const JsonView hello(ctl.recv(10000, err));
    if (!hello.ok() || hello.type() != "hello") {
        err = "no hello: " + (hello.ok() ? hello.type() : err);
        return false;
    }
    if (hello.num("proto") != 1) {
        err = "the client speaks protocol " + hello.str("proto") + ", this daemon speaks 1";
        return false;
    }
    const Json welcome = Json().add("type", "welcome").add("proto", 1)
                             .add("budget_gb", (long long) (cache->budget() >> 30))
                             .add("cap", (long long) cache->capacity_bytes())
                             .add("vram", (long long) cache->vram_capacity())
                             .add("compute", compute_caps(workers));
    if (!ctl.send(welcome.str(), err)) return false;
    // the manifest: names + sizes + the client's own fingerprint of its copy - a mismatch refuses
    const JsonView man(ctl.recv(30000, err));
    if (!man.ok() || man.type() != "manifest") {
        err = "no manifest: " + (man.ok() ? man.type() : err);
        return false;
    }
    const long long n_shards = man.num("shards");
    if (n_shards <= 0 || n_shards > 4096) {
        err = "a manifest of " + man.str("shards") + " shards makes no sense";
        return false;
    }
    std::vector<std::string> names;
    std::vector<uint64_t> sizes;
    for (long long i = 0; i < n_shards; ++i) {
        names.push_back(man.str("shard" + std::to_string(i)));
        sizes.push_back((uint64_t) man.num("size" + std::to_string(i)));
        if (names.back().empty()) {
            err = "manifest shard " + std::to_string(i) + " has no name";
            return false;
        }
    }
    const std::string mine = pack_fingerprint(pack_dir, names, sizes, err);
    if (mine.empty()) return false;
    if (mine != man.str("fp")) {
        err = "pack fingerprint mismatch: the client's copy is not this pack (" + mine.substr(0, 32) +
              "... vs " + man.str("fp").substr(0, 32) + "...)";
        ctl.send(Json().add("type", "error").add("msg", err).str(), err);
        return false;
    }
    if (!cache->open_shards(names, sizes, pack_dir, err)) return false;
    if (!ctl.send(Json().add("type", "fingerprint").add("ok", 1).str(), err)) return false;
    // the row table (hex-framed, 16 bytes a row)
    const JsonView rows_line(ctl.recv(30000, err));
    if (!rows_line.ok() || rows_line.type() != "rows") {
        err = "no row table: " + (rows_line.ok() ? rows_line.type() : err);
        return false;
    }
    const std::string hex = rows_line.str("hex");
    if (hex.size() % 32 != 0 || hex.size() < 32) {
        err = "the row table's " + std::to_string(hex.size()) + " hex digits frame no whole rows";
        return false;
    }
    rows.clear();
    auto nib = [](char c) -> int {
        return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
    };
    for (size_t at = 0; at < hex.size(); at += 32) {
        uint16_t shard = 0, pad = 0;
        uint64_t off = 0;
        uint32_t len = 0;
        uint8_t* rec[4] = {(uint8_t*) &shard, (uint8_t*) &pad, (uint8_t*) &off, (uint8_t*) &len};
        const int widths[4] = {4, 4, 16, 8};             // nibbles per field
        int at_hex = (int) at;
        for (int f = 0; f < 4; ++f) {
            for (int b = 0; b < widths[f] / 2; ++b) {
                const int hi = nib(hex[at_hex++]), lo = nib(hex[at_hex++]);
                if (hi < 0 || lo < 0) {
                    err = "the row table's hex broke at record " + std::to_string(rows.size());
                    return false;
                }
                rec[f][b] = (uint8_t) ((hi << 4) | lo);
            }
        }
        rows.push_back(Row{shard, off, len});
    }
    if (!ctl.send(Json().add("type", "rows").add("count", (long long) rows.size()).str(), err))
        return false;
    say("row table: " + std::to_string(rows.size()) + " rows");
    // the QP comes up by hand (llama.cpp's ggml-rpc recipe): the client's qp-up carries its
    // {qpn, psn, gid}, we create and arm our side, and the ready answers with ours
    hello_window = (int) hello.num("window", 48);
    const int window = hello_window;
    const JsonView qpup(ctl.recv(30000, err));
    if (!qpup.ok() || qpup.type() != "qp-up") {
        err = "no qp-up: " + (qpup.ok() ? qpup.type() : err);
        return false;
    }
    if (!qp.create(*dev, window * 2 + 8, 8, err)) return false;
    if (!qp.arm(*dev, (uint32_t) qpup.num("qpn"), (uint32_t) qpup.num("psn"), qpup.str("gid"), err))
        return false;
    if (!ctl.send(Json().add("type", "ready").add("qpn", (long long) qp.qpn())
                      .add("psn", (long long) qp.psn()).add("gid", qp.gid()).str(), err)) {
        return false;
    }
    say("session up: budget " + std::to_string(cache->budget() >> 20) + " MiB (capacity " +
        std::to_string(cache->capacity_bytes() >> 20) + " MiB with its VRAM slabs" +
        (!workers.empty() ? ", compute " + compute_caps(workers) : ", storage only") + "), window " +
        std::to_string(window));
    return true;
}

bool Session::on_hint(const JsonView& hint, std::string& err) {
    const std::vector<long long> want = hint.nums("rows");
    std::vector<Cached> got;
    if (!cache->fill(want, rows, ++clock_stamp, got, err)) {
        ctl.send(Json().add("type", "error").add("msg", err).str(), err);
        return false;
    }
    // fill answers one entry per requested row, in the request's order: the parallel arrays line
    // straight up against it
    Json map;
    map.add("type", "map");
    std::vector<long long> idx, addr, len, rkey;
    for (size_t k = 0; k < got.size(); ++k) {
        idx.push_back(want[k]);
        addr.push_back((long long) got[k].addr);
        len.push_back(got[k].len);
        rkey.push_back(got[k].rkey);
    }
    map.add("i", idx).add("addr", addr).add("len", len).add("rkey", rkey);
    map.add("dropped", cache->dropped());
    if (!ctl.send(map.str(), err)) return false;
    say("hint: " + std::to_string(want.size()) + " rows, " +
        std::to_string(cache->cached_bytes() >> 20) + " MiB cached (" +
        std::to_string(cache->cached_rows()) + " rows, " + std::to_string(cache->arenas()) +
        " arenas" + (cache->vram_bytes() ? ", " + std::to_string(cache->vram_bytes() >> 20) +
                                                    " MiB of it in VRAM" : "") + ")");
    return true;
}

bool Session::on_compute(const JsonView& line, std::string& err) {
    const long long n = line.num("n", 0);
    const long long n_embd = line.num("n_embd", 0), n_ff = line.num("n_ff", 0);
    const std::vector<long long> nrows = line.nums("rows");
    // EVERY LENGTH OF THE BODY IS DERIVED FROM THIS HEADER, so a header that does not frame one cannot be
    // skipped over and the stream cannot be resynchronised: it ends the session with the reason, the way a map
    // line with mismatched arrays does.  (A job this daemon cannot serve is still ok = 0 and the session lives -
    // only a header that is not this protocol's is a session's end.)
    if (n < 0 || n > 64 || (long long) nrows.size() != n) {
        err = "a compute header of " + std::to_string(n) + " jobs and " + std::to_string(nrows.size()) +
              " row counts frames no body";
        return false;
    }
    if (n > 0 && (n_embd <= 0 || n_ff <= 0 || n_embd % 32 != 0 || n_embd > 65536 || n_ff > 65536)) {
        err = "a compute batch of " + std::to_string(n_embd) + " x " + std::to_string(n_ff) +
              " is not a layer's geometry";
        return false;
    }
    for (long long rows : nrows) {
        if (rows <= 0 || rows > 8) {   // (one line is one layer's picks: past that it is not a batch)
            err = "a compute job of " + std::to_string(rows) + " rows frames no body";
            return false;
        }
    }
    const bool has_w = line.num("has_w", 0) != 0;   // (all-or-nothing a batch: every job carries its rows)
    // the bodies are read into these and stay alive until the worker's run returns: it reads them in place,
    // the wire is the only copy.  WHICH WORKER A JOB GOES TO is decided here, one job at a time: a --vram row
    // has no host window, so only the worker that computes on the card that allocated it may take it; a RAM
    // slab's rows are host bytes any worker takes (they go to the workers in turn, so two cards' workers each
    // get a share of a layer's orders)
    std::vector<std::vector<int8_t>> xq((size_t) n);
    std::vector<std::vector<float>> xd((size_t) n), wt((size_t) n), y((size_t) n);
    std::vector<strata_peer_job> jobs((size_t) n);
    std::vector<long long> at;   // (the jobs this daemon can serve, in the request's order)
    std::vector<int> wsel((size_t) n, -1);   // (each job's worker: the index into workers, -1: nowhere)
    int n_resident = 0;          // (of those, the ones a worker computes in the card's VRAM)
    for (long long k = 0; k < n; ++k) {
        const std::string p = "j" + std::to_string(k);
        const long long rows = nrows[(size_t) k];
        const int32_t tg = (int32_t) line.num(p + "_tg", -1), td = (int32_t) line.num(p + "_td", -1);
        const size_t nq = (size_t) rows * (size_t) n_embd, nd = (size_t) rows * (size_t) (n_embd / 32);
        // the body is positional: every job's bytes are taken whatever this daemon can serve of it, or the
        // next job's would be read at this one's offsets
        xq[(size_t) k].resize(nq);
        xd[(size_t) k].resize(nd);
        wt[(size_t) k].resize((size_t) rows);
        if (!ctl.recv_body(xq[(size_t) k].data(), nq, err) ||
            !ctl.recv_body(xd[(size_t) k].data(), nd * sizeof(float), err) ||
            (has_w && !ctl.recv_body(wt[(size_t) k].data(), (size_t) rows * sizeof(float), err))) {
            return false;
        }
        if (!has_w) std::fill(wt[(size_t) k].begin(), wt[(size_t) k].end(), 1.0f);
        if (tg < 0 || td < 0) continue;   // (not a job this tier serves)
        // the three rows as this daemon holds them, and in which of its two tiers: a row it does not
        // hold (never hinted, or evicted with its arena), or one whose tier no worker can take, is the
        // caller's own work, and no byte of it moves
        const long long gri = line.num(p + "_gate", -1), uri = line.num(p + "_up", -1),
                        dri = line.num(p + "_down", -1);
        const void* gate = nullptr;
        const void* up = nullptr;
        const void* down = nullptr;
        uint32_t glen = 0, ulen = 0, dlen = 0;
        int resident = 0, w = -1;
        if (cache->host_row(gri, gate, glen) && cache->host_row(uri, up, ulen) &&
            cache->host_row(dri, down, dlen)) {
            resident = 0;   // (the pinned RAM slab: host bytes the worker copies into its own memory)
            if (!workers.empty()) w = w_rr++ % (int) workers.size();   // (any of them takes host bytes)
        } else {
            // a --vram row has no host window at all, so the worker is handed the slab's own device
            // address and computes it in place - on the card that allocated it and no other: the slab knows
            // its ordinal, the worker named its own in its version() line, and the two have to agree.
            // (a row on another card is the caller's own work: the address means nothing there)
            uint64_t ga = 0, ua = 0, da = 0;
            int gg = -1, ug = -1, dg = -1;
            if (!cache->dev_row(gri, ga, glen, gg) || !cache->dev_row(uri, ua, ulen, ug) ||
                !cache->dev_row(dri, da, dlen, dg) || gg != ug || gg != dg)
                continue;   // (not one slab's three rows: no worker reads across two cards)
            for (size_t i = 0; i < workers.size(); ++i)
                if (workers[i]->gpu() == gg) w = (int) i;
            if (w < 0) continue;   // (in VRAM, and no worker of this daemon computes on that card)
            gate = (const void*) (uintptr_t) ga;
            up = (const void*) (uintptr_t) ua;
            down = (const void*) (uintptr_t) da;
            resident = 1;
        }
        if (ulen != glen) continue;   // (one expert's gate and up are the same shape: a table says otherwise)
        if (w < 0) continue;
        y[(size_t) k].resize(nq);   // (the worker's own sums: written only for a job it serves)
        strata_peer_job& j = jobs[(size_t) k];
        j.n_embd = (int32_t) n_embd;
        j.n_ff = (int32_t) n_ff;
        j.tg = tg;
        j.td = td;
        j.n_rows = (int32_t) rows;
        // (the scanner keeps a number as its text: strtod reads the float the client wrote)
        j.limit = (float) std::strtod(line.str(p + "_lim", "0").c_str(), nullptr);
        j.gate = (const uint8_t*) gate;
        j.up = (const uint8_t*) up;
        j.down = (const uint8_t*) down;
        j.gate_bytes = glen;
        j.down_bytes = dlen;
        j.xq = xq[(size_t) k].data();
        j.xd = xd[(size_t) k].data();
        j.w = wt[(size_t) k].data();
        j.y = y[(size_t) k].data();
        j.resident = resident;
        wsel[(size_t) k] = w;
        at.push_back(k);
        if (resident != 0) ++n_resident;
    }
    std::vector<uint8_t> ok(at.size(), 0);
    // one batch a worker: the jobs it took, in the request's order (a worker that fails the batch answers 0
    // for every job of its own and the others go on - a card's worker is not the daemon)
    for (size_t wi = 0; wi < workers.size(); ++wi) {
        std::vector<strata_peer_job> batch;
        std::vector<size_t> whence;
        for (size_t i = 0; i < at.size(); ++i)
            if (wsel[(size_t) at[i]] == (int) wi) batch.push_back(jobs[(size_t) at[i]]), whence.push_back(i);
        if (batch.empty()) continue;
        std::vector<uint8_t> o(batch.size(), 0);
        std::string why;
        if (!workers[wi]->run(batch.data(), (int) batch.size(), o.data(), why)) say("compute: " + why);
        for (size_t i = 0; i < whence.size(); ++i) ok[whence[i]] = o[i];
    }
    // the answer: the request's own indices and one ok each (a job the worker refused, and every job of a
    // daemon with no worker at all, is 0 - the caller computes those itself), then the sums as one body in the
    // same order (41 KB where hex was 82 KB, and neither side pays a hex encode a buffer)
    Json out;
    out.add("type", "computed");
    std::vector<long long> idx, served;
    size_t done = 0, bytes = 0;
    for (size_t i = 0; i < at.size(); ++i) {
        idx.push_back(at[i]);
        served.push_back(ok[i] ? 1 : 0);
        if (ok[i] == 0) continue;
        ++done;
        bytes += y[(size_t) at[i]].size() * sizeof(float);
    }
    out.add("i", idx).add("ok", served).add("body", (long long) bytes);
    std::string body;
    body.reserve(bytes);
    for (size_t i = 0; i < at.size(); ++i) {
        if (ok[i] == 0) continue;
        const std::vector<float>& sum = y[(size_t) at[i]];
        body.append((const char*) sum.data(), sum.size() * sizeof(float));
    }
    if (!ctl.send(out.str(), err) || !ctl.send_body(body.data(), body.size(), err)) return false;
    say("compute: " + std::to_string(done) + " of " + std::to_string(n) + " jobs served (" +
        (!workers.empty() ? std::string("with ") + compute_caps(workers) : std::string("no worker, storage only")) +
        (n_resident != 0 ? ", " + std::to_string(n_resident) + " in VRAM" : "") +
        ", window " + std::to_string(hello_window) + ")");
    return true;
}

bool Session::serve(std::string& err) {
    if (!handshake(err)) return false;
    while (true) {
        const JsonView line(ctl.recv(-1, err));      // no timeout: the read loop parks in poll
        if (!line.ok()) {
            err = "session idle: " + err;
            return false;
        }
        if (line.type() == "hint") {
            say("hint in: " + std::to_string(line.nums("rows").size()) + " rows (window " +
                std::to_string(hello_window) + ")");
            if (!on_hint(line, err)) return false;
        } else if (line.type() == "compute") {
            say("compute in: " + std::to_string(line.num("n", 0)) + " jobs (window " +
                std::to_string(hello_window) + ")");
            if (!on_compute(line, err)) return false;
        } else if (line.type() == "ping") {
            if (!ctl.send(Json().add("type", "pong").str(), err)) return false;
        } else if (line.type() == "bye") {
            say("the client said bye");
            return true;
        } else {
            say("an unknown line of type " + line.type() + " ignored");
        }
    }
}

uint64_t clock_stamp = 0;            // (unused here; the session carries its own - kept for symmetry)

}   // namespace

}   // namespace strata::peer

// ------------------------------------------------------------------------------------- main

int main(int argc, char** argv) {
    std::string pack_dir = "pack", bind = "0.0.0.0", device = "";
    int port = 9447;
    long long ram_gb = 0, arena_mb = 0;
    std::vector<std::pair<int, uint64_t>> vram;      // <gpu ordinal>:<GiB>, in the order given
    std::vector<std::pair<std::string, std::string>> computes;   // --compute's .so, its own options (a card each)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            return i + 1 < argc ? argv[++i] : "";
        };
        if (a == "--pack") pack_dir = next();
        else if (a == "--bind") bind = next();
        else if (a == "--port") port = std::atoi(next().c_str());
        else if (a == "--ram-gb") ram_gb = std::atoll(next().c_str());
        else if (a == "--arena-mb") arena_mb = std::atoll(next().c_str());
        else if (a == "--device") device = next();
        else if (a == "--vram") {
            const std::string spec = next();
            const size_t colon = spec.find(':');
            if (colon == std::string::npos) {
                std::fprintf(stderr, "strata-peer: --vram wants <gpu>:<gb> (e.g. --vram 0:12), not \"%s\"\n",
                             spec.c_str());
                return 2;
            }
            const int gpu = std::atoi(spec.substr(0, colon).c_str());
            const double gb = std::atof(spec.substr(colon + 1).c_str());
            if (gpu < 0 || gb <= 0.0) {
                std::fprintf(stderr, "strata-peer: --vram %s names no GPU or no bytes\n", spec.c_str());
                return 2;
            }
            vram.emplace_back(gpu, (uint64_t) (gb * (double) (1ULL << 30)));
        }
        else if (a == "--compute") computes.emplace_back(next(), "");
        else if (a == "--compute-options") {
            if (computes.empty()) {
                std::fprintf(stderr, "strata-peer: --compute-options comes after the --compute it is for\n");
                return 2;
            }
            computes.back().second = next();
        }
        else if (a == "-h" || a == "--help") {
            std::printf("usage: strata-peer --pack <dir> [--bind <ip>] [--port 9447] [--ram-gb <n>] "
                        "[--arena-mb <n>] [--device <ibv device>] [--vram <gpu>:<gb>]... "
                        "[--compute <worker .so> [--compute-options <str>]]...\n");
            return 0;
        } else {
            std::fprintf(stderr, "strata-peer: unknown flag %s\n", a.c_str());
            return 2;
        }
    }
    std::string err;
    strata::peer::Device dev;       // ONE device for the whole daemon: the cache's arena MRs and
    if (!dev.open(device, err)) {   // every session's QP must share its PD, or the reads break
        strata::peer::say("no IB device: " + err);
        return 1;
    }
    strata::peer::PeerCache cache;
    // the arena allocation happens inside a request: log it (the first one pins GiBs and can take
    // a second - a hung-looking gap here is the arena being born, not a lost reply)
    strata::peer::PeerCache::set_alloc_log([](uint64_t bytes) {
        strata::peer::say("allocating a " + std::to_string(bytes >> 20) + " MiB arena (pinning, be patient)");
    });
    if (!cache.open(&dev, (uint64_t) ram_gb << 30, (uint64_t) arena_mb << 20, err)) {
        strata::peer::say(err);
        return 1;
    }
    // the VRAM tier, one slab per spec: allocated and registered here, so a card that cannot take it
    // (busy, absent, or a NIC that cannot read it) is a start-up error with the reason
    for (const auto& v : vram) {
        if (!cache.add_vram(v.first, v.second, err)) {
            strata::peer::say(err);
            return 1;
        }
    }
    // the compute workers, when asked for: one per --compute, each a card's (its options name the card).  A path
    // that does not load is a start-up error, and a daemon without one serves storage only - every compute order
    // then answers ok=0 and the caller computes it itself
    std::vector<std::unique_ptr<strata::peer::ComputeWorker>> workers;
    for (const auto& c : computes) {
        std::unique_ptr<strata::peer::ComputeWorker> w = strata::peer::make_compute_worker(c.first, c.second, err);
        if (w == nullptr) {
            strata::peer::say(err);
            return 1;
        }
        workers.push_back(std::move(w));
    }
    // the listener: one session at a time; a session's end keeps the slab for the next one
    addrinfo hint = {}, *res = nullptr;
    hint.ai_family = AF_INET;
    hint.ai_socktype = SOCK_STREAM;
    hint.ai_flags = AI_PASSIVE;
    if (getaddrinfo(bind.c_str(), std::to_string(port).c_str(), &hint, &res) != 0 || res == nullptr) {
        strata::peer::say(std::string("getaddrinfo bind failed: ") + gai_strerror(errno));
        return 1;
    }
    const int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    const int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (::bind(fd, res->ai_addr, res->ai_addrlen) != 0 || listen(fd, 4) != 0) {
        strata::peer::say(std::string("bind/listen failed: ") + strerror(errno));
        return 1;
    }
    freeaddrinfo(res);
    // (the workers as the welcome's field: their lines joined, so the client prints the daemon's own description)
    std::vector<strata::peer::ComputeWorker*> wp;
    for (const auto& w : workers) wp.push_back(w.get());
    const std::string wcaps = strata::peer::compute_caps(wp);
    strata::peer::say("strata-peer listening on " + bind + ":" + std::to_string(port) +
                      ", pack " + pack_dir +
                      (vram.empty() ? "" : ", " + std::to_string(vram.size()) + " VRAM slab(s)") +
                      (wcaps.empty() ? "" : ", compute " + wcaps));
    while (true) {
        // the previous session's qp async errors (a NAK'd read, a transport failure) surface here:
        // the device's async queue drains between connections
        pollfd ap = { dev.ctx()->async_fd, POLLIN, 0 };
        while (poll(&ap, 1, 0) > 0) {
            ibv_async_event ae = {};
            if (ibv_get_async_event(dev.ctx(), &ae) != 0) break;
            strata::peer::say(std::string("device async event: ") +
                              ibv_event_type_str(ae.event_type));
            ibv_ack_async_event(&ae);
            ap = { dev.ctx()->async_fd, POLLIN, 0 };
        }
        const int conn = accept(fd, nullptr, nullptr);
        if (conn < 0) continue;
        strata::peer::Session s;
        s.cache = &cache;
        s.workers = wp;
        s.pack_dir = pack_dir;
        s.dev = &dev;
        if (!s.ctl.adopt(conn, err)) continue;
        if (s.serve(err)) {
            strata::peer::say("session closed cleanly, " +
                              std::to_string(cache.cached_bytes() >> 20) + " MiB kept warm");
        } else {
            strata::peer::say("session ended (window " + std::to_string(s.hello_window) + "): " + err);
        }
        s.qp.destroy();
        s.ctl.close();
    }
}
