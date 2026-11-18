#include "peer_read.hpp"

#include <algorithm>
#include <cerrno>
#include <time.h>

#include <cstring>

namespace strata::peer {
namespace {

int64_t now_ms() {
    timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

}   // namespace

Reader::Reader(ibv_qp* qp, ibv_cq* cq, int window, size_t chunk)
    : qp_(qp), cq_(cq), window_(window), chunk_(chunk) {
    inflight_.resize((size_t) window * 2 + 8);
}

bool Reader::poll_once(int64_t deadline_ms, std::string& err) {
    ibv_wc wc[8];
    const int n = ibv_poll_cq(cq_, 8, wc);
    if (n < 0) {
        err = "ibv_poll_cq failed";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            err = std::string("read completion failed: ") + ibv_wc_status_str(wc[i].status) +
                  " (opcode " + std::to_string(wc[i].opcode) + ")";
            return false;
        }
        if (wc[i].wr_id >= inflight_.size() || !inflight_[wc[i].wr_id].posted) {
            err = "a completion for an unknown wr_id " + std::to_string(wc[i].wr_id);
            return false;
        }
        inflight_[wc[i].wr_id].posted = false;
        --outstanding_;
    }
    if (n == 0 && now_ms() > deadline_ms) {
        err = "read timed out past the deadline";
        return false;
    }
    return true;
}

bool Reader::read(uint64_t src_addr, uint32_t src_rkey, ibv_mr* dst_mr, uint64_t dst_off, size_t bytes,
                  int timeout_ms, std::string& err) {
    std::vector<ReadReq> one(1);
    one[0] = ReadReq{src_addr, src_rkey, dst_mr, dst_off, bytes};
    return read_many(one, timeout_ms, err);
}

bool Reader::read_many(const std::vector<ReadReq>& reqs, int timeout_ms, std::string& err) {
    if (reqs.empty()) return true;
    const int64_t t0 = now_ms();
    const int64_t deadline = t0 + timeout_ms;
    // the flat chunk list: every request's chunks in request order.  Posting interleaves them, so a
    // layer's twenty-four small parts ride one window - the whole point of batching (a part is one to
    // a few MB, so a row at a time leaves the window nearly idle and pays a round trip each)
    struct Chunk {
        size_t req;
        size_t at;
        size_t len;
    };
    std::vector<Chunk> chunks;
    size_t total = 0;
    for (size_t r = 0; r < reqs.size(); ++r) {
        const size_t bytes = reqs[r].bytes;
        total += bytes;
        for (size_t at = 0; at < bytes; at += chunk_) chunks.push_back({r, at, std::min(chunk_, bytes - at)});
    }
    if (!chunks.empty()) {
        if (inflight_.size() < chunks.size()) inflight_.resize(chunks.size());
        ibv_sge sge = {};                   // the driver copies the WR and its sge at post time, so
        size_t next = 0;                    // one reusable struct serves every chunk
        while (next < chunks.size()) {
            // keep the window: post up to it, then drain - the ENOSPC guard
            while (next < chunks.size() && outstanding_ < window_) {
                const Chunk& c = chunks[next];
                const ReadReq& q = reqs[c.req];
                sge.addr = (uint64_t) (uintptr_t) q.dst_mr->addr + q.dst_off + c.at;
                sge.length = (uint32_t) c.len;
                sge.lkey = q.dst_mr->lkey;
                ibv_send_wr wr = {};
                wr.wr_id = next;
                wr.opcode = IBV_WR_RDMA_READ;
                wr.sg_list = &sge;
                wr.num_sge = 1;
                wr.send_flags = IBV_SEND_SIGNALED;   // one completion per chunk keeps the ring exact
                wr.wr.rdma.remote_addr = q.src_addr + c.at;
                wr.wr.rdma.rkey = q.src_rkey;
                ibv_send_wr* bad = nullptr;
                if (ibv_post_send(qp_, &wr, &bad) != 0) {
                    err = "ibv_post_send failed (errno " + std::to_string(errno) + ", chunk " +
                          std::to_string(next) + " of " + std::to_string(chunks.size()) + ")";
                    return false;
                }
                inflight_[next] = {true};
                ++outstanding_;
                ++next;
            }
            if (!poll_once(deadline, err)) return false;
        }
        while (outstanding_ > 0) {
            if (!poll_once(deadline, err)) return false;
        }
    }
    rows_ += reqs.size();
    bytes_ += total;
    us_ += (uint64_t) (now_ms() - t0);
    return true;
}

}  // namespace strata::peer
