// The initiator's peer session: the control channel's client half, the row table it hands over, the
// QP it brings up against the daemon's, and the map of advertised rows (row -> addr/len/rkey) the
// READs go through.  Phase 3's engine tier wraps exactly this class; the test binary drives it
// directly.
//
// One session at a time, and a failed row fails the session: a completion error leaves the QP in an
// undefined state, so the honest recovery is close + reconnect (the daemon keeps its slab, the map
// is re-advertised on hello), never a retry through a half-dead QP.
#pragma once

#include "peer_cache.hpp"     // (Row; a daemon type, but the row table's wire format is shared)
#include "peer_ctl.hpp"
#include "peer_read.hpp"
#include "peer_rdma.hpp"

#include <unordered_map>

namespace strata::peer {

class PeerClient {
public:
    /// window/chunk/timeout as Reader's (env MAYA_PEER_* defaults, see the daemon's flags).
    PeerClient(int window = 48, size_t chunk = 2ULL << 20, int timeout_ms = 400,
               const std::string& device = "");
    ~PeerClient();
    PeerClient(const PeerClient&) = delete;
    PeerClient& operator=(const PeerClient&) = delete;

    /// Full handshake against host:port: hello, manifest (names/sizes plus the fingerprint of the
    /// initiator's own copy under pack_dir; the daemon refuses a mismatch), the row table, the QP
    /// exchange.  On success the map is whatever the daemon advertised at hello (empty until the
    /// first hints).
    bool connect(const std::string& host, int port, const std::string& pack_dir,
                 const std::vector<std::string>& shard_names, const std::vector<uint64_t>& shard_sizes,
                 const std::vector<Row>& rows, std::string& err);
    void close();
    bool is_up() const { return up_; }

    /// Registers a destination buffer (the rows land here; the engine's pinned staging in phase 3).
    bool add_dst(void* buf, size_t bytes, std::string& err);
    /// This session's PD - a dst MR must be registered on IT (a second Device's PD registers an
    /// lkey the QP cannot use: the READ's local sink then fails with local protection error)
    ibv_pd* pd() const;
    /// The i-th registered destination (add_dst's order), for a batched read's RowRead.
    Mr* dst_mr(size_t i) const { return i < dst_.size() ? dst_[i].get() : nullptr; }
    void clear_dst();

    /// Hints rows (the daemon faults them in and advertises them); merges the answer into the map
    /// and applies any drops.  Call before reading rows the map does not have yet.
    bool hint(const std::vector<long long>& rows, std::string& err, int timeout_ms = -1);

    /// RDMA READs row `ri` into dst_mr at dst_off.  False (with the session closed) when the row
    /// is unadvertised - the caller falls back to its disk tier - or the read failed.
    bool read_row(long long ri, Mr& dst, uint64_t dst_off, std::string& err);

    /// One row's destination: `dst_mr` (a registered destination from add_dst) at `dst_off`.  A
    /// sub-range (`rel`, `bytes`) pulls only part of the row - the DS4 flow's tensor-parallel member
    /// reads a slice of an expert's matrix - and is bounds-checked against the advertised row.
    struct RowRead {
        long long ri = 0;
        Mr* dst = nullptr;
        uint64_t dst_off = 0;
        uint64_t rel = 0;    ///< from the row's start (0: the whole row)
        uint32_t bytes = 0;  ///< 0: the rest of the row from `rel`
    };
    /// **A LAYER'S ROWS IN ONE WINDOW** (the decode path's shape: read_row's one-row-at-a-time is a
    /// round trip per part, and an expert's parts are only a MB or two).  All or nothing: every row
    /// must still be advertised, else false before anything is posted.  A failed batch closes the
    /// session - the caller falls back to its disk tier for the whole batch.
    bool read_rows(const std::vector<RowRead>& reqs, std::string& err);
    bool advertised(long long ri) const { return map_.count(ri) != 0; }

    /// One expert's compute order (docs/PEER_COMPUTE.md, "The protocol"): the daemon's rows for its
    /// gate, up and down, and the activation the three of them share.  `n_rows` activation rows, each
    /// `n_embd` int8 values with a scale a 32 (`xd`, n_rows x n_embd / 32 floats) - the device kernels'
    /// rounding, so the peer's worker re-quantizes once to the weight type's own form.  `w` (n_rows routing
    /// weights) makes the peer apply the weighted sum the caller would otherwise do; null: weight 1.
    struct ComputeReq {
        long long gate = -1;
        long long up = -1;
        long long down = -1;
        int n_embd = 0;
        int n_ff = 0;
        int tg = 0;              ///< the gate/up and down ggml types (the daemon's iq_cpu_fmt)
        int td = 0;
        float limit = 0.0f;       ///< swiglu's clamp (0: none)
        int n_rows = 1;
        const int8_t* xq = nullptr;
        const float* xd = nullptr;
        const float* w = nullptr;
    };

    /// **ONE LAYER'S PICKS IN ONE MESSAGE** (the same batching the reads get, and for the same reason:
    /// the cost of a message is its latency, not its bytes).  `ok[k]` is 1 for a job the peer computed and
    /// 0 for one it cannot serve - a row it does not hold, a geometry its worker refuses, or every job of a
    /// daemon started without --compute - and `y[k]` then holds the weighted sum (n_rows x n_embd floats).
    /// The caller computes the zeros itself.  False (with the session closed) only when the wire failed.
    bool compute(const std::vector<ComputeReq>& jobs, std::vector<char>& ok,
                 std::vector<std::vector<float>>& y, std::string& err);

    /// Liveness: ping the daemon, refresh the map (drops included).  Cheap; the engine calls it
    /// between requests, the test at its own pace.
    bool keepalive(std::string& err);

    uint64_t rows_read() const { return reader_ ? reader_->rows() : 0; }
    uint64_t bytes_read() const { return reader_ ? reader_->bytes_read() : 0; }
    uint64_t us_read() const { return reader_ ? reader_->us() : 0; }
    size_t advertised_rows() const { return map_.size(); }

    /// The bytes the daemon can hold at all (its RAM slab plus every --vram slab), 0 when its welcome did not say.
    /// The tier's resident hint is capped by it.
    uint64_t capacity() const { return capacity_; }
    /// The peer's --vram slabs in all (0: no card on the peer's side of this tier): the rows the resident hint
    /// fills go into those, since a compute worker reads a --vram row in place.
    uint64_t vram_capacity() const { return vram_; }
    /// The compute worker's own version() line as the daemon reported it ("" when it has none): a compute order is
    /// only worth sending when this names one (peer_daemon.cpp's welcome).
    const std::string& compute_caps() const { return compute_caps_; }

private:
    bool handshake(const std::string& host, int port, const std::vector<std::string>& shard_names,
                   const std::vector<uint64_t>& sizes,
                   const std::vector<Row>& rows, std::string& err);

    int window_, timeout_ms_;
    size_t chunk_;
    std::string device_;
    Device dev_;
    Ctl ctl_;
    Qp qp_;
    std::unique_ptr<Reader> reader_;
    std::vector<std::unique_ptr<Mr>> dst_;       // registered destination buffers
    std::unordered_map<long long, Cached> map_;  // the daemon's advertisement, as received
    std::vector<Row> rows_;
    std::string pack_dir_;           // the initiator's own copy: fingerprinted for the manifest
    bool up_ = false;
    uint64_t capacity_ = 0;          // (the daemon's welcome: what it can hold, and its worker's caps)
    uint64_t vram_ = 0;              // (of that, its --vram slabs)
    std::string compute_caps_;
};

}   // namespace strata::peer
