// src/ds4/ds4_peer_tier.hpp - Phase 3's peer expert tier for the DeepSeek engine: an expert the flow would read
// from the model's files is pulled from the OTHER machine's pinned slab over RoCE instead (`strata-peer --pack ...`
// running there).
//
// WHAT THIS REPLACES.  The flow's file items (flow_pump, the plain file branch) are read unbuffered off the local
// disk into their staging slot by IoQueue.  This tier offers the same bytes from a peer's registered arena: its NIC
// answers the READs, so the peer's disk and CPU never enter the decode.  The bytes are identical by construction - a
// row is a shard byte range the initiator named, and the handshake refuses a pack whose fingerprint differs - so a
// token computed over the peer is the token the disk would have produced.
//
// EVERYTHING HERE IS OPTIONAL.  A destination the NIC would not register, a row the daemon does not advertise (it
// never filled it, or an arena was evicted), a session that just died: the order is handed to the local IoQueue as
// exactly the job it stands in for, so the flow never learns this tier exists.  Nothing here may turn a working
// decode into a failing one.
//
// THE ROW TABLE.  One row per (layer, expert, role), row = base[layer] + expert * 3 + role, where base is the prefix
// sum over the layers of their own expert count (DeepSeek's MoE layers need not all route to the same number, unlike
// GLM's fast path).  A layer the tier does not serve - a dense one, or every one of its shards is on this machine's
// disk rather than the peer's - keeps row_of() at -1 and is read locally as before.
//
// TWO OBJECTS.  PeerTier is the process-wide session (a split's parts share one QP; a chunk's worker the only
// poster at a time), deliberately leaked like glmfast::PeerTier.  PeerQueue is one flow's thread: the flow pushes
// orders from flow_pump, the thread pulls them from the peer in one window and accounts them down the same way
// IoQueue counts a job (left / bad), so Chunk::io and Chunk::peerq are interchangeable to the pump.
#pragma once

#include "ds4_pool.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::ds4 {

/// One slice of the pack, per (layer, role): expert e's part is [off + e * stride, + len) of shard `shard`, which is
/// an index into the manifest's names (the subset of the pack's files the peer holds).  `shard` < 0 or `len` 0: this
/// layer is not served and every expert of it stays on the local disk path.
struct PeerRowSpec {
    int32_t shard = -1;
    uint64_t off = 0;
    uint64_t stride = 0;
    uint32_t len = 0;
};

/// A layer's three slices (gate, up, down - `Layer::mx_file` / `mx_off` / `mx_stride` as the flow reads them) and
/// the number of experts they hold.  `n_exp` 0: not a MoE layer, no rows.
struct PeerLayerSpec {
    int64_t n_exp = 0;
    PeerRowSpec part[3];
};

class PeerTier {
public:
    static constexpr int kRoles = 3;   ///< gate, up, down - the flow's three parts of an expert

    /// The session's settings.  The environment (`STRATA_DS4_PEER*`) fills what the caller leaves empty.
    struct Config {
        std::string host;                ///< the peer's control address (its IPv4 on the RoCE link)
        int port = 9447;
        std::string device;              ///< the local ibverbs device name ("" = the first one found)
        std::string pack_dir;            ///< the LOCAL copy of the pack: fingerprinted, never read here
        std::vector<std::string> shard_names;    ///< the pack files the PEER holds (its manifest)
        std::vector<uint64_t> shard_sizes;
        std::vector<PeerLayerSpec> layers;
        int window = 48;                 ///< outstanding 2 MB chunks (the daemon sizes its QP to it)
        int chunk_mb = 2;
        int timeout_ms = 1000;           ///< one batch's deadline
        /// (--peer-compute) THE COMPUTE TIER: the peer computes the experts it holds instead of only serving their
        /// bytes (docs/PEER_COMPUTE.md).  Off unless asked, and even then an order is only sent for an expert whose
        /// three parts this session knows the peer holds (PeerTier::computable).
        bool compute = false;
        /// THE ROWS KEPT RESIDENT AT HELLO, in the order they are wanted (the engine's most-routed experts first):
        /// the tier hints them once the session is up, truncated where the peer's slab runs out - a row the peer
        /// holds without a read is a row a compute order can be sent for before any read has hinted it.
        std::vector<long long> resident;
        /// How long a resident fill may take on the peer's own disk (a 24 GB fill is not a read's timeout).
        int resident_timeout_ms = 600000;
    };

    /// Opens the process-wide session (the first flow opens it; the others attach).  nullptr on refusal, with `err`
    /// saying why - the engine reports it and runs on the local disks.
    static PeerTier* open_session(const Config& cfg, std::string& err);

    /// Registers `[base, base + bytes)` as a READ destination (a flow's staging slots).  false when the NIC refuses
    /// it: that flow then keeps every read on its own disk.
    bool register_dst(void* base, size_t bytes);

    /// The row of (layer, expert, `role`) - gate at role 0, up 1, down 2 - or -1 when the tier does not serve that
    /// slice (a dense layer, an expert past its count, or its shard lives on this machine's disk and not the peer's).
    long long row_of(int64_t layer, int64_t expert, int role) const;

    /// One batch's pull, all through one window: `rows[k]`'s bytes [rel[k], rel[k] + bytes[k]) into `dst[k]` (the
    /// exact place the local read would have written).  `ok[k]` is 1 for a landed row and 0 for one the caller must
    /// read itself.  Never throws, never blocks past the batch timeout, serialised across the flows.
    void serve(int n, const long long* rows, const uint64_t* rel, const size_t* bytes, uint8_t* const* dst, char* ok);

    /// One batch of compute orders (docs/PEER_COMPUTE.md, stage 1): `rows[3k]`, `[3k+1]`, `[3k+2]` are order k's gate, up
    /// and down rows (this tier's row_of), `xq[k]` its activation - n_rows tokens of n_embd int8 values with a scale a
    /// 32 (`xd[k]`, n_rows x n_embd / 32 floats) - `w[k]` the routing weights, and `y[k]` (n_rows x n_embd floats)
    /// gets the weighted sum.  `ok[k]` is 1 for an order the peer computed and 0 for one the caller must compute
    /// itself: a row the peer does not hold (this tier never hints here - residency is decided at hello), a geometry its
    /// worker refuses, or every order of a session that is down.  Residency is checked before a byte moves, exactly as
    /// serve's is, and the same `fail()` / THE PEER IS DOWN contract covers a wire that dies under the batch.
    void compute(int n, const long long* rows, int n_rows, int n_embd, int n_ff, int tg, int td, float limit,
                 const int8_t* const* xq, const float* const* xd, const float* const* w, float* const* y, char* ok);

    /// The same batch with a row count A JOB (`n_rows[k]`): a layer's experts are picked by a different number of rows
    /// (one a token in a decode, more in a verify window), and one order carries one expert's rows alone.
    void compute_rows(int n, const long long* rows, const int* n_rows, int n_embd, int n_ff, int tg, int td, float limit,
                      const int8_t* const* xq, const float* const* xd, const float* const* w, float* const* y, char* ok);

    bool up() const { return up_; }

    /// Whether compute orders may be sent at all: --peer-compute was asked for, the daemon welcomed a worker, and
    /// the session is up.
    bool compute_on() const { return compute_ && up_; }

    /// Whether the peer holds all three of (layer, expert)'s parts right now - a resident one, or one a read has
    /// already hinted.  The flow's test before it leaves an expert's bytes unread: true is a promise that a compute
    /// order for it will be served.
    bool computable(int64_t layer, int64_t expert) const;

    /// The rows the peer reported holding after the resident hint (0: it kept none).
    size_t resident_rows() const { return resident_rows_; }

    /// The compute worker the daemon welcomed ("" when its welcome named none): the daemon's own worker line.
    const std::string& compute_caps() const { return compute_caps_; }

    /// The tiers line's peer counters, said once a process.
    void report();

private:
    PeerTier() = default;
    bool region_of(const void* p, size_t bytes, void*& mr, uint64_t& off) const;
    void fail(const std::string& why);

    struct Region {
        uint8_t* base = nullptr;
        size_t bytes = 0;
        void* mr = nullptr;   ///< the PeerClient's Mr (kept void here: the verbs types stay in the .cpp)
    };

    struct Impl;   // (the session lives behind this: the header stays free of <infiniband/verbs.h>)
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mu_;
    std::vector<Region> regions_;
    std::vector<long long> base_of_;   ///< layer -> its first row (-1: not served)
    std::vector<int64_t> n_exp_of_;    ///< layer -> its experts (0: no rows, every role on the disk path)
    std::vector<uint32_t> len_of_;     ///< row -> its length (0: never served)
    int64_t n_layers_ = 0;
    bool up_ = false;
    bool compute_ = false;              ///< (--peer-compute, and a worker the daemon welcomed)
    size_t resident_rows_ = 0;          ///< (what the resident hint left held)
    std::string compute_caps_;          ///< (the daemon's worker line, "" when it has none)
    // stats (under mu_)
    uint64_t rows_read_ = 0, bytes_ = 0, hints_ = 0, unadvertised_ = 0, batches_ = 0, us_ = 0;
    uint64_t computed_ = 0, compute_orders_ = 0, compute_us_ = 0;
    std::string why_;
};

/// One flow's end of the tier: a thread that takes the orders flow_pump would have pushed to IoQueue, pulls them
/// from the peer, and hands every one the wire did not deliver to `local` as the job it is.  Lives beside its
/// Chunk's IoQueue (declared after it, so the thread is joined before that queue goes).
class PeerQueue {
public:
    PeerQueue(PeerTier& tier, IoQueue& local);
    ~PeerQueue();   // (joins: the session itself is process-wide and stays)

    /// A job the peer may serve, and where its bytes are: `job` is exactly what IoQueue would read (its `dst`, `off`,
    /// `len`, `fd` / `dfd`, `left`, `bad`, and the DS4_FLOW_DOWN_PART compaction), `row` / `rel` / `bytes` name the
    /// pack range over the wire.  `row` < 0 or a destination outside a registered piece: read from `local` instead.
    struct Order {
        IoQueue::Job job;
        long long row = -1;
        uint64_t rel = 0;
        size_t bytes = 0;
    };
    void push(const Order* o, int n);

    PeerTier& tier() { return tier_; }
    /// The tier's row for this (layer, expert, role), -1 when the peer does not serve it: the caller then pushes the
    /// job to its own queue.
    long long row_of(int64_t layer, int64_t expert, int role) { return tier_.row_of(layer, expert, role); }
    uint64_t orders() const { return taken_.load(std::memory_order_relaxed); }

private:
    void loop();
    void drain(std::vector<Order>& batch);

    PeerTier& tier_;
    IoQueue& local_;
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Order> q_;
    bool quit_ = false;
    std::atomic<uint64_t> taken_{0};
};

}  // namespace strata::ds4
