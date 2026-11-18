// Phase 3's peer tier, engine side.  See glm_peer_tier.hpp for the contract; this file is the two
// pieces of state the contract needs - the row table the daemon holds a copy of, and the set of
// local buffers a READ may land in - plus the counters the tiers line reports.
#include "glm_peer_tier.hpp"

#include "peer_cache.hpp"
#include "peer_client.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>

namespace strata::core {
namespace glmfast {

namespace {

/// The process-wide session.  Deliberately leaked: a split's parts each point at it, the daemon
/// serves one initiator, and tearing it down at static-destruction time (after the verbs device
/// has been closed by some other static's destructor) is a crash for no gain - the socket closes
/// when the process does, which is when the daemon should let go anyway.
std::mutex g_open_mu;
PeerTier* g_session = nullptr;
std::string g_open_err;
bool g_tried = false;
bool g_reported = false;

}  // namespace

/// `PeerClient` and the row table, behind a pointer so this header stays free of <infiniband/verbs.h>
/// (the engine's other translation units must not see it: it shadows nothing, it just does not belong).
struct PeerTier::Impl {
    explicit Impl(const Config& c) : cli(c.window, (size_t) c.chunk_mb << 20, c.timeout_ms, c.device) {}
    strata::peer::PeerClient cli;
    std::vector<strata::peer::Row> table;
};

PeerTier* PeerTier::open_session(const Config& cfg, std::string& err) {
    std::lock_guard<std::mutex> lk(g_open_mu);
    if (g_tried) {                 // one attempt a process: a retry storm during a load helps nobody
        if (g_session == nullptr) err = g_open_err;
        return g_session;
    }
    g_tried = true;
    if (cfg.host.empty() || cfg.n_layers <= 0 || cfg.n_expert <= 0 ||
        cfg.specs.size() != (size_t) (cfg.n_layers * kRoles)) {
        g_open_err = "the peer tier was asked to open with no session settings";
        err = g_open_err;
        return nullptr;
    }
    // THE ROW TABLE: one row per (layer, expert, role), the exact slice fast_read_part would read.
    // A layer with no native experts (a dense lead, an EXL3 pack) keeps every row's length at 0 and
    // is never hinted, so the table covers every layer of the model and the arithmetic stays flat:
    // row = (layer * n_expert + expert) * 3 + role.
    const int64_t n_rows = cfg.n_layers * cfg.n_expert * kRoles;
    if (n_rows <= 0 || n_rows > (int64_t) 1 << 22) {
        g_open_err = "a peer row table of " + std::to_string(n_rows) + " rows is not a pack";
        err = g_open_err;
        return nullptr;
    }
    PeerTier* t = new PeerTier();
    t->n_layers_ = cfg.n_layers;
    t->n_expert_ = cfg.n_expert;
    t->len_of_.assign((size_t) n_rows, 0);
    t->served_layer_.assign((size_t) cfg.n_layers, 0);
    t->impl_.reset(new Impl(cfg));
    t->impl_->table.assign((size_t) n_rows, strata::peer::Row{0, 0, 0});
    for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
        const PeerRowSpec& r0 = cfg.specs[(size_t) (layer * kRoles)];
        if (r0.shard < 0) continue;              // not served (dense, or a layout the tier leaves out)
        t->served_layer_[(size_t) layer] = 1;
        for (int role = 0; role < kRoles; ++role) {
            const PeerRowSpec& s = cfg.specs[(size_t) (layer * kRoles + role)];
            for (int64_t e = 0; e < cfg.n_expert; ++e) {
                const size_t row = (size_t) ((layer * cfg.n_expert + e) * kRoles + role);
                t->impl_->table[row] = strata::peer::Row{(uint16_t) s.shard, s.off + (uint64_t) e * s.stride, s.len};
                t->len_of_[row] = s.len;
            }
        }
    }
    std::string why;
    if (!t->impl_->cli.connect(cfg.host, cfg.port, cfg.pack_dir, cfg.shard_names, cfg.shard_sizes,
                               t->impl_->table, why)) {
        g_open_err = why;
        err = why;
        delete t;
        return nullptr;
    }
    t->up_ = true;
    g_session = t;
    std::fprintf(stderr, "glm peer: session up with %s:%d - %lld rows of %lld layers x %lld experts over %s\n",
                 cfg.host.c_str(), cfg.port, (long long) n_rows, (long long) cfg.n_layers,
                 (long long) cfg.n_expert, cfg.device.empty() ? "the first ibverbs device" : cfg.device.c_str());
    return t;
}

bool PeerTier::register_dst(void* base, size_t bytes) {
    if (base == nullptr || bytes == 0 || !up_) return false;
    std::lock_guard<std::mutex> lk(mu_);
    std::string err;
    if (!impl_->cli.add_dst(base, bytes, err)) {
        std::fprintf(stderr, "glm peer: a %.1f MB destination did not register (%s) - its experts stay on the\n"
                             "           disk path\n", (double) bytes / 1048576.0, err.c_str());
        return false;
    }
    Region r;
    r.base = (uint8_t*) base;
    r.bytes = bytes;
    r.mr = impl_->cli.dst_mr(regions_.size());
    regions_.push_back(r);
    return true;
}

long long PeerTier::row_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return -1;
    if (served_layer_[(size_t) layer] == 0) return -1;
    return (layer * n_expert_ + expert) * kRoles;
}

bool PeerTier::region_of(const void* p, size_t bytes, void*& mr, uint64_t& off) const {
    const uintptr_t a = (uintptr_t) p;
    for (const Region& r : regions_) {
        const uintptr_t b = (uintptr_t) r.base;
        if (a >= b && a + bytes <= b + r.bytes) {
            mr = r.mr;
            off = (uint64_t) (a - b);
            return true;
        }
    }
    return false;
}

void PeerTier::fail(const std::string& why) {
    if (!up_) return;
    up_ = false;
    why_ = why;
    // Said loudly and once: from here every miss is a disk read again (the decode is unchanged, only
    // slower), and a silent regression to the slow path is the one thing this tier must not do.
    std::fprintf(stderr, "glm peer: THE PEER IS DOWN (%s) - every miss is a local disk read from here on\n",
                 why.c_str());
}

void PeerTier::serve(int n, const long long* rows, uint8_t* const* dst, char* ok) {
    if (ok == nullptr || n <= 0) return;
    std::memset(ok, 0, (size_t) n);
    std::lock_guard<std::mutex> lk(mu_);
    if (!up_) return;
    // a layer's misses are at most kSpares experts, three parts each; anything past the array takes
    // the disk path for its tail rather than growing this hot path
    constexpr int kMax = 32;
    if (n > kMax) n = kMax;
    const auto t0 = std::chrono::steady_clock::now();
    std::string err;
    void* mr[kMax];
    uint64_t off[kMax];
    long long hint[kMax];
    bool valid[kMax];
    int want = 0;
    // 1. which rows can be pulled at all: the destination must lie in a registered piece (a RAM-tier
    //    block or a staging buffer), and a row a dense layer has no bytes for is never asked
    for (int k = 0; k < n; ++k) {
        valid[k] = false;
        const long long ri = rows[k];
        if (ri < 0 || (size_t) ri >= len_of_.size() || len_of_[(size_t) ri] == 0) continue;
        void* m = nullptr;
        uint64_t o = 0;
        if (!region_of(dst[k], len_of_[(size_t) ri], m, o)) continue;
        valid[k] = true;
        mr[k] = m;
        off[k] = o;
        if (!impl_->cli.advertised(ri)) hint[want++] = ri;
    }
    // 2. ONE hint round trip for everything the daemon has not faulted in yet: a row it already holds
    //    replies instantly, a cold one costs its pread on the peer's disk, and either way the reply
    //    advertises the rows, so the batch below sees a settled map
    if (want > 0) {
        ++hints_;
        if (!impl_->cli.hint(std::vector<long long>(hint, hint + want), err)) {
            fail("the hint failed: " + err);
            return;
        }
    }
    // 3. one window for the whole batch (peer_read's read_many: the layer's parts interleaved, not one
    //    round trip each)
    std::vector<strata::peer::PeerClient::RowRead> reqs;
    std::vector<int> at;
    reqs.reserve((size_t) n);
    at.reserve((size_t) n);
    for (int k = 0; k < n; ++k) {
        if (!valid[k]) continue;
        if (!impl_->cli.advertised(rows[k])) {
            ++unadvertised_;                     // (filled and dropped again: the caller's disk answers)
            continue;
        }
        reqs.push_back(strata::peer::PeerClient::RowRead{rows[k], (strata::peer::Mr*) mr[k], off[k]});
        at.push_back(k);
    }
    if (!reqs.empty()) {
        if (!impl_->cli.read_rows(reqs, err)) {
            fail("a batch read failed: " + err);
            return;
        }
        for (size_t j = 0; j < reqs.size(); ++j) ok[at[j]] = 1;
        ++batches_;
        rows_read_ += reqs.size();
        for (const auto& q : reqs) bytes_ += len_of_[(size_t) q.ri];
    }
    us_ += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - t0).count();
}

void PeerTier::report() {
    std::lock_guard<std::mutex> lk(mu_);
    if (g_reported) return;
    g_reported = true;
    std::fprintf(stderr,
                 "glm peer: %llu rows (%.1f GB) pulled in %llu batches (%.1f GB/s on the wire), %llu hints, "
                 "%llu rows the daemon did not hold%s\n",
                 (unsigned long long) rows_read_, (double) bytes_ / 1073741824.0,
                 (unsigned long long) batches_,
                 us_ > 0 ? (double) bytes_ / ((double) us_ / 1e6) / 1073741824.0 : 0.0,
                 (unsigned long long) hints_, (unsigned long long) unadvertised_,
                 up_ ? "" : " - the session is DOWN");
}

}  // namespace glmfast
}  // namespace strata::core
