// Phase 3's peer tier, DeepSeek engine side.  See ds4_peer_tier.hpp for the contract; this file is the session the
// row table describes, the set of staging buffers a READ may land in, and the flow's thread that keeps a batch of
// orders on the wire - plus the counters the tier reports once.
#include "ds4_peer_tier.hpp"

#include "ds4_os.hpp"

#include <cstring>

#if defined(STRATA_HAVE_PEER)   // (a build without ibverbs: the stubs at the bottom, and the tier says it is off)
#include "peer_cache.hpp"
#include "peer_client.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>

namespace strata::ds4 {

namespace {
/// The process-wide session.  Deliberately leaked, as glmfast::PeerTier's: every flow of the process points at it,
/// the daemon serves one initiator, and tearing it down at static-destruction time (after the verbs device has been
/// closed by some other static's destructor) is a crash for no gain - the socket closes when the process does.
std::mutex g_open_mu;
PeerTier* g_session = nullptr;
std::string g_open_err;
bool g_tried = false;
bool g_reported = false;
}  // namespace

/// `PeerClient` and the row table it published, behind a pointer so this header stays free of
/// <infiniband/verbs.h>.  mu_ serialises the flows' serves (one QP, one link).
struct PeerTier::Impl {
    explicit Impl(const Config& c) : cli(c.window, (size_t) c.chunk_mb << 20, c.timeout_ms, c.device) {}
    strata::peer::PeerClient cli;
    std::vector<strata::peer::Row> table;
};

PeerTier* PeerTier::open_session(const Config& cfg, std::string& err) {
    std::lock_guard<std::mutex> lk(g_open_mu);
    if (g_tried) {   // one attempt a process: a retry storm during a load helps nobody
        if (g_session == nullptr) err = g_open_err;
        return g_session;
    }
    g_tried = true;
    if (cfg.host.empty() || cfg.layers.empty()) {
        g_open_err = "the peer tier was asked to open with no session settings";
        err = g_open_err;
        return nullptr;
    }
    // THE ROW TABLE: one row per (layer, expert, role), laid out layer by layer so a served layer's rows are a
    // run - row = base_of_[layer] + expert * 3 + role.  A layer with no experts (a dense one) or with a role whose
    // shard this tier does not host keeps its rows' lengths at 0, so they are never hinted and never read over the
    // wire; the flow reads those parts off its own disk, as it would without the tier.
    std::vector<PeerLayerSpec> specs = cfg.layers;
    int64_t n_rows = 0;
    for (const PeerLayerSpec& l : specs)
        if (l.n_exp > 0) n_rows += l.n_exp * kRoles;
    if (n_rows <= 0 || n_rows > (int64_t) 1 << 22) {
        g_open_err = "a peer row table of " + std::to_string(n_rows) + " rows is not a pack";
        err = g_open_err;
        return nullptr;
    }
    PeerTier* t = new PeerTier();
    t->n_layers_ = (int64_t) specs.size();
    t->base_of_.assign(specs.size(), -1);
    t->n_exp_of_.assign(specs.size(), 0);
    t->len_of_.assign((size_t) n_rows, 0);
    t->impl_.reset(new Impl(cfg));
    t->impl_->table.assign((size_t) n_rows, strata::peer::Row{0, 0, 0});
    int64_t row = 0;
    for (size_t il = 0; il < specs.size(); ++il) {
        const PeerLayerSpec& l = specs[il];
        if (l.n_exp <= 0) continue;
        t->base_of_[il] = row;
        t->n_exp_of_[il] = l.n_exp;
        for (int role = 0; role < kRoles; ++role) {
            const PeerRowSpec& s = l.part[role];
            if (s.shard < 0 || s.len == 0) continue;   // (a role the peer does not hold: its rows stay 0)
            for (int64_t e = 0; e < l.n_exp; ++e) {
                const size_t r = (size_t) (row + e * kRoles + role);
                t->impl_->table[r] = strata::peer::Row{(uint16_t) s.shard, s.off + (uint64_t) e * s.stride, s.len};
                t->len_of_[r] = s.len;
            }
        }
        row += l.n_exp * kRoles;
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
    std::fprintf(stderr,
                 "ds4 peer: session up with %s:%d - %lld rows of %zu layers over %s, %zu pack files its manifest\n",
                 cfg.host.c_str(), cfg.port, (long long) n_rows, specs.size(),
                 cfg.device.empty() ? "the first ibverbs device" : cfg.device.c_str(), cfg.shard_names.size());
    // (--peer-compute) THE COMPUTE TIER: the daemon's welcome names its worker ("" = storage only), and the rows the
    // caller ranked first are asked to be kept resident at hello - the experts a decode's picks hit, held by the peer's
    // slab rather than read off its disk the first time an order needs them.  Neither is fatal: a daemon with no worker,
    // or a hint it cannot honour, leaves the tier serving storage exactly as stage 1 does.
    t->compute_caps_ = t->impl_->cli.compute_caps();
    t->compute_ = cfg.compute && !t->compute_caps_.empty();
    if (cfg.compute && !t->compute_)
        std::fprintf(stderr, "ds4 peer: no compute worker on the peer (its welcome names none) - the tier serves\n"
                             "          storage only\n");
    else if (t->compute_)
        std::fprintf(stderr, "ds4 peer: compute tier on (%s)\n", t->compute_caps_.c_str());
    if (t->compute_ && !cfg.resident.empty()) {
        // the slice: the peer's --vram slabs, which its worker reads in place; a daemon with no --vram keeps the hint to
        // half its RAM slab, so the flow's own reads still have room to land there
        const uint64_t cap = t->impl_->cli.vram_capacity() > 0 ? t->impl_->cli.vram_capacity()
                                                              : t->impl_->cli.capacity() / 2;
        std::vector<long long> want;
        uint64_t bytes = 0;
        for (long long r : cfg.resident) {
            if (r < 0 || (size_t) r >= t->len_of_.size() || t->len_of_[(size_t) r] == 0) continue;
            if (bytes + t->len_of_[(size_t) r] > cap) break;   // (the caller's order IS the ranking: it stops at the edge)
            bytes += t->len_of_[(size_t) r];
            want.push_back(r);
        }
        if (!want.empty()) {
            std::string hw;
            const auto t0 = std::chrono::steady_clock::now();
            if (!t->impl_->cli.hint(want, hw, cfg.resident_timeout_ms)) {
                std::fprintf(stderr, "ds4 peer: the resident hint did not complete (%s) - its experts are read as they\n"
                                     "          are picked instead\n", hw.c_str());
            } else {
                t->resident_rows_ = t->impl_->cli.advertised_rows();
                std::fprintf(stderr,
                             "ds4 peer: %zu resident rows hinted (%.1f GB of the peer's %.1f GB slab) in %.0f s, %zu\n"
                             "          rows held, the most-routed first\n",
                             want.size(), (double) bytes / 1073741824.0, (double) cap / 1073741824.0,
                             (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - t0).count() / 1000.0, t->resident_rows_);
            }
        }
    }
    return t;
}

bool PeerTier::register_dst(void* base, size_t bytes) {
    if (base == nullptr || bytes == 0 || !up_) return false;
    std::lock_guard<std::mutex> lk(mu_);
    std::string err;
    if (!impl_->cli.add_dst(base, bytes, err)) {
        std::fprintf(stderr,
                     "ds4 peer: a %.1f MB staging buffer did not register (%s) - that flow's experts stay on the\n"
                     "          disk path\n",
                     (double) bytes / 1048576.0, err.c_str());
        return false;
    }
    Region r;
    r.base = (uint8_t*) base;
    r.bytes = bytes;
    r.mr = impl_->cli.dst_mr(regions_.size());
    regions_.push_back(r);
    return true;
}

long long PeerTier::row_of(int64_t layer, int64_t expert, int role) const {
    if (layer < 0 || (size_t) layer >= base_of_.size() || role < 0 || role >= kRoles) return -1;
    const size_t il = (size_t) layer;
    const long long base = base_of_[il];
    // (an expert past the layer's own count would land on the NEXT layer's rows: the same lengths, another expert's
    // bytes - so the bound is the layer's, never the table's)
    if (base < 0 || expert < 0 || expert >= n_exp_of_[il]) return -1;
    const long long r = base + expert * kRoles + role;
    if ((size_t) r >= len_of_.size() || len_of_[(size_t) r] == 0) return -1;
    return r;
}

bool PeerTier::computable(int64_t layer, int64_t expert) const {
    if (!compute_ || !up_) return false;
    std::lock_guard<std::mutex> lk(mu_);
    for (int role = 0; role < kRoles; ++role) {
        const long long r = row_of(layer, expert, role);
        if (r < 0 || !impl_->cli.advertised(r)) return false;
    }
    return true;
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
    // Said loudly and once: from here every flow reads its misses off the local disks again (the decode is unchanged,
    // only slower), and a silent regression to the slow path is the one thing this tier must not do.
    std::fprintf(stderr, "ds4 peer: THE PEER IS DOWN (%s) - every flow reads its misses locally from here on\n",
                 why.c_str());
}

void PeerTier::serve(int n, const long long* rows, const uint64_t* rel, const size_t* bytes, uint8_t* const* dst,
                     char* ok) {
    if (ok == nullptr || n <= 0) return;
    std::memset(ok, 0, (size_t) n);
    std::unique_lock<std::mutex> lk(mu_);   // (dropped for the hint's round trip below: see there)
    if (!up_) return;
    const auto t0 = std::chrono::steady_clock::now();
    const size_t un = (size_t) n;
    std::vector<char> valid(un, 0);
    std::vector<void*> mrs(un, nullptr);
    std::vector<uint64_t> offs(un, 0), rels(un, 0);
    std::vector<uint32_t> lens(un, 0);
    std::vector<long long> hint;
    // 1. which rows can be pulled at all: a row this tier does not serve, or a destination outside a registered
    //    staging piece, is the caller's own read (and asking the daemon for it would only fail the batch)
    for (int k = 0; k < n; ++k) {
        const long long ri = rows[k];
        if (ri < 0 || (size_t) ri >= len_of_.size() || len_of_[(size_t) ri] == 0) continue;
        const uint64_t row_len = len_of_[(size_t) ri], r0 = rel[k];
        if (r0 > row_len) continue;
        const uint64_t nb = bytes[k] != 0 ? (uint64_t) bytes[k] : row_len - r0;
        if (nb == 0 || nb > row_len - r0 || nb > UINT32_MAX) continue;   // (a table this tier built cannot)
        void* m = nullptr;
        uint64_t o = 0;
        if (!region_of(dst[k], (size_t) nb, m, o)) continue;
        valid[(size_t) k] = 1;
        mrs[(size_t) k] = m;
        offs[(size_t) k] = o;
        rels[(size_t) k] = r0;
        lens[(size_t) k] = (uint32_t) nb;
        if (!impl_->cli.advertised(ri)) hint.push_back(ri);
    }
    // 2. ONE hint round trip for everything the daemon has not faulted in yet: a row it already holds answers at
    //    once, a cold one costs its pread on the peer's disk, and either way the reply advertises the rows so the
    //    batch below sees a settled map
    if (!hint.empty()) {
        // (the round trip is off the lock: a cold hint costs the peer's own disk - a 30 s budget at least, minutes for
        //  a slab's worth - while the decode lane's compute orders and every other queue wait on mu_, which is what
        //  made a tiered decode stall behind a single hint.  The map is re-read below: a row that arrived is used, and
        //  one that did not is the caller's own read)
        ++hints_;
        std::string err;
        lk.unlock();
        const bool hinted = impl_->cli.hint(hint, err);
        lk.lock();
        if (!hinted) {
            if (!impl_->cli.is_up()) fail("the hint failed: " + err);   // (a refused hint leaves the session up)
            return;
        }
    }
    // 3. one window for the whole batch (peer_read's read_many: a layer's parts interleaved, not one round trip each)
    std::vector<strata::peer::PeerClient::RowRead> reqs;
    std::vector<int> at;
    reqs.reserve(un);
    at.reserve(un);
    uint64_t moved = 0;
    for (int k = 0; k < n; ++k) {
        if (valid[(size_t) k] == 0) continue;
        if (!impl_->cli.advertised(rows[k])) {
            ++unadvertised_;   // (filled and dropped again - an evicted arena: the caller's own disk answers)
            continue;
        }
        reqs.push_back(strata::peer::PeerClient::RowRead{rows[k], (strata::peer::Mr*) mrs[(size_t) k],
                                                        offs[(size_t) k], rels[(size_t) k], lens[(size_t) k]});
        at.push_back(k);
        moved += lens[(size_t) k];
    }
    if (!reqs.empty()) {
        std::string err;
        if (!impl_->cli.read_rows(reqs, err)) {
            if (!impl_->cli.is_up()) fail("a batch read failed: " + err);
            return;
        }
        for (size_t j = 0; j < at.size(); ++j) ok[at[j]] = 1;
        ++batches_;
        rows_read_ += reqs.size();
        bytes_ += moved;
    }
    us_ += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0)
               .count();
}

void PeerTier::compute(int n, const long long* rows, int n_rows, int n_embd, int n_ff, int tg, int td, float limit,
                        const int8_t* const* xq, const float* const* xd, const float* const* w, float* const* y,
                        char* ok) {
    if (ok == nullptr || n <= 0) return;
    const std::vector<int> nr((size_t) n, n_rows);   // (one batch, one row count: the flow's own uses compute_rows)
    compute_rows(n, rows, nr.data(), n_embd, n_ff, tg, td, limit, xq, xd, w, y, ok);
}

void PeerTier::compute_rows(int n, const long long* rows, const int* n_rows, int n_embd, int n_ff, int tg, int td,
                            float limit, const int8_t* const* xq, const float* const* xd, const float* const* w,
                            float* const* y, char* ok) {
    if (ok == nullptr || n <= 0) return;
    std::memset(ok, 0, (size_t) n);
    std::lock_guard<std::mutex> lk(mu_);
    if (!up_ || !compute_) return;   // (a session without --peer-compute: every order is the caller's own work)
    const auto t0 = std::chrono::steady_clock::now();
    // RESIDENCY BEFORE A BYTE MOVES, as serve's: an order whose three rows this tier does not serve, or which the
    // daemon no longer advertises (never hinted - stage 1 hints nothing - or an arena that was evicted), is the caller's
    // own work.  Asking for it would only fail the batch, and a hint is not this tier's to send: residency is one-shot.
    std::vector<strata::peer::PeerClient::ComputeReq> jobs;
    std::vector<int> at;
    jobs.reserve((size_t) n);
    at.reserve((size_t) n);
    for (int k = 0; k < n; ++k) {
        const long long gate = rows[3 * (size_t) k], up = rows[3 * (size_t) k + 1], down = rows[3 * (size_t) k + 2];
        if (gate < 0 || up < 0 || down < 0) continue;
        if ((size_t) gate >= len_of_.size() || len_of_[(size_t) gate] == 0) continue;
        if ((size_t) up >= len_of_.size() || len_of_[(size_t) up] == 0) continue;
        if ((size_t) down >= len_of_.size() || len_of_[(size_t) down] == 0) continue;
        if (!impl_->cli.advertised(gate) || !impl_->cli.advertised(up) || !impl_->cli.advertised(down)) {
            ++unadvertised_;
            continue;
        }
        if (xq[k] == nullptr || xd[k] == nullptr || y[k] == nullptr) continue;
        strata::peer::PeerClient::ComputeReq q;
        q.gate = gate;
        q.up = up;
        q.down = down;
        q.n_embd = n_embd;
        q.n_ff = n_ff;
        q.tg = tg;
        q.td = td;
        q.limit = limit;
        q.n_rows = n_rows[k];
        q.xq = xq[k];
        q.xd = xd[k];
        q.w = w != nullptr ? w[k] : nullptr;
        jobs.push_back(q);
        at.push_back(k);
    }
    compute_orders_ += (uint64_t) n;
    if (jobs.empty()) return;
    std::vector<char> served;
    std::vector<std::vector<float>> sums;
    std::string err;
    if (!impl_->cli.compute(jobs, served, sums, err)) {
        if (!impl_->cli.is_up()) fail("a compute batch failed: " + err);   // (a refused batch leaves the session up)
        return;
    }
    for (size_t i = 0; i < at.size(); ++i) {
        if (served[i] == 0) continue;
        const size_t want = (size_t) n_rows[at[i]] * (size_t) n_embd;
        if (sums[i].size() != want) continue;   // (the client checked it: never half a sum)
        std::memcpy(y[at[i]], sums[i].data(), want * sizeof(float));
        ok[at[i]] = 1;
        ++computed_;
    }
    compute_us_ += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
}

void PeerTier::report() {
    std::lock_guard<std::mutex> lk(mu_);
    if (g_reported) return;
    g_reported = true;
    std::fprintf(stderr,
                 "ds4 peer: %llu rows (%.2f GB) pulled in %llu batches (%.2f GB/s on the wire), %llu hints, %llu "
                 "rows the daemon did not hold%s\n",
                 (unsigned long long) rows_read_, (double) bytes_ / 1073741824.0,
                 (unsigned long long) batches_, us_ > 0 ? (double) bytes_ / ((double) us_ / 1e6) / 1073741824.0 : 0.0,
                 (unsigned long long) hints_, (unsigned long long) unadvertised_, up_ ? "" : " - the session is DOWN");
    if (compute_)
        std::fprintf(stderr, "ds4 peer: compute tier on, %zu resident rows held at hello\n", resident_rows_);
    if (compute_orders_ > 0)
        std::fprintf(stderr, "ds4 peer: %llu of %llu compute orders served (%.2f ms an order on the peer)\n",
                     (unsigned long long) computed_, (unsigned long long) compute_orders_,
                     computed_ > 0 ? (double) compute_us_ / (double) computed_ / 1000.0 : 0.0);
}

// ------------------------------------------------------------------ the flow's queue
PeerQueue::PeerQueue(PeerTier& tier, IoQueue& local) : tier_(tier), local_(local) {
    th_ = std::thread([this] { loop(); });
}

PeerQueue::~PeerQueue() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
    tier_.report();   // (the counters, once a process: at the teardown of the last flow of it)
}

void PeerQueue::push(const Order* o, int n) {
    if (o == nullptr || n <= 0) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < n; ++i) q_.push_back(o[i]);
    }
    cv_.notify_one();
}

void PeerQueue::drain(std::vector<Order>& batch) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return quit_ || !q_.empty(); });
    batch.assign(q_.begin(), q_.end());
    q_.clear();
}

void PeerQueue::loop() {
    std::vector<Order> batch;
    std::vector<long long> rows;
    std::vector<uint64_t> rel;
    std::vector<size_t> bytes;
    std::vector<uint8_t*> dst;
    std::vector<char> ok;
    for (;;) {
        batch.clear();
        drain(batch);
        if (quit_) return;
        const size_t n = batch.size();
        rows.resize(n);
        rel.resize(n);
        bytes.resize(n);
        dst.resize(n);
        ok.resize(n);
        for (size_t k = 0; k < n; ++k) {
            const Order& o = batch[k];
            const uint64_t lead = o.job.off % os::kDirectAlign;   // (a read lands at dst + its offset's lead)
            rows[k] = o.row;
            rel[k] = o.rel;
            bytes[k] = o.bytes;
            dst[k] = o.job.dst + lead;
        }
        tier_.serve((int) n, rows.data(), rel.data(), bytes.data(), dst.data(), ok.data());
        for (size_t k = 0; k < n; ++k) {
            IoQueue::Job& j = batch[k].job;
            if (ok[k] == 0) {   // (unserved, dropped, or the session just died: exactly the read it stands in for)
                local_.push(&j, 1);
                continue;
            }
            if (j.crows > 0)   // (DS4_FLOW_DOWN_PART: the member's columns of the down, where the read left them)
                compact_rows(j.dst + j.off % os::kDirectAlign, j.crows, j.cpitch, j.coff, j.clen);
            if (j.left != nullptr) j.left->fetch_sub(1, std::memory_order_acq_rel);
        }
        taken_.fetch_add(n, std::memory_order_relaxed);
        local_.notify_done();   // (the flow's wait_any sleeps on this queue: a landed batch is a finished job to it)
    }
}

}  // namespace strata::ds4

#else   // !STRATA_HAVE_PEER
// A build without ibverbs (the tier is opt-in, and so is its build): the tier never comes up, every order goes the
// local way, and the engine runs as it did before Phase 3.
namespace strata::ds4 {

struct PeerTier::Impl {};

PeerTier* PeerTier::open_session(const Config&, std::string& err) {
    err = "this build has no ibverbs (strata_peer was not built)";
    return nullptr;
}
bool PeerTier::register_dst(void*, size_t) { return false; }
long long PeerTier::row_of(int64_t, int64_t, int) const { return -1; }
bool PeerTier::computable(int64_t, int64_t) const { return false; }
void PeerTier::serve(int n, const long long*, const uint64_t*, const size_t*, uint8_t* const*, char* ok) {
    if (ok != nullptr && n > 0) std::memset(ok, 0, (size_t) n);
}
void PeerTier::compute(int n, const long long*, int, int, int, int, int, float, const int8_t* const*, const float* const*,
                       const float* const*, float* const*, char* ok) {
    if (ok != nullptr && n > 0) std::memset(ok, 0, (size_t) n);
}
void PeerTier::compute_rows(int n, const long long*, const int*, int, int, int, int, float, const int8_t* const*,
                            const float* const*, const float* const*, float* const*, char* ok) {
    if (ok != nullptr && n > 0) std::memset(ok, 0, (size_t) n);
}
bool PeerTier::region_of(const void*, size_t, void*&, uint64_t&) const { return false; }
void PeerTier::fail(const std::string&) {}
void PeerTier::report() {}

PeerQueue::PeerQueue(PeerTier& tier, IoQueue& local) : tier_(tier), local_(local) {}
PeerQueue::~PeerQueue() = default;
void PeerQueue::push(const Order* o, int n) {
    for (int i = 0; i < n; ++i) local_.push(&o[i].job, 1);
}
long long PeerQueue::row_of(int64_t, int64_t, int) { return -1; }
void PeerQueue::drain(std::vector<Order>&) {}
void PeerQueue::loop() {}

}  // namespace strata::ds4

#endif  // STRATA_HAVE_PEER
