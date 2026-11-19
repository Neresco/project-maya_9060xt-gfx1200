// The peer's HIP compute worker (docs/PEER_COMPUTE.md, stage 2): one routed expert's arithmetic on the
// peer's own card, with the kernels the engine's device lane runs (src/ds4/ds4_iq.cu: iq::gate_up and
// iq::down), so a job this worker serves is the job that lane would have computed - and, because
// STRATA_DS4_IQ_STRICT_FP is defined for this target (CMakeLists.txt), on the rounding a local gfx1200 takes
// rather than the gfx11 default's.
//
// WHY A SHARED OBJECT.  strata-peer loads this file's .so by path (--compute) and links no GPU runtime at all:
// a peer machine with no ROCm still builds, starts and serves the storage tier, and this is the only place a HIP
// call is made on that machine's behalf.  The calls here are HIP's own (hipMalloc, hipMemcpyAsync,
// hipStreamSynchronize) for that reason: dlopen is the daemon's business, not the worker's, and a worker that IS
// HIP by design has no reason to reach its own runtime through a handle.
//
// WHAT THE PROTOCOL ALREADY IS.  The activation arrives as int8 values with a scale a 32 values, which is exactly
// iq::Args::xq / xd - the form the kernels take - so it is handed over as it came off the wire and no rounding is
// lost on the way in.  The h between gate/up and down is this worker's own: it is quantized the way the engine's lane
// quantizes it (Q8_K's rounding, a 256 values a scale - the CPU lane's own int8 values and scale), so the down sums
// are that lane's for those bytes.
//
// RESIDENCY.  A row in the daemon's --vram slab has no host window at all (peer_vram.cpp: the pages belong to the
// GPU), so the daemon hands its device address over with resident = 1 and this worker reads it in place - no copy of
// 10.6 MB of weights, which is the whole point of the tier.  Such an address means something on one card alone, so a
// resident pointer that is not this device's own allocation is refused (ok = 0) rather than read.  resident = 0 is a
// host row: the RAM slab's bytes, copied into this card's memory once a batch.
#include "peer_compute_abi.h"

#include "ds4_iq.cuh"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

namespace iq = strata::ds4::gguf::iq;

/// A token's picks are what this tier serves (the daemon's own limit of 8 rows a job): a prompt's rows per expert
/// are the caller's, which reads them where they already are.
constexpr int kMaxRows = 8;
/// One launch's entries are 8 apart in the scratch (iq::Args' grp_ent: entry e reads activation row e >> 3), so one
/// job's rows are entries 0, 8, 16 .. - the shape a token's plan has.
constexpr int kEntryStride = 8;
/// The expert kernels' grid (ds4_iq.cu's kWarps = 8 warps a block, kGuRW = 1 and kDnRW = 2 rows of a warp): iq::gate_up
/// and iq::down answer false when a geometry does not divide it, so such a job is refused before anything is launched.
constexpr int kRowAlign = 256;   // (a copied region's start: what the kernels' row addressing expects)

/// What one launch's kernels' arguments are, i.e. what its jobs have to share: the blob's role offsets (a resident job's
/// from the slab's own addresses, a host job's from this worker's staging layout) and the geometry the kernels decode with.
/// A token's picks are one (a layer's experts), so a batch is one plan; another shape is another launch.
struct PlanKey {
    size_t tr[3];
    int n_embd, n_ff, tg, td;
    float limit;
    bool same(const PlanKey& o) const {
        return tr[0] == o.tr[0] && tr[1] == o.tr[1] && tr[2] == o.tr[2] && n_embd == o.n_embd &&
               n_ff == o.n_ff && tg == o.tg && td == o.td && limit == o.limit;
    }
};

/// One launch: the jobs whose arguments are the key's (grp_ptr, grp_start and grp_ent cover them all, 8 entries a group),
/// so gate/up is one launch and down is one launch for the whole plan rather than one each a job.  A plan's groups are
/// contiguous in the batch's plan order, so its slices of the plan's arrays are too.
struct Plan {
    PlanKey key;
    std::vector<int> jobs;   // the batch's job indices, in the request's order
    int first = 0;           // its first group in the batch's plan order
    int ne_max = 1;           // the entries of its largest job (the kernels' NE)
    size_t woff = 0;          // its host jobs' slot in the weights' staging buffer
};

/// The down activation, rounded as the engine's lane rounds it: DS4_IQ_Q8K's rule (ggml's Q8_K, which is what the
/// CPU lane's quantize_row_q8_K does) - a 256 values' largest magnitude (the first of equal ones), the scale
/// -127 / that value, the nearest int8 of scale x v, and 1 / scale written to each of the eight 32-blocks.  One
/// warp a 256-block, as the engine's own kernel has it, so the int8 values and scales are its bits.
__global__ void __launch_bounds__(256) hq_quant_kernel(const float* __restrict__ h, int n_ff,
                                                        int8_t* __restrict__ q, float* __restrict__ d) {
    __shared__ float s_am[8];
    __shared__ int s_at[8];
    const int r = (int) blockIdx.y, j = (int) threadIdx.x, lane = j & 31, w = j >> 5;
    const int at = (int) blockIdx.x * 256 + j;      // (the value of its 256-block)
    // (a row's swiglu sits at ENTRY r * kEntryStride in the scratch - the index grp_ent carries and gate_up_kernel
    //  writes - not at r: reading it at r quantized row 0's block for every row after it, and the rows' down then
    //  ran on zeros)
    const size_t e_row = (size_t) r * kEntryStride;
    const float v = h[e_row * n_ff + at];
    float am = fabsf(v);
    int ix = j;                                    // (the first index of the largest magnitude)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, am, o);
        const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
        if (oa > am || (oa == am && oi < ix)) am = oa, ix = oi;
    }
    if (lane == 0) s_am[w] = am, s_at[w] = ix;
    __syncthreads();
    if (j == 0)
        for (int i = 1; i < 8; ++i)
            if (s_am[i] > s_am[0] || (s_am[i] == s_am[0] && s_at[i] < s_at[0]))
                s_am[0] = s_am[i], s_at[0] = s_at[i];
    __syncthreads();
    const float amax = s_am[0];
    int8_t qv = 0;
    float dv = 0.0f;
    if (amax > 0.0f) {
        const float mx = h[e_row * n_ff + (size_t) blockIdx.x * 256 + (size_t) s_at[0]];
        const float iscale = __fdiv_rn(-127.0f, mx);
        // (ggml's nearest_int: the magic number's rounding, halves to even)
        const float val = __fadd_rn(__fmul_rn(iscale, v), 12582912.0f);
        int i;
        memcpy(&i, &val, 4);
        qv = (int8_t) min(127, (i & 0x007fffff) - 0x00400000);
        dv = __fdiv_rn(1.0f, iscale);
    }
    // (the entries are 8 apart, as grp_ent spaces them: entry r * 8 of the scratch)
    const size_t e = e_row;
    q[e * n_ff + at] = qv;
    if (lane == 0) d[e * (n_ff / 32) + at / 32] = dv;
}

/// y[r][i] = w[r] * part[r][i]: the routing weight the caller would otherwise apply itself, on the device rather
/// than in a host loop over a copy of every row.
__global__ void __launch_bounds__(256) scale_rows_kernel(const float* __restrict__ part,
                                                    const float* __restrict__ w,
                                                    float* __restrict__ y, int n_embd) {
    const int r = (int) blockIdx.y, i = (int) blockIdx.x * 256 + (int) threadIdx.x;
    if (i >= n_embd) return;
    y[(size_t) r * n_embd + i] = w[r] * part[(size_t) r * kEntryStride * n_embd + i];
}

/// The worker: the card, the stream its batches run on, and the scratch they need.  The version() line names the card
/// once open() has picked it (peer_compute_abi.h's "gpu<N>" field: the daemon compares it with a --vram slab's own
/// ordinal before it hands that slab's rows over).
struct Worker {
    // (one batch at a time: the daemon serves one initiator, and the scratch below is the batch's)
    std::mutex mu;
    int dev = 0;
    hipStream_t stream = nullptr;
    uint8_t* d_w = nullptr;        // the host jobs' gate, up and down, one slot a job
    size_t w_cap = 0;
    // the batch's activation and its scales in one block, copied by one pinned copy a batch: a pageable
    // async copy is staged by the runtime through a buffer of its own, i.e. one more sync each
    char* d_act = nullptr;
    size_t act_xd_at = 0;          // the batch's scales inside d_act (the activation's rows are n_embd int8 a row, then the
                                  // scales, page-aligned: iq::Args reads xq and xd as two row arrays, so both offsets matter)
    size_t act_cap = 0;
    char* h_act = nullptr;         // (the pinned staging buffer of the copy above)
    size_t h_act_cap = 0;
    // the plan's own arrays in one block: grp_ptr (a group's blob), grp_start (its first and last entry),
    // grp_ent (the entry an activation row reads) and n_groups - the kernels read them on the device
    char* h_plan = nullptr;
    size_t h_plan_cap = 0;
    char* d_plan = nullptr;
    size_t plan_cap = 0;
    float* d_wt = nullptr;         // [rows] floats: the routing weights, one a row
    size_t wt_cap = 0;
    float* h_wt = nullptr;         // (their pinned staging buffer)
    size_t h_wt_cap = 0;
    float* d_h = nullptr;          // [kEntryStride rows][n_ff] floats: gate/up's swiglu
    size_t h_cap = 0;
    int8_t* d_hq = nullptr;        // [kEntryStride rows][n_ff] int8: the down activation
    size_t hq_cap = 0;
    float* d_hd = nullptr;         // [kEntryStride rows][n_ff / 32] floats: its scales
    size_t hd_cap = 0;
    float* d_part = nullptr;       // [kEntryStride rows][n_embd] floats: each entry's down output
    size_t part_cap = 0;
    float* d_y = nullptr;          // [rows][n_embd] floats: the weighted sums, before the copy back
    size_t y_cap = 0;
    float* h_y = nullptr;          // (their pinned staging buffer)
    size_t h_y_cap = 0;

    ~Worker() { release(); }

    /// Grows a device buffer to `bytes`, keeping what it holds.  False (err) when this card cannot take it: a worker
    /// whose allocation fails answers ok = 0 for that batch, which is never the daemon's failure.
    template <typename T>
    bool need(T*& p, size_t& cap, size_t bytes, const char* what, std::string& err) {
        if (p != nullptr && cap >= bytes) return true;
        if (p != nullptr) (void) hipFree(p);
        p = nullptr;
        cap = 0;
        void* q = nullptr;
        const hipError_t st = hipMalloc(&q, bytes);
        if (st != hipSuccess) {
            err = std::string("hipMalloc of ") + std::to_string(bytes) + " bytes for " + what + " failed (" +
                  hipGetErrorString(st) + ")";
            return false;
        }
        p = (T*) q;
        cap = bytes;
        return true;
    }

    /// The same for a host staging buffer, which is pinned: a copy into it does not wait for the copy it stages.
    template <typename T>
    bool need_pin(T*& p, size_t& cap, size_t bytes, const char* what, std::string& err) {
        if (p != nullptr && cap >= bytes) return true;
        if (p != nullptr) (void) hipHostFree(p);
        p = nullptr;
        cap = 0;
        void* q = nullptr;
        const hipError_t st = hipHostMalloc(&q, bytes);
        if (st != hipSuccess) {
            err = std::string("hipHostMalloc of ") + std::to_string(bytes) + " bytes for " + what + " failed (" +
                  hipGetErrorString(st) + ")";
            return false;
        }
        p = (T*) q;
        cap = bytes;
        return true;
    }


    /// Frees the scratch and the stream (a failed open's, or the daemon's close).
    void release() {
        if (dev >= 0) (void) hipSetDevice(dev);   // (a free means the card that allocated it: the daemon's current
                                                   //  device is the last worker's or a slab's, not this one's)
        if (stream != nullptr) (void) hipStreamDestroy(stream);
        stream = nullptr;
        if (d_w != nullptr) (void) hipFree(d_w);
        if (d_act != nullptr) (void) hipFree(d_act);
        if (h_act != nullptr) (void) hipHostFree(h_act);
        if (h_plan != nullptr) (void) hipHostFree(h_plan);
        if (d_plan != nullptr) (void) hipFree(d_plan);
        if (d_wt != nullptr) (void) hipFree(d_wt);
        if (h_wt != nullptr) (void) hipHostFree(h_wt);
        if (d_h != nullptr) (void) hipFree(d_h);
        if (d_hq != nullptr) (void) hipFree(d_hq);
        if (d_hd != nullptr) (void) hipFree(d_hd);
        if (d_part != nullptr) (void) hipFree(d_part);
        if (d_y != nullptr) (void) hipFree(d_y);
        if (h_y != nullptr) (void) hipHostFree(h_y);
        d_w = nullptr, d_act = nullptr, h_act = nullptr, h_plan = nullptr, d_plan = nullptr, d_wt = nullptr;
        h_wt = nullptr;
        d_h = nullptr, d_hq = nullptr, d_hd = nullptr, d_part = nullptr, d_y = nullptr, h_y = nullptr;
        w_cap = act_cap = h_act_cap = h_plan_cap = plan_cap = wt_cap = h_wt_cap = 0;
        h_cap = hq_cap = hd_cap = part_cap = y_cap = h_y_cap = 0;
    }
};

/// The options the daemon's --compute-options carried: whitespace- (or comma-) separated tokens, of which this
/// worker knows one - "gpu:<ordinal>", the peer machine's own HIP ordinal.  Anything else is refused rather than
/// ignored: an operator who named a card this worker does not understand has to hear about it at start-up, not find out
/// from a decode that quietly ran on the wrong card.
bool parse_options(const char* options, int& dev, std::string& err) {
    const std::string s = options != nullptr ? options : "";
    size_t at = 0;
    while (at < s.size()) {
        while (at < s.size() && (s[at] == ' ' || s[at] == '\t' || s[at] == ',')) ++at;
        size_t end = at;
        while (end < s.size() && s[end] != ' ' && s[end] != '\t' && s[end] != ',') ++end;
        const std::string tok = s.substr(at, end - at);
        at = end;
        if (tok.empty()) continue;
        if (tok.compare(0, 4, "gpu:") != 0 || tok.size() == 4 ||
            tok.find_first_not_of("0123456789", 4) != std::string::npos) {
            err = "the compute worker takes \"gpu:<ordinal>\" and nothing else, not \"" + tok + "\"";
            return false;
        }
        dev = std::atoi(tok.c_str() + 4);
    }
    return true;
}

/// The card's name as the runtime reports it, without the feature suffix ("gfx1151:sramecc+:xnack-").
std::string arch_of(const char* gcn) {
    const std::string s = gcn != nullptr ? gcn : "";
    const size_t colon = s.find(':');
    return colon == std::string::npos ? s : s.substr(0, colon);
}

/// Whether this .so carries code for the card's architecture: STRATA_PEER_HIP_ARCHS is the list the build was given
/// (CMake's HIP_ARCHITECTURES for this target, the same variable), so the two cannot disagree.  The card is asked
/// first, because a launch on a card this .so has no code for is not an error code the worker can answer: ROCm 7
/// aborts the process on the first such launch (hip_code_object.cpp's assertion), which is exactly the decode this
/// check exists to keep a peer from failing.
bool arch_carried(const std::string& arch) {
    const std::string list = STRATA_PEER_HIP_ARCHS;
    size_t at = 0;
    while (at <= list.size()) {
        const size_t end = list.find(',', at);
        const std::string tok = list.substr(at, end == std::string::npos ? std::string::npos : end - at);
        if (!tok.empty() && tok == arch) return true;
        if (end == std::string::npos) break;
        at = end + 1;
    }
    return false;
}

/// A host job's three matrices in one contiguous copy, as the engine's own blob lays them out: gate, up and down,
/// each at a 256-byte start so the kernels' row addressing is the pack's own.
struct Copy {
    size_t up_at, down_at, bytes;
};

Copy copy_of(const strata_peer_job& j) {
    const size_t gu = (size_t) j.gate_bytes;
    const size_t up_at = (gu + kRowAlign - 1) / kRowAlign * kRowAlign;
    const size_t down_at = (2 * gu + kRowAlign - 1) / kRowAlign * kRowAlign;
    return Copy{up_at, down_at, down_at + (size_t) j.down_bytes};
}

/// The kernels' arguments for one plan: the batch's activation as the wire carried it (one buffer, a row a
/// row), the geometry as the daemon's row table advertised it, and the plan's own arrays as the device holds
/// them.  `p` is the plan's slice of them: its groups, its rows, its n_groups.
void args_of(const Worker& w, const Plan& p, size_t pi, const size_t* at, iq::Args& a) {
    // the activation as the wire carried it: one int8 block for every row, then the scales, a row a row - iq::Args reads
    // them by the entry's own row (e >> 3 takes the row of the entries the plan gave it), so the base of both is the batch's
    a.xq = (const int8_t*) w.d_act;
    a.xd = (const float*) (w.d_act + w.act_xd_at);
    a.grp_ptr = (const unsigned long long*) (w.d_plan + at[0]) + p.first;
    a.grp_start = (const int*) (w.d_plan + at[1]) + p.first;
    a.grp_ent = (const int*) (w.d_plan + at[2]);
    a.n_groups = (const int*) (w.d_plan + at[3]) + pi;
    a.tr[0] = p.key.tr[0];
    a.tr[1] = p.key.tr[1];
    a.tr[2] = p.key.tr[2];
    a.n_embd = p.key.n_embd;
    a.n_ff = p.key.n_ff;
    a.limit = p.key.limit;
    a.hq = w.d_hq;
    a.hd = w.d_hd;
    a.h = w.d_h;
    a.part = w.d_part;
    a.tg = p.key.tg;
    a.td = p.key.td;
    a.act = iq::kActSwiglu;   // (a routed expert's activation: DeepSeek's clamped swiglu)
    a.pexp = 0;               // (DS4_LANE_PARITY is the CPU lane's exp: not this worker's)
}

/// Whether every one of a resident job's three pointers is this card's own allocation.  A device address means something
/// on the card that allocated it and nowhere else, so one from another card (or from host memory) is refused rather than
/// read - the daemon compares the slab's ordinal first, and this is the check that does not trust it.
bool resident_here(const strata_peer_job& j, int dev) {
    const void* p[3] = {j.gate, j.up, j.down};
    for (const void* q : p) {
        hipPointerAttribute_t at{};
        if (hipPointerGetAttributes(&at, q) != hipSuccess || (int) at.device != dev) return false;
    }
    return true;
}

/// Whether this job is one this worker takes at all (no copies, no launches): its geometry against the daemon's own byte
/// counts, the types the expert kernels cover, its residency flag, and every buffer it names.
bool check(const strata_peer_job& j, std::string& err) {
    if (j.resident != 0 && j.resident != 1) {
        err = "a job of residency " + std::to_string(j.resident) + " is not one this worker takes";
        return false;
    }
    if (j.n_rows <= 0 || j.n_rows > kMaxRows) {
        err = "a job of " + std::to_string(j.n_rows) + " rows is not one this worker takes";
        return false;
    }
    if (j.gate == nullptr || j.up == nullptr || j.down == nullptr || j.xq == nullptr || j.xd == nullptr ||
        j.y == nullptr) {
        err = "a job with a missing buffer is not one this worker takes";
        return false;
    }
    const int E = j.n_embd, FF = j.n_ff;
    if (E <= 0 || FF <= 0 || E % 32 != 0 || FF % 256 != 0) {
        // (the gate's scales are a 32 values, and the down activation is Q8_K's: whole 256-blocks, as the lane rounds them)
        err = "a job of " + std::to_string(E) + " x " + std::to_string(FF) + " is not one this worker takes";
        return false;
    }
    if (!iq::supported(j.tg) || !iq::supported(j.td)) {
        err = "a job of types " + std::to_string(j.tg) + "/" + std::to_string(j.td) +
              " is not one this worker takes";
        return false;
    }
    // the row table's own byte counts against this geometry: a table built by another pack (a requantized one) is caught
    // here, before a row of it is read - and so is a job whose matrices are not the shape it names
    const size_t gu_row = iq::row_bytes(j.tg, E), d_row = iq::row_bytes(j.td, FF);
    if (j.gate_bytes != gu_row * (size_t) FF || j.down_bytes != d_row * (size_t) E) {
        err = "a job of " + std::to_string(j.gate_bytes) + " + " + std::to_string(j.down_bytes) +
              " bytes is not this geometry's (" + std::to_string(gu_row * (size_t) FF) + " + " +
              std::to_string(d_row * (size_t) E) + ")";
        return false;
    }
    return true;
}

/// Serves one batch this worker has already checked.  One plan a batch, not a plan a job: every plan's gate/up is
/// one launch, the batch's down activation is quantized in one, every plan's down is one, the routing weights are
/// one, and the sums come back in one - five launches where the per-job plan had four a job, which is the ~22 us a
/// pick stage 2 measured beside 50.1 us of pure compute.  A host job's three matrices are copied per job into the
/// plan's own slot of one staging buffer (9.1 MB a job, which is what the residency tier exists to avoid), never
/// waited for a job.
bool serve(const strata_peer_job* jobs, int n, const std::vector<char>& take, std::vector<Plan>& plans,
           const std::vector<int>& grp_of, const std::vector<int>& grp_row,
           const std::vector<unsigned long long>& grp_blob, size_t rows_total, Worker& w,
           std::vector<char>& served, std::string& err) {
    const int E = plans[0].key.n_embd, FF = plans[0].key.n_ff;
    hipStream_t s = w.stream;
    // the batch's activation and its scales in one block, each row a row as iq::Args takes them, and the routing
    // weights and the sums beside it - one pinned copy a batch where the per-job plan paid one a job
    const size_t xq_bytes = rows_total * (size_t) E;
    const size_t xd_bytes = rows_total * (size_t) (E / 32) * sizeof(float);
    const size_t xd_at = (xq_bytes + kRowAlign - 1) / kRowAlign * kRowAlign;
    const size_t act_bytes = xd_at + xd_bytes;
    if (!w.need(w.d_act, w.act_cap, act_bytes, "the batch's activation", err) ||
        !w.need_pin(w.h_act, w.h_act_cap, act_bytes, "its staging", err) ||
        !w.need(w.d_wt, w.wt_cap, rows_total * sizeof(float), "the routing weights", err) ||
        !w.need_pin(w.h_wt, w.h_wt_cap, rows_total * sizeof(float), "their staging", err) ||
        !w.need(w.d_h, w.h_cap, (size_t) kEntryStride * rows_total * (size_t) FF * sizeof(float), "h", err) ||
        !w.need(w.d_hq, w.hq_cap, (size_t) kEntryStride * rows_total * (size_t) FF, "the down activation",
                err) ||
        !w.need(w.d_hd, w.hd_cap, (size_t) kEntryStride * rows_total * (size_t) (FF / 32) * sizeof(float),
                "its scales", err) ||
        !w.need(w.d_part, w.part_cap, (size_t) kEntryStride * rows_total * (size_t) E * sizeof(float),
                "the down output", err) ||
        !w.need(w.d_y, w.y_cap, rows_total * (size_t) E * sizeof(float), "the sums", err) ||
        !w.need_pin(w.h_y, w.h_y_cap, rows_total * (size_t) E * sizeof(float), "their staging", err)) {
        return false;
    }
    for (int k = 0; k < n; ++k) {
        if (take[(size_t) k] == 0) continue;
        const strata_peer_job& j = jobs[k];
        const size_t row = (size_t) grp_row[(size_t) k];
        std::memcpy(w.h_act + row * (size_t) E, j.xq, (size_t) j.n_rows * (size_t) E);
        std::memcpy(w.h_act + xd_at + row * (size_t) (E / 32) * sizeof(float), j.xd,
                    (size_t) j.n_rows * (size_t) (E / 32) * sizeof(float));
        // (a job without routing weights is the expert's own output: every row's weight is 1)
        if (j.w != nullptr) std::memcpy(w.h_wt + row, j.w, (size_t) j.n_rows * sizeof(float));
        else std::fill(w.h_wt + row, w.h_wt + row + j.n_rows, 1.0f);
    }
    w.act_xd_at = xd_at;   // (args_of reads the scales beside the rows, at this batch's own offset)
    if (hipMemcpyAsync(w.d_act, w.h_act, act_bytes, hipMemcpyHostToDevice, s) != hipSuccess ||
        hipMemcpyAsync(w.d_wt, w.h_wt, rows_total * sizeof(float), hipMemcpyHostToDevice, s) != hipSuccess) {
        err = "the batch's activation did not reach this card";
        return false;
    }
    // a host job's three matrices, one slot a job of the weights' staging buffer: the copy is a job's (9.1 MB), the
    // wait is the batch's - the per-job plan synchronized a job, which is the other half of its ~22 us a pick
    for (int k = 0; k < n; ++k) {
        if (take[(size_t) k] == 0 || jobs[k].resident != 0) continue;
        // (a host job's own slot: the group reads grp_blob[k], which was handed its slot above - a plan's jobs are not
        //  one slot, they are one launch: copying them all into the plan's first slot left every group but the first
        //  reading a slot nothing had written)
        uint8_t* const slot = (uint8_t*) (uintptr_t) grp_blob[(size_t) k];
        const Copy c = copy_of(jobs[k]);
        if (hipMemcpyAsync(slot, jobs[k].gate, (size_t) jobs[k].gate_bytes, hipMemcpyHostToDevice, s) != hipSuccess ||
            hipMemcpyAsync(slot + c.up_at, jobs[k].up, (size_t) jobs[k].gate_bytes, hipMemcpyHostToDevice, s) !=
                hipSuccess ||
            hipMemcpyAsync(slot + c.down_at, jobs[k].down, (size_t) jobs[k].down_bytes, hipMemcpyHostToDevice,
                            s) != hipSuccess) {
            err = "the expert's rows did not reach this card";
            return false;
        }
    }
    // the plan's own arrays in one block, one copy a batch: grp_ptr (a group's blob), grp_start (its first and last
    // entry, 8 entries a group as iq::Args' grp_ent spaces them), grp_ent (the entry a row reads) and n_groups
    const size_t ptr_at = 0, start_at = (size_t) n * sizeof(unsigned long long);
    const size_t ent_at = start_at + ((size_t) n + 1) * sizeof(int);
    const size_t ng_at = ent_at + rows_total * sizeof(int);
    const size_t plan_bytes = ng_at + plans.size() * sizeof(int);
    const size_t at[4] = {ptr_at, start_at, ent_at, ng_at};
    if (!w.need(w.d_plan, w.plan_cap, plan_bytes, "the plan's arrays", err) ||
        !w.need_pin(w.h_plan, w.h_plan_cap, plan_bytes, "their staging", err)) {
        return false;
    }
    {
        char* const h = w.h_plan;
        auto* const ptr = (unsigned long long*) (h + ptr_at);
        auto* const start = (int*) (h + start_at);
        auto* const ent = (int*) (h + ent_at);
        auto* const ng = (int*) (h + ng_at);
        for (int k = 0; k < n; ++k) {
            if (take[(size_t) k] == 0) continue;
            const int g = grp_of[(size_t) k];   // (its group in the batch's plan order)
            ptr[g] = grp_blob[(size_t) k];
            start[g] = grp_row[(size_t) k];
        }
        start[n] = (int) rows_total;
        for (size_t t = 0; t < rows_total; ++t) ent[t] = (int) t * kEntryStride;
        for (size_t pi = 0; pi < plans.size(); ++pi) ng[pi] = (int) plans[pi].jobs.size();
        if (hipMemcpyAsync(w.d_plan, w.h_plan, plan_bytes, hipMemcpyHostToDevice, s) != hipSuccess) {
            err = "the plan did not reach this card";
            return false;
        }
    }
    if (std::getenv("STRATA_PEER_COMPUTE_DEBUG") != nullptr) {
        const char* const h = w.h_plan;
        const auto* const ptr = (const unsigned long long*) (h + ptr_at);
        const auto* const start = (const int*) (h + start_at);
        std::fprintf(stderr, "  start:");
        for (int g = 0; g <= n; ++g) std::fprintf(stderr, " %d", start[g]);
        std::fprintf(stderr, "\n  ptr:");
        for (int g = 0; g < n; ++g) std::fprintf(stderr, " 0x%llx", ptr[g]);
        std::fprintf(stderr, "\n");
    }
    // gate/up for every plan (the engine's own call, so the same sums), then the batch's down activation, then down
    // for every plan, then the routing weights: a geometry their grids cannot cover answers false here, before anything
    // is launched, which is how such a batch is refused rather than run
    std::vector<iq::Args> args(plans.size());
    std::vector<char> bad(plans.size(), 0);
    for (size_t pi = 0; pi < plans.size(); ++pi) args_of(w, plans[pi], pi, at, args[pi]);
    for (size_t pi = 0; pi < plans.size(); ++pi) {
        if (!iq::gate_up(args[pi], (int) plans[pi].jobs.size(), plans[pi].ne_max, s)) bad[pi] = 1;
    }
    hq_quant_kernel<<<dim3((unsigned) (FF / 256), (unsigned) rows_total), 256, 0, s>>>(w.d_h, FF, w.d_hq,
                                                                                  w.d_hd);
    for (size_t pi = 0; pi < plans.size(); ++pi) {
        if (bad[pi] != 0) continue;
        if (!iq::down(args[pi], (int) plans[pi].jobs.size(), plans[pi].ne_max, s)) bad[pi] = 1;
    }
    scale_rows_kernel<<<dim3((unsigned) ((E + 255) / 256), (unsigned) rows_total), 256, 0, s>>>(
        w.d_part, w.d_wt, w.d_y, E);
    const hipError_t st = hipGetLastError();
    if (st != hipSuccess) {
        err = std::string("the expert's kernels failed (") + hipGetErrorString(st) + ")";
        return false;
    }
    // the sums, one copy a batch into one pinned buffer and one memcpy a job from it: a job whose plan refused
    // answers ok = 0 and is computed by the caller instead.  The wait is here, not in run(): an async copy into a
    // pinned buffer has not touched it when the stream returns, so a job read before this sync would get the
    // buffer's previous (empty) contents - the sums are read once the stream has actually drained.
    if (hipMemcpyAsync(w.h_y, w.d_y, rows_total * (size_t) E * sizeof(float), hipMemcpyDeviceToHost, s) !=
            hipSuccess ||
        hipStreamSynchronize(s) != hipSuccess) {
        err = "the sums did not reach the caller";
        return false;
    }
    for (size_t pi = 0; pi < plans.size(); ++pi) {
        if (bad[pi] != 0) continue;
        for (int k : plans[pi].jobs) {
            const strata_peer_job& j = jobs[k];
            std::memcpy(j.y, w.h_y + (size_t) grp_row[(size_t) k] * (size_t) E,
                         (size_t) j.n_rows * (size_t) E * sizeof(float));
            served[(size_t) k] = 1;
        }
    }
    return true;
}

/// The worker's own version() line.  open() rewrites it once it knows the card: peer_compute_abi.h's "gpu<N>" field is what
/// the daemon compares with a --vram slab's own ordinal, and it can only be named after hipSetDevice.
std::string g_caps = "strata-peer-compute/2 hip (no device)";

}   // namespace

extern "C" {

const char* strata_peer_compute_version(void) {
    // (the daemon reads this twice: once for the ABI number before open, and once for the capabilities after it - a worker
    // only learns which card it runs on at open, so its "gpu<N>" field appears in the second reading)
    return g_caps.c_str();
}

void* strata_peer_compute_open(const char* options) {
    int dev = 0;
    std::string err;
    if (!parse_options(options, dev, err)) {
        std::fprintf(stderr, "strata-peer-compute-hip: %s\n", err.c_str());
        return nullptr;
    }
    int count = 0;
    const hipError_t counted = hipGetDeviceCount(&count);
    if (counted != hipSuccess || count <= 0) {
        std::fprintf(stderr, "strata-peer-compute-hip: no ROCm device here (%d HIP devices, %s)\n", count,
                     hipGetErrorString(counted));
        return nullptr;
    }
    if (dev < 0 || dev >= count) {
        std::fprintf(stderr, "strata-peer-compute-hip: GPU %d does not exist here (%d HIP devices)\n", dev, count);
        return nullptr;
    }
    hipError_t st = hipSetDevice(dev);
    if (st != hipSuccess) {
        std::fprintf(stderr, "strata-peer-compute-hip: hipSetDevice(%d) failed: %s\n", dev,
                     hipGetErrorString(st));
        return nullptr;
    }
    hipDeviceProp_t prop{};
    st = hipGetDeviceProperties(&prop, dev);
    if (st != hipSuccess) {
        std::fprintf(stderr, "strata-peer-compute-hip: no properties for GPU %d: %s\n", dev,
                     hipGetErrorString(st));
        return nullptr;
    }
    const std::string arch = arch_of(prop.gcnArchName);
    auto w = std::make_unique<Worker>();
    w->dev = dev;
    st = hipStreamCreate(&w->stream);
    if (st != hipSuccess) {
        std::fprintf(stderr, "strata-peer-compute-hip: hipStreamCreate on GPU %d failed: %s\n", dev,
                     hipGetErrorString(st));
        return nullptr;
    }
    // a card this .so carries no code for refuses to open here rather than fail a decode's third layer (a worker built
    // for the peer's gfx1102/gfx1151 on a gfx1200 box, and the other way round)
    if (!arch_carried(arch)) {
        std::fprintf(stderr, "strata-peer-compute-hip: GPU %d (%s) is not one this worker was built for (%s)\n", dev,
                     arch.c_str(), STRATA_PEER_HIP_ARCHS);
        w->release();
        return nullptr;
    }
    g_caps = "strata-peer-compute/2 hip " + arch + " gpu" + std::to_string(dev);
    return w.release();
}

int strata_peer_compute_run(void* worker, const strata_peer_job* jobs, int n, uint8_t* ok) {
    if (worker == nullptr || jobs == nullptr || ok == nullptr || n <= 0) return 1;
    Worker& w = *(Worker*) worker;
    std::lock_guard<std::mutex> lk(w.mu);
    // THE DAEMON'S CURRENT DEVICE IS NOT THIS WORKER'S: two workers live in one process (one a card), each with the
    // stream it made at open - and a stream, a copy and a launch mean the card they were made on.  Launched with
    // another card current, a batch answers hipErrorInvalidResourceHandle and the async copies around it fault the
    // context (which then fails even a hipMalloc).  Every call below is this worker's card, so it is made current here.
    if (hipSetDevice(w.dev) != hipSuccess) {
        std::fprintf(stderr, "strata-peer-compute-hip: hipSetDevice(%d) failed (%s) - no batch on this worker\n", w.dev,
                     hipGetErrorString(hipGetLastError()));
        return 1;
    }
    for (int k = 0; k < n; ++k) ok[k] = 0;
    std::string err;
    // the batch's plan, a job at a time: which jobs this worker takes (a job it cannot serve is the caller's own
    // work, never the batch's - the ABI has no channel to say why, so a refused job is silent by design), each one's
    // role offsets, and the plan whose kernels' arguments they are one launch's
    std::vector<char> take((size_t) n, 0);
    std::vector<size_t> tr((size_t) n * 3);
    std::vector<unsigned long long> blob((size_t) n, 0);
    std::vector<Plan> plans;
    size_t rows_total = 0, wcopy_bytes = 0;
    for (int k = 0; k < n; ++k) {
        const strata_peer_job& j = jobs[k];
        if (!check(j, err)) continue;
        // the protocol carries one geometry a batch (the client refuses two and so does the daemon), so a job of
        // another one is not a plan's: it is the caller's own work, as any other job this worker refuses is
        if (!plans.empty() && (j.n_embd != plans[0].key.n_embd || j.n_ff != plans[0].key.n_ff)) continue;
        size_t* t = &tr[(size_t) k * 3];
        PlanKey key{};
        key.n_embd = j.n_embd, key.n_ff = j.n_ff, key.tg = j.tg, key.td = j.td, key.limit = j.limit;
        if (j.resident == 1) {
            if (!resident_here(j, w.dev)) continue;
            // the slab's own addresses: the blob base is the lowest of the three, since a row's address is inside
            // one allocation and the three are one region of it - and the offsets are then the differences
            const uint8_t* base = j.gate;
            if (j.up < base) base = j.up;
            if (j.down < base) base = j.down;
            blob[(size_t) k] = (unsigned long long) (uintptr_t) base;
            t[0] = (size_t) (j.gate - base);
            t[1] = (size_t) (j.up - base);
            t[2] = (size_t) (j.down - base);
        } else {
            // a host row: this card's own copy, whose layout is this worker's to choose (the aligned one, so a
            // job's three matrices start on 256-byte boundaries)
            const Copy c = copy_of(j);
            t[0] = 0;
            t[1] = c.up_at;
            t[2] = c.down_at;
            wcopy_bytes += c.bytes;
        }
        key.tr[0] = t[0], key.tr[1] = t[1], key.tr[2] = t[2];
        // a job whose arguments a launch already carries joins it: a layer's picks are one plan, a batch of two
        // layers is two
        int p = -1;
        for (size_t i = 0; i < plans.size() && p < 0; ++i)
            if (plans[i].key.same(key)) p = (int) i;
        if (p < 0) {
            plans.push_back(Plan{key});
            p = (int) plans.size() - 1;
        }
        plans[(size_t) p].jobs.push_back(k);
        if (j.n_rows > plans[(size_t) p].ne_max) plans[(size_t) p].ne_max = j.n_rows;
        rows_total += (size_t) j.n_rows;
        take[(size_t) k] = 1;
    }
    // the batch's own order: each plan's groups contiguous, every job's rows in it, and a host job's slot in the
    // weights' staging buffer (one copy a job, into one buffer: not a copy and a wait a job)
    std::vector<int> grp_of((size_t) n, -1), grp_row((size_t) n, 0);
    std::vector<unsigned long long> grp_blob((size_t) n, 0);
    size_t rows = 0, woff = 0;
    int g = 0;
    for (Plan& p : plans) {
        p.first = g;
        p.woff = woff;
        for (int k : p.jobs) {
            grp_of[(size_t) k] = g++;
            grp_row[(size_t) k] = (int) rows;
            rows += (size_t) jobs[k].n_rows;
            if (jobs[k].resident == 0) {
                grp_blob[(size_t) k] = (unsigned long long) (uintptr_t) (w.d_w + woff);
                woff += copy_of(jobs[k]).bytes;
            } else {
                grp_blob[(size_t) k] = blob[(size_t) k];
            }
        }
    }
    std::vector<char> served((size_t) n, 0);
    if (std::getenv("STRATA_PEER_COMPUTE_DEBUG") != nullptr) {   // (the batch's shape, as the worker reads it)
        std::fprintf(stderr, "strata-peer-compute-hip: debug: %d jobs, %zu rows, %zu plans\n", n, rows_total,
                     plans.size());
        for (size_t pi = 0; pi < plans.size(); ++pi) {
            std::fprintf(stderr, "  plan %zu: first %d, woff %zu, ne_max %zu, tr %zu/%zu/%zu, jobs", pi, plans[pi].first,
                         plans[pi].woff, plans[pi].ne_max, plans[pi].key.tr[0], plans[pi].key.tr[1],
                         plans[pi].key.tr[2]);
            for (int k : plans[pi].jobs) std::fprintf(stderr, " %d(row %d blob 0x%llx)", k, grp_row[(size_t) k],
                                                      grp_blob[(size_t) k]);
            std::fprintf(stderr, "\n");
        }
        std::fprintf(stderr, "  grp_of:");
        for (int k = 0; k < n; ++k) std::fprintf(stderr, " %d", grp_of[(size_t) k]);
        std::fprintf(stderr, "\n");
    }
    if (!plans.empty()) {
        // (a card that cannot take the batch's scratch answers nothing: every ok stays 0 and the caller computes them)
        if ((wcopy_bytes != 0 && !w.need(w.d_w, w.w_cap, wcopy_bytes, "the host jobs' weights", err)) ||
            !serve(jobs, n, take, plans, grp_of, grp_row, grp_blob, rows_total, w, served, err)) {
            std::fprintf(stderr, "strata-peer-compute-hip: %s\n", err.c_str());
            return 1;
        }
    }
    // one wait a batch, and a fault loses all of it: a job whose sums did not reach the caller's buffer answers ok = 0 and
    // is computed there instead - the same contract a worker that cannot run at all has (docs/PEER_COMPUTE.md, "The fallback")
    const hipError_t st = hipStreamSynchronize(w.stream);
    if (st != hipSuccess) {
        std::fprintf(stderr, "strata-peer-compute-hip: a batch failed on GPU %d: %s\n", w.dev,
                     hipGetErrorString(st));
        // (what the batch held, so a fault names its own addresses: a resident job's three rows are separate
        //  cache entries - not one contiguous expert blob - and each is read through its own offset in the plan)
        std::fprintf(stderr,
                     "strata-peer-compute-hip: the batch was %d jobs, %zu rows, act_cap %zu (xq %zu B), plan_cap %zu, "
                     "%zu plans\n",
                     n, rows_total, w.act_cap, (size_t) (rows_total * (size_t) plans[0].key.n_embd), w.plan_cap,
                     plans.size());
        for (int k = 0; k < n; ++k) {
            if (take[(size_t) k] == 0) continue;
            hipPointerAttribute_t pa{};
            const bool known = hipPointerGetAttributes(&pa, (void*) grp_blob[(size_t) k]) == hipSuccess;
            std::fprintf(stderr,
                         "strata-peer-compute-hip: job %d resident=%d rows=%d tg=%d td=%d E=%d FF=%d gate_bytes=%zu "
                         "down_bytes=%zu gate=%p up=%p down=%p blob=0x%llx (device %d, base %p) tr=%zu/%zu/%zu\n",
                         k, jobs[k].resident, jobs[k].n_rows, jobs[k].tg, jobs[k].td, jobs[k].n_embd, jobs[k].n_ff,
                         (size_t) jobs[k].gate_bytes, (size_t) jobs[k].down_bytes, (const void*) jobs[k].gate,
                         (const void*) jobs[k].up, (const void*) jobs[k].down,
                         (unsigned long long) grp_blob[(size_t) k], known ? (int) pa.device : -1,
                         known ? pa.devicePointer : nullptr, tr[(size_t) k * 3], tr[(size_t) k * 3 + 1],
                         tr[(size_t) k * 3 + 2]);
        }
        for (int k = 0; k < n; ++k) ok[k] = 0;
        return 1;
    }
    for (int k = 0; k < n; ++k) ok[k] = served[(size_t) k];
    return 0;
}

void strata_peer_compute_close(void* worker) {
    if (worker != nullptr) delete (Worker*) worker;
}

}   // extern "C"
