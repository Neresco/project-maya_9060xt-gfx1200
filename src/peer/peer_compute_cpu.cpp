// The peer's CPU compute worker (docs/PEER_COMPUTE.md, stage 1): one routed expert's arithmetic on the
// peer's own cores, exactly as the engine's CPU lane runs it - gate/up rows, swiglu, the down activation
// quantized, down rows, the routing weight applied.  It is a shared object the daemon loads by path
// (strata-peer --compute), so a peer without ROCm still computes; the arithmetic is ds4_iq_cpu.cpp's own
// (iq_cpu_gate_up / iq_cpu_quant / iq_cpu_down), so the sums are the lane's for the same bytes.
//
// THE ONE ROUNDING.  The activation arrives as the device kernels round theirs - int8 values with a scale a 32
// (q8_round's form, the protocol's body) - and the lane quantizes the raw floats to the weight type's
// vec_dot_type (Q8_K: a scale a 256).  So this file dequantizes to f32 and quantizes again, which is one
// rounding more than the lane's own path takes from the floats: the sums agree bit for bit with a lane fed
// these bytes, and with a lane fed the raw floats to within that second rounding (tests/peer_compute_test).
#include "peer_compute_abi.h"

#include "ds4_gguf_cpu.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

namespace gg = strata::ds4::gguf;

/// The worker's scratch, per activation row: the f32 window the activation is dequantized into, the gate/up
/// activation as the weight type takes it, the f32 h between the two matrices, h as the down type takes it,
/// and the down output.  One batch at a time (the daemon serves one initiator, and the lane's own thread_local
/// buffers make it so anyway): a mutex is the honest statement of that.
struct Worker {
    std::mutex mu;
    std::vector<float> x;      // n_embd a row
    std::vector<uint8_t> aq;    // act_bytes a row
    std::vector<float> ff;     // n_ff a row
    std::vector<uint8_t> hq;    // h_bytes a row
    std::vector<float> out;    // n_embd a row
};

/// Dequantizes `n` values: the int8 values with their scales, as the device kernels round them.
void dequant(const int8_t* q, const float* scales, int n, float* x) {
    for (int i = 0; i < n; ++i) x[i] = (float) q[i] * scales[i / 32];
}

/// One expert: the lane's four steps, on the activation the protocol carried.
bool serve(const strata_peer_job& j, Worker& w, std::string& err) {
    gg::IqCpu f;
    if (!gg::iq_cpu_fmt(j.tg, j.td, j.n_embd, j.n_ff, f, err)) return false;
    if (j.n_rows <= 0 || j.n_rows > gg::kMxCpuMaxN) {
        err = "a job of " + std::to_string(j.n_rows) + " rows is not one this worker takes";
        return false;
    }
    if (j.gate == nullptr || j.up == nullptr || j.down == nullptr || j.xq == nullptr || j.xd == nullptr ||
        j.y == nullptr) {
        err = "a job with a missing buffer is not one this worker takes";
        return false;
    }
    // (a resident job names the peer's own device addresses: there is no host window behind them, and this
    // worker's arithmetic reads host memory - so it answers ok=0 and the caller computes that row locally)
    if (j.resident != 0) return false;
    // the row table's own byte counts against this geometry: a table built by another pack (a requantized one)
    // is caught here, before a row of it is read
    if (j.gate_bytes != (uint64_t) f.gu_row * (uint64_t) j.n_ff ||
        j.down_bytes != (uint64_t) f.d_row * (uint64_t) j.n_embd) {
        err = "a job of " + std::to_string(j.gate_bytes) + " + " + std::to_string(j.down_bytes) +
              " bytes is not this geometry's (" + std::to_string((uint64_t) f.gu_row * (uint64_t) j.n_ff) +
              " + " + std::to_string((uint64_t) f.d_row * (uint64_t) j.n_embd) + ")";
        return false;
    }
    const int nt = j.n_rows, E = j.n_embd, FF = j.n_ff;
    w.x.resize((size_t) nt * E);
    w.aq.resize((size_t) nt * f.act_bytes);
    w.ff.resize((size_t) nt * FF);
    w.hq.resize((size_t) nt * f.h_bytes);
    w.out.resize((size_t) nt * E);
    const void* ap[gg::kMxCpuMaxN];
    const void* hp[gg::kMxCpuMaxN];
    float* fp[gg::kMxCpuMaxN];
    float* yp[gg::kMxCpuMaxN];
    for (int t = 0; t < nt; ++t) {
        float* x = w.x.data() + (size_t) t * E;
        dequant(j.xq + (size_t) t * E, j.xd + (size_t) t * (E / 32), E, x);
        gg::iq_cpu_quant(f.gu_act, x, E, w.aq.data() + (size_t) t * f.act_bytes);
        ap[t] = w.aq.data() + (size_t) t * f.act_bytes;
        fp[t] = w.ff.data() + (size_t) t * FF;
        hp[t] = w.hq.data() + (size_t) t * f.h_bytes;
        yp[t] = w.out.data() + (size_t) t * E;
    }
    // gate/up rows, then swiglu - the lane's own call, so the same sums
    gg::iq_cpu_gate_up(f, j.gate, j.up, ap, nt, fp, 0, FF, j.limit);
    // h as the down type takes it, down rows, then the routing weight: the weighted sum the caller would
    // otherwise do over the picks it made
    for (int t = 0; t < nt; ++t)
        gg::iq_cpu_quant(f.d_act, w.ff.data() + (size_t) t * FF, FF, w.hq.data() + (size_t) t * f.h_bytes);
    gg::iq_cpu_down(f, j.down, hp, nt, yp, 0, E);
    for (int t = 0; t < nt; ++t) {
        const float wt = j.w != nullptr ? j.w[t] : 1.0f;
        const float* y = w.out.data() + (size_t) t * E;
        float* o = j.y + (size_t) t * E;
        for (int i = 0; i < E; ++i) o[i] = wt * y[i];
    }
    return true;
}

}   // namespace

extern "C" {

const char* strata_peer_compute_version(void) {
    // (the daemon refuses a worker that names another ABI: the line begins with it)
    static const std::string line = "strata-peer-compute/" + std::to_string(STRATA_PEER_COMPUTE_ABI) + " cpu";
    return line.c_str();
}

void* strata_peer_compute_open(const char*) {
    // (nothing to open: the scratch is the batch's, and ggml-cpu's kernel selection is the lane's own)
    return new Worker();
}

int strata_peer_compute_run(void* worker, const strata_peer_job* jobs, int n, uint8_t* ok) {
    if (worker == nullptr || jobs == nullptr || ok == nullptr || n <= 0) return 1;
    Worker& w = *(Worker*) worker;
    std::lock_guard<std::mutex> lk(w.mu);
    for (int k = 0; k < n; ++k) {
        ok[k] = 0;
        std::string err;
        // (a job this worker cannot serve is the caller's own work: never the batch's - and the ABI has
        // no channel to say why, so a refused job is silent by design)
        if (!serve(jobs[k], w, err)) continue;
        ok[k] = 1;
    }
    return 0;
}

void strata_peer_compute_close(void* worker) {
    delete (Worker*) worker;
}

}   // extern "C"
