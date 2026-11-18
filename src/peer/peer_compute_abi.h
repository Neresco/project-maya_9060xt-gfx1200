/* The peer compute worker's ABI: one routed expert's arithmetic, in plain C, with no GPU runtime
 * in sight.  strata-peer loads a worker shared object by path (--compute) and hands it batches of these
 * jobs; which worker it is - the peer's cores (stage 1, peer_compute_cpu.cpp) or its cards (stage 2,
 * peer_compute_hip.cu) - is the peer's business, because the engine only ever sees "computed" or
 * "cannot serve this" (docs/PEER_COMPUTE.md).
 *
 * A job is one expert of one layer for n_rows activation rows: gate/up rows, swiglu, the down
 * activation quantized, down rows, and the weighted sum of the routing weight.  The arithmetic is the
 * engine's CPU lane's (src/ds4/ds4_iq_cpu.cpp: iq_cpu_gate_up / iq_cpu_quant / iq_cpu_down) or, on a
 * card, the token kernels' the engine's own device lane runs (src/ds4/ds4_iq.cu: iq::gate_up / iq::down),
 * so a job this ABI serves is the job one of those lanes would have computed.
 *
 * THE CONTRACT.  A worker exports the four symbols below and nothing else is required of it:
 *
 *   strata_peer_compute_version()  the worker's name and capabilities, one printable line.  The daemon
 *                                  reads it twice - once for the ABI number before open(), once for the
 *                                  capabilities after it - and logs the second.
 *   strata_peer_compute_open(o)    a worker, or NULL with nothing to serve.  `o` is the daemon's
 *                                  opaque options string ("" when it has none).
 *   strata_peer_compute_run(...)   0 when the worker ran the batch - ok[k] then says which jobs it
 *                                  computed (1) and which it cannot serve (0).  Non-zero: the
 *                                  worker itself cannot serve anything (every ok is 0).  Either way the
 *                                  caller falls back locally for the zeros; a worker must never take
 *                                  the daemon down over one job it cannot do.
 *   strata_peer_compute_close(w)   releases the worker (NULL is a no-op).
 *
 * NOTHING HERE MAY CHANGE WITHOUT STRATA_PEER_COMPUTE_ABI going up: the daemon refuses a worker
 * whose version() names another number.  The struct is passed by pointer and never by value, so a field
 * added at the end keeps an old worker loadable only if the daemon's own version check allows it.
 *
 * ABI 2 adds strata_peer_job::resident (appended, so stage 1's field order stands): a row the daemon holds
 * in a --vram slab has no host window to read, so the job names the peer's own device addresses and the worker
 * computes straight out of that card's VRAM.  A worker without a device (the CPU worker) is never handed one: it
 * answers ok=0 for a resident job, which is stage 1's behaviour for those rows unchanged.
 *
 * WHICH DEVICE.  A worker that computes on a card names it in its version() line as a "gpu<N>" field (the peer
 * machine's own ordinal, the one hipSetDevice takes):
 *     strata-peer-compute/2 hip gfx1151 gpu1
 * The daemon compares that ordinal with the slab's before it hands a resident row over - the row's address is
 * meaningful on one card only, so a resident job for a worker on another device is ok=0 and the caller computes it
 * locally.  A worker that names no "gpu<N>" field has no device (the CPU worker) and is never handed a resident row.
 */
#ifndef STRATA_PEER_COMPUTE_ABI_H
#define STRATA_PEER_COMPUTE_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STRATA_PEER_COMPUTE_ABI 2

/* One expert's compute order, in the peer's own memory (the daemon resolves the row table to
 * pointers: the worker never sees a row index).  Every pointer stays valid for the duration of the run
 * call and no buffer overlaps another.
 */
typedef struct strata_peer_job {
    /* the layer's geometry, as iq_cpu_fmt takes it (ggml type ids for gate/up and down) */
    int32_t n_embd;
    int32_t n_ff;
    int32_t tg;
    int32_t td;
    /* activation rows (tokens) this job carries */
    int32_t n_rows;
    /* swiglu's clamp (0: none), the device grouped path's limit */
    float limit;
    /* the expert's matrices: n_ff rows of gate/up, n_embd rows of down.  The byte counts are what
     * the daemon's row table advertised for them; a worker refuses a count its own geometry does not
     * produce, which is how a row table built by another pack is caught here and not in the sums. */
    const uint8_t* gate;
    const uint8_t* up;
    const uint8_t* down;
    uint64_t gate_bytes;
    uint64_t down_bytes;
    /* the activation, as the device kernels round theirs: n_rows x n_embd int8 values, each row with a
     * scale a 32 values (n_rows x n_embd / 32 floats).  A device worker hands these bytes to the
     * kernels in place (they are exactly what those kernels take, so no rounding is lost on the way in); the
     * CPU worker dequantizes and quantizes again to the weight type's vec_dot_type, so its sums are the
     * lane's for these bytes and differ from the lane's for the raw floats by that second rounding alone. */
    const int8_t* xq;
    const float* xd;
    /* n_rows routing weights: y[t][j] = w[t] * (the expert's down output), i.e. the weighted sum the
     * caller would otherwise do.  NULL: every row's weight is 1. */
    const float* w;
    /* n_rows x n_embd floats out, written by the worker only for a job it serves */
    float* y;
    /* 0: gate/up/down are host buffers, exactly as stage 1 has them - a worker copies them into
     *    whatever memory its arithmetic reads.  1: they are the peer's own device addresses on the worker's
     *    device (a --vram row's address in the daemon's slab, resident there and never copied): the worker
     *    uses them in place, and refuses the job (ok=0) when they are not on its own device.  Appended in
     *    ABI 2. */
    int32_t resident;
} strata_peer_job;

/* The worker's name and capabilities, one printable line ("strata-peer-compute/2 cpu",
 * "strata-peer-compute/2 hip gfx1151 gpu1"), never NULL. */
const char* strata_peer_compute_version(void);

/* A worker for the daemon's options string, or NULL when this worker cannot serve at all. */
void* strata_peer_compute_open(const char* options);

/* Runs a batch (see the contract above): 0 when the worker ran it, non-zero when it cannot serve. */
int strata_peer_compute_run(void* worker, const strata_peer_job* jobs, int n, uint8_t* ok);

/* Releases a worker (NULL is a no-op). */
void strata_peer_compute_close(void* worker);

#ifdef __cplusplus
}
#endif

#endif /* STRATA_PEER_COMPUTE_ABI_H */
