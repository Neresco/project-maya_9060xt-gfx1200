// The daemon's compute worker: one shared object loaded by path (the daemon's --compute), its ABI
// (peer_compute_abi.h) resolved with dlopen/dlsym and nothing else.  Same discipline as peer_vram.cpp
// and for the same reason: strata-peer must still build and run on a peer machine with no ROCm at all,
// so the only place a GPU runtime could ever appear is inside the worker .so, never in the daemon.
//
// A path that does not load is a start-up error, not a slow path: a worker that cannot start must say so
// on the spot, not at the third layer of a decode (docs/PEER_COMPUTE.md, "Risks and traps").  A worker
// that loads but cannot serve a particular job answers ok=0 for it, and the caller computes locally.
#pragma once

#include "peer_compute_abi.h"

#include <cstddef>
#include <memory>
#include <string>

namespace strata::peer {

/// One loaded worker: the dlopen handle, the ABI's three entry points and the worker itself.  Not
/// copyable (the handle is one) and closed by its destructor, so a daemon that loses its worker at
/// start-up never leaks the .so.
class ComputeWorker {
public:
    ComputeWorker() = default;
    ~ComputeWorker();
    ComputeWorker(const ComputeWorker&) = delete;
    ComputeWorker& operator=(const ComputeWorker&) = delete;

    /// Loads `path` and opens a worker.  False (err) when the file is not loadable, when it is not a
    /// worker at all (an entry point missing), when it speaks another ABI, or when it refuses to open.
    /// `options` is the daemon's --compute-options string, handed to the worker as it is ("" when the
    /// operator gave none: the worker picks its own device, as the CPU worker has nothing to pick).
    bool open(const std::string& path, const std::string& options, std::string& err);
    void close();
    bool is_open() const { return worker_ != nullptr; }

    /// The worker's own name and capabilities (its version() line), for the daemon's log.
    const std::string& caps() const { return caps_; }
    const std::string& path() const { return path_; }

    /// The HIP device ordinal the worker computes on, or -1 when it has no device (the CPU worker, or a
    /// worker that names none).  The worker reports it in its version() line (peer_compute_abi.h's
    /// "gpu<N>" field); the daemon compares it with a --vram slab's own ordinal before it hands a
    /// resident row over, because such an address means something on that one card only.
    int gpu() const { return gpu_; }

    /// One batch: ok[k] is 1 for a job the worker computed and 0 for one it cannot serve (the caller
    /// computes that one itself).  False (err) when the worker as a whole could not run the batch: every
    /// ok is then 0, and the session goes on.
    bool run(const strata_peer_job* jobs, int n, uint8_t* ok, std::string& err);

private:
    void* handle_ = nullptr;                       // the .so
    const char* (*version_)() = nullptr;
    void* (*open_)(const char*) = nullptr;
    int (*run_)(void*, const strata_peer_job*, int, uint8_t*) = nullptr;
    void (*close_)(void*) = nullptr;
    void* worker_ = nullptr;                       // the worker itself
    int gpu_ = -1;                                 // (the worker's device, -1: none)
    std::string caps_, path_;
};

/// A worker loaded from `path` (empty: nothing is loaded, and the daemon serves storage only), or null
/// with the reason - the daemon's --compute contract is --vram's: given and broken is a start-up error.
/// `options` is its --compute-options string (empty: whatever the worker picks).
std::unique_ptr<ComputeWorker> make_compute_worker(const std::string& path, const std::string& options,
                                                 std::string& err);

/// `bytes` at `p` as lowercase hex (the protocol's body encoding for the activation, the scales and the
/// sums: arbitrary bytes would carry newlines and break the line framing).
std::string hex_of(const void* p, size_t bytes);

/// Exactly `bytes` bytes of `hex` decoded into `out`, false on a stray digit or a length that is not the
/// one the field needs (a job whose body is short is a job the daemon refuses, not one it misreads).
bool unhex(const std::string& hex, void* out, size_t bytes);

}   // namespace strata::peer
