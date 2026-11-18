// The daemon's opt-in VRAM tier: a GPU allocation registered as a dmabuf MR, so the peer can hold
// expert rows in VRAM and let the initiator RDMA READ them straight out of it - no pinned system RAM
// spent on rows the peer GPU already owns, and no copy on the wire.
//
// This is the one place the daemon touches a GPU runtime, and it reaches it through dlopen on
// purpose: strata-peer must still start and serve the RAM slab on a peer machine with no ROCm at all,
// and --vram is the only thing that needs the runtime.  Nothing here may include a HIP or HSA header.
#pragma once

#include "peer_cache.hpp"     // Slab

#include <infiniband/verbs.h>

#include <cstddef>
#include <memory>
#include <string>

namespace strata::peer {

/// A slab of exactly `bytes` on HIP device `gpu` (the peer machine's own ordinal), registered as a
/// dmabuf MR on `pd` - the daemon's protection domain, since the reader reaches it through the same
/// port as the RAM arenas.  Null and `err` set when the ROCm runtime is absent, the allocation does
/// not fit on that card, or the NIC cannot register a dmabuf.
std::unique_ptr<Slab> make_vram_slab(int gpu, size_t bytes, ibv_pd* pd, std::string& err);

}   // namespace strata::peer
