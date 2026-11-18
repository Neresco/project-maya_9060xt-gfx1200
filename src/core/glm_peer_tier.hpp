// Phase 3: the engine's end of the peer expert tier (`strata-peer` on the other machine).
//
// WHAT THIS REPLACES.  A decode miss that is in neither VRAM nor the RAM tier is read off the local
// disk (`fast_read_part`, three slices of the pack per expert).  This tier offers the same bytes from
// the PEER's pinned slab over RoCE: the daemon holds the pack's shards in a budget-capped registered
// arena and its NIC answers RDMA READs, so the engine's service thread posts the reads and the peer's
// disk and CPU never enter the decode.  The bytes are identical by construction - a row is a shard
// byte range the initiator named, and the handshake refuses a pack whose fingerprint differs - so a
// token computed over the peer is the token the disk would have produced.
//
// EVERYTHING HERE IS OPTIONAL.  A row the daemon does not advertise (it never filled it, or an arena
// was evicted), a destination the NIC would not register, a dead session: each answers "not served"
// and the caller's disk read runs exactly as before, byte for byte.  Nothing in this file may turn a
// working decode into a failing one, which is why the sticky `down_` is the only error state.
//
// ONE SESSION PER PROCESS.  The daemon serves one initiator at a time (its slab is the point), so a
// layer split's parts share this session: one QP, one control channel, and a mutex around each
// part's serves - the wire is one link, and serialising the parts' bursts costs far less than a
// second session nobody would accept.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {
namespace glmfast {

/// One slice of the pack, per (layer, role): expert `e`'s part is `[off + e * stride, + len)` of
/// shard `shard`.  role 0 = gate, 1 = up, 2 = down, exactly as `fast_read_part` lays them out.
/// `shard` < 0 or `len` 0: this layer has no such row (a dense layer, or a pack layout the tier does
/// not serve) and every expert of it stays on the disk path.
struct PeerRowSpec {
    int32_t shard = -1;
    uint64_t off = 0;
    uint64_t stride = 0;
    uint32_t len = 0;
};

class PeerTier {
public:
    /// The three parts of an expert's blob, in `fast_read_part`'s order.
    static constexpr int kRoles = 3;

    /// The session's settings, read from the environment by the caller (the engine's flags).
    struct Config {
        std::string host;            ///< the peer's control address (its IPv4 on the RoCE link)
        int port = 9447;
        std::string device;          ///< the local ibverbs device name ("" = the first one found)
        std::string pack_dir;        ///< the LOCAL copy of the pack: fingerprinted, never read
        std::vector<std::string> shard_names;
        std::vector<uint64_t> shard_sizes;
        int64_t n_layers = 0;
        int64_t n_expert = 0;
        std::vector<PeerRowSpec> specs;   ///< `n_layers * kRoles`
        int window = 48;             ///< outstanding 2 MB chunks (the daemon sizes its QP to it)
        int chunk_mb = 2;
        int timeout_ms = 1000;       ///< one batch's deadline (a layer's misses, not one row)
    };

    /// Opens the process-wide session (the first part of a split opens it; the others attach).
    /// nullptr on refusal, with `err` saying why - the engine reports it and runs on the disk path.
    static PeerTier* open_session(const Config& cfg, std::string& err);

    /// Registers `[base, base + bytes)` as a READ destination (a RAM-tier piece, a staging buffer).
    /// false when the NIC refuses it: that memory then simply is not offered to the tier.
    bool register_dst(void* base, size_t bytes);

    /// The first row of `(layer, expert)` (roles follow at +0/+1/+2), or -1 when this layer is not
    /// served over the peer at all.
    long long row_of(int64_t layer, int64_t expert) const;

    /// **THE DECODE PATH.**  Reads `n` rows into `dst[k]` (the exact place the disk read would have
    /// written), hinting the ones the daemon has not advertised yet in one round trip first.  `ok[k]`
    /// is 1 for a landed row and 0 for one the caller must read from its own disk: unadvertised,
    /// dropped, a destination the NIC would not register, or a session that just died.  Never throws,
    /// never blocks past the batch timeout, and serialised across a split's parts.
    void serve(int n, const long long* rows, uint8_t* const* dst, char* ok);

    bool up() const { return up_; }
    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }

    /// The tiers line's peer counters, said once a process (the split's parts share the session).
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

    // (the peer types stay out of this header: the session lives behind this pointer)
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mu_;
    std::vector<Region> regions_;
    std::vector<uint32_t> len_of_;    ///< row -> its length (0: never served)
    std::vector<uint8_t> served_layer_;   ///< layer -> 1 when every role has a row
    int64_t n_layers_ = 0, n_expert_ = 0;
    bool up_ = false;
    // stats (under mu_)
    uint64_t rows_read_ = 0, bytes_ = 0, hints_ = 0, unadvertised_ = 0, batches_ = 0, us_ = 0;
    std::string why_;
};

}  // namespace glmfast
}  // namespace strata::core
