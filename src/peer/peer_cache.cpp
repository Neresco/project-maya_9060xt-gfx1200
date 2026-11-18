#include "peer_cache.hpp"

#include "peer_budget.hpp"
#include "peer_vram.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <utility>

namespace strata::peer {

namespace {

/// Pinned system RAM: the Mr over a page-aligned malloc.  Rows are pread'd straight into it, so
/// put() is never reached (host() is the window itself).
class RamSlab final : public Slab {
public:
    bool open(ibv_pd* pd, size_t bytes, std::string& err) {
        if (!mr_.alloc(pd, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, err)) return false;
        buf_ = mr_.buf();
        bytes_ = mr_.bytes();
        rkey_ = mr_.rkey();
        return true;
    }
    void* host() override { return buf_; }
    bool put(size_t, const void*, size_t, std::string&) override { return false; }   // (host() covers it)
    uint64_t addr_at(size_t off) const override { return (uint64_t) (uintptr_t) ((char*) buf_ + off); }
    size_t bytes() const override { return bytes_; }
    uint32_t rkey() const override { return rkey_; }

private:
    Mr mr_;
    void* buf_ = nullptr;
    size_t bytes_ = 0;
    uint32_t rkey_ = 0;
};

}   // namespace

void (*PeerCache::alloc_log_)(uint64_t bytes) = nullptr;

bool PeerCache::open(Device* dev, uint64_t budget, uint64_t arena_bytes, std::string& err) {
    dev_ = dev;
    budget_ = budget ? budget : budget_probe(32ULL << 30);
    arena_bytes_ = arena_bytes ? arena_bytes : 2ULL << 30;
    if (budget_ == 0) {
        err = "no RAM budget: the cgroup-aware probe found nothing free";
        return false;
    }
    if (arena_bytes_ > budget_) arena_bytes_ = budget_;
    say("slab budget " + std::to_string(budget_ >> 20) + " MiB in " +
        std::to_string(arena_bytes_ >> 20) + " MiB arenas");
    return true;
}

void PeerCache::close() {
    close_shards();
    arenas_.clear();
    where_.clear();
    dropped_.clear();
    stage_.clear();
    std::vector<char>().swap(stage_);
    cached_bytes_ = 0;
    ram_bytes_ = 0;
    next_arena_ = 0;
}

bool PeerCache::add_vram(int gpu, uint64_t bytes, std::string& err) {
    if (dev_ == nullptr) {
        err = "a VRAM slab needs the daemon's device open first";
        return false;
    }
    say("allocating a " + std::to_string(bytes >> 20) + " MiB VRAM slab on GPU " +
        std::to_string(gpu) + " (exporting a dmabuf, be patient)");
    std::unique_ptr<Slab> slab = make_vram_slab(gpu, (size_t) bytes, dev_->pd(), err);
    if (slab == nullptr) return false;
    ArenaRec rec;
    rec.slab = std::move(slab);
    rec.pool = (int) next_pool_++;
    auto* out = &arenas_[next_arena_++];
    *out = std::move(rec);
    vram_total_ += bytes;   // (the slab's spec, not what it has taken: capacity is a start-up number)
    return true;
}

std::pair<uint32_t, PeerCache::ArenaRec*> PeerCache::pick_vram(size_t bytes) {
    for (auto& kv : arenas_) {
        if (kv.second.pool == 0) continue;
        if (kv.second.used + bytes <= kv.second.slab->bytes()) return {kv.first, &kv.second};
    }
    return {0, nullptr};
}

bool PeerCache::open_shards(const std::vector<std::string>& names, const std::vector<uint64_t>& sizes,
                            const std::string& pack_dir, std::string& err) {
    close_shards();
    shard_fd_.reserve(names.size());
    shard_size_.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        // the manifest's names are basenames; the daemon resolves them inside its own pack dir (the
        // two machines' mount points differ; the fingerprint is what actually binds the copies)
        std::string clean = names[i];
        const size_t slash = clean.find_last_of("/\\");
        if (slash != std::string::npos) clean = clean.substr(slash + 1);
        const std::string path = pack_dir + (pack_dir.back() == '/' ? "" : "/") + clean;
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "shard " + clean + " not found under " + pack_dir + " (" + std::strerror(errno) + ")";
            close_shards();
            return false;
        }
        const off_t real = lseek(fd, 0, SEEK_END);
        if (real < 0 || (uint64_t) real != sizes[i]) {
            err = "shard " + clean + " is " + std::to_string((uint64_t) std::max<off_t>(real, 0)) +
                  " bytes, the manifest said " + std::to_string(sizes[i]);
            close_shards();
            return false;
        }
        shard_fd_.push_back(fd);
        shard_size_.push_back(sizes[i]);
    }
    return true;
}

void PeerCache::close_shards() {
    for (int fd : shard_fd_) ::close(fd);
    shard_fd_.clear();
    shard_size_.clear();
}

std::string pack_fingerprint(const std::string& dir, const std::vector<std::string>& names,
                             const std::vector<uint64_t>& sizes, std::string& err) {
    if (names.size() != sizes.size() || names.empty()) {
        err = "a pack of " + std::to_string(names.size()) + " shards and " +
              std::to_string(sizes.size()) + " sizes is not a pack";
        return "";
    }
    char piece[80];
    const size_t edge = 1ULL << 20;                  // first and last MiB of each shard
    std::vector<char> buf(edge);
    uint64_t total = 0;
    std::string out;
    for (size_t i = 0; i < names.size(); ++i) {
        std::string clean = names[i];
        const size_t slash = clean.find_last_of("/\\");
        if (slash != std::string::npos) clean = clean.substr(slash + 1);
        const std::string path = dir + (dir.back() == '/' ? "" : "/") + clean;
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "shard " + clean + " not found under " + dir + " (" + strerror(errno) + ")";
            return "";
        }
        const uint64_t size = sizes[i];
        total += size;
        uint64_t h = 1469598103934665603ULL;         // the 64-bit FNV-1a offset basis
        const size_t head = (size_t) std::min<uint64_t>(edge, size);
        if (pread(fd, buf.data(), head, 0) != (ssize_t) head) {
            err = "shard " + clean + ": head pread failed";
            close(fd);
            return "";
        }
        for (size_t b = 0; b < head; ++b) h = (h ^ (unsigned char) buf[b]) * 1099511628211ULL;
        if (size > edge) {
            if (pread(fd, buf.data(), edge, (off_t) (size - edge)) != (ssize_t) edge) {
                err = "shard " + clean + ": tail pread failed";
                close(fd);
                return "";
            }
            for (size_t b = 0; b < edge; ++b) h = (h ^ (unsigned char) buf[b]) * 1099511628211ULL;
        }
        close(fd);
        std::snprintf(piece, sizeof piece, "%s%llu:%016llx", out.empty() ? "" : ";",
                      (unsigned long long) size, (unsigned long long) h);
        out += piece;
    }
    std::snprintf(piece, sizeof piece, "|%llu", (unsigned long long) total);
    return out + piece;
}

void PeerCache::drop_arena(uint32_t id) {
    const auto arena = arenas_.find(id);
    if (arena == arenas_.end()) return;
    const bool ram = arena->second.pool == 0;
    for (auto it = where_.begin(); it != where_.end();) {
        if (it->second.arena == id) {
            dropped_.push_back(it->first);
            cached_bytes_ -= it->second.len;
            if (ram) ram_bytes_ -= it->second.len;
            it = where_.erase(it);
        } else {
            ++it;
        }
    }
    // (cached_bytes_ is the sum of the rows' lens, which is what an arena's used counter is too: subtracting it as
    // well underflowed the budget - one eviction made every later fill evict the whole slab and then refuse)
    arenas_.erase(arena);
}

PeerCache::ArenaRec* PeerCache::room_for(size_t bytes, uint64_t stamp, uint32_t* id_out, std::string& err) {
    for (auto& kv : arenas_) {
        if (kv.second.pool != 0) continue;      // (a VRAM slab is chosen by pick_vram, never by the RAM pool)
        if (kv.second.used + bytes <= kv.second.slab->bytes()) {
            kv.second.stamp = stamp;
            *id_out = kv.first;
            return &kv.second;
        }
    }
    // nothing fits: a fresh arena under the budget, else LRU arenas out whole until it does
    const size_t cap = std::max<size_t>(bytes, arena_bytes_);
    while (ram_bytes_ + cap > budget_ && !arenas_.empty()) {
        uint32_t oldest = 0;
        uint64_t best = UINT64_MAX;
        for (const auto& kv : arenas_) {
            if (kv.second.pool != 0) continue;  // (VRAM rows do not compete with the RAM budget)
            if (kv.second.stamp < best) {
                best = kv.second.stamp;
                oldest = kv.first;
            }
        }
        if (best == UINT64_MAX) break;          // (only VRAM slabs are left: nothing to evict here)
        drop_arena(oldest);
    }
    if (ram_bytes_ + cap > budget_) {
        err = "the slab's budget (" + std::to_string(budget_ >> 20) + " MiB) is exhausted and its " +
              "rows are all in use; raise the daemon's --ram-gb";
        return nullptr;
    }
    ArenaRec rec;
    auto ram = std::make_unique<RamSlab>();
    log_alloc(cap);                      // GiBs of pinning mid-request: the caller may stall here
    if (!ram->open(dev_->pd(), cap, err)) return nullptr;
    rec.slab = std::move(ram);
    rec.stamp = stamp;
    const uint32_t id = next_arena_++;
    auto* out = &arenas_[id];
    *out = std::move(rec);
    *id_out = id;
    return out;
}

bool PeerCache::host_row(long long ri, const void*& p, uint32_t& len) const {
    p = nullptr;
    len = 0;
    const auto it = where_.find(ri);
    if (it == where_.end()) return false;
    const auto arena = arenas_.find(it->second.arena);
    if (arena == arenas_.end() || arena->second.slab->host() == nullptr) return false;
    // the row's window is the arena's host base plus its bump offset (the offset is the difference between
    // the row's address and the arena's own address at 0 - never the address itself, which is 0 for the first row)
    p = (const char*) arena->second.slab->host() + (it->second.addr - arena->second.slab->addr_at(0));
    len = it->second.len;
    return true;
}

bool PeerCache::dev_row(long long ri, uint64_t& addr, uint32_t& len, int& gpu) const {
    addr = 0;
    len = 0;
    gpu = -1;
    const auto it = where_.find(ri);
    if (it == where_.end()) return false;
    const auto arena = arenas_.find(it->second.arena);
    if (arena == arenas_.end()) return false;
    // (the slab's own ordinal, not the row's: a row's address in a device slab is that card's alone)
    gpu = arena->second.slab->gpu();
    if (gpu < 0) return false;             // (host memory: host_row's business)
    addr = it->second.addr;
    len = it->second.len;
    return true;
}

bool PeerCache::fill(const std::vector<long long>& rows, const std::vector<Row>& table, uint64_t stamp,
                     std::vector<Cached>& out, std::string& err) {
    out.clear();
    dropped_.clear();
    for (long long ri : rows) {
        if (ri < 0 || (size_t) ri >= table.size()) {
            err = "row " + std::to_string(ri) + " is outside the row table (" +
                  std::to_string(table.size()) + " rows)";
            return false;
        }
        const Row& r = table[(size_t) ri];
        // (a row with no bytes - a dense layer, or a shard this daemon does not hold - is never asked: room_for(0)
        // would hand out a whole arena.  A shard index past the manifest's is a bug on the initiator's side; the
        // pread below would read through an fd that is not a shard at all)
        if (r.len == 0) {
            err = "row " + std::to_string(ri) + " has no bytes to fill";
            return false;
        }
        if ((size_t) r.shard >= shard_fd_.size()) {
            err = "row " + std::to_string(ri) + " names shard " + std::to_string(r.shard) + ", the manifest " +
                  "carried " + std::to_string(shard_fd_.size());
            return false;
        }
        if (r.off > shard_size_[(size_t) r.shard] || r.len > shard_size_[(size_t) r.shard] - r.off) {
            err = "row " + std::to_string(ri) + "'s [" + std::to_string(r.off) + ", +" + std::to_string(r.len) +
                  ") runs past the end of shard " + std::to_string(r.shard) + " (" +
                  std::to_string(shard_size_[(size_t) r.shard]) + " bytes)";
            return false;
        }
        const auto have = where_.find(ri);
        if (have != where_.end()) {
            arenas_[have->second.arena].stamp = stamp;
            out.push_back(have->second);
            continue;
        }
        // VRAM first (the explicit, scarce tier, in --vram order), the RAM slab after it: a row the
        // VRAM slab cannot hold spills instead of failing, and the row map does not care which it got
        std::pair<uint32_t, ArenaRec*> land = pick_vram((size_t) r.len);
        if (land.second == nullptr) {
            uint32_t id = 0;
            ArenaRec* ram = room_for((size_t) r.len, stamp, &id, err);
            if (ram == nullptr) return false;
            land = std::make_pair(id, ram);
        }
        ArenaRec& a = *land.second;
        Slab* slab = a.slab.get();
        void* dst = slab->host();
        if (dst != nullptr) dst = (char*) dst + a.used;
        else {
            if (stage_.size() < (size_t) r.len) stage_.resize((size_t) r.len);
            dst = stage_.data();
        }
        const ssize_t got = pread(shard_fd_[r.shard], dst, r.len, (off_t) r.off);
        if (got < 0 || (size_t) got != r.len) {
            err = "shard " + std::to_string(r.shard) + " pread at " + std::to_string(r.off) +
                  " failed (" + std::strerror(errno) + ")";
            return false;
        }
        if (slab->host() == nullptr && !slab->put(a.used, stage_.data(), r.len, err)) return false;
        Cached c;                     // (the map's arena id: the daemon's own accounting only)
        c.addr = slab->addr_at(a.used);
        c.len = r.len;
        c.rkey = slab->rkey();
        c.arena = land.first;
        a.used += r.len;
        a.stamp = stamp;
        cached_bytes_ += r.len;
        if (a.pool == 0) ram_bytes_ += r.len;
        where_[ri] = c;
        out.push_back(c);
    }
    ++clock_;
    return true;
}

}   // namespace strata::peer
