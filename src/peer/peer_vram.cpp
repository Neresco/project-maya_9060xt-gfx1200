#include "peer_vram.hpp"

#include <dlfcn.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace strata::peer {
namespace {

constexpr int kHipOk = 0;                    // hipSuccess
constexpr int kHipMemcpyHostToDevice = 1;    // hipMemcpyHostToDevice's value in the HIP ABI
constexpr uint32_t kHsaOk = 0;               // HSA_STATUS_SUCCESS

/// ibv_reg_dmabuf_mr's own signature.  It is resolved by name, not called from the header: the verb
/// only exists from rdma-core 27 on, and a peer built against an older rdma-core must still compile
/// and run (it just cannot offer the VRAM tier).
using reg_dmabuf_mr_fn = ibv_mr* (*)(ibv_pd*, uint64_t offset, size_t length, uint64_t iova, int fd,
                                     int access);

/// The ROCm entry points, resolved by dlopen.  The signatures are the HIP/HSA C ABI spelled out by
/// hand so that no toolkit header (and so no toolkit) is needed to build the daemon.
struct Rocm {
    bool loaded = false;
    std::string why;                         // (why it is not usable, when it is not)
    int (*device_count)(int*) = nullptr;
    int (*set_device)(int) = nullptr;
    int (*mem_get_info)(size_t*, size_t*) = nullptr;
    int (*malloc)(void**, size_t) = nullptr;
    int (*free)(void*) = nullptr;
    int (*memcpy)(void*, const void*, size_t, int) = nullptr;
    int (*device_synchronize)() = nullptr;
    const char* (*error_string)(int) = nullptr;
    uint32_t (*export_dmabuf)(const void*, size_t, int*, uint64_t*) = nullptr;
    reg_dmabuf_mr_fn reg_dmabuf = nullptr;
};

template <typename Fn>
Fn sym(void* handle, const char* name) {
    return (Fn) dlsym(handle, name);
}

void* dlopen_first(const char* const* names, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (void* h = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL)) return h;
    }
    return nullptr;
}

Rocm load_rocm() {
    static const char* const hip_libs[] = {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6",
                                           "/opt/rocm/lib/libamdhip64.so",
                                           "/opt/rocm/lib/libamdhip64.so.7"};
    static const char* const hsa_libs[] = {"libhsa-runtime64.so", "libhsa-runtime64.so.1",
                                           "/opt/rocm/lib/libhsa-runtime64.so.1",
                                           "/opt/rocm/lib/libhsa-runtime64.so"};
    static const char* const ibv_libs[] = {"libibverbs.so.1", "libibverbs.so",
                                           "/usr/lib/x86_64-linux-gnu/libibverbs.so.1"};
    Rocm r;
    void* hip = dlopen_first(hip_libs, sizeof hip_libs / sizeof hip_libs[0]);
    if (hip == nullptr) {
        r.why = "no ROCm runtime: libamdhip64 is not loadable here";
        return r;
    }
    void* hsa = dlopen_first(hsa_libs, sizeof hsa_libs / sizeof hsa_libs[0]);
    if (hsa == nullptr) {
        r.why = "no ROCr runtime: libhsa-runtime64 is not loadable here";
        return r;
    }
    // every entry point the tier needs, by the name the runtime carries it under
    r.device_count = sym<decltype(r.device_count)>(hip, "hipGetDeviceCount");
    r.set_device = sym<decltype(r.set_device)>(hip, "hipSetDevice");
    r.mem_get_info = sym<decltype(r.mem_get_info)>(hip, "hipMemGetInfo");
    r.malloc = sym<decltype(r.malloc)>(hip, "hipMalloc");
    r.free = sym<decltype(r.free)>(hip, "hipFree");
    r.memcpy = sym<decltype(r.memcpy)>(hip, "hipMemcpy");
    r.device_synchronize = sym<decltype(r.device_synchronize)>(hip, "hipDeviceSynchronize");
    if (r.device_count == nullptr || r.set_device == nullptr || r.mem_get_info == nullptr ||
        r.malloc == nullptr || r.free == nullptr || r.memcpy == nullptr ||
        r.device_synchronize == nullptr) {
        r.why = "libamdhip64 is missing an entry point the VRAM tier needs";
        return r;
    }
    // (hipGetErrorString is optional: it is deprecated, and the numeric code is enough to debug)
    r.error_string = sym<decltype(r.error_string)>(hip, "hipGetErrorString");
    r.export_dmabuf = sym<decltype(r.export_dmabuf)>(hsa, "hsa_amd_portable_export_dmabuf");
    if (r.export_dmabuf == nullptr) {
        r.why = "libhsa-runtime64 has no hsa_amd_portable_export_dmabuf";
        return r;
    }
    void* ibv = dlopen_first(ibv_libs, sizeof ibv_libs / sizeof ibv_libs[0]);
    r.reg_dmabuf = ibv == nullptr ? nullptr : sym<reg_dmabuf_mr_fn>(ibv, "ibv_reg_dmabuf_mr");
    if (r.reg_dmabuf == nullptr) {
        r.why = "this rdma-core has no ibv_reg_dmabuf_mr (27 or newer is needed to register VRAM)";
        return r;
    }
    r.loaded = true;
    return r;
}

const Rocm& rocm() {
    static const Rocm r = load_rocm();       // (resolved once, on the first --vram slab)
    return r;
}

std::string hip_why(int code) {
    const Rocm& r = rocm();
    const char* s = r.error_string ? r.error_string(code) : nullptr;
    return std::to_string(code) + (s ? std::string(" (") + s + ")" : "");
}

std::string mib(size_t bytes) { return std::to_string(bytes >> 20); }

/// A GPU allocation that the NIC can read: hipMalloc, exported as a dmabuf, registered as an MR at
/// the allocation's own device address (so a row's advertised address is its offset from there).
class VramSlab final : public Slab {
public:
    VramSlab(int gpu, size_t bytes) : gpu_(gpu), bytes_(bytes) {}
    ~VramSlab() override { release(); }

    bool open(ibv_pd* pd, std::string& err) {
        const Rocm& r = rocm();
        if (!r.loaded) {
            err = r.why;
            return false;
        }
        int count = 0;
        if (r.device_count(&count) != kHipOk || count <= 0) {
            err = "the peer machine reports " + std::to_string(count) + " HIP devices";
            return false;
        }
        if (gpu_ < 0 || gpu_ >= count) {
            err = "GPU " + std::to_string(gpu_) + " does not exist on the peer (" +
                  std::to_string(count) + " HIP devices)";
            return false;
        }
        const int set = r.set_device(gpu_);
        if (set != kHipOk) {
            err = "hipSetDevice(" + std::to_string(gpu_) + ") failed: " + hip_why(set);
            return false;
        }
        size_t free_bytes = 0, total = 0;
        if (r.mem_get_info(&free_bytes, &total) == kHipOk) {
            say("vram tier: GPU " + std::to_string(gpu_) + " has " + mib(free_bytes) + " MiB free of " +
                mib(total) + " MiB");
        }
        const int alloc = r.malloc(&buf_, bytes_);
        if (alloc != kHipOk) {
            err = "hipMalloc of " + mib(bytes_) + " MiB on GPU " + std::to_string(gpu_) +
                  " failed: " + hip_why(alloc) + " (is something else using that card?)";
            buf_ = nullptr;
            return false;
        }
        int fd = -1;
        uint64_t offset = 0;
        const uint32_t st = r.export_dmabuf(buf_, bytes_, &fd, &offset);
        if (st != kHsaOk) {
            err = "hsa_amd_portable_export_dmabuf failed (status " + std::to_string(st) +
                  "): this allocation cannot be shared with the NIC";
            r.free(buf_);
            buf_ = nullptr;
            return false;
        }
        // the iova is ours to choose; the allocation's own device address keeps it unique per MR and
        // makes an advertised row address traceable back to the slab in a log
        iova_ = (uint64_t) (uintptr_t) buf_;
        mr_ = r.reg_dmabuf(pd, offset, bytes_, iova_, fd,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
        if (mr_ == nullptr) {
            err = "ibv_reg_dmabuf_mr failed (errno " + std::to_string(errno) + ": " +
                  std::strerror(errno) + "): this NIC cannot read this GPU's memory";
            ::close(fd);
            r.free(buf_);
            buf_ = nullptr;
            return false;
        }
        fd_ = fd;
        rkey_ = mr_->rkey;
        char hex[32];
        std::snprintf(hex, sizeof hex, "0x%llx", (unsigned long long) iova_);
        char key[16];
        std::snprintf(key, sizeof key, "0x%x", rkey_);
        say("vram slab up: GPU " + std::to_string(gpu_) + ", " + mib(bytes_) + " MiB at " + hex +
            ", rkey " + key);
        return true;
    }

    void* host() override { return nullptr; }          // (a pread cannot land on GPU pages)

    bool put(size_t off, const void* src, size_t bytes, std::string& err) override {
        const Rocm& r = rocm();
        const int st = r.memcpy((char*) buf_ + off, src, bytes, kHipMemcpyHostToDevice);
        if (st != kHipOk) {
            err = "hipMemcpy of " + std::to_string(bytes) + " bytes into VRAM failed: " + hip_why(st);
            return false;
        }
        // the initiator reads the row through the dmabuf, not through this process's mapping: wait
        // until the copy has actually landed before the row is advertised
        const int sync = r.device_synchronize();
        if (sync != kHipOk) {
            err = "hipDeviceSynchronize after a VRAM row failed: " + hip_why(sync);
            return false;
        }
        return true;
    }

    uint64_t addr_at(size_t off) const override { return iova_ + off; }
    size_t bytes() const override { return bytes_; }
    uint32_t rkey() const override { return rkey_; }
    // (the card this slab was allocated on: a compute worker on that same ordinal can read a row in place)
    int gpu() const override { return gpu_; }

private:
    void release() {
        if (mr_ != nullptr) ibv_dereg_mr(mr_);
        if (fd_ >= 0) ::close(fd_);
        if (buf_ != nullptr && rocm().loaded) rocm().free(buf_);
        mr_ = nullptr;
        fd_ = -1;
        buf_ = nullptr;
    }

    int gpu_ = 0;
    size_t bytes_ = 0;
    void* buf_ = nullptr;
    int fd_ = -1;
    uint64_t iova_ = 0;
    ibv_mr* mr_ = nullptr;
    uint32_t rkey_ = 0;
};

}   // namespace

std::unique_ptr<Slab> make_vram_slab(int gpu, size_t bytes, ibv_pd* pd, std::string& err) {
    if (bytes == 0) {
        err = "a VRAM slab of 0 bytes is not a slab";
        return nullptr;
    }
    auto slab = std::make_unique<VramSlab>(gpu, bytes);
    if (!slab->open(pd, err)) return nullptr;
    return slab;
}

}   // namespace strata::peer
