#include "peer_compute.hpp"

#include "peer_rdma.hpp"     // say (the daemon's log voice)

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>

namespace strata::peer {
namespace {

template <typename Fn>
Fn sym(void* handle, const char* name) {
    return (Fn) dlsym(handle, name);
}

int nib(char c) {
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
}

}   // namespace

ComputeWorker::~ComputeWorker() {
    close();
}

void ComputeWorker::close() {
    // (the worker goes before its code does: close() is a call into the .so, and dlclose after it
    // would unmap the function pointer that is still running)
    if (worker_ != nullptr && close_ != nullptr) close_(worker_);
    worker_ = nullptr;
    version_ = nullptr;
    open_ = nullptr;
    run_ = nullptr;
    close_ = nullptr;
    if (handle_ != nullptr) dlclose(handle_);
    handle_ = nullptr;
    caps_.clear();
    path_.clear();
    gpu_ = -1;
}

bool ComputeWorker::open(const std::string& path, const std::string& options, std::string& err) {
    close();
    if (path.empty()) {
        err = "a compute worker needs a path";
        return false;
    }
    // RTLD_NOW: a worker with a symbol it cannot resolve must fail here, at start-up, not on the
    // third layer of a decode.  RTLD_LOCAL: the worker's own symbols (ggml's among them) stay inside
    // it - a peer machine's daemon must not have its own globals shadowed by a worker's.
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
        const char* why = dlerror();
        err = "the compute worker " + path + " did not load: " + (why != nullptr ? why : "no reason given");
        return false;
    }
    version_ = sym<decltype(version_)>(h, "strata_peer_compute_version");
    open_ = sym<decltype(open_)>(h, "strata_peer_compute_open");
    run_ = sym<decltype(run_)>(h, "strata_peer_compute_run");
    close_ = sym<decltype(close_)>(h, "strata_peer_compute_close");
    if (version_ == nullptr || open_ == nullptr || run_ == nullptr || close_ == nullptr) {
        err = "the compute worker " + path + " is not one: strata_peer_compute_" +
              (version_ == nullptr ? "version" : open_ == nullptr ? "open" : run_ == nullptr ? "run" : "close") +
              " is missing";
        dlclose(h);
        return false;
    }
    const char* caps = version_();
    if (caps == nullptr) {
        err = "the compute worker " + path + " names no version";
        dlclose(h);
        return false;
    }
    caps_ = caps;
    // the ABI is versioned by the worker's own line, which begins "strata-peer-compute/<abi>"
    const std::string mine = "strata-peer-compute/" + std::to_string(STRATA_PEER_COMPUTE_ABI);
    if (caps_.compare(0, mine.size(), mine) != 0) {
        err = "the compute worker " + path + " speaks \"" + caps_ + "\", this daemon speaks " + mine;
        dlclose(h);
        caps_.clear();
        return false;
    }
    handle_ = h;
    worker_ = open_(options.c_str());
    if (worker_ == nullptr) {
        err = "the compute worker " + path + " refused to open";
        close();
        return false;
    }
    // the capabilities again, now that the worker is open: a worker that only learns which card it runs on at
    // open (the HIP worker) names it in the second reading, and its "gpu<N>" field is what a --vram slab's ordinal is
    // compared with.  (the first reading is the ABI's, which cannot wait for open: another ABI's worker is never opened)
    caps_ = version_();
    for (size_t at = 0; at < caps_.size();) {
        const size_t sp = caps_.find(' ', at);
        const size_t end = sp == std::string::npos ? caps_.size() : sp;
        const std::string tok = caps_.substr(at, end - at);
        if (tok.size() > 3 && tok.compare(0, 3, "gpu") == 0 &&
            tok.find_first_not_of("0123456789", 3) == std::string::npos) {
            gpu_ = std::atoi(tok.c_str() + 3);
            break;
        }
        at = end + 1;
    }
    path_ = path;
    say("compute worker " + caps_ + " loaded from " + path);
    return true;
}

bool ComputeWorker::run(const strata_peer_job* jobs, int n, uint8_t* ok, std::string& err) {
    if (worker_ == nullptr) {
        err = "no compute worker is loaded";
        return false;
    }
    if (jobs == nullptr || ok == nullptr || n <= 0) {
        err = "a compute batch of " + std::to_string(n) + " jobs is not a batch";
        return false;
    }
    std::memset(ok, 0, (size_t) n);
    const int rc = run_(worker_, jobs, n, ok);
    if (rc != 0) {
        std::memset(ok, 0, (size_t) n);   // (a worker that cannot serve answers nothing)
        err = "the compute worker " + caps_ + " cannot serve this batch (rc " + std::to_string(rc) + ")";
        return false;
    }
    return true;
}

std::unique_ptr<ComputeWorker> make_compute_worker(const std::string& path, const std::string& options,
                                                 std::string& err) {
    auto w = std::make_unique<ComputeWorker>();
    if (!w->open(path, options, err)) return nullptr;
    return w;
}

std::string hex_of(const void* p, size_t bytes) {
    static const char* digits = "0123456789abcdef";
    const unsigned char* b = (const unsigned char*) p;
    std::string out;
    out.reserve(bytes * 2);
    for (size_t i = 0; i < bytes; ++i) {
        out += digits[b[i] >> 4];
        out += digits[b[i] & 15];
    }
    return out;
}

bool unhex(const std::string& hex, void* out, size_t bytes) {
    if (hex.size() != bytes * 2) return false;
    unsigned char* b = (unsigned char*) out;
    for (size_t i = 0; i < bytes; ++i) {
        const int hi = nib(hex[2 * i]), lo = nib(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        b[i] = (unsigned char) ((hi << 4) | lo);
    }
    return true;
}

}   // namespace strata::peer
