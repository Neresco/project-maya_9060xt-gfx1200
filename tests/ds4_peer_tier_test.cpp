// The DeepSeek peer tier's end-to-end test, no GPU and no pack: a fixture of shard files, the real strata-peer
// daemon as a subprocess, and ds4_peer_tier's session and queue against it - the row table's arithmetic (unequal
// per-layer expert counts, a role the peer does not host), sub-range pulls, the DS4_FLOW_DOWN_PART compaction, and
// the fallback that must always keep the flow's accounting honest: a destination outside a registered piece, a row
// the tier does not serve, and a daemon killed under it all end as the local job IoQueue would have run.
//
// Skipped with a note when the machine has no RDMA device.  STRATA_PEER_TEST_DEV names the ibv device; the daemon
// binary comes from the STRATA_PEER_BIN define (CMake).
#include "ds4_os.hpp"
#include "ds4_peer_tier.hpp"
#include "ds4_pool.hpp"
#include "peer_ctl.hpp"
#include "peer_rdma.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace strata::ds4;

int fails = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "[ok]" : "[FAIL]", what.c_str());
    std::fflush(stdout);
    if (!ok) ++fails;
}

/// The fixture's byte at a shard's absolute offset (the content and its verification both use it).
uint8_t pattern(uint64_t i) {
    i = i * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint8_t) (i >> 33);
}

// (--peer-compute) a REAL expert for the tier's compute section: DeepSeek's shape at a size the test can afford, whole
// IQ2_XXS blocks both ways (peer_compute_test's geometry), laid into shard0 past the pattern ranges the storage checks
// use - the peer's worker sums what a pack's bytes really are, not what a pattern decodes to
constexpr int kPcNEmbd = 512, kPcNFf = 256, kPcTG = 16, kPcTD = 16, kPcRows = 2;   // (ggml's IQ2_XXS is type 16)
constexpr size_t kPcGu = 132 * (size_t) kPcNFf, kPcDown = 66 * (size_t) kPcNEmbd;
constexpr uint64_t kPcAt = 0x200000;   // (in shard0, past every range the storage section reads)

void fill_iq2_xxs(uint8_t* w, size_t bytes) {
    for (size_t b = 0; b < bytes; b += 66) {
        w[b] = 0x00;      // (fp16 0x1c00 = 2^-8: small finite values, a scale the sums stay inside)
        w[b + 1] = 0x1c;
        for (size_t i = 2; i < 66; ++i) w[b + i] = pattern(b + i);
    }
}

/// The one expert (gate, up, down back to back) at kPcAt of shard0.
void place_expert(const std::string& dir) {
    std::vector<uint8_t> body(2 * kPcGu + kPcDown);
    fill_iq2_xxs(body.data(), body.size());
    const std::string f = dir + "/shard0.bin";
    std::fstream out(f, std::ios::binary | std::ios::in | std::ios::out);
    out.seekp((std::streamoff) kPcAt);
    out.write((const char*) body.data(), (std::streamsize) body.size());
}

void make_shards(const std::string& dir, int n) {
    std::filesystem::create_directories(dir);
    for (int s = 0; s < n; ++s) {
        std::ofstream f(dir + "/shard" + std::to_string(s) + ".bin", std::ios::binary);
        std::vector<char> buf(1ULL << 20);
        for (int a = 0; a < 4; ++a) {
            for (size_t i = 0; i < buf.size(); ++i) buf[i] = (char) pattern((size_t) a * buf.size() + i);
            f.write(buf.data(), (std::streamsize) buf.size());
        }
    }
}

pid_t start_daemon(const std::string& bin, const std::string& pack, int port, const std::string& so = "") {
    const pid_t pid = fork();
    if (pid != 0) return pid;
    const int fd = open("/tmp/ds4-peer-tier-test-daemon.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
    }
    if (so.empty())
        execl(bin.c_str(), "strata-peer", "--pack", pack.c_str(), "--bind", "0.0.0.0", "--port",
              std::to_string(port).c_str(), "--ram-gb", "2", (char*) nullptr);
    else   // (--compute: the peer sums the experts it holds, not only serves their bytes)
        execl(bin.c_str(), "strata-peer", "--pack", pack.c_str(), "--bind", "0.0.0.0", "--port",
              std::to_string(port).c_str(), "--ram-gb", "2", "--compute", so.c_str(), (char*) nullptr);
    _exit(127);
}

void stop_daemon(pid_t pid) {
    kill(pid, SIGTERM);
    int status = 0;
    waitpid(pid, &status, 0);
}

/// The RDMA link's own IPv4 (the device's netdev), as peer_rdma_test finds it: rdma_cm cannot loop back here.
std::string link_ip_of(const strata::peer::Device& dev) {
    const std::string want = dev.netdev();
    if (want.empty()) return "";
    ifaddrs* ifs = nullptr;
    if (getifaddrs(&ifs) != 0) return "";
    std::string out;
    for (ifaddrs* it = ifs; it != nullptr; it = it->ifa_next) {
        if (it->ifa_addr == nullptr || it->ifa_addr->sa_family != AF_INET || want != it->ifa_name) continue;
        char buf[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &((sockaddr_in*) it->ifa_addr)->sin_addr, buf, sizeof buf) &&
            std::strncmp(buf, "127.", 4) != 0)
            out = buf;
    }
    freeifaddrs(ifs);
    return out;
}

/// One shard's bytes at [off, off + n) are `pattern(off + i)` in every shard: true when dst holds them.
bool holds_pattern(const uint8_t* dst, uint64_t off, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (dst[i] != pattern(off + i)) return false;
    return true;
}

}  // namespace

int main() {
    const std::string bin = std::getenv("STRATA_PEER_BIN") ? std::getenv("STRATA_PEER_BIN") : STRATA_PEER_BIN;
    strata::peer::Device probe;
    std::string err;
    if (!probe.open(std::getenv("STRATA_PEER_TEST_DEV") ? std::getenv("STRATA_PEER_TEST_DEV") : "", err)) {
        std::printf("[skip] no RDMA device: %s\n", err.c_str());
        return 0;
    }
    const std::string link_ip = link_ip_of(probe);
    if (link_ip.empty()) {
        std::printf("[skip] no IPv4 address on the RDMA device's netdev\n");
        return 0;
    }

    const std::string tmp = "/tmp/ds4-peer-tier-test-" + std::to_string(getpid());
    const std::string pack = tmp + "/pack";
    make_shards(pack, 4);   // (shard 3 is the one the tier's manifest does NOT carry)
    place_expert(pack);     // (the compute section's one real expert, past every pattern range)
#if defined(STRATA_PEER_COMPUTE_SO)
    const std::string so = STRATA_PEER_COMPUTE_SO;   // (the CPU worker: the compute section's peer)
#else
    const std::string so = std::getenv("STRATA_PEER_COMPUTE_SO") ? std::getenv("STRATA_PEER_COMPUTE_SO") : "";
#endif
    const int port = 19449;
    const pid_t daemon = start_daemon(bin, pack, port, so);
    bool up = false;
    for (int i = 0; i < 100 && !up; ++i) {
        usleep(100000);
        strata::peer::Ctl probe_ctl;
        up = probe_ctl.connect("127.0.0.1", port, 500, err);
    }
    check(up, "the daemon came up on 127.0.0.1:19449");
    if (!up) {
        stop_daemon(daemon);
        return 1;
    }

    // ---- the row table: three layers, unequal expert counts, one dense, one role on a shard the peer has not
    PeerTier::Config cfg;
    cfg.host = link_ip;
    cfg.port = port;
    cfg.pack_dir = pack;
    cfg.shard_names = {"shard0.bin", "shard1.bin", "shard2.bin"};
    cfg.shard_sizes = {4ULL << 20, 4ULL << 20, 4ULL << 20};
    cfg.window = 8;
    cfg.chunk_mb = 1;
    cfg.timeout_ms = 5000;
    PeerLayerSpec l0, l1, l2;
    l0.n_exp = 2;
    l0.part[0] = PeerRowSpec{0, 0x2000, 0x8000, 0x4000};    // (16 KiB a part, 4 KiB aligned offsets: a local read's
    l0.part[1] = PeerRowSpec{1, 0x1000, 0x8000, 0x4000};    //  fallback lands at dst + 0 - ds4_os's kDirectAlign)
    l0.part[2] = PeerRowSpec{2, 0x1000, 0x8000, 0x4000};
    l1.n_exp = 0;                                           // (a dense layer: no rows at all)
    l2.n_exp = 3;
    l2.part[0] = PeerRowSpec{0, 0x40000, 0x4000, 0x2000};
    l2.part[1] = PeerRowSpec{1, 0x40000, 0x4000, 0x2000};
    l2.part[2] = PeerRowSpec{-1, 0, 0, 0};                  // (its down lives on shard 3: the peer has not got it)
    // (--peer-compute) a third MoE layer too, one REAL expert (IQ2_XXS, the fixtures above the pattern ranges): the
    // tier's resident set at hello, and the order it is summed by.  `resident` is the engine's ranking - this machine's
    // most-routed experts first (ds4_model.cu builds it off its profile) - and the tier truncates it at the peer's slab.
    PeerLayerSpec l3;
    l3.n_exp = 1;
    l3.part[0] = PeerRowSpec{0, kPcAt, kPcGu + kPcDown, (uint32_t) kPcGu};
    l3.part[1] = PeerRowSpec{0, kPcAt + kPcGu, kPcGu + kPcDown, (uint32_t) kPcGu};
    l3.part[2] = PeerRowSpec{0, kPcAt + 2 * kPcGu, kPcGu + kPcDown, (uint32_t) kPcDown};
    cfg.layers = {l0, l1, l2, l3};
    cfg.compute = !so.empty();   // (the opt-in: no worker on the peer, no compute orders)
    cfg.resident = {15, 16, 17}; // (the third layer's expert 0: gate, up, down - the layout's own row ids)

    PeerTier* tier = PeerTier::open_session(cfg, err);
    check(tier != nullptr, "the session came up over " + link_ip + (tier ? "" : (" (" + err + ")")));
    if (tier == nullptr) {
        stop_daemon(daemon);
        std::filesystem::remove_all(tmp);
        return 1;
    }

    check(tier->row_of(0, 0, 0) == 0 && tier->row_of(0, 1, 0) == 3 && tier->row_of(0, 1, 2) == 5,
          "the first layer's rows: expert 1's gate at 3, its down at 5");
    check(tier->row_of(1, 0, 0) == -1, "a dense layer has no row");
    check(tier->row_of(2, 0, 0) == 6 && tier->row_of(2, 2, 1) == 13, "the third layer's rows follow the second's");
    check(tier->row_of(2, 0, 2) == -1, "the role the peer does not hold is never asked");
    check(tier->row_of(2, 3, 0) == -1, "an expert past the layer's count is never asked");
    check(tier->row_of(9, 0, 0) == -1, "a layer past the table is never asked");
    check(tier->row_of(3, 0, 2) == 17, "the compute fixture's expert is the table's last three rows (15, 16, 17)");

    // ---- (--peer-compute) RESIDENCY AT HELLO: the rows the engine ranks first are asked for once, before any read has
    //      hinted them, and an order for them is served at once; a daemon killed under the next one is THE PEER IS DOWN
    //      with every order 0 - the caller's own work, as the storage tier's fallback is
    if (tier->compute_on()) {
        check(tier->resident_rows() == 3, "the resident hint at hello left the peer holding the three rows asked for");
        check(tier->computable(3, 0), "and a resident expert is computable with no read having hinted it");
        check(!tier->computable(2, 2), "an expert whose down the peer does not serve is not computable (its rows are 0)");
        check(!tier->computable(1, 0), "nor is one of a layer with no rows at all");
    } else {
        std::printf("[skip] no compute worker to test residency against (%s)\n",
                    so.empty() ? "STRATA_PEER_COMPUTE_SO names none" : "the daemon's welcome named none");
    }

    // ---- one window, three sub-ranges: the bytes land where the local read would have put them, and nothing beside
    //      them moves
    std::vector<uint8_t> dst(96 << 10, 0xAB);   // (a registered piece: the session's READ destination)
    check(tier->register_dst(dst.data(), dst.size()), "the 96 KiB destination registered");
    {
        uint8_t* at[3] = {dst.data() + 0x100, dst.data() + 0x9000, dst.data() + 0x12000};
        const long long rows[3] = {tier->row_of(0, 0, 0), tier->row_of(0, 1, 0), tier->row_of(2, 2, 1)};
        const uint64_t rel[3] = {0, 0x1000, 0x40};   // (a tensor-parallel member's slice: from the middle)
        const size_t bytes[3] = {0x4000, 0x2000, 0x800};
        char ok[3] = {0, 0, 0};
        std::memset(at[0], 0xAB, 0x4000);
        std::memset(at[1], 0xAB, 0x2000);
        std::memset(at[2], 0xAB, 0x800);
        tier->serve(3, rows, rel, bytes, at, ok);
        check(ok[0] == 1 && ok[1] == 1 && ok[2] == 1, "three sub-ranges in one batch landed");
        check(holds_pattern(at[0], 0x2000 + 0, 0x4000), "the first expert's gate: its whole 16 KiB part");
        check(holds_pattern(at[1], 0x2000 + 0x8000 + 0x1000, 0x2000),
              "the second expert's up: the slice from +0x1000, byte for byte");
        check(holds_pattern(at[2], 0x40000 + 2 * 0x4000 + 0x40, 0x800), "a 2 KiB slice of the third expert's up");
        check(dst[0xFF] == 0xAB, "the byte before the first range was left alone");
        check(dst[0x100 + 0x4000] == 0xAB, "the byte after it was left alone");
        check(dst[0x9000 + 0x2000] == 0xAB && dst[0x12000 + 0x800] == 0xAB, "the neighbours of both slices too");
    }

    // ---- (--peer-compute) one order for the resident expert: no read of it ever happened, and the peer sums it
    std::vector<int8_t> pc_xq((size_t) kPcRows * kPcNEmbd);
    std::vector<float> pc_xd((size_t) kPcRows * kPcNEmbd / 32, 0.001f);
    std::vector<float> pc_y((size_t) kPcRows * kPcNEmbd, 0.0f);
    bool pc_ran = false;
    if (tier->compute_on()) {
        pc_ran = true;
        for (size_t i = 0; i < pc_xq.size(); ++i) pc_xq[i] = (int8_t) ((pattern(i) & 0x7f) - 64);
        const long long order[3] = {15, 16, 17};
        const int8_t* xqs[1] = {pc_xq.data()};
        const float* xds[1] = {pc_xd.data()};
        float* ys[1] = {pc_y.data()};
        char ok[1] = {0};
        tier->compute(1, order, kPcRows, kPcNEmbd, kPcNFf, kPcTG, kPcTD, 0.0f, xqs, xds, nullptr, ys, ok);
        check(ok[0] == 1, "the resident expert was summed by the peer's worker (its bytes were never read here)");
        bool finite = true;
        for (float v : pc_y) finite = finite && std::isfinite(v);
        check(finite, "and the sums it returned are finite numbers (two rows of 512)");
    }

    // ---- the flow's queue: a served order accounts down like IoQueue's job, and everything else IS the job
    IoQueue local(2);
    PeerQueue pq(*tier, local);
    const std::string shard0 = pack + "/shard0.bin";
    const os::fd_t fd0 = os::open_ro(shard0);
    check(fd0 != os::kBadFd, "the fixture's shard0 opened for the local fallback");
    std::vector<uint8_t> peer_dst(0x4000, 0x5A);                          // (never registered: the local way)
    std::vector<uint8_t> peer_dst2(0x4000 + os::kDirectAlign, 0x5A);
    std::vector<uint8_t> fallback(0x4000 + os::kDirectAlign, 0x5A);
    {
        std::atomic<int> left_a{1}, left_b{1}, left_c{1};
        std::atomic<bool> bad_a{false}, bad_b{false}, bad_c{false};
        PeerQueue::Order o[3];
        // (a) served: a registered destination
        o[0].job.fd = fd0;
        o[0].job.off = 0x2000;   // (0x2000 % kDirectAlign is 0: a local read would land at dst)
        o[0].job.len = 0x2000;
        o[0].job.dst = dst.data() + 0x4000;
        o[0].job.left = &left_a;
        o[0].job.bad = &bad_a;
        o[0].row = tier->row_of(0, 0, 0);
        o[0].rel = 0;
        o[0].bytes = 0x2000;
        // (b) a served row into a piece the session never registered: the job IoQueue would have run
        o[1].job.fd = fd0;
        o[1].job.off = 0x2000;
        o[1].job.len = 0x2000;
        o[1].job.dst = peer_dst.data();
        o[1].job.left = &left_b;
        o[1].job.bad = &bad_b;
        o[1].row = tier->row_of(0, 0, 0);
        o[1].rel = 0;
        o[1].bytes = 0x2000;
        // (c) no row at all (the tier does not serve this layer, role, or expert): the same
        o[2].job.fd = fd0;
        o[2].job.off = 0x2000;
        o[2].job.len = 0x2000;
        o[2].job.dst = peer_dst2.data();
        o[2].job.left = &left_c;
        o[2].job.bad = &bad_c;
        o[2].row = -1;
        pq.push(o, 3);
        for (int i = 0; i < 200 && (left_a.load() || left_b.load() || left_c.load()); ++i) usleep(20000);
        check(left_a.load() == 0 && left_b.load() == 0 && left_c.load() == 0, "all three orders accounted down");
        check(!bad_a.load() && !bad_b.load() && !bad_c.load(), "none of them failed");
        check(holds_pattern(dst.data() + 0x4000, 0x2000, 0x2000), "the served order landed the peer's bytes");
        check(holds_pattern(peer_dst.data(), 0x2000, 0x2000), "the unregistered destination read them locally");
        check(holds_pattern(peer_dst2.data(), 0x2000, 0x2000), "so did the row the tier does not serve");
    }

    // ---- DS4_FLOW_DOWN_PART: the read lands whole, then the member's columns are compacted where it landed
    {
        std::atomic<int> left{1};
        std::atomic<bool> bad{false};
        const size_t pitch = 0x1000, coff = 0x400, clen = 0x800, rows_n = 2;
        uint8_t* at = dst.data() + 0x6000;
        PeerQueue::Order o;
        o.job.fd = fd0;
        o.job.off = 0x2000;
        o.job.len = pitch * rows_n;
        o.job.dst = at;
        o.job.left = &left;
        o.job.bad = &bad;
        o.job.crows = rows_n;   // (the member's columns of every row, gathered to the front: ds4::compact_rows)
        o.job.cpitch = pitch;
        o.job.coff = coff;
        o.job.clen = clen;
        o.row = tier->row_of(0, 0, 0);
        o.rel = 0;
        o.bytes = pitch * rows_n;
        pq.push(&o, 1);
        for (int i = 0; i < 200 && left.load(); ++i) usleep(20000);
        check(left.load() == 0 && !bad.load(), "the down-part order finished");
        bool compacted = true;
        for (size_t r = 0; r < rows_n && compacted; ++r)
            compacted = holds_pattern(at + r * clen, 0x2000 + r * pitch + coff, clen);
        check(compacted, "its rows' kept columns are compacted to the front, byte for byte");
    }

    // ---- a peer that dies under the flow: the whole batch falls back, and the queue keeps working
    stop_daemon(daemon);
    usleep(200000);
    {
        std::atomic<int> left{1};
        std::atomic<bool> bad{false};
        const long long rows[1] = {tier->row_of(0, 0, 1)};
        const uint64_t rel[1] = {0};
        const size_t bytes[1] = {0x1000};
        uint8_t* at[1] = {dst.data() + 0x8000};
        char ok[1] = {0};
        tier->serve(1, rows, rel, bytes, at, ok);
        check(ok[0] == 0, "a batch against the dead daemon is served by nobody");
        check(!tier->up(), "the session says so (THE PEER IS DOWN), once");
        PeerQueue::Order o;
        o.job.fd = fd0;
        o.job.off = 0x2000;
        o.job.len = 0x1000;
        o.job.dst = fallback.data();
        o.job.left = &left;
        o.job.bad = &bad;
        o.row = rows[0];
        o.rel = 0;
        o.bytes = 0x1000;
        pq.push(&o, 1);
        for (int i = 0; i < 200 && left.load(); ++i) usleep(20000);
        check(left.load() == 0 && !bad.load(), "the flow's order still finished (on the local disk)");
        check(holds_pattern(fallback.data(), 0x2000, 0x1000), "its bytes came off the disk, byte for byte");
        if (pc_ran) {
            // (--peer-compute) a daemon killed under the tier: the next order is nobody's, every job 0, and the caller's
            // own computation - the lane's - is the one it would have got
            const long long order2[3] = {15, 16, 17};
            const int8_t* xqs2[1] = {pc_xq.data()};
            const float* xds2[1] = {pc_xd.data()};
            std::fill(pc_y.begin(), pc_y.end(), 0.0f);
            float* ys2[1] = {pc_y.data()};
            char ok2[1] = {1};
            tier->compute(1, order2, kPcRows, kPcNEmbd, kPcNFf, kPcTG, kPcTD, 0.0f, xqs2, xds2, nullptr, ys2, ok2);
            check(ok2[0] == 0, "an order after the kill is served by nobody (the caller computes it)");
        }
    }

    std::filesystem::remove_all(tmp);
    std::printf("%s ds4_peer_tier_test: %s\n", fails ? "FAIL" : "PASS",
                fails ? "there were failures" : "all checks passed");
    return fails ? 1 : 0;
}
