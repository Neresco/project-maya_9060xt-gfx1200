// The peer compute tier's end-to-end test, no GPU and no pack: a fixture of one shard file holding two
// synthetic IQ2_XXS experts, the real strata-peer daemon as a subprocess with --compute pointing at the
// built CPU worker, and the orders the engine would send it - (a) a batch's sums against the same
// arithmetic computed here, byte for byte, (b) a job naming a row the daemon does not hold answered 0 while
// its neighbour is served, and (c) a daemon killed under a batch surfacing as THE PEER IS DOWN, with every
// order marked 0 so the caller's fallback computes it.
//
// Skipped with a note when the machine has no RDMA device (the daemon brings a QP up even for storage only).
// STRATA_PEER_TEST_DEV names the ibv device; the binaries come from CMake's STRATA_PEER_BIN and
// STRATA_PEER_COMPUTE_SO defines.
#include "ds4_gguf_cpu.hpp"
#include "ds4_peer_tier.hpp"
#include "peer_client.hpp"
#include "peer_ctl.hpp"
#include "peer_rdma.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <sys/wait.h>
#include <unistd.h>

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

/// The fixture's byte at an offset (the fabricated bodies use it, so the test is the same twice).
uint8_t pattern(uint64_t i) {
    i = i * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint8_t) (i >> 33);
}

// the fixture's geometry: DeepSeek's shape at a size the test can afford, whole IQ2_XXS blocks both ways
constexpr int kNEmbd = 512;                         // (2 blocks a gate/up row, 132 B each)
constexpr int kNFf = 256;                           // (1 block a down row, 66 B each)
constexpr int kNExperts = 2;
constexpr int kRows = 2;                             // activation rows (tokens) an order
constexpr int kTG = 16;                              // ggml IQ2_XXS
constexpr int kTD = 16;
constexpr size_t kGuRow = 132, kDRow = 66;
constexpr size_t kGateBytes = kGuRow * (size_t) kNFf;    // 33792
constexpr size_t kDownBytes = kDRow * (size_t) kNEmbd;   // 33792
constexpr size_t kExpertBytes = 2 * kGateBytes + kDownBytes;

/// One expert's three matrices as ggml lays an IQ2_XXS block out (66 bytes a 256 values: an fp16 scale,
/// then 32 words of grid indices and signs).  The scale is a small finite half (2^-8), so a fabricated body
/// decodes to small finite values and the comparison is about the sums rather than about infinities.
void fill_iq2_xxs(uint8_t* w, size_t bytes) {
    for (size_t b = 0; b < bytes; b += 66) {
        w[b] = 0x00;      // (fp16 0x1c00 = 2^-8, little end first)
        w[b + 1] = 0x1c;
        for (size_t i = 2; i < 66; ++i) w[b + i] = pattern(b + i);
    }
}

/// The shard: the two experts back to back, each [gate | up | down].
void make_shard(const std::string& dir) {
    std::filesystem::create_directories(dir);
    std::vector<uint8_t> body((size_t) kNExperts * kExpertBytes);
    fill_iq2_xxs(body.data(), body.size());
    std::ofstream f(dir + "/shard0.bin", std::ios::binary);
    f.write((const char*) body.data(), (std::streamsize) body.size());
}

/// The activation as the device kernels round it: a scale a 32 values of max|x|/127 and the nearest int8
/// (q8_round's form, which is what the protocol carries).  A row of zeros gets a zero scale.
void quant_rows(const float* x, int n, int8_t* q, float* s) {
    for (int i = 0; i < n; i += 32) {
        float mx = 0.0f;
        for (int j = 0; j < 32; ++j) mx = std::fmax(mx, std::fabs(x[i + j]));
        const float sc = mx > 0.0f ? mx / 127.0f : 0.0f;
        s[i / 32] = sc;
        for (int j = 0; j < 32; ++j) {
            const float v = sc > 0.0f ? x[i + j] / sc : 0.0f;
            q[i + j] = (int8_t) std::nearbyint(v);   // (round half to even, as q8_round does)
        }
    }
}

/// The local lane's expert on one activation: iq_cpu_gate_up, the down activation quantized, iq_cpu_down
/// and the routing weight - the calls src/ds4/ds4_iq_cpu.cpp's lane makes, in the same order.
void lane(const uint8_t* gate, const uint8_t* up, const uint8_t* down, const float* x, const float* w, float* y) {
    gguf::IqCpu f;
    std::string err;
    if (!gguf::iq_cpu_fmt(kTG, kTD, kNEmbd, kNFf, f, err)) {
        std::printf("[FAIL] the fixture's geometry: %s\n", err.c_str());
        ++fails;
        return;
    }
    std::vector<uint8_t> aq((size_t) kRows * f.act_bytes), hq((size_t) kRows * f.h_bytes);
    std::vector<float> ff((size_t) kRows * kNFf), out((size_t) kRows * kNEmbd);
    const void* ap[kRows];
    const void* hp[kRows];
    float* fp[kRows];
    float* yp[kRows];
    for (int t = 0; t < kRows; ++t) {
        gguf::iq_cpu_quant(f.gu_act, x + (size_t) t * kNEmbd, kNEmbd, aq.data() + (size_t) t * f.act_bytes);
        ap[t] = aq.data() + (size_t) t * f.act_bytes;
        fp[t] = ff.data() + (size_t) t * kNFf;
        hp[t] = hq.data() + (size_t) t * f.h_bytes;
        yp[t] = out.data() + (size_t) t * kNEmbd;
    }
    gguf::iq_cpu_gate_up(f, gate, up, ap, kRows, fp, 0, kNFf, 0.0f);
    for (int t = 0; t < kRows; ++t)
        gguf::iq_cpu_quant(f.d_act, ff.data() + (size_t) t * kNFf, kNFf, hq.data() + (size_t) t * f.h_bytes);
    gguf::iq_cpu_down(f, down, hp, kRows, yp, 0, kNEmbd);
    for (int t = 0; t < kRows; ++t)
        for (int i = 0; i < kNEmbd; ++i) y[(size_t) t * kNEmbd + i] = w[t] * out[(size_t) t * kNEmbd + i];
}

pid_t start_daemon(const std::string& bin, const std::string& pack, const std::string& so, int port) {
    const pid_t pid = fork();
    if (pid != 0) return pid;
    const int fd = open("/tmp/peer-compute-test-daemon.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
    }
    if (so.empty())
        execl(bin.c_str(), "strata-peer", "--pack", pack.c_str(), "--bind", "0.0.0.0", "--port",
              std::to_string(port).c_str(), "--ram-gb", "1", (char*) nullptr);
    else
        execl(bin.c_str(), "strata-peer", "--pack", pack.c_str(), "--bind", "0.0.0.0", "--port",
              std::to_string(port).c_str(), "--ram-gb", "1", "--compute", so.c_str(), (char*) nullptr);
    _exit(127);
}

// (--compute twice: one worker a card.  The two CPU workers both compute the RAM slab's host bytes - the daemon
// hands a batch's jobs to them in turn - so a job's sums are the same from either, which is what the check after
// this daemon's session leans on)
pid_t start_daemon_two(const std::string& bin, const std::string& pack, const std::string& so, int port) {
    const pid_t pid = fork();
    if (pid != 0) return pid;
    const int fd = open("/tmp/peer-compute-test-daemon.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
    }
    execl(bin.c_str(), "strata-peer", "--pack", pack.c_str(), "--bind", "0.0.0.0", "--port",
          std::to_string(port).c_str(), "--ram-gb", "1", "--compute", so.c_str(), "--compute-options", "cpu",
          "--compute", so.c_str(), "--compute-options", "cpu", (char*) nullptr);
    _exit(127);
}

void stop_daemon(pid_t pid) {
    kill(pid, SIGTERM);
    int status = 0;
    waitpid(pid, &status, 0);
}

/// The daemon is listening (the control channel takes a connection): the wait the other peer tests do too.
bool wait_listening(int port) {
    std::string err;
    for (int i = 0; i < 100; ++i) {
        usleep(100000);
        strata::peer::Ctl probe_ctl;
        if (probe_ctl.connect("127.0.0.1", port, 500, err)) return true;
    }
    return false;
}

/// The RDMA link's own IPv4 (the device's netdev), as peer_rdma_test finds it: rdma_cm cannot loop
/// back here.
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

}   // namespace

int main() {
    const std::string bin = std::getenv("STRATA_PEER_BIN") ? std::getenv("STRATA_PEER_BIN") : STRATA_PEER_BIN;
    const std::string so =
        std::getenv("STRATA_PEER_COMPUTE_SO") ? std::getenv("STRATA_PEER_COMPUTE_SO") : STRATA_PEER_COMPUTE_SO;
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

    const std::string tmp = "/tmp/peer-compute-test-" + std::to_string(getpid());
    const std::string pack = tmp + "/pack";
    make_shard(pack);
    std::vector<uint8_t> shard((size_t) kNExperts * kExpertBytes);
    fill_iq2_xxs(shard.data(), shard.size());

    // ---- the row table: two experts of one layer, six rows, each a whole matrix of the fixture's shard
    std::vector<strata::peer::Row> table;
    for (int e = 0; e < kNExperts; ++e) {
        const uint64_t base = (uint64_t) e * kExpertBytes;
        table.push_back(strata::peer::Row{0, base, (uint32_t) kGateBytes});                  // gate
        table.push_back(strata::peer::Row{0, base + kGateBytes, (uint32_t) kGateBytes});         // up
        table.push_back(strata::peer::Row{0, base + 2 * kGateBytes, (uint32_t) kDownBytes});     // down
    }

    // ---- the activation, and the sums the local lane computes from the same bytes
    std::vector<float> x((size_t) kRows * kNEmbd);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = (float) ((double) pattern(i * 7 + 1) / 128.0 - 1.0);   // (-1, 1)
    std::vector<int8_t> xq((size_t) kRows * kNEmbd);
    std::vector<float> xd((size_t) kRows * (kNEmbd / 32));
    for (int t = 0; t < kRows; ++t)   // (a row at a time, as the caller's does)
        quant_rows(x.data() + (size_t) t * kNEmbd, kNEmbd, xq.data() + (size_t) t * kNEmbd,
                   xd.data() + (size_t) t * (kNEmbd / 32));
    std::vector<float> xq32(x.size());
    for (size_t i = 0; i < x.size(); ++i) xq32[i] = (float) xq[i] * xd[i / 32];
    const float w[kRows] = {0.75f, -0.25f};
    std::vector<float> ref((size_t) kRows * kNEmbd), ref1((size_t) kRows * kNEmbd);
    lane(shard.data(), shard.data() + kGateBytes, shard.data() + 2 * kGateBytes, xq32.data(), w, ref.data());
    lane(shard.data() + kExpertBytes, shard.data() + kExpertBytes + kGateBytes,
         shard.data() + kExpertBytes + 2 * kGateBytes, xq32.data(), w, ref1.data());

    const int port = 19451;
    const pid_t daemon = start_daemon(bin, pack, so, port);
    const bool up = wait_listening(port);
    check(up, "the daemon came up on 127.0.0.1:19451 with the CPU worker");
    if (!up) {
        stop_daemon(daemon);
        std::filesystem::remove_all(tmp);
        return 1;
    }

    // ---- the session, and both experts' rows hinted in (the daemon faults them in from its shard)
    strata::peer::PeerClient cli(8, 1ULL << 20, 5000, "");
    check(cli.connect(link_ip, port, pack, {"shard0.bin"}, {shard.size()}, table, err),
          "the session came up over " + link_ip + (err.empty() ? "" : " (" + err + ")"));
    check(cli.hint({0, 1, 2, 3, 4, 5}, err), "both experts hinted");
    check(cli.advertised(0) && cli.advertised(5), "the daemon holds all six rows");

    // ---- (a) a batch of two orders against the local lane on the same bytes: byte for byte
    {
        std::vector<strata::peer::PeerClient::ComputeReq> jobs(2);
        for (int k = 0; k < 2; ++k) {
            jobs[(size_t) k].gate = 3 * k;
            jobs[(size_t) k].up = 3 * k + 1;
            jobs[(size_t) k].down = 3 * k + 2;
            jobs[(size_t) k].n_embd = kNEmbd;
            jobs[(size_t) k].n_ff = kNFf;
            jobs[(size_t) k].tg = kTG;
            jobs[(size_t) k].td = kTD;
            jobs[(size_t) k].n_rows = kRows;
            jobs[(size_t) k].xq = xq.data();
            jobs[(size_t) k].xd = xd.data();
            jobs[(size_t) k].w = w;
        }
        std::vector<char> ok;
        std::vector<std::vector<float>> y;
        const bool sent = cli.compute(jobs, ok, y, err);
        check(sent && ok.size() == 2 && ok[0] == 1 && ok[1] == 1,
              "both compute orders were served" + (sent ? "" : " (" + err + ")"));
        if (sent && ok.size() == 2 && ok[0] == 1 && ok[1] == 1) {
            check(y[0].size() == ref.size() && std::memcmp(y[0].data(), ref.data(), ref.size() * 4) == 0,
                  "the first expert's sums are the lane's, byte for byte");
            check(y[1].size() == ref1.size() && std::memcmp(y[1].data(), ref1.data(), ref1.size() * 4) == 0,
                  "and so are the second expert's");
        }
    }

    // ---- the same order from the raw floats: one rounding apart (the wire's int8 activation against the
    //      lane's own quantization), which is what the worker's second quantization costs
    {
        std::vector<float> raw(ref.size());
        lane(shard.data(), shard.data() + kGateBytes, shard.data() + 2 * kGateBytes, x.data(), w, raw.data());
        double worst = 0.0, scale = 0.0;
        for (size_t i = 0; i < raw.size(); ++i) {
            worst = std::fmax(worst, std::fabs((double) raw[i] - (double) ref[i]));
            scale = std::fmax(scale, std::fabs((double) raw[i]));
        }
        const double rel = scale > 0.0 ? worst / scale : worst;
        std::printf("      (the raw-float lane against the wire's activation: worst %.3g of %.3g, %.3g relative)\n",
                     worst, scale, rel);
        // the wire's activation is int8 with a scale a 32 values, so the lane fed the raw floats and the lane
        // fed those bytes differ by that rounding alone - measured at 1.4% here, and bounded at 5% (a decode's
        // picks are never asked of this tier for experts whose sums cancel to nothing)
        check(rel < 5e-2, "the raw-float lane agrees to within 5% of the largest value");
    }

    // ---- (b) a job naming a row the daemon does not hold: 0 for it, its neighbour served
    {
        std::vector<strata::peer::PeerClient::ComputeReq> jobs(2);
        for (int k = 0; k < 2; ++k) {
            jobs[(size_t) k].gate = k == 0 ? 0 : 6;   // (the table holds six rows: 6 is not one)
            jobs[(size_t) k].up = k == 0 ? 1 : 7;
            jobs[(size_t) k].down = k == 0 ? 2 : 8;
            jobs[(size_t) k].n_embd = kNEmbd;
            jobs[(size_t) k].n_ff = kNFf;
            jobs[(size_t) k].tg = kTG;
            jobs[(size_t) k].td = kTD;
            jobs[(size_t) k].n_rows = kRows;
            jobs[(size_t) k].xq = xq.data();
            jobs[(size_t) k].xd = xd.data();
            jobs[(size_t) k].w = w;
        }
        std::vector<char> ok;
        std::vector<std::vector<float>> y;
        const bool sent = cli.compute(jobs, ok, y, err);
        check(sent && ok.size() == 2 && ok[0] == 1 && ok[1] == 0,
              "a job of rows the daemon does not hold is 0, its neighbour served" +
                  (sent ? "" : " (" + err + ")"));
        check(sent && y.size() == 2 && y[1].empty(), "and it carries no sum at all");
        check(cli.is_up(), "the session is still up after a job nobody can serve");
    }
    cli.close();
    stop_daemon(daemon);

    // ---- a daemon with no --compute at all: storage only, and every compute order answers 0
    {
        const int plain_port = port + 1;
        const pid_t plain = start_daemon(bin, pack, "", plain_port);
        const bool came_up = wait_listening(plain_port);
        check(came_up, "a daemon without --compute came up (storage only)");
        if (came_up) {
            strata::peer::PeerClient store(8, 1ULL << 20, 5000, "");
            const bool sess = store.connect(link_ip, plain_port, pack, {"shard0.bin"}, {shard.size()}, table, err) &&
                               store.hint({0, 1, 2, 3, 4, 5}, err);
            check(sess, "its storage session came up" + (sess ? "" : " (" + err + ")"));
            if (sess) {
                std::vector<strata::peer::PeerClient::ComputeReq> jobs(1);
                jobs[0].gate = 0;
                jobs[0].up = 1;
                jobs[0].down = 2;
                jobs[0].n_embd = kNEmbd;
                jobs[0].n_ff = kNFf;
                jobs[0].tg = kTG;
                jobs[0].td = kTD;
                jobs[0].n_rows = kRows;
                jobs[0].xq = xq.data();
                jobs[0].xd = xd.data();
                jobs[0].w = w;
                std::vector<char> ok;
                std::vector<std::vector<float>> y;
                const bool sent = store.compute(jobs, ok, y, err);
                check(sent && ok.size() == 1 && ok[0] == 0,
                      "a daemon with no worker answers 0 for every order (the caller computes it)" +
                          (sent ? "" : " (" + err + ")"));
                store.close();
            }
        }
        stop_daemon(plain);
    }

    // ---- (b2) a daemon with TWO --compute workers: its welcome names both, and the two share a batch - one worker a
    //      card, so two cars' workers each take their own orders (the RAM slab's rows go to them in turn)
    {
        const int two_port = port + 3;
        const pid_t two = start_daemon_two(bin, pack, so, two_port);
        const bool came_up = wait_listening(two_port);
        check(came_up, "a daemon with two --compute workers came up");
        if (came_up) {
            strata::peer::PeerClient cli2(8, 1ULL << 20, 5000, "");
            const bool sess = cli2.connect(link_ip, two_port, pack, {"shard0.bin"}, {shard.size()}, table, err) &&
                              cli2.hint({0, 1, 2, 3, 4, 5}, err);
            check(sess, "its session came up over " + link_ip + (sess ? "" : " (" + err + ")"));
            if (sess) {
                check(cli2.compute_caps().find(" | ") != std::string::npos,
                      "its welcome names both workers (\"" + cli2.compute_caps() + "\")");
                std::vector<strata::peer::PeerClient::ComputeReq> jobs(2);
                for (int k = 0; k < 2; ++k) {
                    jobs[(size_t) k].gate = 3 * k;
                    jobs[(size_t) k].up = 3 * k + 1;
                    jobs[(size_t) k].down = 3 * k + 2;
                    jobs[(size_t) k].n_embd = kNEmbd;
                    jobs[(size_t) k].n_ff = kNFf;
                    jobs[(size_t) k].tg = kTG;
                    jobs[(size_t) k].td = kTD;
                    jobs[(size_t) k].n_rows = kRows;
                    jobs[(size_t) k].xq = xq.data();
                    jobs[(size_t) k].xd = xd.data();
                    jobs[(size_t) k].w = w;
                }
                std::vector<char> ok;
                std::vector<std::vector<float>> y;
                const bool sent = cli2.compute(jobs, ok, y, err);
                check(sent && ok.size() == 2 && ok[0] == 1 && ok[1] == 1,
                      "two orders served by the two workers" + (sent ? "" : " (" + err + ")"));
                if (sent && ok.size() == 2 && ok[0] == 1 && ok[1] == 1)
                    check(y[0].size() == ref.size() && std::memcmp(y[0].data(), ref.data(), ref.size() * 4) == 0 &&
                              std::memcmp(y[1].data(), ref1.data(), ref1.size() * 4) == 0,
                          "each worker's sums are the lane's, byte for byte");
                cli2.close();
            }
        }
        stop_daemon(two);
    }

    // ---- (c) the engine's end: the tier computes a batch, then a daemon killed under the next one is
    //      THE PEER IS DOWN with every order 0, so the caller's fallback computes all of them
    {
        const int tier_port = port + 2;
        const pid_t second = start_daemon(bin, pack, so, tier_port);
        const bool came_up = wait_listening(tier_port);
        check(came_up, "a second daemon came up for the tier's session");
        PeerTier::Config cfg;
        cfg.host = link_ip;
        cfg.port = tier_port;
        cfg.pack_dir = pack;
        cfg.shard_names = {"shard0.bin"};
        cfg.shard_sizes = {shard.size()};
        cfg.window = 8;
        cfg.chunk_mb = 1;
        cfg.timeout_ms = 5000;
        cfg.compute = true;   // (--peer-compute: the tier only sends an order when it was asked to, and its welcome named a
                              //  worker - the default is storage only)
        PeerLayerSpec l0;
        l0.n_exp = kNExperts;
        l0.part[0] = PeerRowSpec{0, 0, kGateBytes, (uint32_t) kGateBytes};
        l0.part[1] = PeerRowSpec{0, kGateBytes, kGateBytes, (uint32_t) kGateBytes};
        l0.part[2] = PeerRowSpec{0, 2 * kGateBytes, kDownBytes, (uint32_t) kDownBytes};
        cfg.layers = {l0};
        PeerTier* tier = PeerTier::open_session(cfg, err);
        check(tier != nullptr, "the tier's session came up" + (tier ? "" : " (" + err + ")"));
        if (tier != nullptr) {
            check(tier->compute_on(), "the tier takes compute orders (the daemon's welcome named its worker)");
            check(!tier->computable(0, 0), "an expert whose rows nobody has hinted is not promised to a compute order");
            // residency: the tier's own read path faults the rows in, and a compute order never hints
            std::vector<uint8_t> dst((size_t) 6 * kGateBytes);
            check(tier->register_dst(dst.data(), dst.size()), "the tier's staging buffer registered");
            const long long rows[6] = {0, 1, 2, 3, 4, 5};
            const uint64_t rel[6] = {0, 0, 0, 0, 0, 0};
            const size_t bytes[6] = {kGateBytes, kGateBytes, kDownBytes, kGateBytes, kGateBytes, kDownBytes};
            uint8_t* at[6];
            for (int k = 0; k < 6; ++k) at[k] = dst.data() + (size_t) k * kGateBytes;
            char served[6] = {0, 0, 0, 0, 0, 0};
            tier->serve(6, rows, rel, bytes, at, served);
            check(served[0] == 1 && served[5] == 1, "the tier read both experts' rows in");
            check(tier->computable(0, 0) && tier->computable(0, 1),
                  "and now holds all three parts of both experts (an order is promised, not hoped)");

            long long order[3] = {0, 1, 2};   // (expert 0's gate, up and down)
            std::vector<float> y((size_t) kRows * kNEmbd);
            const int8_t* xqs[1] = {xq.data()};
            const float* xds[1] = {xd.data()};
            const float* ws[1] = {w};
            float* ys[1] = {y.data()};
            char ok[1] = {0};
            tier->compute(1, order, kRows, kNEmbd, kNFf, kTG, kTD, 0.0f, xqs, xds, ws, ys, ok);
            check(ok[0] == 1, "the tier's compute order was served");
            check(std::memcmp(y.data(), ref.data(), ref.size() * 4) == 0,
                  "and its sums are the lane's, byte for byte");

            // the daemon dies under the next batch: the tier says so once and every order is 0
            stop_daemon(second);
            usleep(200000);
            char dead[1] = {0};
            std::memset(y.data(), 0, y.size() * 4);
            tier->compute(1, order, kRows, kNEmbd, kNFf, kTG, kTD, 0.0f, xqs, xds, ws, ys, dead);
            check(dead[0] == 0, "a batch against the dead daemon is served by nobody");
            check(!tier->up(), "the tier says so (THE PEER IS DOWN), once");
            // the caller's fallback is the lane, and it computes the order the peer did not
            std::vector<float> fallback((size_t) kRows * kNEmbd);
            lane(shard.data(), shard.data() + kGateBytes, shard.data() + 2 * kGateBytes, xq32.data(), w,
                 fallback.data());
            check(std::memcmp(fallback.data(), ref.data(), ref.size() * 4) == 0,
                  "the caller's own computation is the one it would have got");
        } else {
            stop_daemon(second);
        }
    }

    std::filesystem::remove_all(tmp);
    std::printf("%s peer_compute_test: %s\n", fails ? "FAIL" : "PASS",
                fails ? "there were failures" : "all checks passed");
    return fails ? 1 : 0;
}
