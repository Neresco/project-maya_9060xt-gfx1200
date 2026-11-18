# Two machines: computing on the peer's GPUs and CPU

**Status: stages 1 and 2 are built and the compute path's framing is tightened; stages 3-4 are not.** This page is the case for a third tier beside the RAM/VRAM one: the
peer machine is not just storage, it also *computes* the experts it holds. What exists today is the **CPU worker**
(`src/peer/peer_compute_cpu.cpp`) and the **HIP worker** (`src/peer/peer_compute_hip.cu`, built for the peer's
`gfx1102;gfx1151` cards), a `.so` the daemon loads with `--compute`; the `compute`/`computed` protocol; the engine side of
it (`PeerTier::compute`) and the fallback - proven by `tests/peer_compute_test` and `tests/peer_compute_hip_test`, not yet
wired into a live decode. The routing lane (stage 3) and the APU residency pool (stage 4) are still paper. The case for
them is the measurement in "What stage 0 measured", and it came out in favour: with the framing tightened the peer
beats the CPU lane by 4.4-5.8x on the numbers below, which is what the arithmetic said it could (see "Where the time
goes now").

Read [RDMA_PEER_TIER.md](RDMA_PEER_TIER.md) first: same daemon, same link, same pack, same `--vram` slab.

## Why this is a residency question, not a FLOPs one

The pack's experts are 10,598,400 B each (gate 3,041,760 + up 3,041,760 + down 4,514,880), and a decoded token picks
240 of them - **2.54 GB of weights per token**. Move that across the wire instead of keeping it where it is and the
arithmetic ends the idea on the spot:

| | bytes | at the link's 2.6-3.0 GB/s |
|---|---|---|
| one expert's weights, sent per pick | 10,598,400 | **3.4 ms** |
| that expert's compute, on a peer card | - | **0.041-0.066 ms** |
| one pick's activation + result (weights resident) | 26,240 | **0.019 ms** |

So sending *weights* to be computed costs ~80x more wire time than the compute it buys. Sending an *activation*
costs 400x less than the weights and the same order as the compute itself: 5,120 B of int8 + 640 B of scales in,
20,480 B of float2 out. **Peer compute therefore only pays for experts that are already resident on the peer** - in
the `--vram` slab the daemon already registers for RDMA, or in its RAM slab. It is a residency play: the same
`--vram`/`--ram-gb` tiers, now with a kernel at the end of them instead of a row address.

That also kills the shape that looks most attractive on paper: a **remote prefill GEMM**. With 6-of-384 routing a
1000-token batch gives ~15 rows per expert, so the weights cross the wire anyway, on top of the output - strictly
worse than pulling 10.6 MB into a local 6.5-14.4 GB/s PCIe lane and using the fused prompt path there.

## What stage 0 measured

Two boxes, one working tree. Strix (192.168.100.20, RX 7600 XT `gfx1102` + Ryzen AI Max 8060S APU `gfx1151`, 32
threads) got the tests built for both of its cards; this side is 4x RX 9060 XT `gfx1200` plus 8 physical cores
(16 threads). All numbers are for the pack's pair IQ2_XXS/IQ3_XXS (`tests/ds4_gguf_test --bench-q`, 8 copies of a
6-expert plan, 50 rounds; `tests/ds4_lane_bench` for the CPU lane).

**A decoded token's pick (gate/up + down for one expert):**

| where | µs/pick | note |
|---|---|---|
| peer APU, `gfx1151` | **50.1** | 300.5 µs for a token's 6 picks; 212 GB/s of weights |
| peer RX 7600 XT, `gfx1102` | **65.4** | 392.5 µs for 6; 162 GB/s; gate/up 243.9 + down 175.3 |
| this machine's cards, `gfx1200` | 41-52 | from `maya-ds4-hip.log`, the same shape |
| this machine's CPU lane | **478** | as the engine times it under load; 293 with the box idle |
| the peer's CPU lane, 32 threads | **120** | 88 GB/s; 271 at 8 threads |

Three things fall out of that table:

- **The peer's APU is the fastest thing on the peer, and the 7600 XT is the slower of the two** - 1.3x - despite the
  APU having no dedicated memory at all (its `--vram` slab is the machine's own RAM behind the GTT aperture). It wins
  because the decode is bandwidth-bound and the APU's unified memory is faster than Navi 33's GDDR6.
- **The peer's *CPU* lane at 32 threads beats ours by 2.4-4x**, because this box only has 8 physical cores while
  strix has 32 threads. Offloading to a peer CPU worker is a speed win, not merely spare capacity.
- **A peer card is not faster than a local card.** 50-65 µs against 41-52 µs: nobody should move an expert that is
  already in a local card's VRAM. The win is entirely *against the CPU lane* and *against a disk read*.

**A prompt-shaped set (32 experts x rows, the prompt path's tiles):**

| rows | peer APU | peer 7600 XT |
|---|---|---|
| 128 | 19.59 ms, 14.8 TFLOP/s (17 GB/s of weights) | 24.69 ms, 11.7 TFLOP/s (14 GB/s) |
| 8 | 2.98 ms | 4.23 ms |
| 2 | 1.80 ms | 1.85 ms |

Only meaningful for experts resident there - see the prefill note above.

**The wire, this link (RoCEv2, perftest, both directions):**

| bytes | write | read |
|---|---|---|
| 26,240 (one pick's activation/result) | 17.9 µs, p99 25.6 | 19.4 µs, p99 35.4 |
| 524,288 | 199.3 µs | 198.1 µs (**2.6 GB/s**) |

The cost of a message is **~17-19 µs whether it carries 26 KiB or 26 bytes**: latency-bound, not bandwidth-bound, up
to a few hundred KiB. That single fact shapes the protocol more than anything else - one pick per message wastes the
whole budget, so compute orders are batched per layer.

## The gate, and what it says

Continue only if `peer compute + wire + round trip` is well under what the local CPU lane costs for the same pick,
which the engine times at **478 µs**:

- APU: 50.1 + ~38 (activation in, result out) = **88 µs**
- 7600 XT: 65.4 + ~38 = **103 µs**
- batched per layer: ~300 µs of APU compute + ~90 µs of wire (35 KB in, 123 KB out) = **~390 µs against 1,760 µs**
  on an idle local lane and 2,870 µs at the 478 µs the engine actually pays

**It passes by 3.2-3.7x per pick and ~4.5x per layer.** That is the margin the rest of this page spends.

### Where the time goes now (stage 2's numbers before, `peer-compute-tighten` after)

| path | µs/pick before | after | what it is |
|---|---|---|---|
| the APU kernel alone, weights resident in `--vram` | 72.7 | - | 50.1 of compute + the plan/launch/copy the batch now amortizes |
| the APU kernel, rows copied from the RAM slab | 206.2 | - | a 9.1 MB H2D a pick: the tier that residency replaces |
| worker + daemon + wire, 1 pick a message | 252-265 | **108.6-114.0** | a fixed ~64 us a message (a round trip, the plan's copy) plus ~50 us a pick |
| the same, 2 picks a message | 268-307 | **81.7** | the fixed part spreads: 163.6 us for two picks |

Two of the three cuts are taken, and they are what moved those numbers:

1. **Taken: a binary body instead of hex** (`Ctl::send_body`/`recv_body`, `PEER_COMPUTE.md`'s "The protocol"). Every
   buffer travels as raw bytes after its header line - no hex, no `Json` parse per buffer, no `unhex` - while the
   storage exchanges (manifest, hint, map) keep the framing they had, which `peer_rdma_test` still proves byte for byte.
2. **Taken: one plan a batch in the worker**, not a plan a job: gate/up and down are one launch each for the whole
   batch, the down activation is quantized in one, the routing weights and the sums are one copy each.
3. **Left: stage 3's `ya<i>`/`yr<i>`** - the result written into the peer's slab and pulled by the existing
   `Reader::read_many` as `{addr, rkey}`, which removes the outbound body entirely (41 KB a pick becomes one 40-byte
   line) and the round trip with it.

The win against the 478 µs CPU lane is now **4.4x at one pick a message and 5.8x at two**. A token's layer sends six
picks in one message, where the fixed ~64 us spreads over all of them: the measured pair implies **~60 us a pick, 8x
the lane** (a projection from the 1- and 2-pick points, not yet measured at six - the test times one and two).

That is worth stating plainly, because it is where the two boxes' arithmetic said this should land: an APU pick costs
~50 us of compute and ~10 us of wire, and the rest was framing.

## The design

### One protocol, two workers, chosen by the peer

The daemon loads a worker shared object at start-up and never links HIP itself (the `--vram` contract: a peer without
ROCm still builds and runs the storage tier). Two implementations of one plain-C ABI:

| worker | built | runs on | today's number |
|---|---|---|---|
| `strata-peer-compute-cpu.so` | always (`peer_compute_cpu.cpp`) | the peer's cores, `iq_cpu_gate_up`/`iq_cpu_quant`/`iq_cpu_down` as the local lane uses them | 120 µs/expert at 32 threads |
| `strata-peer-compute-hip.so` | only with `STRATA_ENABLE_HIP`, for `gfx1102;gfx1151` (`HIP_ARCHITECTURES` on that target alone) | the peer's cards, `ds4_iq.cu`'s `iq::gate_up`/`iq::down`, `STRATA_DS4_IQ_STRICT_FP=1` | 72.7 µs/pick (APU, resident rows), 259 through the protocol |

The ABI is `src/peer/peer_compute_abi.h`: one `strata_peer_job` per expert (geometry, gate/up/down pointers and their
byte counts as the row table advertised them, the activation's int8 values and scales, the routing weight, `y`), and
`strata_peer_compute_{version,open,run,close}`. `run` returns 0 when it ran the batch and then says per job whether it
computed it; a job it cannot serve is `ok=0` and the caller computes it locally, so a worker is never a single point
of failure.

```
strata-peer --compute <path to the worker .so>
```

Absent: storage only, exactly as before. Given and unloadable (`dlerror`, a missing symbol, another ABI's version
line): a start-up error, the same contract as `--vram`. The daemon logs the worker once -
`compute worker strata-peer-compute/1 cpu loaded from ...` - and names it in its `listening on` line.

The engine does not care which one answers - an order is offered, the peer returns `y` or says it cannot, and anything
the peer cannot do runs locally. That is what makes the CPU worker the right first stage: it proves the protocol,
parity and fallback with no GPU-specific code at all, and it is already worth 4x.

**One worker a card** (`--compute` may be given more than once, each its own `--compute-options`): the daemon keeps
the workers in the order given and decides job by job which one takes it - a `--vram` slab's row is an address on the
card that allocated it, so only the worker whose `gpu<N>` is that card is offered it (no worker on that card: the job
is `ok=0`), while the pinned RAM slab's rows are host bytes **any** worker takes, handed out in turn so two cards'
workers each get a share of a layer. A batch is split into one run a worker; a worker that fails its run answers 0 for
its own jobs only and the rest of the batch goes on. The welcome's `compute` field is then the workers' `version()`
lines joined with ` | ` - still an opaque string to the client, which gates the whole tier on it being non-empty.
`peer-tier.sh` builds the pairs from `PEER_COMPUTE_GPU` (a comma list, `1,0`, one worker each) and, when a worker will
not open, drops the last pair and starts again on what is left - the reason said, never assumed (a slab and a worker
both fail at start-up by design, so the log names which of the two went).

### The three lanes

`src/ds4/ds4_pcie.hpp` already splits a token's picks between the cards, the CPU lane and the streamed lane by
measured time. A peer pick becomes a fourth cost in the same table (`Costs::r`, infinity while the session is down),
so routing decides by time like everything else, and the peer is chosen only for experts resident there.

### The protocol

Reuses the hint/map control channel (`strata::peer::Ctl`, NDJSON, no line cap), so the daemon's row table is the
address space - no new addressing:

- `{"type":"compute","n":k,"n_embd":5120,"n_ff":2304,"rows":[r0,r1,...],"has_w":0|1}` - the layer's geometry and its
  rows once a batch, then per job `j<i>_gate`, `j<i>_up`, `j<i>_down` (row indices), `j<i>_tg`, `j<i>_td` (the two
  quant types) and `j<i>_lim`; **then a raw body**, `body` bytes by the same `rows` arithmetic: for each job in request
  order `xq` (rows x 5120 int8), `xd` (rows x 160 f32), and - when `has_w` - its `rows` f32 routing weights
- reply `{"type":"computed","i":[...],"ok":[...],"body":bytes}` - the request's own indices, one `ok` each - then a
  raw body of `y` (rows x 5120 f32) for the jobs served, in `i` order
- a later stage replaces the outbound body with `ya<i>`/`yr<i>` = `{addr, rkey}` in the peer's slab that the existing
  `Reader::read_many` pulls, which is where `REMOTE_WRITE` on the slab and a writable initiator MR come in

The body is raw bytes (`Ctl::send_body`/`recv_body`), not hex: the line's send is the flush and the body is its
continuation, so nothing is escaped, parsed or doubled on the way.  The storage exchanges (manifest, hint, map) keep
their own framing untouched - `peer_rdma_test` is what proves that.  A pick is ~41 KB out and ~5.8 KB in, i.e. ~10 µs of
wire; what the exchange costs beyond it is now a round trip and the plan's copy, see "Where the time goes now".

Stage 2's contract, verified by `tests/peer_compute_hip_test`: a job names the daemon's own device address with
`resident = 1` when its three rows live in a `--vram` slab (`PeerCache::dev_row`, the slab's `addr_at`), and the worker reads
them in place - no copy of an expert's 9.1 MB.  A resident row is meaningful on the card that allocated it alone, so the daemon
compares the slab's HIP ordinal with the worker's own (the `gpu<N>` field of its `version()` line) and refuses the job (`ok = 0`)
on any other card; the worker checks the pointer's own allocation again before it reads it.  `--compute-options gpu:<n>` tells the
worker which peer card to use (it picks the machine's first otherwise).  The worker is built for `gfx1102;gfx1151` from a tree
configured for anything else (`HIP_ARCHITECTURES` on that one target), with `STRATA_DS4_IQ_STRICT_FP=1` so its rounding is a
local gfx1200 card's and not gfx11's default, and it refuses to open on a card it has no code for rather than fail a decode.

Stage 1's limit, by design: a batch of at most 8 rows (`n_rows` is a token's picks, not a prompt), and a worker with no
device (the CPU one) is only handed rows that live in the **pinned RAM slab** - it answers `ok=0` for a `--vram` row, which
stage 2's worker computes in place. Stage 2 adds one measured limit worth knowing: the wire carries the activation as int8 a 32
values where both lanes quantize to Q8_K (a 256 values a scale), so a peer card's rows agree with a local lane's in norm to
4.7-9.2% and in their worst element to 29-39% of the largest sum - a rounding of the activation, which a device-side Q8_K
requant would close at one kernel and 4 bytes a value, and which carrying Q8_K on the wire would close for free.

### The fallback

Verbatim from the storage tier (`src/ds4/ds4_peer_tier.cpp`): residency and registration are checked *before a byte
moves* (`ok[k]=0`), a failed batch calls `fail()` once ("THE PEER IS DOWN"), and from then on every order goes to
`local_.push(...)` as the read it stands in for. A busy peer is bounded by `Config::timeout_ms` (1000 ms), so a hung
compute never stalls a decode; `Costs::r` becomes infinity and new picks stay local.

### Where it plugs into the engine (built: stage 3's flow hook)

`src/ds4/ds4_model.cu`'s flow (`flow_push`, `flow_peer_layer`, the batch step in `flow_layer`):

- `flow_push` asks the tier once an item (`PeerTier::computable(il, e)`: this tier serves all three of the expert's
  rows and the daemon currently advertises them) and, when it says yes, the item is marked `pcm = 1`: no staging
  slot, no role-orders, no DMA - `flow_pump` records its ring-fill event and moves on.
- `flow_peer_layer` runs once a layer, before its batches: every `pcm = 1` item of the layer has its activation rows
  gathered out of `c.x` by `row_tok` and rounded to int8 **on the card** (`gg::q8_quant_rows`, `q8_quant_block`'s own
  bits), copied once to pinned host memory, and sent as one message a layer (`PeerTier::compute_rows`, `w = nullptr` -
  `moe_combine` applies the routing weights as it does for a local set, so the sums are added exactly where they would
  have been). `y` comes back once a layer and is copied onto the card in one H2D.
- the batch step adds `moe_combine(c.acc, pc_y + row0 * dim, c.pos, c.rw, T, k_exp, a, b, dim)` for each such item in
  item order and calls `run_set` for the runs between them, so the batch's sums keep their order whatever the reads'
  timing. `--peer-compute` off: `pcm` is never set and the code is the pre-stage-3 flow, byte for byte.

No `y` is ever silently lost: an item whose order the peer refuses (a row evicted, a worker that will not take the
geometry, a session that died between two layers) is read off this machine's files at once (`peer_read`: `pk.read_at`
into a chunk-held buffer, `src` pointing at it, `pcm = 2`) and `run_set` computes it - the lane's arithmetic, as
before.

### ... and the decode lane (built: stage 3's second hook, where a decode's picks actually are)

The flow hook above is the **prompt** path. A decoded token's CPU-lane picks never reach `flow_pump`: with the pack's
`iq` expert types the token path routes them to `serve_lane()` (`ds4_model.cu`, the service thread that answers a
layer's routing), which groups the picks by expert and computes them with `cpu_experts` on the host - that call, not
the streamed ring, is the 0.38-0.41 ms a pick. So stage 3's hook there:

- before the lane's own call, the picks are partitioned: an expert `pc_layer_ok` and `PeerTier::computable` both say yes
to becomes **one order** (its rows the picks' tokens, `n_rows` a job), and its x rows - already on the host in `r.x` -
are rounded to int8 there (`gg::q8_quant_row_host`, `q8_quant_row`'s arithmetic, so the codes are the ones a device's
`q8_quant_rows` would have written), and the batch goes over the wire once for the request.
- the lane then computes only what is left, exactly as before, and the peer's `y` rows are added with their routing
weights into the same per-token part (`resps[slot].part`, `cpu[t] = 1`), so the token sums in the order it always did.
- a refusal puts those picks straight back into the lane's set and the local call computes them: same fallback, same
  guarantee - a peer's no may cost a lane's time, never an expert's contribution.
- the gate keeps out what the flow hook also keeps out: a draft (`mtp`) layer, a tensor-parallel group, `DS4_LANE_PARITY`
  layers (a pick's own part is the device's, not a weighted sum) and any layer whose three parts are not one `iq` matrix
  of the row sizes the table names.

## Stages

Each stage must earn the next one. Nothing after stage 0 has been built.

| stage | what | must measure | worth keeping if |
|---|---|---|---|
| 0 | the tests above, both peer cards + its CPU lane + perftest | µs/pick, wire µs | peer + wire < 0.33 ms/pick - **done, 88-103 µs** |
| 1 | CPU worker + `compute`/`computed` + fallback, `tests/peer_compute_test.cpp` | `y` bit-identical to the local lane, per-pick time, a daemon killed mid-batch | **done**: `y` is `memcmp`-identical to the lane fed the wire's own int8 bytes, within 1.4% of the row max of the lane fed the raw floats (one extra Q8_K rounding, asserted <5%); a 6/7/8-job batch with one unheld row answers `ok=0` for that job only; a daemon killed mid-batch gives THE PEER IS DOWN and all-zero `ok`. `peer_compute_test`, `peer_rdma_test`, `ds4_peer_tier_test` pass |
| 2 | HIP worker, APU first then the 7600 XT, IQ2_XXS only, `MAYA_DS4_IQ_ONLY` | µs/pick end to end, `y` against the local kernel | **done on the APU, and it misses the target**: 72.7 µs a pick in the worker (against 50.1 µs of pure compute, so ~22 µs of per-job launch), 252-307 µs through the protocol against the 250 µs target - the gap is the hex body, not the kernels. `y` agrees with the local lane's norms to 4.7-9.2%, worst element 29-39% (see the rounding note above) |
| 3 | residency hinting at hello + the flow hook + the fourth lane in `ds4_pcie` | decode tok/s on a 512+128 greedy run (baseline: 8.81 no-draft, 9.77 dspark), 16k prompt tok/s | >=1.15x decode, <=2% prompt regression - **built, not yet measured**: the A/B is the next step (`./peer-tier.sh up` with `PEER_COMPUTE`) |
| 4 | the APU slab as the residency pool (`--vram 1:64`-class) | hit rate per token, peer RSS against 112 GB | >=1.4x decode, safely inside memory |

## Risks and traps

- **`DS4_IQ_STRICT_FP` is on for `gfx1200`/`906` and off for `gfx11`** (`src/ds4/ds4_iq.cu`), so a peer card's result
  differs from a local one's in the last bits. The worker wants an override to keep parity; it costs ~7% of `gfx11`
  decode, paid only in the worker. Otherwise document the ULP drift rather than pretending it is exact.
- **A `--vram` slab is not counted by the daemon's budget.** `--ram-gb 80` plus `--vram 1:64` is 144 GB on a 112 GB
  APU box, and the APU's slab *is* system RAM. The sum must be capped before this tier is used at scale.
- **Residency is one-shot.** VRAM rows are never evicted and never re-pinned - the hit rate is decided at hello. If
  the hint set does not match the routing the win disappears with it. The engine's set is its own profile's ranking
  (the same order `DS4_WARM_CAP_GB` takes), truncated at the peer's slab by `PeerTier::open_session`.
- **The compute path is serialised with the layer's reads** (the flow's own thread computes the layer's orders before
  its batches run), so the peer's round trip is not hidden behind that layer's streaming yet - that overlap is the
  peer queue's next step, and until then a layer with many orders pays a message's latency each.
- **A peer's sums are not this machine's.** The storage tier is byte-exact; a computed expert's sums differ in the
  activation's rounding and the reduction's order (~1e-2 relative), which is why `DS4_LANE_PARITY` and `--tp` keep
  the local arithmetic.
- **Batch or don't bother.** One pick in one message costs 108 µs on the 7600 XT - more than all six of a layer's
  picks together.
- **The peer's card may be doing something else.** ComfyUI on this box is the normal case; a worker that cannot start
  must fail at start-up, never at the third layer of a decode.
- **Do not move a pick that is already local** - a local card is 41-52 µs and no wire at all.
- The daemon must stay buildable and runnable without ROCm: the worker is a separate `.so`, and `--compute` with a
  missing file is a start-up error, not a slow path.

## Reproducing the stage 0 numbers

```
rsync -a --exclude='.git/' --exclude='.venv/' --exclude='build-*/' ./ peer:/home/<user>/peer-bench-src/
cmake -S peer-bench-src -B peer-bench-build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON \
      -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES="gfx1102;gfx1151" \
      -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1 -Dhip_DIR=/opt/rocm/core-10.1/lib/cmake/hip
cmake --build peer-bench-build --target ds4_gguf_test ds4_lane_bench -j 24      # 2m53s on 24 threads

HIP_VISIBLE_DEVICES=0 ./ds4_gguf_test --bench-q      # RX 7600 XT
HIP_VISIBLE_DEVICES=1 ./ds4_gguf_test --bench-q      # APU (its "VRAM" is system RAM: expect hipMalloc to be GTT)
./ds4_lane_bench --threads 32                        # and --threads 8 for the engine's lane

ib_write_lat -d rocep198s0 -x 3 -F -s 26240 -n 20000      # server, peer side
ib_write_lat -d rocep7s0  -x 3 -F -s 26240 -n 20000 <peer> # client, this side (same for ib_read_lat)
```

`-x 3` is the RoCEv2 over IPv4 GID on both cards here (`ibv_devinfo -v`); `-F` allows the non-power-of-two size.
