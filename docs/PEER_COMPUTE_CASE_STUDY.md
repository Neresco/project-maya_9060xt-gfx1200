# Peer compute tier - measured case study (anonymised)

A worked example of the `--peer-compute` tier (see [PEER_COMPUTE.md](PEER_COMPUTE.md) for the contract and
[PEER_TIER_RUNBOOK.md](PEER_TIER_RUNBOOK.md) for the operations), measured on a real pair of machines. All host names,
addresses, paths and pack names are replaced by "machine A", "machine B" and "the model"; the numbers, code paths and
commands are the ones actually used.

## 1. The question the tier answers

A Mixture-of-Experts model routes every token to a handful of experts. Each of those experts is several megabytes of
weights, and they are spread over far more bytes than any single machine's VRAM holds. The tier's proposition is:

> for an expert this machine does not hold in VRAM, is it cheaper to have the other machine compute it, or to do it
> here?

Everything below is an answer to that sentence, and the answer turns out to depend on one variable: **how starved is
this machine's own host lane**.

## 2. The pair

| | machine A (runs the engine) | machine B (the peer) |
|---|---|---|
| CPU | 8 cores / 16 threads, desktop class, 96 MB L3 | 16 cores / 32 threads, APU with unified memory |
| RAM | 125 GiB DDR4, 2 channels, ~51 GB/s peak | 112 GiB LPDDR5X 256-bit, ~256 GB/s peak |
| GPUs | 4 discrete cards, 16 GiB each, all gen5 x16 negotiated (the board bifurcates them) | one 16 GiB discrete card (PCIe 4.0 x8) + the APU's integrated GPU, whose "VRAM" is system RAM |
| storage | NVMe, 341 GB pack in 6 shard files | NVMe, its own copy of the same pack |
| link | 25 Gb/s RoCEv2 each end, direct cable | same |

The model: 40 MoE layers x 384 experts, 6 experts a token, 10.60 MB or 12.63 MB an expert depending on the layer's
quant class. That is 240 picks a token and **2.641 GB of expert weights a token** if nothing is resident - which is
what makes residency, not bandwidth, the primary design constraint.

## 3. What was measured on the hardware

Per direction unless stated; "peak" is the platform figure, everything else was measured.

| path | machine A | machine B |
|---|---|---|
| RAM peak | ~51 GB/s | ~256 GB/s |
| RAM, 1 GiB memcpy (single core) | 17.7 GiB/s copy = ~38 GB/s traffic | 26 GiB/s copy = ~56 GB/s traffic |
| host -> device, pinned, 1 GiB, best of 4 | 7.2 / 6.5 / 14.4 / 14.5 GB/s (cards 0..3) | 7.1 GB/s (discrete card); 83.8 GB/s (integrated, unified memory) |
| storage, cache bypassed | 2.0 GB/s (512 MiB, O_DIRECT) | 974 MB/s (2 GiB) |
| NIC line rate / measured | 25 Gb/s / **2.6 GB/s** between the two machines (2 MB frames) | same |

Two facts from that table shaped everything else:

* the four cards negotiate gen5 x16 yet half of them deliver ~7 GB/s of pinned copy and half ~14 GB/s - the board
  splits the lanes. That PCIe path is still ~3x the wire, so a local expert that is already in RAM is *cheaper* to
  handle locally than over the link;
* machine B's integrated GPU reads its slab out of system RAM (83.8 GB/s measured, no PCIe crossing), which is why
  the peer's bigger slab is the APU's, not the discrete card's.

## 4. What the tier does with them

* **row table**: one row per (layer, expert, role), three roles (gate, up, down) - 46 080 rows for this model, laid
  out so a served layer's rows are a run.
* **residency, once at load**: the engine ranks the experts by its own hit profile (most-routed first) and names rows
  until the peer's slabs run out. Measured: **13 387 rows = 46.0 GB across both of machine B's slabs, filled in 80 s**,
  i.e. ~4 462 experts - roughly doubling the resident expert pool without touching machine A (whose four cards hold
  3 894 experts, and whose host lane keeps 11 824 more "in place" on NVMe).
* **reads**: a row the daemon does not hold is hinted and pulled (2 MB chunks, 48 outstanding) into the engine's
  staging; the peer's RAM slab keeps filling from those hints.
* **compute orders**: an expert whose three rows are resident becomes one order - the worker reads them in place on
  the card that allocated them, sums them, and returns the weighted sums. A refusal (not resident, wrong card,
  geometry) puts the picks straight back on the local lane, so a peer's "no" costs time, never an expert's
  contribution.
* **one QP per session, no reattach**, and the engine never writes into the peer's memory: the flow is read-only.

Measured end to end: **99.4-99.6 % of compute orders served** (49 946 / 50 232 and 49 546 / 49 729), with 21 of
18 728 batches refused - the engine computes those locally.

## 5. The result, and why

Decode, 400 tokens, temperature 0, same build both arms, request *i* against request *i*:

| local lane threads | tier on | no tier | verdict |
|---|---|---|---|
| 8 (the machine's normal) | 16.91 / 17.02 tok/s @ 94.7-95.0 % expert cache | 16.66 / 16.80 @ 89.1-89.2 % | **parity, +1.3-1.5 %** |
| 2 (deliberately starved) | 12.02 / 13.29 / 14.23 tok/s | 9.54 / 11.22 / 11.40 | **+26 / +19 / +25 %** |

The reason is in the per-pick costs:

| doing one pick | cost |
|---|---|
| machine A's own lane, 8 threads busy | 393 us an expert (its own log) |
| the peer, rows resident | 119-164 us a pick through the wire (50-65 us of it is the card's own arithmetic) |
| the peer, one message with two picks | 82 us a pick (batching amortises the round trip) |
| machine A's local disk | 6.4-7.8 ms an expert |

The peer is 2-3x faster per pick than the local lane per pick - but the lane runs several picks at once, so at 8
threads it is not the bottleneck at all: only 5-11 % of a decode's experts go to the lane, and the decode waits on the
GPU links. Cut the lane to 2 threads and it becomes the bottleneck, and then the tier pays.

Costs that came with the tier, unchanged by the above:

* **the prompt path is a loss**: the first prompt read took 63 s with the tier against 9-12 s without it. Prompt
  misses are many, and the peer's disk (974 MB/s) plus the wire (2.6 GB/s) are slower than the local NVMe (2.0 GB/s)
  for a burst of cold experts;
* **not bit-exact**: the peer's sums agree with the local lane to ~1e-4 relative (verified offline, element by
  element), but a different rounding order occasionally flips a token at temperature 0. Two runs of the same prompt
  produced 1 019 vs 1 022 characters.

## 6. Is it a VRAM extension? Capacity yes, context no

**Expert weights - yes, with a hit-rate condition.** The peer added 46 GB of GPU residency plus 40 GB of RAM slab.
But the capacity only pays while most picks are already resident: a token needs 2.641 GB of weights, so a
fully-missed token at 2.6 GB/s would cost ~1 s - 1 token/s. The tier is a *residency* mechanism; as a paging tier for
a cold model it would be unusable. Against "just keep the pack on local disk" the wire is only ~1.3x the local NVMe
(2.6 vs 2.0 GB/s), so the advantage comes from residency, not from the link.

**Context - no.** The tier moves expert weights only. KV cache is read every token, for every layer, with latency
rather than bandwidth dominating; a ~120 us round trip cannot substitute for local VRAM. More context is not
addressable by this design.

## 7. Break-even, in one paragraph

The tier wins if the peer's per-pick cost is lower than what the local machine pays for the same pick *including the
parallelism it has left*. On 8 spare cores the local lane does a token's six picks in parallel and the tier is a wash.
On two cores it is a 19-26 % win. Lowering the peer's per-pick cost (a layer's picks batched into one message: 119 ->
82 us measured at 2 picks, and a whole layer's 6-8 should go further) moves the break-even towards machines with
normal core counts; making the prompt path stay local removes the one measured regression. Neither changes the
structural ceiling: **the wire is 2.6 GB/s and the local PCIe path is 7-14 GB/s**, so a peer expert can only ever win
on the arithmetic, never on moving the bytes.

## 8. Bugs this exercise found (worth knowing if the tier is revived)

The tier had never served a single order before this measurement, for four separate reasons, all now fixed:

1. the engine never set `PeerTier::Config::compute` nor filled the ranked `resident` list, so `pc_on` was always
   false - `--peer-compute` / `MAYA_DS4_PEER_COMPUTE` and the profile-ordered residency list had to be restored;
2. the HIP worker never made its own card current (`hipSetDevice`) before a batch, so with one worker per card in one
   process a batch launched on the other card's stream: `hipErrorInvalidResourceHandle`, and a poisoned context
   afterwards;
3. `hq_quant_kernel` read a row's swiglu at the plain row index while `gate_up_kernel`/`down_kernel` address it by
   entry (`row * kEntryStride`), so only row 0 was ever quantized correctly and the rest of a batch's down ran on
   zeros - which also produced the `HSA_STATUS_ERROR_MEMORY_FAULT` seen in the engine's batches;
4. every host job of a plan copied its weights into the plan's first staging slot while each group read its own slot,
   silently wrong for all but one order of a plan.

Reproduction without the engine: `build-hip/peer_compute_hip_test` (built with `STRATA_BUILD_TESTS=ON`) started on
machine B, pointed at the peer binary and the worker module with `STRATA_PEER_BIN`, `STRATA_PEER_COMPUTE_HIP_SO`,
`STRATA_PEER_COMPUTE_HIP_GPU` and optionally `STRATA_PEER_TEST_TWO_SLABS=1`. It builds a two-expert synthetic pack,
loads the real daemon with a `--vram` slab, and checks the sums against the local lane - including an eight-order
batch and two host rows in one message, which are the two cases that catch bugs 3 and 4. The worker module contains
code for its two target architectures only, so the test skips on a machine whose cards it was not built for.

## 9. Where the numbers came from

`lscpu`, `/proc/meminfo`, `lsblk`, `/sys/class/infiniband/*/ports/1/rate`,
`/sys/bus/pci/devices/*/current_link_{speed,width}`; a 1 GiB `memcpy` for RAM; `dd bs=4M iflag=direct` (cache bypassed
where the volume does not support `O_DIRECT`) for the disks; a small HIP program (`hipMemcpy`, 1 GiB pinned, best of
four) per card; the A/B runs are detached scripts that stop the production engine, run both arms cold, and put
production back at the end. The served-order counts are the daemon's own log lines on machine B.
