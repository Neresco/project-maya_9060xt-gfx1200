# Two machines: the peer expert tier (RoCE)

A Maya engine reads the experts it misses from disk. This page is about the alternative that puts those bytes on a
**second machine's RAM** and pulls them over **RoCE** (RDMA over Ethernet): a small daemon there (`strata-peer`) holds
an unpacked slice of the model's pack in a pinned, registered arena, and its NIC answers the engine's RDMA READs. The
peer's disk and CPU never enter the decode; the engine's own disk does the rest, so a run is never *worse* than a
single machine.

It is optional and off unless you ask for it. Back to the [README](../README.md), every setting in
[docs/SETTINGS.md](SETTINGS.md).

## What it is good for

- **Two machines, one model:** the pack stays where it is on each side. The engine reads the shards the peer holds over
  the link and the ones only it holds from its own disk.
- **More RAM for the experts:** the RAM tier is the cheap tier; a second machine's RAM is more of it.
- **Measured link:** two ConnectX-4 Lx cards over a direct 100 GbE link, RDMA READ of 2 MiB messages -
  **24.5 Gb/s, about 3.0 GB/s** (`ib_read_bw -s 2097152 -n 3000`, MTU 4096). A cold expert's parts (3-9 MB) arrive
  in a millisecond or two of wire time; the engine's per-layer staging is what queues the reads.

## What you need

- **RDMA hardware on both machines** and a link between them that both ends reach as the same GID family: the engine
  brings the queue pair up by hand (no `rdma_cm`), exchanges `{qpn, psn, gid}` over TCP, and prints the device, GID
  index and GID it chose at bring-up. Check with `ibv_devinfo` / `rping` first.
- **The same pack on both machines** - the daemon serves byte ranges of *its* copy. The handshake fingerprints the
  files (size plus the first and last MiB of each) and refuses a mismatch, so a half-copied or re-quantised pack
  cannot serve silent garbage.
- The link's MTU on both ends (9000 or the interface default; the QP takes the port's active MTU).

## The daemon

```
strata-peer --pack <dir> [--bind <ip>] [--port 9447] [--ram-gb <n>] [--arena-mb <n>] [--device <ibv device>]
            [--vram <gpu>:<gb>]...
```

- `--pack` is a directory holding the pack's shard files under their real names (the engine's manifest names them; the
  daemon resolves the basenames inside this directory). It holds everything it needs to have: with the DeepSeek pack's
  341 GB and a peer with 214 GB free, that is a subset - see `--peer-shards` below. A name the pack's tensor list keeps
  **beside** the model directory instead (`../dspark/dspark.bin`, `../vision/vision.bin` in DeepSeek's) is not a shard
  and is never offered: the daemon would look for its basename in `--pack` and, worse, the pack fingerprint would then
  fail for the six files that ARE there. The engine says so once and puts the rest on the wire.
- `--ram-gb` is the slab the daemon may pin for the experts it has faulted in (default: a quarter of what the cgroup
  lets it have, capped); `--arena-mb` is the registration granularity (2 GiB default: few, big arenas; 256 MiB suits a
  big pack better - an evicted arena is dropped whole).
- `--device` names the ibverbs device (`rocep7s0`, `rocep198s0`, ...); empty: the first one found.
- `--vram <gpu>:<gb>` (repeatable, **off by default**) puts the peer's rows in a **VRAM slab** instead of pinned
  RAM: the slab is one `hipMalloc` held in that GPU, exported as a dmabuf (`hsa_amd_portable_export_dmabuf`) and
  registered as a memory region (`ibv_reg_dmabuf_mr`), so the initiator's RDMA READs are served straight out of the
  card's memory. `<gpu>` is the peer machine's own HIP ordinal (`rocm-smi`/`rocminfo` order: on the strix box 0 is the
  RX 7600 XT, 1 the Ryzen AI Max APU), `<gb>` the slab's size and may be fractional (`--vram 0:12`, `--vram 1:64`).
  It is **start-up-fatal when it cannot be honoured** - the runtime is missing, the ordinal does not exist, the card
  cannot give the bytes (something else is using it), or the NIC cannot register a dmabuf of it - because a peer whose
  card is busy should say so at start-up, not fail a fill an hour into a decode.
  On an APU the slab is the machine's own RAM behind the GPU's GTT aperture (no extra memory); on a discrete card it is
  dedicated VRAM, which is the pool worth taking. Rows take VRAM first, in the order the flags were given; a row no VRAM
  slab can hold (every slab full) **spills into the RAM slab** and the row map does not care which it got, so a small VRAM slab is a fast path, never a cliff. VRAM
  rows are never evicted (a slab is exactly the size you asked for and is filled once), and they do not count against
  `--ram-gb`. The daemon's log says both halves: `hint: 12 rows, 48 MiB cached (12 rows, 2 arenas, 32 MiB of it in
  VRAM)`. The daemon reaches the GPU runtime through `dlopen` (`src/peer/peer_vram.cpp`), so a peer machine without
  ROCm still builds and runs the RAM tier; nothing needs HIP headers to build it.
- It **serves one initiator at a time** and never reads the pack itself unless the engine hints a row: a hint says
  "my routing is about to need these", the daemon preads them from its own disk into the arena and advertises
  `{address, length, rkey}`; the engine then reads them all in one window.

Building it is part of the normal build (`cmake --build build-hip --target strata-peer`) whenever ibverbs is installed;
without it the flag below does nothing and the engine runs as it always did.

## The engine's side

Both engines take the same three flags; the pack files the peer holds are named as the pack's own numbering
(`maya-ds4 --pack-info` lists them for DeepSeek):

```
maya            --peer <host> [--peer-port 9447] [--peer-device <ibv device>]      (GLM-5.3-Flash / Maya packs)
maya-ds4        --peer <host> [--peer-port 9447] [--peer-device <ibv device>] [--peer-shards 0,2,3]
```

`--peer-shards` (DeepSeek only) names the pack files **the peer holds**; every expert whose shard is not in that list
is read from this machine's disk, exactly as before. Empty: the peer holds the whole pack. The same knobs live in the
environment - `MAYA_GLM_PEER*` / `STRATA_GLM_PEER*` and `MAYA_DS4_PEER*` / `STRATA_DS4_PEER*` (`_PORT`, `_DEVICE`,
`_WINDOW`, `_CHUNK_MB`, `_TIMEOUT_MS`, and for DeepSeek `_SHARDS` as the comma list `--peer-shards` takes) - for a
setup that starts the engine from a script, whose config file it does not touch.

Example, two machines, the DeepSeek pack on this side (all six shards) and shards 2 and 3 on the peer:

```
# on the peer (192.168.100.20), a directory holding shard 2 and 3 under their real names:
strata-peer --pack /models/ds4-peer --ram-gb 80 --arena-mb 256 --device rocep198s0

# on this machine:
maya-ds4 --pack /models/maya-ds4 --peer 192.168.100.20 --peer-shards 2,3 --ids 1,2,3 --n 200
```

On this pair of machines the three steps (stop what holds the cards, bring the peer up, start the engine with the
tier on) are one script: `./peer-tier.sh up` - it warns first, because what it stops is whatever serves the
API on :8080, and `./peer-tier.sh status` says where both halves stand (`down` stops the engine and the
daemon). Its `PEER_VRAM` setting is the peer's own VRAM tier (`--vram`, see above): `0:14` by default, which takes
the 7600 XT's 16 GiB of dedicated memory; empty leaves the tier off. On an APU (`--vram 1:N` on this box) the slab is
**not extra memory** - its "VRAM" is the same system RAM through the GPU's GTT aperture - so only the discrete card
adds a pool the peer did not already have.

Both sides print what they are doing: the daemon its arenas and faults, the engine one line at start-up
(`ds4 peer: session up with ...`) and its counters once at the end (`ds4 peer: N rows (x GB) pulled in M batches
(y GB/s on the wire), ... hints`).

## `--peer-compute`: the peer sums the experts it holds

The tier above moves bytes. `--peer-compute` (or `MAYA_DS4_PEER_COMPUTE=1`) goes one step further: an expert the peer
already holds is **computed there**, and only its weighted sum comes back - 28 KB for a decode row instead of the
10.6 MB of gate, up and down that the same row would otherwise pull. It is opt-in, DeepSeek only, and off by default:
without the flag the tier is exactly the storage tier it was.

What it does, per layer of a decode-shaped run:

- **Residency at hello.** The engine ranks its experts the way its own read-ahead does (`DS4_WARM_CAP_GB`: "the
  most-routed N GB") - by the routing profile - keeps the ones whose three parts the peer serves whole in types a
  worker computes, and asks the daemon for them in that order when the session comes up. The list is truncated where
  the peer's slab runs out: `--vram` bytes when it has a card in this tier, else half the RAM slab (so a later
  hint still has room). The engine logs `ds4 peer: N experts may be held resident (X GB of rows)`, the tier
  `N resident rows hinted ... M rows held`.
- **The flow hook** (`src/ds4/ds4_model.cu`, `flow_push` / `flow_peer_layer`): a picked expert whose three rows the
  tier says it holds (`PeerTier::computable`) is not read at all - the flow skips its three role-orders, stages
  nothing and copies nothing. Once a layer, before its batches run, the layer's such experts go to the peer as **one
  compute order each** (the activation gathered from `row_tok`, rounded to int8 on the card by `gg::q8_quant_rows`,
  weights left to the engine: `moe_combine` applies them as it always did), and the returned sums are added where
  `run_set` would have added the expert's own.
- **One message a layer**, not one a pick: the wire's cost is a message's latency, so a layer's orders travel
  together (the daemon's framing takes up to 64 jobs of 1-8 rows).
- **A peer that says no loses nothing.** An order the peer refuses - a row its arena evicted, a worker that will not
  take the geometry, a session that died between two layers - is read off this machine's files at the same spot and
  computed locally by `run_set`, so the contribution is the lane's. What is *not* identical is when the peer does
  answer: its sums are the peer's arithmetic, not this machine's tiles' (a different activation rounding and
  reduction order, ~1e-2 relative on a sum, as `docs/PEER_COMPUTE.md` measured) - the storage tier is byte-exact,
  this one is not.

Limits worth knowing before trusting a number:

- **Decode-shaped chunks only**: a chunk of more than 8 tokens (one peer job's row limit) reads its experts as it
  always did, so a prompt gets no compute help - only its decode does. `DS4_LANE_PARITY`'s exact sums and
  tensor-parallel members (`--tp`) are excluded the same way: they are the local lanes' arithmetic by definition.
- **A decode asks the peer's disk for nothing.** In a chunk whose tokens one peer order takes, a row the peer does not
  already hold is not hinted: the flow reads it off this machine's own files, so the peer's SSD is read once (the
  resident fill at hello) and its link carries sums instead of weights. A prompt's chunk (more tokens than an order
  takes) keeps the storage tier exactly as it was - and so does a run with `--peer-compute` off.
- **Serialised with the reads, for now**: a layer's orders are computed and awaited on the flow's own thread at the
  layer's start, so the peer's round trip does not overlap that layer's streaming. Overlapping it (the orders riding
  the peer queue beside the reads) is the next step, not this one.
- **The decode lane's own hook.** A decoded token's CPU-lane picks never go through the flow: with the pack's `iq`
  expert types the token path hands them to `serve_lane()` (`src/ds4/ds4_model.cu`), which computes them with
  `cpu_experts` - the 0.38-0.41 ms a pick. There too an expert the tier holds becomes one order (its rows the picks'
  tokens, its activation rounded on the host by `gg::q8_quant_row_host`, `q8_quant_row`'s own arithmetic) and its
  weighted sum is added into the token's part where the lane's would have been; a refusal puts those picks back in the
  lane's set, so the fallback is the same one. Draft (`mtp`) layers, tensor-parallel members and `DS4_LANE_PARITY`
  layers keep the local arithmetic, as chunks over 8 tokens do.
- **One worker a card** (built): `--compute` may be given more than once, each with its own `--compute-options`; a
  `--vram` row goes only to the worker whose `gpu<N>` is that slab's card, the pinned RAM slab's rows to the workers in
  turn, and the welcome's `compute` field is their lines joined with ` | `. `peer-tier.sh` makes the pairs from
  `PEER_COMPUTE_GPU` (a comma list) and drops a worker the peer will not open instead of taking the run down.

## When it goes wrong: it falls back

Everything here is optional and every failure is local:

- a row the daemon does not hold (never hinted, or its arena was evicted), a destination the NIC will not register, a
  session that just died - the engine reads **that** slice from its own disk, byte for byte, and the decode is
  unchanged (only slower).
- Kill the daemon mid-decode: the batch in flight fails, the engine prints `THE PEER IS DOWN` once, and from there
  every miss is a local read again. The run finishes.
- A fingerprint mismatch, a missing shard, a full slab: said once at start-up, and the run proceeds on the local disks.

Nothing in this tier may turn a working decode into a failing one; that is the contract the code is written to
(`src/ds4/ds4_peer_tier.hpp`, `src/core/glm_peer_tier.hpp`) and what the tests assert.

## What it does not do (yet)

- **One peer, one session, no reattach.** A dead session stays dead for the run (a re-connect would have to re-advertise
  the map mid-decode; the local disk is the cheaper answer).
- **No GPU-direct on the engine's side.** The *peer* can serve rows out of its own VRAM (`--vram`), but the bytes
  still land in the engine's pinned host staging and take the usual DMA into its cards from there: a NIC writing into
  the engine's VRAM (or registering its allocations as the read's destination) is not done.
- **No write-back.** The peer never asks the engine for anything; the arena is read-only from its side.
- **GLM's decode path only** (`src/core/glm_peer_tier.cpp`): a prompt's reads stay on the disk path, and GLM has no
  compute tier at all - `--peer-compute` is DeepSeek's.
- **DeepSeek's flow path only** (the in-place expert flow, `ds4_model.cu`'s `flow_pump`): RAM-tier copies that pass
  through staging, the layer cache and `DS4_FLOW_DOWN_SHARE`'s shared down read keep their own bookkeeping and stay
  local.
- **A hinted row is a whole matrix slice:** a tensor-parallel member reads a sub-range of it (`rel`, `bytes`), which is
  bounds-checked on both ends.

## Verifying a build

Two tests run without a model or a GPU, against the real daemon on the machine's own RDMA device:

```
cmake --build build-hip --target peer_rdma_test ds4_peer_tier_test strata-peer -j
pkill -x strata-peer; ./build-hip/peer_rdma_test        # transport: handshake, hints, reads, a killed daemon
./build-hip/ds4_peer_tier_test                          # DeepSeek's end: the row table, sub-ranges, the fallback
```

Both read a fixture pack written into `/tmp` and print `PASS`. They skip themselves (and say so) on a machine with no
RDMA device.

The engine's READ destination is not an ordinary heap buffer but a pinned `cudaHostAlloc` (see `halloc` in
`ds4_model.cu`), and registering those with the NIC is the one thing a GPU-less test cannot cover. It was checked
once, by hand, on ROCm 7 / RX 9060 XT: a 96 KiB and a 256 MiB `hipHostMalloc` buffer both registered as memory
regions and a row read into each landed byte for byte (the far end of the 256 MiB one included). No VRAM is involved
in that - a pinned buffer is host memory the NIC can reach.

The daemon's own `--vram` tier needs a GPU, so it is not in those two tests either. It was checked on the two-machine
pair with the real pack: a daemon on the peer (`--vram 1:8` on the APU, `--vram 0:1` on the RX 7600 XT) exported and
registered a dmabuf MR on both cards, served hinted rows out of VRAM, and the initiator's RDMA READs landed byte for
byte identical to the pack on disk (`realpack_probe`: 5 rows, 14 MiB). A deliberately small slab (`--vram 0:0.008`,
8 MiB) filled with 8 MiB of rows and spilled the 6 MiB row that no longer fit into the RAM arena - also byte for byte -
and a slab the card cannot give (`--vram 0:200` on a 16 GiB card) stops the daemon at start-up with `hipMalloc ...
failed: 2 (out of memory)`.

## How the transport works (for the curious)

- No `rdma_cm`: both sides create their queue pair on the device's own PD and CQ, then exchange
  `{qpn, psn, gid16hex}` over the TCP control channel (`NDJSON` lines). `rdma_cm`'s accept path forces a
  `REMOTE_WRITE`-only QP, and then an incoming READ is NAKed with *remote access error* - which is why this is done by
  hand.
- QP: INIT with `LOCAL_WRITE | REMOTE_WRITE | REMOTE_READ`; RTR `path_mtu` = the port's active MTU,
  `max_dest_rd_atomic` 1, `min_rnr_timer` 1, `hop_limit` 1, GID index as printed; RTS `timeout` 14, `retry_cnt` /
  `rnr_retry` 7, `max_rd_atomic` 1.
- Reads are posted as ~2 MB chunks through a bounded window (`_WINDOW`, 48 by default) and drained once per batch:
  posting faster than the completions come back ends in `ENOSPC` on `ibv_post_send`.
- The row table travels once, at hello: row `i` is `{shard, offset, length}` of the initiator's pack, and the daemon's
  map answers `{address, length, rkey}` per row it holds. A hint is a row index list; an arena eviction is answered by
  a drop list, and the engine simply reads those rows locally.

## Our link, for reference

| | |
| --- | --- |
| Cards | ConnectX-4 Lx at both ends |
| Interfaces | `rocep7s0` / `enp7s0np0` at 192.168.100.101, `rocep198s0` / `enp198s0np0` at 192.168.100.20 |
| MTU | 4096 (the interface default here); RDMA READ, 2 MiB messages: 24.5 Gb/s |
| GID | the engine picks a RoCE v2 entry when the port has one, else a v1 entry, else index 0 (it prints which) |
