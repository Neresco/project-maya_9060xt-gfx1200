# Runbook: the tiered run, and how to get back if it does not come up

Two machines, one model: this box runs the engine, strix (192.168.100.20) serves rows out of its RAM/VRAM slab and,
once the compute tier is wired in, computes some of the experts it holds. `peer-tier.sh` starts it all; this page is
what to type when it goes wrong.

**The one thing to know first:** the engine behind `:8080` is what serves the chat, so a tiered run means the chat
(and any assistant session attached to it) is down while the test runs. Everything below is meant to be usable from a
plain terminal, with no assistant in the loop.

## Start a tiered run

```
cd /home/lunarbuntu/Kobold/project-maya_9060xt-gfx1200
PEER_RAM_GB=40 PEER_VRAM=1:24 PEER_COMPUTE=strata-peer-compute-hip.so PEER_COMPUTE_GPU=1 ./peer-tier.sh up
```

Both of strix's cards computing (`PEER_COMPUTE_GPU` a comma list, one worker a card - the daemon hands a job to the
worker on its slab's own card, and the RAM slab's rows to them in turn):

```
PEER_RAM_GB=70 PEER_VRAM=1:32,0:14 PEER_COMPUTE=strata-peer-compute-hip.so PEER_COMPUTE_GPU=1,0 ./peer-tier.sh up
```

A worker the peer cannot open is dropped (the last pair first) and the run goes on with what is left - `up` says so and
keeps going; only a daemon that comes up with no listener at all is fatal.

- `PEER_VRAM=1:24` is the APU's slab. The APU is **GPU 1** on strix (`rocm-smi` order; GPU 0 is the RX 7600 XT) and its
  "VRAM" is the machine's own RAM behind the GTT aperture, so the slab is *system* memory, and the daemon's `--ram-gb`
  does not count it: `40 + 24 = 64 GB` of strix's ~112 GB leaves ~48 GB for the daemon's own pages and the page cache
  that carries the rest of the pack. 32 GB works too (72 GB total) but has less room; the default 80 GB RAM slab plus
  any APU slab is the combination that runs a 112 GB box out of memory.
- A worker and a slab must be on the **same card** (a resident row is an address on the card that allocated it), hence
  `PEER_COMPUTE_GPU=1` with `PEER_VRAM=1:32`. The RX 7600 XT would be `0:14` / `gpu:0`.
- `PEER_COMPUTE=` empty keeps the peer a storage tier. The worker is copied to `~/strata-peer-compute.so` on the peer;
  strix needs no build of its own.
- `up` first stops the engine it finds, so the running production engine is *replaced*. It waits for the API, then
  proves the tier by looking for `session up` in the peer's log, and dies with a reason if the engine on `:8080` is not
  the tier's own (it reads `MAYA_DS4_PEER` out of the running process).

## Watch it

```
./peer-tier.sh status            # engine, server, tier on/OFF, cards, the peer's daemon, the last log lines
```
Logs: engine `maya-ds4-hip.log`, server `/tmp/maya-peer-server.log`, peer `/tmp/strata-peer-real.log` **on strix**
(`ssh lunarbuntu@192.168.100.20 'tail -f /tmp/strata-peer-real.log'`).

## If the engine does not come up again

Do this, in order, from a terminal. Nothing here depends on an assistant being attached.

```
cd /home/lunarbuntu/Kobold/project-maya_9060xt-gfx1200
./peer-tier.sh down                              # stops its engine and the peer's daemon
pkill -f "serve/server.py"; pkill -x maya-ds4    # belt and braces: nothing of the failed run survives
pkill -x strata-peer                             # local daemon, if one was started here
ssh lunarbuntu@192.168.100.20 'pkill -x strata-peer'   # the peer's, which holds the 32 GB slab until it exits
```

Then check what the failure left behind:

```
ss -ltnp | grep ':8080\|:9447'                   # nothing should listen
free -g                                          # local RAM back to ~32 used / ~93 cache
rocm-smi --showmeminfo vram                       # four local cards: ~61.5 GiB accounted for by the engine
ssh lunarbuntu@192.168.100.20 'free -g; rocm-smi --showmeminfo vram'   # the 32 GB slab is gone with the daemon
```

And start the normal host again — this is exactly what the production run is:

```
.venv/bin/python maya.py --gpus 0,1,2,3 --host 0.0.0.0 --vram-reserve-mib 1536
```

Wait for `:8080` (`ss -ltnp | grep 8080`), then `./peer-tier.sh status` must say `tier: OFF (the running engine has no
MAYA_DS4_PEER)`. That is the pre-test state; the chat is back.

Notes for the two ways this usually fails:

- **The daemon died on `--vram 1:24`** (strix memory short, or the card busy): `up` says so and falls back to the RAM
  slab by itself. To free the card, close whatever uses it (`rocm-smi --showpids` on strix; ComfyUI is the usual one).
- **The engine died while loading**: `/tmp/maya-peer-server.log` and the tail of `maya-ds4-hip.log` say why. The tier
  never takes the run down with it - a dead peer only means the engine reads from its own disk - so a failure here is
  usually memory (the engine's own read-ahead) rather than the tier.

## After the run

`./peer-tier.sh down` restores the state described above; then start the host the normal way. If the session that was
driving the test is gone (it usually is - it lives on this box's `:8080`), nothing is lost: the numbers are in
`maya-ds4-hip.log` (`grep 'ds4 peer'`) and in the peer's log, and the tree on disk is unchanged - the tier is
started from the environment, never from the config.
