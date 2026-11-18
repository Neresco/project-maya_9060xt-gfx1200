#!/bin/sh
# Two machines, one model: this machine runs the engine with the peer expert tier on, the other
# machine's strata-peer serves the expert rows this one misses over RoCE.  Runs on its own:
#
#   ./peer-tier.sh up       1) stop what holds the cards  2) bring the peer up  3) start the engine
#   ./peer-tier.sh status   where both halves stand, and what the two logs' peer lines say
#   ./peer-tier.sh down     stop the engine and the peer (the cards go back to nobody)
#
# Overrides (environment, all optional - the defaults are this pair of machines):
#   PEER_HOST=192.168.100.20      the machine that serves (passwordless ssh)
#   PEER_USER=lunarbuntu
#   PEER_PORT=9447                the daemon's control port
#   PEER_PACK=.../Maya-S          the copy of the pack THE PEER holds
#   PEER_RAM_GB=80                the peer's pinned slab
#   PEER_VRAM="0:14"              the peer's GPU 0, 14 GB of it, held in VRAM and read straight out of it
#                                 (opt-in, a comma list of GPU:GB; empty: RAM only).  GPU numbers are HIP's on
#                                 the peer (`rocm-smi` / `rocminfo` order).  On the strix box GPU 0 is the
#                                 7600 XT's 16 GiB of dedicated memory - the one pool the peer does not
#                                 already have - while GPU 1 is the APU, whose "VRAM" is the same system RAM
#                                 (GTT), so a slab there adds no memory.  Empty keeps that card for image
#                                 generation; a slab the card cannot take falls back to RAM-only (see below).
#   PEER_SHARDS=""                the pack files the peer holds, as the pack numbers them (empty: all)
#   PEER_COMPUTE=""               the compute worker's .so the daemon loads (e.g. strata-peer-compute-hip.so
#                                 from build-hip), copied to the peer as ~/strata-peer-compute.so.  Empty:
#                                 storage only.  A worker on the APU needs the slab there too (PEER_VRAM=1:N),
#                                 because a resident row means something on the card that holds it alone.
#   PEER_COMPUTE_GPU=""           the peer card(s) the workers run on (a comma list: "1", or "1,0" for one worker
#                                 each, --compute-options gpu:N a worker).  Empty: one worker, the machine's first card.
#                                 A card's worker serves that card's slab's rows; the RAM slab's rows go to the
#                                 workers in turn.  A worker that will not open is dropped (the last one first) and
#                                 the run goes on with the rest - said on the spot.
#   PEER_COMPUTE_ENGINE=""        "1": the ENGINE takes the compute tier too (MAYA_DS4_PEER_COMPUTE=1) - the peer
#                                 sums the experts it holds, not only serves them (docs/PEER_COMPUTE.md).  Empty:
#                                 storage only, with the worker loaded beside it - the A/B's other arm.
#   CONFIG=$ROOT/maya-ds4-hip.json, PORT=8080, ENGINE_GPUS=0,1,2,3
set -u

ROOT=/home/lunarbuntu/Kobold/project-maya_9060xt-gfx1200
PEER_HOST=${PEER_HOST:-192.168.100.20}
PEER_USER=${PEER_USER:-lunarbuntu}
PEER_PORT=${PEER_PORT:-9447}
PEER_PACK=${PEER_PACK:-/home/lunarbuntu/Maya-data/models/DeepSeek-V4.1-Flash-1017-Maya-S/Maya-S}
PEER_RAM_GB=${PEER_RAM_GB:-80}
PEER_VRAM=${PEER_VRAM:-0:14}
PEER_SHARDS=${PEER_SHARDS:-}
PEER_COMPUTE=${PEER_COMPUTE:-}
PEER_COMPUTE_GPU=${PEER_COMPUTE_GPU:-}
PEER_COMPUTE_ENGINE=${PEER_COMPUTE_ENGINE:-}
CONFIG=${CONFIG:-$ROOT/maya-ds4-hip.json}
PORT=${PORT:-8080}
ENGINE_GPUS=${ENGINE_GPUS:-0,1,2,3}

PEER_LOG=/tmp/strata-peer-real.log
SERVER_LOG=/tmp/maya-peer-server.log
ENGINE_LOG=$ROOT/maya-ds4-hip.log
DAEMON=$ROOT/build-hip/strata-peer
PY=$ROOT/.venv/bin/python

say() { printf '[peer-tier] %s\n' "$*"; }
die() { printf '[peer-tier] %s\n' "$*" >&2; exit 1; }

peer_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=6 "$PEER_USER@$PEER_HOST" "$@"; }

peer_ok() {
    peer_ssh true 2>/dev/null || die "the peer $PEER_USER@$PEER_HOST does not answer ssh (passwordless key, direct link up?)"
    peer_ssh 'command -v ibv_devinfo >/dev/null' 2>/dev/null || die "the peer has no ibv_devinfo (rdma-core missing on $PEER_HOST?)"
    peer_ssh "test -d $PEER_PACK" 2>/dev/null || die "the peer does not have the pack at $PEER_PACK (set PEER_PACK)"
}

# the cards' used memory in MiB, one number (the four of them summed)
vram_mib() {
    rocm-smi --showmeminfo vram 2>/dev/null | awk -F: '
        /VRAM Total Used Memory/ { used = $3; gsub(/ /, "", used); s += used } END { printf "%d", s / 1048576 }' || printf '0'
    }

# ------------------------------------------------------------------ 1. stop what holds the cards
engine_pids() { pgrep -f "$ROOT/build-hip/maya-ds4 --serve"; pgrep -f "$ROOT/serve/server.py"; }

# the peer the RUNNING engine was started against, "host:port" (empty: the engine on :$PORT is not this
# tier's - a launcher of its own, maya.py or a dashboard, can take the port, and then the daemon waits for
# a session nobody opens while the tier looks up)
engine_peer() {
    for p in $(pgrep -x maya-ds4 2>/dev/null); do
        h=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^MAYA_DS4_PEER=//p' | head -1)
        [ -n "$h" ] || continue
        pt=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^MAYA_DS4_PEER_PORT=//p' | head -1)
        printf '%s:%s' "$h" "${pt:-9447}"
        return 0
    done
    return 1
}

stop_engine() {
    pids=$(engine_pids)
    if [ -z "$pids" ]; then say "no engine was running"; else
        say "stopping the engine (pids $(echo $pids))"
        say "NOTE: this is whatever serves the API on :$PORT - a dashboard or chat in flight loses its model with it"
        for p in $pids; do kill "$p" 2>/dev/null; done
        i=0
        while [ $i -lt 60 ]; do
            [ -z "$(engine_pids)" ] && break
            i=$((i + 1)); sleep 1
        done
        # only what is still alive gets the hard kill (a reused pid must not)
        for p in $(engine_pids); do kill -9 "$p" 2>/dev/null; done
    fi
    say "waiting for the cards to come free (now $(vram_mib) MiB used)"
    i=0
    while [ $i -lt 90 ]; do
        now=$(vram_mib)
        [ "$now" -lt 4096 ] && break
        i=$((i + 1)); sleep 2
    done
    now=$(vram_mib)
    [ "$now" -lt 4096 ] || die "the cards still hold $now MiB (four cards, 17.1 GB each) - something else runs on them (rocm-smi, pgrep -a maya-ds4)"
    say "the cards are free ($now MiB used)"
}

# ------------------------------------------------------------------------- 2. the peer, up first
peer_up() {
    peer_ok
    [ -x "$DAEMON" ] || die "$DAEMON is not built (cd $ROOT/build-hip && make strata-peer)"
    say "the peer $PEER_HOST: $PEER_PACK"
    # a stale daemon of another build answers the port wrongly, and the engine then says 'no welcome'
    local_md5=$(md5sum < "$DAEMON" | awk '{print $1}')
    remote_md5=$(peer_ssh "md5sum ~/strata-peer 2>/dev/null" | awk '{print $1}')
    if [ "$local_md5" != "$remote_md5" ]; then
        say "copying strata-peer to $PEER_HOST (~/strata-peer)"
        scp -q "$DAEMON" "$PEER_USER@$PEER_HOST:~/strata-peer" || die "scp to $PEER_HOST failed"
    fi
    peer_ssh 'pkill -x strata-peer' 2>/dev/null
    i=0
    while [ $i -lt 20 ] && peer_ssh 'pgrep -x strata-peer >/dev/null' 2>/dev/null; do i=$((i + 1)); sleep 0.5; done
    peer_ssh 'pkill -9 -x strata-peer' 2>/dev/null

    vram_opt=""
    [ -n "$PEER_VRAM" ] && vram_opt=$(printf '%s' "--vram $(printf '%s' "$PEER_VRAM" | sed 's/,/ --vram /g')")
    # the compute workers travel the way the daemon does - a .so copied to the peer's home and loaded by path, so the
    # peer's own build is never needed and a machine with no ROCm still serves storage when this is left empty.
    # ONE WORKER A CARD: PEER_COMPUTE_GPU takes a comma list ("1,0") and each card's worker goes to the daemon as its
    # own --compute / --compute-options pair - a resident row is an address of the slab on its own card, so a slab's
    # rows are the business of the worker on that card, while the RAM slab's rows go to the workers in turn
    cpairs=""   # (a line a pair: a pair the peer cannot open is dropped off the end, so a line is what drops)
    if [ -n "$PEER_COMPUTE" ]; then
        [ -f "$ROOT/build-hip/$PEER_COMPUTE" ] || die "$ROOT/build-hip/$PEER_COMPUTE is not built (make ${PEER_COMPUTE%.so})"
        scp -q "$ROOT/build-hip/$PEER_COMPUTE" "$PEER_USER@$PEER_HOST:~/strata-peer-compute.so" || \
            die "scp of the compute worker to $PEER_HOST failed"
        if [ -n "$PEER_COMPUTE_GPU" ]; then
            for g in $(printf '%s' "$PEER_COMPUTE_GPU" | sed 's/,/ /g'); do
                cpairs="$cpairs--compute ~/strata-peer-compute.so --compute-options gpu:$g
"
                # (a worker on a card with no slab there still computes the RAM slab's rows: said, not refused)
                printf '%s' "$PEER_VRAM" | sed 's/,/\n/g' | cut -d: -f1 | grep -qx -- "$g" || \
                    say "note: no VRAM slab on GPU $g - that card's worker computes the RAM slab's rows only"
            done
        else
            cpairs="--compute ~/strata-peer-compute.so
"
        fi
    fi
    npairs=0
    [ -n "$cpairs" ] && npairs=$(printf '%s' "$cpairs" | grep -c .)
    pairs_head() {   # ($1 pairs to keep) - the first $1 lines of cpairs
        printf '%s' "$cpairs" | head -n "$1"
    }
    # one start: the listener up is the proof (a --vram slab the card cannot hand out and a worker that cannot open
    # are both start-up errors by design - so each is said on the spot, not at the third layer of a decode)
    try_daemon() {   # ($1 vram flags, $2 compute flags)
        peer_ssh "setsid nohup ~/strata-peer --pack $PEER_PACK --bind 0.0.0.0 --port $PEER_PORT --ram-gb $PEER_RAM_GB $1 $2 >$PEER_LOG 2>&1 </dev/null &"
        sleep 0.5
        local_j=0
        while [ $local_j -lt 40 ]; do
            peer_ssh "grep -q 'listening on' $PEER_LOG" 2>/dev/null && return 0
            local_j=$((local_j + 1)); sleep 0.5
        done
        return 1
    }
    say "starting the daemon (RAM slab $PEER_RAM_GB GB${PEER_VRAM:+, VRAM $PEER_VRAM}${PEER_COMPUTE:+, compute $PEER_COMPUTE${PEER_COMPUTE_GPU:+ on GPU $PEER_COMPUTE_GPU}})"
    vopt="$vram_opt"
    nkeep="$npairs"
    while :; do
        copt=""
        # (the pairs are kept a line each - that is how one drops off the end - but the daemon gets ONE command line:
        #  an unjoined pair is read by the peer's shell as a command of its own ("--compute: command not found"), and
        #  the daemon then comes up with the workers it happened to parse, saying nothing)
        [ "$nkeep" -gt 0 ] && copt=$(pairs_head "$nkeep" | tr '\n' ' ')
        if try_daemon "$vopt" "$copt"; then break; fi
        peer_ssh "tail -20 $PEER_LOG" 2>/dev/null
        # WHAT THE LOG NAMES IS WHAT GOES, and only that: a card's worker that would not open takes one pair off the
        # list (the last one first - a busy card is the usual cause), a slab the peer cannot take takes the slabs, and
        # the RAM tier alone is still worth having.  The run goes on with what is left; the reason is said here.
        if peer_ssh "grep -qE 'compute worker|strata-peer-compute' $PEER_LOG" 2>/dev/null && [ "$nkeep" -gt 0 ]; then
            nkeep=$((nkeep - 1))
            say "a compute worker would not start on $PEER_HOST - the run goes on with $nkeep of $npairs"
            peer_ssh 'pkill -x strata-peer' 2>/dev/null
            continue
        fi
        if [ -n "$vopt" ]; then
            vopt=""
            say "the VRAM slab could not be taken on $PEER_HOST - the peer comes up on its RAM slab only"
            peer_ssh 'pkill -x strata-peer' 2>/dev/null
            continue
        fi
        if [ "$nkeep" -gt 0 ]; then
            nkeep=$((nkeep - 1))
            say "a compute worker would not start on $PEER_HOST - the run goes on with $nkeep of $npairs"
            peer_ssh 'pkill -x strata-peer' 2>/dev/null
            continue
        fi
        die "the daemon did not come up - its log is on the peer at $PEER_LOG"
    done
    peer_ssh "grep -E 'device .* up|slab budget|VRAM' $PEER_LOG" 2>/dev/null
    peer_ssh "grep -q 'listening on' $PEER_LOG" 2>/dev/null || die "no listener on $PEER_HOST:$PEER_PORT"
    say "the daemon listens on $PEER_HOST:$PEER_PORT"
}

# ---------------------------------------------------------------------- 3. the engine, connecting
engine_up() {
    [ -x "$ROOT/build-hip/maya-ds4" ] || die "$ROOT/build-hip/maya-ds4 is not built (make maya-ds4)"
    [ -f "$CONFIG" ] || die "$CONFIG is not there (CONFIG=...)"
    [ -x "$PY" ] || die "$PY is not there (a setup without .venv?)"
    # the engine reads the tier's address from the environment (the config's args stay untouched: a
    # hand-run engine and this script start the same way, and 'down' leaves nothing behind to undo)
    MAYA_DS4_PEER=$PEER_HOST
    MAYA_DS4_PEER_PORT=$PEER_PORT
    [ -n "$PEER_SHARDS" ] && MAYA_DS4_PEER_SHARDS=$PEER_SHARDS
    say "starting the engine with the tier on ($PEER_HOST:$PEER_PORT${PEER_SHARDS:+, shards $PEER_SHARDS}${PEER_COMPUTE_ENGINE:+, compute})"
    setsid nohup env MAYA_DS4_PEER="$MAYA_DS4_PEER" MAYA_DS4_PEER_PORT="$MAYA_DS4_PEER_PORT" \
        ${PEER_SHARDS:+MAYA_DS4_PEER_SHARDS="$PEER_SHARDS"} \
        ${PEER_COMPUTE_ENGINE:+MAYA_DS4_PEER_COMPUTE="$PEER_COMPUTE_ENGINE"} \
        "$PY" "$ROOT/serve/server.py" --engine maya --config "$CONFIG" --port "$PORT" --host 0.0.0.0 \
        --gpu "$ENGINE_GPUS" >"$SERVER_LOG" 2>&1 </dev/null &
    say "the engine is loading (its log: $ENGINE_LOG, the server's: $SERVER_LOG)"
    i=0
    while [ $i -lt 600 ]; do
        if ss -lnt 2>/dev/null | grep -q ":$PORT "; then break; fi
        if ! pgrep -f "$ROOT/serve/server.py" >/dev/null; then
            tail -30 "$SERVER_LOG" 2>/dev/null
            die "the server exited while the engine loaded - $SERVER_LOG says why"
        fi
        i=$((i + 1)); sleep 2
    done
    ss -lnt 2>/dev/null | grep -q ":$PORT " || die "nothing listens on :$PORT after 20 minutes - $SERVER_LOG"
    say "the API is up on http://127.0.0.1:$PORT (the model may still be loading: watch $ENGINE_LOG)"
    # a listening port is not proof that THIS tier's engine owns it: check the running engine's own environment
    t=$(engine_peer)
    if [ "$t" != "$PEER_HOST:$PEER_PORT" ]; then
        tail -20 "$SERVER_LOG" 2>/dev/null
        die "the engine on :$PORT is not this tier's (it was started against '${t:-nothing}', this script wanted $PEER_HOST:$PEER_PORT) - something else owns the port; stop it and run 'up' again"
    fi
    # the proof the two halves met: the daemon logs every session it accepts
    i=0
    while [ $i -lt 300 ]; do
        if peer_ssh "grep -q 'session up' $PEER_LOG" 2>/dev/null; then
            peer_ssh "grep 'session up' $PEER_LOG" 2>/dev/null | tail -1
            say "the engine's session is on the wire"
            return 0
        fi
        i=$((i + 1)); sleep 2
    done
    say "the API is up but the peer session has not come up yet - the engine's log says why (grep 'ds4 peer' $ENGINE_LOG)"
    return 0
}

# -------------------------------------------------------------------------------- what is running
status() {
    printf 'engine:  '
    p=$(pgrep -f "$ROOT/build-hip/maya-ds4 --serve" | head -1)
    [ -n "$p" ] && printf 'maya-ds4 pid %s\n' "$p" || printf 'not running\n'
    printf 'server:  '
    p=$(pgrep -f "$ROOT/serve/server.py" | head -1)
    [ -n "$p" ] && printf 'serve/server.py pid %s, API :%s\n' "$p" "$PORT" || printf 'not running\n'
    printf 'tier:    '
    t=$(engine_peer)
    if [ -n "$t" ]; then printf 'on, the engine reads its rows from %s\n' "$t"
    else printf 'OFF (the running engine has no MAYA_DS4_PEER)\n'; fi
    printf 'cards:   %s MiB used (four cards, 17.1 GB each)\n' "$(vram_mib)"
    if peer_ssh 'true' 2>/dev/null; then
        printf 'peer:    '
        p=$(peer_ssh 'pgrep -x strata-peer' 2>/dev/null | head -1)
        if [ -n "$p" ]; then
            printf 'strata-peer pid %s on %s:%s\n' "$p" "$PEER_HOST" "$PEER_PORT"
            peer_ssh "grep -E 'slab budget|VRAM' $PEER_LOG" 2>/dev/null | tail -2 | sed 's/^/         /'
        else printf 'not running on %s\n' "$PEER_HOST"; fi
    else printf 'peer:    %s does not answer ssh\n' "$PEER_HOST"; fi
    printf 'engine log, peer lines:\n'
    grep 'ds4 peer' "$ENGINE_LOG" 2>/dev/null | tail -5 | sed 's/^/         /' || true
    printf 'peer log, sessions:\n'
    peer_ssh "grep -E 'session up|hint reply|served' $PEER_LOG" 2>/dev/null | tail -5 | sed 's/^/         /' || true
}

# ------------------------------------------------------------------------------------ down again
down() {
    stop_engine
    if peer_ssh 'true' 2>/dev/null; then
        say "stopping the peer's daemon"
        peer_ssh 'pkill -x strata-peer' 2>/dev/null
    fi
    say "done"
}

case "${1:-up}" in
    up)     stop_engine; peer_up; engine_up ;;
    status) status ;;
    down)   down ;;
    *)      die "usage: $0 [up|status|down]" ;;
esac
