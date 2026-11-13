#!/usr/bin/env bash
# Start (or resume) every benchmark job that is not already running. Safe to rerun any time.
# Run from your own terminal so jobs outlive the Claude session:
#   nohup /truenas/home/arigra/permuter/ariGranevich/realness/benchmarks/run_all.sh > /dev/null 2>&1 &
B=/truenas/home/arigra/permuter/ariGranevich/realness/benchmarks
cd $B
stamp=$(date +%m%d_%H%M)
start() {  # name, pattern, command
    if [ -f runs/$1.done ]; then return; fi                      # finished cleanly earlier
    if pgrep -f "$2" > /dev/null; then echo "$1 already running"; return; fi
    [ -f logs/$1.log ] && mv logs/$1.log logs/$1_$stamp.log
    echo "starting $1"; nohup bash -c "$3" > logs/$1.log 2>&1 < /dev/null &
}
start radial "1-Train.py" "./run_radial.sh && touch runs/radial.done"
start carrada "mvrss/train.py" "./run_carrada.sh && touch runs/carrada.done"
if grep -q "^done" logs/raddet_cache.log 2>/dev/null; then
    start raddet "\.venv_tf/bin/python train.py" "./run_raddet.sh && touch runs/raddet.done"
else
    start raddet_cache "cache_raddet_logmag.py" "python3 scripts/cache_raddet_logmag.py --workers 6 && ./run_raddet.sh > logs/raddet.log 2>&1 && touch runs/raddet.done"
fi
[ "${WAIT:-0}" = 1 ] && wait   # keep the launcher alive (needed when started from a Claude background task)
