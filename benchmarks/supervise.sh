#!/usr/bin/env bash
# Keeps the benchmark jobs alive: every 5 min, run_all.sh (re)starts any job that is neither
# running nor finished (runs/<name>.done). Stop it with:  touch benchmarks/STOP
B=/truenas/home/arigra/permuter/ariGranevich/realness/benchmarks
cd $B
while [ ! -f STOP ]; do
    echo "$(date '+%m-%d %H:%M') $(./run_all.sh | tr '\n' ' ')" >> logs/supervisor.log
    sleep 300
done
echo "$(date '+%m-%d %H:%M') STOP file found, supervisor exiting (jobs keep running)" >> logs/supervisor.log
