#!/usr/bin/env bash
# One fresh CARLA server per town (avoids map-switch memory peaks), capture, stop server. Resumable.
# Usage: carla_capture_all.sh <CARLA_DIR> <OUT> <positions-per-town> <frames-per-position> <towns...>
set -uo pipefail
cd "$(dirname "$0")/.."
CARLA=$1; OUT=$2; NP=$3; NF=$4; shift 4
for town in "$@"; do
  ./scripts/carla_server.sh "$CARLA" "$town" > "logs/carla_server_$town.log" 2>&1 &
  srv=$!
  for i in $(seq 1 60); do
    .venv/bin/python -c "import carla;c=carla.Client('localhost',2000);c.set_timeout(5);c.get_world()" 2>/dev/null && break
    sleep 5
  done
  .venv/bin/python scripts/carla_capture.py --out "$OUT" --towns "$town" --positions-per-town "$NP" --frames-per-position "$NF"
  echo "$town capture rc=$?"
  # match the server by exact (15-char truncated) process name, never by command line
  pkill -TERM -P $srv 2>/dev/null; kill $srv 2>/dev/null; pkill -x "CarlaUE4-Linux-" 2>/dev/null
  sleep 10
done
