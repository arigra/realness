RadSimReal v0.2 — text-only source edition

All Python source, tests, configurations, bundled text OBJ geometry, licenses,
source locks and documentation are included. These source files are restored
from the verified v2 ZIP with LF line endings; code content is unchanged.
The restorer also accepts the bundle saved with CRLF line endings by an editor.
The following small execution records
are also preserved: native_pilots.json, unet_test_metrics.json,
unet_training/history.json and ARCHIVE_VALIDATION.json under examples/v2/.

Binary sample datasets (.npz), figures (.png), and trained checkpoints (.pt)
are intentionally not part of this text edition. README commands that load the
included binary examples require those artifacts; instead start by generating
your own outputs with the commands below. A 400-update historical run report
does not mean its checkpoint is distributed in this edition. No user datasets
are distributed in either edition.

From this restored directory, use an appropriate Python environment:

python -m pip install -e ".[test]"
python -m pytest -q -p no:cacheprovider --basetemp work/pytest
python scripts/mesh_demo.py --out runs/mesh_demo
python -m radsimreal native generate --config configs/cartesian_demo.yaml --vehicle-assets configs/assets/vehicle_assets.json --out runs/cartesian_data --train 16 --val 4 --test 4 --max-samples 2000
python -m radsimreal bev-train --model unet --data runs/cartesian_data --out runs/unet --epochs 2 --max-steps 4 --base-channels 8 --batch-size 4
python -m radsimreal bev-evaluate --data runs/cartesian_data --checkpoint runs/unet/best.pt --out runs/unet/test_metrics.json

The cluster agent should then read docs/HPC_GUIDE.md and use the actual locally
mounted RADDet/CARRADA recordings. Tests and supplied pilot reports use synthetic
fixtures; they do not establish real-data accuracy or sensor calibration.
