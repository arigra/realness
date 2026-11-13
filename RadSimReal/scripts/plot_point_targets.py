"""Overlay normalized range/azimuth/Doppler cuts of chosen stationary candidates (coherent frame mean)."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from radsimreal.native.workflows import reader_from_prepared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--candidates", required=True)
    ap.add_argument("--pick", nargs="+", required=True, help="scene:range:azimuth")
    ap.add_argument("--out", required=True); ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--half", nargs=3, type=int, default=[8, 6, 48])
    a = ap.parse_args()
    reader, manifest = reader_from_prepared(a.data)
    cand = json.loads(Path(a.candidates).read_text())
    zb = cand["zero_doppler_canonical_bin"]
    byid = {r.id: r for r in reader.records}
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    summary = {}
    for pick in a.pick:
        scene, r, az = pick.split(":"); r, az = int(r), int(az)
        ids = cand["scenes"][scene]["frames_used"][: a.frames]
        cubes = []
        for i in ids:
            c = np.asarray(reader.load_tensor(byid[i]))
            if not np.iscomplexobj(c):
                c = np.sqrt(c)
            cubes.append(c)
        st = np.stack(cubes)
        if np.iscomplexobj(st):  # align each capture to the peak phase before averaging
            st = st * np.exp(-1j * np.angle(st[:, r, zb, az]))[:, None, None, None]
        m = np.abs(st.mean(0)) ** 2
        pk = m[r, zb, az]
        hr, hd, ha = a.half
        rr = np.arange(-hr, hr + 1); dd = np.arange(-hd, hd + 1); aa = np.arange(-ha, ha + 1)
        cut_r = m[np.clip(r + rr, 0, m.shape[0] - 1), zb, az] / pk
        cut_d = m[r, np.mod(zb + dd, m.shape[1]), az] / pk
        cut_a = m[r, zb, np.clip(az + aa, 0, m.shape[2] - 1)] / pk
        for k, (x, y, name) in enumerate([(rr, cut_r, "range bins"), (dd, cut_d, "Doppler bins"), (aa, cut_a, "azimuth bins")]):
            ax[0, k].plot(x, 10 * np.log10(np.maximum(y, 1e-9)), label=pick); ax[0, k].set(xlabel=name, ylabel="dB re peak", ylim=(-60, 3))
        img = 10 * np.log10(np.maximum(m[:, zb, :] / pk, 1e-9))
        k = len(summary)
        if k < 3:
            ax[1, k].imshow(img[max(r - 30, 0): r + 31, max(az - 70, 0): az + 71], origin="lower", vmin=-50, vmax=0, cmap="magma", aspect="auto")
            ax[1, k].set_title(f"{pick} zero-Doppler, ±30 r, ±70 az")
        crop = m[max(r - hr, 0): r + hr + 1, :, max(az - ha, 0): az + ha + 1]
        summary[pick] = {"frames": len(ids), "peak_power": float(pk),
                         "crop_energy_in_zero_doppler_plane": float(crop[:, zb, :].sum() / crop.sum())}
    for k in range(3):
        ax[0, k].legend(fontsize=7); ax[0, k].grid(alpha=.3)
    fig.savefig(a.out, dpi=110)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
