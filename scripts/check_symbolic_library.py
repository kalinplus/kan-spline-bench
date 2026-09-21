#!/usr/bin/env python
# Library-sensitivity check for B4's symbolic snap: does widening the candidate function
# library (adding exp / cos / arctan / gaussian, i.e. families present in pykan's SYMBOLIC_LIB
# but absent from ours) change the snapping outcome on the 158 strongest edges?
#
# Read-only over B4 artifacts: common/runs/kan/explain/{edge_curves_all.npz,kan_edge_importance.csv}.
# Reuses run_b4_explain.fit_candidate / CANDIDATES verbatim, so the base-library numbers here are
# the same code path that produced results/B4_explain.md.
#
# Output: common/runs/kan/explain/library_compare.csv (per edge: base best / extended best / gain).
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_b4_explain import CANDIDATES, fit_candidate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPLAIN = ROOT / "common" / "runs" / "kan" / "explain"

EXTRA = {
    "exp": lambda t: np.exp(t),
    "cos": lambda t: np.cos(t),
    "arctan": lambda t: np.arctan(t),
    "gaussian": lambda t: np.exp(-t * t),
}


def main():
    npz = np.load(EXPLAIN / "edge_curves_all.npz", allow_pickle=True)
    xs, phi, names = npz["xs"].astype(np.float64), npz["phi"], list(npz["feat_names"])
    imp = pd.read_csv(EXPLAIN / "kan_edge_importance.csv")
    strongest = dict(zip(imp["feature"], imp["strongest_hidden"]))
    lib = {**CANDIDATES, **EXTRA}

    rows = []
    for f in imp["feature"]:  # 158 strongest edges, one per feature
        i, j = names.index(f), int(strongest[f])
        y = phi[:, i, j].astype(np.float64)
        base = {k: fit_candidate(xs, y, fn)["r2"] for k, fn in CANDIDATES.items()}
        ext = {k: fit_candidate(xs, y, fn)["r2"] for k, fn in lib.items()}
        bb, be = max(base, key=base.get), max(ext, key=ext.get)
        rows.append({"feature": f, "base_best": bb, "base_r2": base[bb],
                     "ext_best": be, "ext_r2": ext[be], "gain": ext[be] - base[bb]})
        print(f"[lib] {f:8s} {bb:20s} {base[bb]:.5f} -> {be:12s} {ext[be]:.5f}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(EXPLAIN / "library_compare.csv", index=False)
    flip = d[d.ext_best != d.base_best]
    print(f"\n[lib] base R2 mean {d.base_r2.mean():.5f} min {d.base_r2.min():.4f} | "
          f"extended mean {d.ext_r2.mean():.5f} max gain {d.gain.max():.4f}")
    print(f"[lib] gain > 1e-3: {(d.gain > 1e-3).sum()} | gain > 1e-4: {(d.gain > 1e-4).sum()}")
    print(f"[lib] best-family label changed: {len(flip)}/{len(d)}")
    print(f"[lib] of those, gain < 1e-6 (numerical tie): {(flip.gain < 1e-6).sum()}")
    print(f"[lib] extended best-family distribution: {d.ext_best.value_counts().to_dict()}")
    print(f"-> {EXPLAIN / 'library_compare.csv'}")


if __name__ == "__main__":
    main()
