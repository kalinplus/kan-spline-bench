#!/usr/bin/env python
# B14 (Issue #25): cross-seed shape stability — do the B4 spline-shape findings hold across
# the 5 seeds of the B7 multi-seed run (kan-baseline_s42..s46, same [158,24,1]/g5/k3 arch as
# the B1 main KAN, CPU-trained)?
#
# Reuse strategy (zero edits to run_b4_explain.py — B13 is modifying its plotting in parallel):
# import run_b4_explain and monkey-patch its module constants OUT / KAN_DIR per seed, then call
# the ORIGINAL functions (extract_edges, run_shape_classification) so grid, thresholds and
# edge formula are literally the same code path as B4. Symbolic snap of the per-feature
# strongest edge reuses b4.fit_candidate / b4.CANDIDATES / b4.SNAP_* constants directly.
#
# Pre-registered verdict rule (stated before looking at results, recorded in the report):
#   list-level PASS  : min pairwise top-12 overlap >= 8 of 12
#   shape-level PASS : among features in >= 3/5 top-12 lists, share with a single
#                      unanimous shape label >= 0.80
#   both PASS -> "consistent: the model stably discovers these structures";
#   otherwise -> "not consistent: single-model post-hoc reading" (say which level failed).
#
# Outputs: common/runs/kan/seed-stability/
#   s<seed>/{edge_curves_all.npz, kan_edge_importance.csv, shape_classification.csv,
#            strongest_edge_snap.csv}   (per-seed, B4-identical format)
#   consistency_pairwise_top12.csv, consistency_labels.csv, consistency_snap_family.csv
#   overlay_strongest_edges.png, summary.json, run.log
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b4_explain as b4  # noqa: E402  (constants + functions reused, file never edited)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "common" / "runs" / "kan"
MS = RUNS / "multi-seed"
OUT = RUNS / "seed-stability"
SEEDS = [42, 43, 44, 45, 46]
TOP_PLOT = b4.TOP_PLOT                      # 12, same atlas size as B4
OVERLAY_FEATURES = ["RESI5", "RESI60", "MA5"]  # B4 top factors per Issue #25

# pre-registered verdict thresholds (see module docstring)
OVERLAP_MIN = 8          # of 12
LABEL_AGREE_MIN = 0.80

# Interpretation-level equivalence, both mappings stated in B4's own report
# (explain_report.md §2): "sigmoid ≡ tanh (affine equivalent family)" — exact;
# "sin is the basis function of flat-shouldered bumps, not a periodic signal" — sin / x^2 / x^3
# all snap flat-shouldered single bumps, an interpretive (not exact) family. Used ONLY as a
# secondary lens reported next to the pre-registered raw-label verdict, never to override it.
FAMILY_CANON = {"sigmoid": "sigmoid_family", "tanh": "sigmoid_family",
                "sin": "bump_family", "x^2": "bump_family", "x^3": "bump_family"}  # others: themselves


def seed_model_dir(seed: int) -> Path:
    """kan-baseline_s42 is a reuse run (no local model.pt) -> follow metrics.json."""
    d = MS / f"kan-baseline_s{seed}"
    if (d / "model.pt").exists():
        return d
    m = json.load(open(d / "metrics.json"))
    src = RUNS / m["reused_from"]           # e.g. sweep/configs/g5_k3_w24
    assert (src / "model.pt").exists(), f"reuse source missing: {src}"
    return src


def extract_seed(seed: int) -> dict:
    """B4's own extraction + shape classification, patched to this seed's weights/outputs,
    plus strongest-edge symbolic snap on the shared family/thresholds."""
    t0 = time.perf_counter()
    sd = OUT / f"s{seed}"
    sd.mkdir(parents=True, exist_ok=True)
    b4.KAN_DIR = seed_model_dir(seed)
    b4.OUT = sd
    xs, phi, imp, names, strongest = b4.extract_edges()
    shape_df = b4.run_shape_classification(xs, phi, imp, names, strongest)

    rows = []
    xg = xs.astype(np.float64)
    for i, name in enumerate(names):
        y = phi[:, i, strongest[i]].astype(np.float64)
        fits = {k: b4.fit_candidate(xg, y, f) for k, f in b4.CANDIDATES.items()}
        bk, bf = max(fits.items(), key=lambda kv: kv[1]["r2"])
        status = ("snapped" if bf["r2"] >= b4.SNAP_R2 else
                  "borderline" if bf["r2"] >= b4.SNAP_R2_BORDER else "not_snappeable")
        rows.append({"feature": name, "strongest_hidden": int(strongest[i]),
                     "edge_maxabs": float(np.abs(y).max()), "best_cand": bk,
                     "best_r2": bf["r2"], "a": bf["a"], "b": bf["b"], "c": bf["c"],
                     "d": bf["d"], "snap_status": status})
    snap_df = pd.DataFrame(rows)
    snap_df.to_csv(sd / "strongest_edge_snap.csv", index=False)

    top12 = list(pd.read_csv(sd / "kan_edge_importance.csv").head(TOP_PLOT)["feature"])
    print(f"[seed {seed}] done in {time.perf_counter()-t0:.0f}s "
          f"(weights: {b4.KAN_DIR.name})", flush=True)
    return {"seed": seed, "top12": top12, "shape": shape_df.set_index("feature"),
            "snap": snap_df.set_index("feature"), "phi": phi, "xs": xs,
            "strongest": strongest, "names": names}


def consistency(res: list) -> dict:
    # (a) pairwise top-12 overlap / Jaccard
    rows = []
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            a, b = set(res[i]["top12"]), set(res[j]["top12"])
            rows.append({"seed_a": res[i]["seed"], "seed_b": res[j]["seed"],
                         "overlap_of_12": len(a & b), "jaccard": round(len(a & b) / len(a | b), 3),
                         "top12_a": "|".join(res[i]["top12"]), "top12_b": "|".join(res[j]["top12"])})
    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(OUT / "consistency_pairwise_top12.csv", index=False)

    # (b) per-feature top-12 presence + shape-label agreement
    counts = pd.Series([f for r in res for f in r["top12"]]).value_counts()
    rows = []
    for f, c in counts.items():
        labels = [r["shape"].loc[f, "shape_label"] for r in res if f in r["top12"]]
        rows.append({"feature": f, "top12_count": int(c), "in_top12_seeds": "|".join(
            str(r["seed"]) for r in res if f in r["top12"]),
            "shape_labels": "|".join(labels), "n_distinct_labels": len(set(labels)),
            "unanimous_among_present": labels.count(labels[0]) == len(labels)})
    lab_df = pd.DataFrame(rows).sort_values(["top12_count", "feature"],
                                            ascending=[False, True])
    lab_df.to_csv(OUT / "consistency_labels.csv", index=False)

    # (c) strongest-edge snap family agreement, eligible = features in >= 3/5 top-12 lists
    eligible = counts[counts >= 3].index.tolist()
    rows = []
    for f in eligible:
        fams_top = [r["snap"].loc[f, "best_cand"] for r in res if f in r["top12"]]
        fams_all = [r["snap"].loc[f, "best_cand"] for r in res]   # snap exists for all 158
        canon = [FAMILY_CANON.get(x, x) for x in fams_all]
        plurality = pd.Series(canon).value_counts().iloc[0] / len(canon)
        rows.append({"feature": f, "top12_count": int(counts[f]),
                     "families_in_top12_seeds": "|".join(fams_top),
                     "families_all_5_seeds": "|".join(fams_all),
                     "n_distinct_families_all5": len(set(fams_all)),
                     "canonical_families": "|".join(canon),
                     "plurality_family_share": round(float(plurality), 2)})
    fam_df = pd.DataFrame(rows)
    fam_df.to_csv(OUT / "consistency_snap_family.csv", index=False)

    stats = {
        "min_pairwise_overlap_of_12": int(pair_df["overlap_of_12"].min()),
        "mean_pairwise_jaccard": float(pair_df["jaccard"].mean()),
        "eligible_features_ge3_of_5": len(eligible),
        "eligible_features": eligible,
        "label_unanimous_rate_eligible": float(
            lab_df.set_index("feature").loc[eligible, "unanimous_among_present"].mean()),
        "snap_family_unanimous_rate_all5": float((fam_df["n_distinct_families_all5"] == 1).mean()),
        "canon_plurality_share_mean": float(fam_df["plurality_family_share"].mean()),
        "canon_plurality_share_min": float(fam_df["plurality_family_share"].min()),
        "curve_corr": pairwise_curve_corr(res),
        "list_pass": bool(pair_df["overlap_of_12"].min() >= OVERLAP_MIN),
        "shape_pass": bool(lab_df.set_index("feature").loc[eligible, "unanimous_among_present"]
                           .mean() >= LABEL_AGREE_MIN),
    }
    stats["verdict"] = ("consistent" if stats["list_pass"] and stats["shape_pass"]
                        else "not_consistent")
    return stats, pair_df, lab_df, fam_df


def overlay_plot(res: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(OVERLAY_FEATURES), figsize=(5 * len(OVERLAY_FEATURES), 4))
    colors = {42: "tab:blue", 43: "tab:orange", 44: "tab:green", 45: "tab:red", 46: "tab:purple"}
    for ax, feat in zip(np.atleast_1d(axes), OVERLAY_FEATURES):
        for r in res:
            i = r["names"].index(feat)
            y = r["phi"][:, i, r["strongest"][i]]
            ax.plot(r["xs"], y, color=colors[r["seed"]], lw=1.4, label=f"seed {r['seed']}")
        ax.axhline(0, color="gray", lw=0.4, ls=":")
        ax.set_title(f"{feat}: strongest edge across seeds", fontsize=10)
        ax.set_xlabel("normalized feature (clip ±3)", fontsize=8)
        ax.set_ylabel("activation", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    fig.suptitle("B14: first-layer strongest edge per seed (kan-baseline s42–s46, CPU path)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "overlay_strongest_edges.png", dpi=160)
    plt.close(fig)
    print(f"[overlay] -> {OUT}/overlay_strongest_edges.png", flush=True)


def pairwise_curve_corr(res: list) -> dict:
    """Eligible feature -> Pearson corr of strongest-edge curves for each seed pair.
    Direct continuity check: do different seeds draw the SAME curve, whatever we call it?"""
    counts = pd.Series([f for r in res for f in r["top12"]]).value_counts()
    eligible = counts[counts >= 3].index.tolist()
    out = {}
    for f in eligible:
        curves = {r["seed"]: r["phi"][:, r["names"].index(f), r["strongest"][r["names"].index(f)]]
                  for r in res}
        cs = []
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                c = float(np.corrcoef(curves[res[i]["seed"]], curves[res[j]["seed"]])[0, 1])
                cs.append(c)
        out[f] = {"min": round(min(cs), 3), "mean": round(float(np.mean(cs)), 3),
                  "max": round(max(cs), 3),
                  "abs_min": round(min(abs(c) for c in cs), 3),
                  "abs_mean": round(float(np.mean(np.abs(cs))), 3),
                  "mirror_pairs_of_10": int(sum(1 for c in cs if c < -0.95))}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = [extract_seed(s) for s in SEEDS]
    stats, pair_df, lab_df, fam_df = consistency(res)
    overlay_plot(res)
    summary = {
        "seeds": SEEDS,
        "arch": "kan-baseline [158,24,1] grid5 order3 (B1-arch CPU retrain, B7 run)",
        "b4_top12_reference": list(pd.read_csv(RUNS / "explain" / "kan_edge_importance.csv")
                                   .head(TOP_PLOT)["feature"]),
        "per_seed_top12": {str(r["seed"]): r["top12"] for r in res},
        "pre_registered_rule": {"min_pairwise_overlap_of_12": OVERLAP_MIN,
                                "label_unanimous_rate_min": LABEL_AGREE_MIN},
        **stats,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n[verdict]", json.dumps({k: summary[k] for k in
          ("min_pairwise_overlap_of_12", "label_unanimous_rate_eligible",
           "snap_family_unanimous_rate_all5", "list_pass", "shape_pass", "verdict")}, indent=2))
    print("run_b14_seed_shapes DONE", flush=True)


if __name__ == "__main__":
    main()
