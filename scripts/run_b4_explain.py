#!/usr/bin/env python
# B4 (Issue #4): post-hoc interpretability of the B1 main KAN — three KAN-specific pieces of
# evidence: (1) first-layer spline curve atlas for the top edge-norm features, each annotated
# with a shape adjudication (monotone inc/dec, U-shaped, saturating, irregular); (2) symbolic
# snap of every first-layer edge onto a candidate function family (x, x^2, x^3, |x|,
# sgn(x)*log(1+|x|), sin, sigmoid, tanh) with affine wrap a*f(b*x+c)+d, fit error and R2,
# non-snappeable edges explicitly marked; (3) KAN edge-norm importance vs LightGBM gain
# importance cross-check (S1b retrained once from the cached parquet, official hyperparams,
# seed 42 — the KAN itself is never retrained). Outputs to common/runs/kan/explain/.
#
# Symbolic route note: pykan is NOT installed and intentionally not used — its suggest_symbolic
# operates on pykan's own KAN class, while the B1 checkpoint is efficient_kan.KAN (incompatible
# state_dict layout); installing pykan also risks upgrading the shared env torch. We therefore
# use the manual family fit (affine-wrapped least squares + curve_fit refinement), which the
# issue explicitly allows.
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit
from scipy.special import expit
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from efficient_kan import KAN

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "common" / "runs" / "kan" / "_cache"
KAN_DIR = ROOT / "common" / "runs" / "kan" / "b1_kan"
OUT = ROOT / "common" / "runs" / "kan" / "explain"
KAN_HIDDEN = 24
KAN_GRID_RANGE = (-3.0, 3.0)
N_GRID = 401
TOP_PLOT = 12          # atlas covers top-12 features (acceptance asks for >= 10)
TOPN = 20              # importance comparison table size
SEED = 42
SNAP_R2 = 0.95         # R2 above this -> snapped
SNAP_R2_BORDER = 0.90  # between this and SNAP_R2 -> borderline
NEGLIGIBLE = 1e-3      # edges with max|phi| below this are "negligible" (dead) edges

# candidate symbolic family: name -> f(t); the fitted form is a*f(b*x+c)+d
CANDIDATES = {
    "x": lambda t: t,
    "x^2": lambda t: t * t,
    "x^3": lambda t: t * t * t,
    "|x|": lambda t: np.abs(t),
    "sgn(x)*log(1+|x|)": lambda t: np.sign(t) * np.log1p(np.abs(t)),
    "sin": lambda t: np.sin(t),
    "sigmoid": expit,
    "tanh": lambda t: np.tanh(t),
}
B0_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
C0_GRID = [-1.5, 0.0, 1.5]


def xy(df):
    # train/valid frames carry qlib's two-level columns (group, name)
    groups = [c[0] if isinstance(c, tuple) else "feature" for c in df.columns]
    feat_cols = [c for c, g in zip(df.columns, groups) if g == "feature"]
    lab_cols = [c for c, g in zip(df.columns, groups) if g == "label"]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[lab_cols[0]].to_numpy(dtype=np.float32) if lab_cols else None
    names = [c[1] if isinstance(c, tuple) else c for c in feat_cols]
    return X, y, names


# ---------------------------------------------------------------- KAN edge extraction


def extract_edges():
    """All first-layer edge curves phi_ij(x) = spline(x) + w_base*SiLU(x) on a [-3,3] grid,
    plus the B1 edge-norm importance (identical definition to run_b1.dump_spline_curves)."""
    names = list(pd.read_parquet(CACHE / "test_feat.parquet").columns)  # 158 flat feature names
    d = len(names)
    model = KAN(layers_hidden=[d, KAN_HIDDEN, 1], grid_size=5, spline_order=3,
                grid_range=list(KAN_GRID_RANGE))
    model.load_state_dict(torch.load(KAN_DIR / "model.pt", map_location="cpu"))
    model.eval()
    layer = model.layers[0]
    xs = torch.linspace(KAN_GRID_RANGE[0], KAN_GRID_RANGE[1], N_GRID)
    xg = xs.unsqueeze(1).repeat(1, layer.in_features)
    with torch.no_grad():
        bases = layer.b_splines(xg)                 # (N, in, coeff)
        sw = layer.scaled_spline_weight             # (out, in, coeff)
        bw = layer.base_weight                       # (out, in)
        phi = torch.einsum("xjk,ijk->xji", bases, sw) \
            + bw.T.unsqueeze(0) * F.silu(xs).unsqueeze(1).unsqueeze(2)
        imp = bw.abs().sum(0) + sw.abs().sum(2).sum(0)   # (in,)
    phi = phi.cpu().numpy().astype(np.float32)      # (N, in, out)
    imp = imp.cpu().numpy()
    assert phi.shape == (N_GRID, d, KAN_HIDDEN), phi.shape
    print(f"[edges] phi {phi.shape}, imp sum {imp.sum():.2f}", flush=True)
    np.savez_compressed(OUT / "edge_curves_all.npz", xs=xs.numpy(), phi=phi,
                        imp=imp.astype(np.float32), feat_names=np.array(names))
    edge_l1 = np.abs(phi).max(axis=0)               # (in, out) max |phi| per edge
    strongest = edge_l1.argmax(axis=1)              # (in,)
    order = np.argsort(-imp)
    df = pd.DataFrame({
        "rank": np.arange(1, d + 1),
        "feature": [names[i] for i in order],
        "edge_norm": imp[order],
        "edge_norm_share": imp[order] / imp.sum(),
        "strongest_hidden": strongest[order],
        "strongest_edge_maxabs": edge_l1[order, strongest[order]],
    })
    df.to_csv(OUT / "kan_edge_importance.csv", index=False)
    print("[edges] top-10 by edge norm:", flush=True)
    print(df.head(10).to_string(index=False), flush=True)
    return xs.numpy(), phi, imp, names, strongest


# ---------------------------------------------------------------- shape adjudication


def classify_curve(x, y):
    """Heuristic shape adjudication for one curve on a uniform grid; returns metrics + labels.
    Issue classes: monotone inc / monotone dec / U-shaped / saturating / irregular."""
    rho = float(spearmanr(x, y).statistic)
    dy = np.gradient(y, x)
    mid_slope = float(np.abs(dy[(x >= -1.0) & (x <= 1.0)]).mean()) + 1e-12
    left_rel = float(np.abs(dy[x <= -2.2]).mean()) / mid_slope
    right_rel = float(np.abs(dy[x >= 2.2]).mean()) / mid_slope
    sat_ratio = min(left_rel, right_rel)   # either flat end => saturation present
    c2, c1, c0 = np.polyfit(x, y, 2)
    yhat = np.polyval([c2, c1, c0], x)
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = float(((y - yhat) ** 2).sum())
    r2_quad = 1.0 - ssr / sst if sst > 1e-12 else (1.0 if ssr < 1e-12 else 0.0)
    vertex = -c1 / (2 * c2) if abs(c2) > 1e-9 else np.nan
    frac_up = float((dy > 0).mean())

    label = None
    if abs(rho) >= 0.995:
        direction = "monotone_inc" if rho > 0 else "monotone_dec"
        label = f"{direction}_saturating" if sat_ratio < 0.25 else direction
    elif abs(rho) >= 0.94 and sat_ratio <= 0.30:
        # strongly monotone with one flat end: delayed-activation sigmoid half
        label = "monotone_inc_saturating" if rho > 0 else "monotone_dec_saturating"
    elif r2_quad >= 0.90 and abs(vertex) <= 2.5 and abs(c2) * 9.0 >= 0.5 * y.std():
        base = "U_shaped" if c2 > 0 else "inverted_U"
        label = f"{base}_saturating" if sat_ratio < 0.25 else base
    else:
        label = "irregular"

    issue_map = {
        "monotone_inc": "monotone increasing", "monotone_dec": "monotone decreasing",
        "monotone_inc_saturating": "monotone increasing (saturating)",
        "monotone_dec_saturating": "monotone decreasing (saturating)",
        "U_shaped": "U-shaped", "inverted_U": "U-shaped (inverted)",
        "U_shaped_saturating": "U-shaped (saturating)", "inverted_U_saturating": "U-shaped (inverted, saturating)",
        "irregular": "no clear pattern",
    }
    return {"spearman_rho": rho, "slope_pos_frac": frac_up, "sat_left": left_rel,
            "sat_right": right_rel, "sat_ratio": sat_ratio,
            "quad_r2": r2_quad, "quad_c2": float(c2), "quad_vertex": float(vertex)
            if np.isfinite(vertex) else np.nan,
            "shape_label": label, "issue_class": issue_map[label]}


def run_shape_classification(xs, phi, imp, names, strongest):
    rows = []
    order = np.argsort(-imp)
    for i in order:
        y = phi[:, i, strongest[i]]
        r = {"feature": names[i], "strongest_hidden": int(strongest[i]),
             "edge_norm": float(imp[i]), "edge_maxabs": float(np.abs(y).max())}
        r.update(classify_curve(xs, y))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "shape_classification.csv", index=False)
    print("\n[shape] counts by label:")
    print(df["shape_label"].value_counts().to_string(), flush=True)
    print("\n[shape] top-20 detail (for manual review):")
    cols = ["feature", "edge_norm", "spearman_rho", "slope_pos_frac", "sat_left",
            "sat_right", "quad_r2", "quad_c2", "quad_vertex", "shape_label"]
    print(df[cols].head(20).to_string(index=False), flush=True)
    return df


# ---------------------------------------------------------------- symbolic snap


def _lin_fit(fv, y):
    """Least-squares y ~ a*f + d (closed form); returns a, d, ssr."""
    A = np.column_stack([fv, np.ones_like(fv)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ssr = float(((A @ coef - y) ** 2).sum())
    return float(coef[0]), float(coef[1]), ssr


def fit_candidate(x, y, f):
    """Fit y ~ a*f(b*x+c)+d: coarse grid over (b, c) with closed-form (a, d), then curve_fit
    refinement from the best grid point. Returns dict with params, r2, rmse."""
    sst = float(((y - y.mean()) ** 2).sum())
    best = None
    for b0 in B0_GRID:
        for c0 in C0_GRID:
            fv = f(b0 * x + c0)
            a0, d0, ssr = _lin_fit(fv, y)
            if best is None or ssr < best["ssr"]:
                best = {"a": a0, "b": b0, "c": c0, "d": d0, "ssr": ssr}

    def model(t, a, b, c, d):
        return a * f(b * t + c) + d

    try:
        p, _ = curve_fit(model, x, y,
                         p0=[best["a"], best["b"], best["c"], best["d"]], maxfev=4000)
        yhat = model(x, *p)
        ssr = float(((yhat - y) ** 2).sum())
        if np.isfinite(ssr) and ssr <= best["ssr"]:
            best = {"a": p[0], "b": p[1], "c": p[2], "d": p[3], "ssr": ssr}
    except (RuntimeError, ValueError):
        pass
    if sst <= 1e-12:
        r2 = 1.0 if best["ssr"] <= 1e-12 else 0.0
    else:
        r2 = 1.0 - best["ssr"] / sst
    rmse = float(np.sqrt(best["ssr"] / len(x)))
    return {**best, "r2": r2, "rmse": rmse}


def run_symbolic_snap(xs, phi, imp, names, strongest):
    rows = []
    t0 = time.perf_counter()
    in_dim = len(names)
    for i in range(in_dim):
        for j in range(KAN_HIDDEN):
            y = phi[:, i, j].astype(np.float64)
            row = {"feature": names[i], "hidden": j,
                   "edge_maxabs": float(np.abs(y).max()),
                   "edge_l1": float(np.abs(y).mean())}
            if row["edge_maxabs"] < NEGLIGIBLE:
                row.update({"best_cand": "(negligible edge)", "best_r2": np.nan, "best_rmse": np.nan,
                            "a": np.nan, "b": np.nan, "c": np.nan, "d": np.nan,
                            "second_cand": "", "second_r2": np.nan,
                            "snap_status": "negligible"})
                rows.append(row)
                continue
            fits = {k: fit_candidate(xs.astype(np.float64), y, f) for k, f in CANDIDATES.items()}
            ranked = sorted(fits.items(), key=lambda kv: -kv[1]["r2"])
            (bk, bf), (sk, _) = ranked[0], ranked[1]
            status = "snapped" if bf["r2"] >= SNAP_R2 else (
                "borderline" if bf["r2"] >= SNAP_R2_BORDER else "not_snappeable")
            row.update({"best_cand": bk, "best_r2": bf["r2"], "best_rmse": bf["rmse"],
                        "a": bf["a"], "b": bf["b"], "c": bf["c"], "d": bf["d"],
                        "second_cand": sk, "second_r2": fits[sk]["r2"],
                        "snap_status": status})
            rows.append(row)
        if (i + 1) % 40 == 0:
            print(f"[snap] {i+1}/{in_dim} features ({time.perf_counter()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "symbolic_snap_edges.csv", index=False)
    live = df[df["snap_status"] != "negligible"]
    print(f"\n[snap] {len(df)} edges ({len(live)} live) in {time.perf_counter()-t0:.0f}s; status counts:")
    print(live["snap_status"].value_counts().to_string(), flush=True)
    print("\n[snap] strongest edges of top-12 features:")
    imp_order = pd.read_csv(OUT / "kan_edge_importance.csv")
    top12 = list(imp_order.head(TOP_PLOT)["feature"])
    hid = dict(zip(imp_order["feature"], imp_order["strongest_hidden"]))
    sel = df[df.apply(
        lambda r: r["feature"] in top12 and int(r["hidden"]) == int(hid[r["feature"]]), axis=1)]
    order_key = {f: k for k, f in enumerate(top12)}
    sel = sel.sort_values("feature", key=lambda s: s.map(order_key))
    print(sel[["feature", "hidden", "edge_maxabs", "best_cand", "best_r2", "best_rmse",
               "snap_status", "a", "b", "c", "d"]].to_string(index=False), flush=True)
    return df


# ---------------------------------------------------------------- LightGBM gain importance


def run_lgb():
    import lightgbm as lgb
    Xtr, ytr, names = xy(pd.read_parquet(CACHE / "train.parquet"))
    Xva, yva, _ = xy(pd.read_parquet(CACHE / "valid.parquet"))
    params = {  # S1b / qlib official Alpha158 LightGBM benchmark hyperparams
        "objective": "regression",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.042,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
        "seed": SEED,
        "verbose": -1,
    }
    t0 = time.perf_counter()
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=names)
    dva = lgb.Dataset(Xva, label=yva, feature_name=names, reference=dtr)
    # valid_sets=[train, valid] replicates qlib LGBModel exactly: the train-set l2 keeps
    # improving and resets early-stopping patience, so the run effectively uses all 1000
    # rounds (this quirk is part of the S1b behavior being copied, seed aside).
    booster = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dtr, dva],
                        valid_names=["train", "valid"],
                        callbacks=[lgb.early_stopping(50, verbose=False),
                                   lgb.log_evaluation(200)])
    secs = time.perf_counter() - t0
    gain = booster.feature_importance("gain")
    df = pd.DataFrame({"feature": names, "gain": gain,
                       "gain_pct": gain / gain.sum() * 100.0})
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(OUT / "lgb_gain_importance.csv", index=False)
    valid_pred = booster.predict(Xva, num_iteration=booster.best_iteration)
    vs = float(spearmanr(valid_pred, yva).statistic)
    # sanity check against the S1b artifact: same data / params / early-stopping behavior,
    # only the seed differs -> predictions should be highly correlated
    s1b_pred = ROOT / "common" / "runs" / "s1b_alpha158_lgb" / "pred.parquet"
    check = None
    if s1b_pred.exists():
        te = pd.read_parquet(CACHE / "test_feat.parquet")
        mine = pd.Series(booster.predict(te.to_numpy(dtype=np.float32)), index=te.index)
        mine.index.names = ["datetime", "instrument"]
        ref = pd.read_parquet(s1b_pred)["score"]
        jj = pd.concat([mine.rename("mine"), ref.rename("s1b")], axis=1).dropna()
        check = {"pearson": float(np.corrcoef(jj["mine"], jj["s1b"])[0, 1]),
                 "spearman": float(spearmanr(jj["mine"], jj["s1b"]).statistic),
                 "rows": int(len(jj))}
    meta = {"params": params, "num_boost_round": 1000, "early_stopping_rounds": 50,
            "best_iteration": booster.best_iteration, "valid_spearman": vs,
            "n_trees_used": booster.num_trees(), "s1b_pred_check": check,
            "train_rows": int(Xtr.shape[0]), "train_seconds": secs,
            "data": "common/runs/kan/_cache/{train,valid}.parquet (same cache as B1 models)",
            "note": "retrained once for gain-importance export only; S1b artifacts untouched. "
                    "valid_sets=[train,valid] mirrors qlib LGBModel; lightgbm's early stopping "
                    "ignores the train entry, so ~16 trees are kept — the same behavior as S1b. "
                    "s1b_pred_check quantifies agreement with the S1b test predictions (only "
                    "the seed differs)."}
    with open(OUT / "lgb_retrain_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[lgb] best_iter={booster.best_iteration} n_trees={booster.num_trees()} "
          f"valid_spearman={vs:.5f} ({secs:.0f}s); s1b_pred_check={check}; top-10 gain:", flush=True)
    print(df.head(10).to_string(index=False), flush=True)
    return df, meta


# ---------------------------------------------------------------- cross-check


def run_compare(imp_df, lgb_df):
    kan = imp_df.set_index("feature")
    lgb = lgb_df.set_index("feature")
    j = pd.DataFrame({"feature": kan.index, "kan_edge_norm": kan["edge_norm"].values,
                      "kan_rank": kan["rank"].values,
                      "lgb_gain_pct": lgb.loc[kan.index, "gain_pct"].values,
                      "lgb_rank": lgb.loc[kan.index, "rank"].values})
    j["kan_top20"] = j["kan_rank"] <= TOPN
    j["lgb_top20"] = j["lgb_rank"] <= TOPN
    rho = float(spearmanr(j["kan_rank"], j["lgb_rank"]).statistic)
    overlap = j[j["kan_top20"] & j["lgb_top20"]]["feature"].tolist()
    only_kan = j[j["kan_top20"] & ~j["lgb_top20"]]["feature"].tolist()
    only_lgb = j[~j["kan_top20"] & j["lgb_top20"]]["feature"].tolist()
    j.to_csv(OUT / "importance_compare.csv", index=False)
    summary = {"spearman_all158": rho, "overlap_top20": overlap,
               "only_kan_top20": only_kan, "only_lgb_top20": only_lgb}
    with open(OUT / "compare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[compare] full-158 rank Spearman {rho:.3f}; top-{TOPN} overlap {len(overlap)}")
    print(f"[compare] both top{TOPN}: {overlap}")
    print(f"[compare] KAN-only top{TOPN}: {only_kan}")
    print(f"[compare] LGB-only top{TOPN}: {only_lgb}", flush=True)
    return j, summary


# ---------------------------------------------------------------- atlas figure


def make_atlas(xs, phi, imp, names, strongest, shape_df, snap_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(-imp)[:TOP_PLOT]
    sidx = shape_df.set_index("feature")
    # per-feature strongest-edge snap result
    snap_best = {}
    for feat in [names[i] for i in order]:
        sub = snap_df[(snap_df["feature"] == feat) &
                      (snap_df["hidden"] == sidx.loc[feat, "strongest_hidden"])]
        if len(sub):
            r = sub.iloc[0]
            snap_best[feat] = "—"
            if r["snap_status"] in ("snapped", "borderline"):
                snap_best[feat] = f"{r['best_cand']} (R²={r['best_r2']:.2f})"
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for ax, i in zip(axes.flat, order):
        curves = phi[:, i, :]                              # (N, out)
        hi = strongest[i]
        name = names[i]
        sh = sidx.loc[name]
        ax.plot(xs, curves, color="steelblue", alpha=0.18, lw=0.7)
        ax.plot(xs, curves[:, hi], color="crimson", lw=1.7)
        ax.axhline(0, color="gray", lw=0.4, ls=":")
        ax.set_title(f"{name}\nshape: {sh['shape_label']} | snap: {snap_best.get(name, '—')}",
                     fontsize=8.5)
        ax.set_xlabel("normalized feature (RobustZScoreNorm, clipped ±3)", fontsize=7)
        ax.set_ylabel("activation", fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle("B4: KAN first-layer spline atlas — top-12 features by edge norm, "
                 "strongest edge highlighted, shape adjudicated + symbolic snap", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "spline_curves_top12_labeled.png", dpi=160)
    plt.close(fig)
    print(f"[atlas] -> {OUT}/spline_curves_top12_labeled.png", flush=True)


# ---------------------------------------------------------------- main


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    xs, phi, imp, names, strongest = extract_edges()
    shape_df = run_shape_classification(xs, phi, imp, names, strongest)
    snap_df = run_symbolic_snap(xs, phi, imp, names, strongest)
    lgb_df, lgb_meta = run_lgb()
    imp_df = pd.read_csv(OUT / "kan_edge_importance.csv")
    cmp_df, summary = run_compare(imp_df, lgb_df)
    make_atlas(xs, phi, imp, names, strongest, shape_df, snap_df)
    print("run_b4_explain DONE", flush=True)


if __name__ == "__main__":
    main()
