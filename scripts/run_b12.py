#!/usr/bin/env python
# B12 (Issue #22): baseline fairness (local LGB / MLP tuning) + daily paired significance
# + expanding portfolio-layer backtest. Training protocol reuses scripts/run_b1.py (read-only
# import); W2/W3 are zero-training analyses over existing artifacts (common/runs/kan/rolling/
# stitched/, b1_*/, s1b_alpha158_lgb/). Outputs to common/runs/kan/b12_baseline_significance/.
#
# Discipline (Issue #22): W1/W1b grids are evaluated on valid only (selection = valid global
# RankIC); test is scored ONCE, only for the selected arm. S1b-LGB / B1-MLP / Ridge / KAN
# references are read from existing frozen artifacts and never re-scored. The W2 main test
# (KAN vs MLP, expanding x norm-global, NW lag=10 + block bootstrap 20/10000) is preregistered
# in the Issue body and run verbatim; every other comparison is labelled exploratory.
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import qlib
from qlib.constant import REG_CN

import run_b1 as b1

DEVICE = "cpu"
b1.DEVICE = DEVICE  # train_torch / predict_torch read this module-global at call time

OUT = b1.RUNS / "b12_baseline_significance"
STITCHED = b1.RUNS / "rolling" / "stitched"

# S1b hyperparams verbatim (common/scripts/run_alpha158_lgb.py "official hyperparams", as in
# scripts/run_rolling.py). The W1 grid overrides exactly three of them; the default arm
# (num_leaves=210, lr=0.042, min_data_in_leaf=20=lgb default) IS the S1b config.
LGB_PARAMS = {
    "objective": "mse",
    "verbosity": -1,
    "colsample_bytree": 0.8879,
    "learning_rate": 0.042,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 20,
}
LGB_NUM_BOOST_ROUND = 1000
LGB_EARLY_STOP = 50
LGB_GRID = {"num_leaves": [31, 63, 127, 210], "learning_rate": [0.02, 0.05, 0.1],
            "min_data_in_leaf": [20, 100]}

# W1b grid: hidden x lr; everything else B1 protocol (wd 1e-4, batch 4096, max 200 epochs,
# early stop on valid RankIC patience 15, seed 42).
MLP_GRID = {"hidden": [64, 128, 224, 448], "lr": [1e-3, 3e-3]}

B7_ALL_THRESHOLD = 0.0083   # 2x pooled seed std, pairs not KAN-vs-KAN (B7 REPORT.md)
# W2 preregistered hyperparams (Issue #22 body -- do not change before running):
W2_NW_LAG = 10
W2_BLOCK = 20
W2_BOOT = 10000


def dump_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def spear(a, b):
    r = float(spearmanr(a, b).statistic)
    return -1.0 if np.isnan(r) else r


# ---------------------------------------------------------------- W1: LGB local tuning


def lgb_arm_params(num_leaves, lr, min_data):
    p = dict(LGB_PARAMS)
    p.update(num_leaves=num_leaves, learning_rate=lr, min_data_in_leaf=min_data)
    return p


def train_lgb_valid(params, Xtr, ytr, Xva, yva, tag):
    import lightgbm as lgb
    t0 = time.perf_counter()
    dtrain = lgb.Dataset(Xtr, label=ytr)
    dvalid = lgb.Dataset(Xva, label=yva, reference=dtrain)
    net = lgb.train(params, dtrain, num_boost_round=LGB_NUM_BOOST_ROUND,
                    valid_sets=[dvalid], valid_names=["valid"],
                    callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False)])
    vp = net.predict(Xva, num_iteration=net.best_iteration)
    return net, spear(vp, yva), int(net.best_iteration or LGB_NUM_BOOST_ROUND), time.perf_counter() - t0


def w1():
    """24-arm LGB grid + S1b default arm, valid-only selection; test scored once for the
    selected arm. S1b default test metrics are cited from the existing s1b_alpha158_lgb run."""
    out = OUT / "w1_lgb"
    out.mkdir(parents=True, exist_ok=True)
    splits = b1.prepare_data()
    Xtr, ytr, _ = b1.xy(splits["train"])
    Xva, yva, _ = b1.xy(splits["valid"])
    yva = yva.astype(np.float64)

    arms = [(nl, lr, md) for nl in LGB_GRID["num_leaves"] for lr in LGB_GRID["learning_rate"]
            for md in LGB_GRID["min_data_in_leaf"]]
    rows = []
    for nl, lr, md in arms:
        tag = f"nl{nl}_lr{lr}_mdl{md}"
        rj = out / f"arm_{tag}.json"
        if rj.exists():
            rec = json.load(open(rj))
        else:
            print(f"[w1] arm {tag}", flush=True)
            _, vic, best_it, secs = train_lgb_valid(lgb_arm_params(nl, lr, md), Xtr, ytr, Xva, yva, tag)
            rec = {"num_leaves": nl, "learning_rate": lr, "min_data_in_leaf": md,
                   "valid_rankic": vic, "best_iteration": best_it, "train_seconds": secs}
            dump_json(rec, rj)
        rows.append(rec)
    # S1b default arm: valid-only local run for the grid table (its test metrics already
    # exist in s1b_alpha158_lgb and are cited, never re-scored)
    rj = out / "arm_s1b_default.json"
    if rj.exists():
        default_rec = json.load(open(rj))
    else:
        print("[w1] arm s1b_default (valid only)", flush=True)
        _, vic, best_it, secs = train_lgb_valid(LGB_PARAMS, Xtr, ytr, Xva, yva, "s1b_default")
        default_rec = {"num_leaves": 210, "learning_rate": 0.042, "min_data_in_leaf": 20,
                       "valid_rankic": vic, "best_iteration": best_it, "train_seconds": secs,
                       "is_s1b_default": True}
        dump_json(default_rec, rj)
    rows.sort(key=lambda r: -r["valid_rankic"])
    cols = ["num_leaves", "learning_rate", "min_data_in_leaf", "valid_rankic",
            "best_iteration", "train_seconds"]
    grid = pd.DataFrame(rows + [default_rec])[cols]
    grid.to_csv(out / "grid_valid_rankic.csv", index=False)
    print(f"[w1] grid table -> {out/'grid_valid_rankic.csv'}", flush=True)

    # selection among the 24 grid arms (default row is the baseline, not a candidate)
    sel = max((r for r in rows if not r.get("is_s1b_default")), key=lambda r: r["valid_rankic"])
    tag = f"nl{sel['num_leaves']}_lr{sel['learning_rate']}_mdl{sel['min_data_in_leaf']}"
    sel_marker = out / "selected.json"
    if sel_marker.exists():
        print(f"[w1] selected arm already test-scored: {json.load(open(sel_marker))}", flush=True)
        return
    print(f"[w1] selected arm {tag} (valid RankIC {sel['valid_rankic']:.5f}); single test eval", flush=True)
    qlib.init(provider_uri=b1.DATA_DIR, region=REG_CN)
    net, _, _, _ = train_lgb_valid(lgb_arm_params(sel["num_leaves"], sel["learning_rate"],
                                                  sel["min_data_in_leaf"]), Xtr, ytr, Xva, yva, tag)
    Xte, _, _ = b1.xy(splits["test_feat"])
    pred = pd.Series(net.predict(Xte, num_iteration=net.best_iteration),
                     index=splits["test_feat"].index).dropna()
    pred.index.names = ["datetime", "instrument"]
    tdir = out / "test_selected"
    tdir.mkdir(parents=True, exist_ok=True)
    metrics = b1.evaluate_dump(pred, tdir)
    pred.to_frame("score").to_parquet(tdir / "pred.parquet")
    metrics["train"] = {**sel, "selection": "valid global RankIC over 24-arm grid",
                        "protocol": "S1b hyperparams except grid-overridden fields; "
                                    f"early stop 50 on valid L2; num_threads 20; device cpu"}
    dump_json(metrics, tdir / "metrics.json")
    dump_json({"selected_arm": sel, "s1b_default_valid": default_rec}, sel_marker)
    print(f"[w1] DONE tuned test RankIC={metrics['rankic']['ic_mean']:.4f}", flush=True)


# ---------------------------------------------------------------- W1b: MLP local tuning


def w1b():
    """12-arm MLP grid (hidden x lr), B1 protocol otherwise, CPU seed 42. Valid-only
    selection; test scored once for the selected arm. B1-MLP [158,224,1] test metrics are
    cited from b1_mlp artifacts, never re-scored."""
    out = OUT / "w1b_mlp"
    out.mkdir(parents=True, exist_ok=True)
    splits = b1.prepare_data()
    Xtr_np, ytr_np, _ = b1.xy(splits["train"])
    Xva_np, yva_np, _ = b1.xy(splits["valid"])
    Xtr = torch_from(Xtr_np)
    ytr = torch_from(ytr_np)
    Xva = torch_from(Xva_np)

    import torch
    rows = []
    for h in MLP_GRID["hidden"]:
        for lr in MLP_GRID["lr"]:
            tag = f"h{h}_lr{lr:g}"
            adir = out / f"arm_{tag}"
            rj = adir / "result.json"
            if rj.exists():
                rec = json.load(open(rj))
            else:
                adir.mkdir(parents=True, exist_ok=True)
                print(f"[w1b] arm {tag}", flush=True)
                b1.TRAIN_CFG = {**b1.TRAIN_CFG, "lr": lr}
                b1.set_seed()
                model = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], h), torch.nn.ReLU(),
                                            torch.nn.Linear(h, 1))
                best_state, best_epoch, secs = b1.train_torch(model, Xtr, ytr, Xva, yva_np, tag)
                torch.save(best_state, adir / "model.pt")
                model.load_state_dict(best_state)
                vic = spear(b1.predict_torch(model, Xva), yva_np)
                rec = {"hidden": h, "lr": lr, "valid_rankic": vic, "best_epoch": int(best_epoch),
                       "train_seconds": secs, "n_params": b1.n_params(model)}
                dump_json(rec, rj)
            rows.append(rec)
    rows.sort(key=lambda r: -r["valid_rankic"])
    pd.DataFrame(rows)[["hidden", "lr", "valid_rankic", "best_epoch", "train_seconds", "n_params"]] \
        .to_csv(out / "grid_valid_rankic.csv", index=False)
    print(f"[w1b] grid table -> {out/'grid_valid_rankic.csv'}", flush=True)

    sel_marker = out / "selected.json"
    if sel_marker.exists():
        print(f"[w1b] selected arm already test-scored: {json.load(open(sel_marker))}", flush=True)
        return
    sel = rows[0]
    tag = f"h{sel['hidden']}_lr{sel['lr']:g}"
    print(f"[w1b] selected arm {tag} (valid RankIC {sel['valid_rankic']:.5f}); single test eval", flush=True)
    qlib.init(provider_uri=b1.DATA_DIR, region=REG_CN)
    model = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], sel["hidden"]), torch.nn.ReLU(),
                                torch.nn.Linear(sel["hidden"], 1))
    model.load_state_dict(torch.load(out / f"arm_{tag}" / "model.pt", map_location=DEVICE))
    pred = pd.Series(b1.predict_torch(model, torch_from(b1.xy(splits["test_feat"])[0])),
                     index=splits["test_feat"].index).dropna()
    pred.index.names = ["datetime", "instrument"]
    tdir = out / "test_selected"
    tdir.mkdir(parents=True, exist_ok=True)
    metrics = b1.evaluate_dump(pred, tdir)
    pred.to_frame("score").to_parquet(tdir / "pred.parquet")
    metrics["train"] = {**sel, "selection": "valid global RankIC over 12-arm grid",
                        "protocol": "B1 (AdamW wd 1e-4, batch 4096, max 200 epoch, valid-RankIC "
                                    "early stop patience 15, seed 42, cpu)"}
    dump_json(metrics, tdir / "metrics.json")
    dump_json({"selected_arm": sel}, sel_marker)
    print(f"[w1b] DONE tuned test RankIC={metrics['rankic']['ic_mean']:.4f}", flush=True)


def torch_from(a):
    import torch
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(DEVICE)


# ---------------------------------------------------------------- W2: daily paired stats


def nw_hac(d, lag):
    """Newey-West HAC variance of the mean (Bartlett kernel) -> (t, se)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    e = d - d.mean()
    s = float(e @ e) / n
    for l in range(1, lag + 1):
        s += 2.0 * (1.0 - l / (lag + 1.0)) * float(e[l:] @ e[:-l]) / n
    se = np.sqrt(max(s, 0.0) / n)
    return d.mean() / se if se > 0 else np.nan, se


def block_bootstrap_p(d, block, b, seed=42):
    """Moving-block bootstrap one-sided p for H1: mean > 0 (H0: mean <= 0)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    n = len(d)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(b)
    for i in range(b):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([s + np.arange(block) for s in starts])[:n]
        means[i] = d[idx].mean()
    return (1.0 + int((means <= 0).sum())) / (b + 1.0)


def w2_selfcheck():
    """Sanity: on iid noise NW t ~ classical t; on a positive-mean series p-values are small."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    t_nw, se_nw = nw_hac(x, W2_NW_LAG)
    t_cl = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(t_nw - t_cl) < 0.35, (t_nw, t_cl)
    p1 = block_bootstrap_p(x + 0.5, W2_BLOCK, 500)
    assert p1 < 0.01, p1
    p0 = block_bootstrap_p(rng.normal(size=1000), W2_BLOCK, 500)
    assert 0.2 < p0 < 0.9, p0
    print("[w2] selfcheck passed", flush=True)


def load_rankic(path):
    s = pd.read_csv(path, index_col=0, parse_dates=True)["rankic"]
    s.index = pd.to_datetime(s.index)
    return s


def paired_row(name_a, name_b, sa, sb):
    j = sa.index.intersection(sb.index)
    assert len(j) == len(sa) == len(sb), \
        f"date misalignment {name_a}({len(sa)}) vs {name_b}({len(sb)}): joined {len(j)}"
    d = (sa - sb).loc[j]
    t, se = nw_hac(d.values, W2_NW_LAG)
    p_nw = float(1.0 - norm.cdf(t))  # one-sided, H0: mean <= 0
    p_boot = block_bootstrap_p(d.values, W2_BLOCK, W2_BOOT)
    p_dm = float(2.0 * min(norm.cdf(t), 1 - norm.cdf(t)))  # DM reference, two-sided
    return {"comparison": f"{name_a} - {name_b}", "n_days": int(len(j)),
            "mean_diff": float(d.mean()), "hac_se": float(se), "nw_t": float(t),
            "p_nw_one_sided": p_nw, "p_block_bootstrap_one_sided": p_boot,
            "p_dm_two_sided_ref": p_dm}


def w2():
    """Preregistered main test (Issue #22 body): KAN vs MLP, expanding x norm-global stitched
    daily RankIC, H0 mean(diff) <= 0, NW HAC t (lag=10) + moving-block bootstrap (20/10000),
    DM as reference column. All other rows are exploratory."""
    w2_selfcheck()
    out = OUT / "w2_significance"
    out.mkdir(parents=True, exist_ok=True)
    kan_e = load_rankic(STITCHED / "kan_expanding_global" / "ic.csv")
    mlp_e = load_rankic(STITCHED / "mlp_expanding_global" / "ic.csv")
    kan_r = load_rankic(STITCHED / "kan_rolling_global" / "ic.csv")
    mlp_r = load_rankic(STITCHED / "mlp_rolling_global" / "ic.csv")
    b1_kan = load_rankic(b1.RUNS / "b1_kan" / "ic.csv")
    b1_mlp = load_rankic(b1.RUNS / "b1_mlp" / "ic.csv")

    main = paired_row("KAN expanding-global", "MLP expanding-global", kan_e, mlp_e)
    main["role"] = "main (preregistered, Issue #22)"
    exploratory = [
        paired_row("KAN expanding-global", "B1 KAN full-window (frozen)", kan_e, b1_kan),
        paired_row("KAN rolling-global", "KAN expanding-global", kan_r, kan_e),
        paired_row("MLP expanding-global", "B1 MLP full-window (frozen)", mlp_e, b1_mlp),
        paired_row("MLP rolling-global", "MLP expanding-global", mlp_r, mlp_e),
    ]
    for r in exploratory:
        r["role"] = "exploratory"
    df = pd.DataFrame([main] + exploratory)
    df.to_csv(out / "significance.csv", index=False)
    dump_json({"main": main, "exploratory": exploratory,
               "prereg": "Main test definition fixed in Issue #22 body before running: KAN vs "
                         "MLP, expanding x norm-global stitched daily RankIC (860 days), H0: "
                         "mean daily diff <= 0, NW HAC lag=10 + moving-block bootstrap "
                         "block=20 B=10000, DM reference. Side tests exploratory.",
               "params": {"nw_lag": W2_NW_LAG, "block": W2_BLOCK, "bootstrap": W2_BOOT}},
              out / "significance.json")
    print(df.to_string(index=False), flush=True)
    print(f"[w2] -> {out/'significance.csv'}", flush=True)


# ---------------------------------------------------------------- W3: expanding backtest


def w3():
    """Expanding stitched predictions (B9) through the B1 backtest pipeline; B1 frozen
    backtests cited from existing artifacts. Zero training."""
    out = OUT / "w3_backtest"
    out.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=b1.DATA_DIR, region=REG_CN)
    rows = []
    for m in ("kan", "mlp", "ridge", "lgb"):
        tdir = out / f"{m}_expanding_global"
        tdir.mkdir(parents=True, exist_ok=True)
        rj = tdir / "metrics.json"
        if rj.exists():
            metrics = json.load(open(rj))
        else:
            pred = pd.read_parquet(STITCHED / f"{m}_expanding_global" / "pred.parquet")["score"]
            metrics = b1.evaluate_dump(pred, tdir)
            dump_json(metrics, rj)
        rows.append({"model": m, "pred_source": f"stitched/{m}_expanding_global (B9)",
                     "rankic": metrics["rankic"]["ic_mean"], "ic": metrics["ic"]["ic_mean"],
                     "rankicir": metrics["rankic"]["icir"],
                     "ann_excess": metrics["backtest"]["mean"] * 252,
                     "ir": metrics["backtest"]["information_ratio"],
                     "mdd": metrics["backtest"]["max_drawdown"],
                     "turnover": metrics["backtest"].get("daily_turnover_mean", float("nan"))})
        print(f"[w3] {m}: ann_excess={rows[-1]['ann_excess']:.4f}", flush=True)

    # tuned arms (W1/W1b test predictions) through the same pipeline
    for m, wdir in (("lgb_tuned", OUT / "w1_lgb" / "test_selected"),
                    ("mlp_tuned", OUT / "w1b_mlp" / "test_selected")):
        rj = wdir / "metrics.json"
        if not rj.exists():
            print(f"[w3] skip {m}: no test prediction yet", flush=True)
            continue
        metrics = json.load(open(rj))
        rows.append({"model": m, "pred_source": f"{wdir} (B12 tuned arm)",
                     "rankic": metrics["rankic"]["ic_mean"], "ic": metrics["ic"]["ic_mean"],
                     "rankicir": metrics["rankic"]["icir"],
                     "ann_excess": metrics["backtest"]["mean"] * 252,
                     "ir": metrics["backtest"]["information_ratio"],
                     "mdd": metrics["backtest"]["max_drawdown"],
                     "turnover": metrics["backtest"].get("daily_turnover_mean", float("nan"))})

    # B1 frozen references (existing artifacts, never re-run)
    refs = {"ridge": b1.RUNS / "b1_linear", "mlp": b1.RUNS / "b1_mlp", "kan": b1.RUNS / "b1_kan",
            "lgb": b1.RUNS.parent / "s1b_alpha158_lgb"}
    for m, d in refs.items():
        metrics = json.load(open(d / "metrics.json"))
        rows.append({"model": f"{m} (B1 frozen)", "pred_source": f"{d} (existing artifact)",
                     "rankic": metrics["rankic"]["ic_mean"], "ic": metrics["ic"]["ic_mean"],
                     "rankicir": metrics["rankic"]["icir"],
                     "ann_excess": metrics["backtest"]["mean"] * 252,
                     "ir": metrics["backtest"]["information_ratio"],
                     "mdd": metrics["backtest"]["max_drawdown"],
                     "turnover": metrics["backtest"].get("daily_turnover_mean", float("nan"))})
    pd.DataFrame(rows).to_csv(out / "backtest_comparison.csv", index=False)
    print(f"[w3] -> {out/'backtest_comparison.csv'}", flush=True)


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["w1", "w1b", "w2", "w3"])
    args = ap.parse_args()
    {"w1": w1, "w1b": w1b, "w2": w2, "w3": w3}[args.stage]()
