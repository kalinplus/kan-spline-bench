#!/usr/bin/env python
# Split-level metric summary for the four B1 arms: train / valid aggregates + per-year test.
# Post-hoc reporting only — no test re-scoring of any published number: NN and LGB test metrics
# are read from the frozen pred.parquet artifacts (b1_*/pred.parquet, s1b_alpha158_lgb/pred.parquet).
# Train/valid predictions are regenerated because they were never dumped: Ridge is refit from the
# cached train frame with the alpha recorded in metrics.json; LGB is retrained from the cache with
# the S1b official hyperparams (same call pattern as run_b12 W1 default arm); MLP/KAN load model.pt.
#
# Label caveat: train/valid use the cached (CSZScoreNorm'd) label, test uses the raw label from
# label_test.parquet. Only RankIC (and RankICIR) are comparable across the two口径 — a per-day
# monotone transform leaves the within-day rank order untouched but does change Pearson IC.
#
# Output: common/runs/kan/metrics-summary/summary.csv (model, split, year, rankic, rankicir, ic, icir, n_days)
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b1 as b1  # noqa: E402  (module-level paths/helpers; main() is guarded)
from common.eval.ic import daily_ic, ic_summary  # noqa: E402
from efficient_kan import KAN  # noqa: E402

OUT = b1.RUNS / "metrics-summary"
S1B = b1.RUNS.parent / "s1b_alpha158_lgb"
PARAMS = {"linear": (b1.RUNS / "b1_linear"), "mlp": (b1.RUNS / "b1_mlp"), "kan": (b1.RUNS / "b1_kan")}

# S1b official LightGBM hyperparams (verbatim, see common/scripts/run_alpha158_lgb.py)
LGB_PARAMS = {
    "objective": "mse", "verbosity": -1, "colsample_bytree": 0.8879, "learning_rate": 0.042,
    "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8,
    "num_leaves": 210, "num_threads": 20, "seed": 42,
}


def score(pred: pd.Series, label: pd.Series) -> dict:
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ic, ric = daily_ic(df["p"], df["y"]), daily_ic(df["p"], df["y"], method="spearman")
    return {"ic": ic_summary(ic)["ic_mean"], "icir": ic_summary(ic)["icir"],
            "rankic": ic_summary(ric)["ic_mean"], "rankicir": ic_summary(ric)["icir"],
            "n_days": int(ric.shape[0])}


def load_linear(splits):
    alpha = json.load(open(PARAMS["linear"] / "metrics.json"))["train"]["selected_alpha"]
    from sklearn.linear_model import Ridge
    Xtr, ytr, _ = b1.xy(splits["train"])
    m = Ridge(alpha=alpha).fit(Xtr, ytr)
    print(f"[linear] refit Ridge(alpha={alpha}) on cached train frame", flush=True)
    return lambda df: pd.Series(m.predict(b1.xy(df)[0]), index=df.index).dropna()


def load_nn(kind, splits):
    Xtr, _, feats = b1.xy(splits["train"])
    if kind == "mlp":
        model = nn.Sequential(nn.Linear(Xtr.shape[1], b1.MLP_HIDDEN), nn.ReLU(), nn.Linear(b1.MLP_HIDDEN, 1))
    else:
        model = KAN(layers_hidden=[Xtr.shape[1], b1.KAN_HIDDEN, 1], grid_size=5, spline_order=3,
                    grid_range=list(b1.KAN_GRID_RANGE))
    model.load_state_dict(torch.load(PARAMS[kind] / "model.pt", map_location="cpu"))
    model.to("cpu")
    print(f"[{kind}] loaded model.pt", flush=True)
    return lambda df: pd.Series(b1.predict_torch(model, torch.from_numpy(b1.xy(df)[0])), index=df.index).dropna()


def load_lgb(splits):
    import lightgbm as lgb
    Xtr, ytr, _ = b1.xy(splits["train"])
    Xva, yva, _ = b1.xy(splits["valid"])
    t0 = time.perf_counter()
    ds_tr, ds_va = lgb.Dataset(Xtr, label=ytr), lgb.Dataset(Xva, label=yva, reference=None)
    net = lgb.train(LGB_PARAMS, ds_tr, num_boost_round=1000, valid_sets=[ds_va],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"[lgb] retrained, best_iteration={net.best_iteration} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return lambda df: pd.Series(net.predict(b1.xy(df)[0]), index=df.index).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="linear,lgb,mlp,kan")
    args = ap.parse_args()
    splits = b1.prepare_data()
    OUT.mkdir(parents=True, exist_ok=True)
    lab_te = pd.read_parquet(b1.CACHE / "label_test.parquet")["label"]
    label = {"train": b1.xy(splits["train"])[1], "valid": b1.xy(splits["valid"])[1]}
    label = {"train": pd.Series(label["train"], index=splits["train"].index).dropna(),
             "valid": pd.Series(label["valid"], index=splits["valid"].index).dropna()}

    rows = []
    for m in args.models.split(","):
        if m == "linear":
            predict = load_linear(splits)
        elif m == "lgb":
            predict = load_lgb(splits)
        else:
            predict = load_nn(m, splits)

        for split in ("train", "valid"):
            r = score(predict(splits[split]), label[split])
            rows.append({"model": m, "split": split, "year": "all", **r})
            print(f"[{m}] {split} RankIC={r['rankic']:.4f} ({r['n_days']}d)", flush=True)

        pred_te = (pd.read_parquet(S1B / "pred.parquet") if m == "lgb"
                   else pd.read_parquet(PARAMS[m] / "pred.parquet"))["score"]
        r = score(pred_te, lab_te)
        rows.append({"model": m, "split": "test", "year": "all", **r})
        print(f"[{m}] test RankIC={r['rankic']:.4f}", flush=True)
        for y in (2023, 2024, 2025, 2026):
            s = slice(f"{y}-01-01", f"{y}-12-31")
            print(f"[{m}] test {y} ...", flush=True)
            r = score(pred_te.loc[s], lab_te)
            rows.append({"model": m, "split": "test", "year": str(y), **r})

    df = pd.DataFrame(rows)[["model", "split", "year", "rankic", "rankicir", "ic", "icir", "n_days"]]
    df.to_csv(OUT / "summary.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print(f"-> {OUT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
