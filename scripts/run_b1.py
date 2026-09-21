#!/usr/bin/env python
# B1 (Issue #1): Linear/Ridge + MLP + KAN on Alpha158, CSI300, identical data / eval / backtest
# pipeline as S1b (common/scripts/run_alpha158_lgb.py): official benchmark processors
# (RobustZScoreNorm+Fillna on features, DropnaLabel+CSZScoreNorm on label), train 2012-2020 /
# valid 2021-2022 / test 2023-2026.07, topk 30 / n_drop 1, cost open 0.05% close 0.15%.
# Hyperparameters are selected on valid only (Ridge alpha grid, NN early stopping); test is
# scored once at the end. Outputs to common/runs/kan/b1_{linear,mlp,kan}/ (physical files land
# in the agentic-feature-mining repo via the `common` symlink; never committed there).
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval.ic import daily_ic, ic_summary
from common.eval.layered import layered_returns
from efficient_kan import KAN

DATA_DIR = "/data1/wujunxi/kailin/data/qlib_data/cn_data"
RUNS = Path(__file__).resolve().parents[1] / "common" / "runs" / "kan"
CACHE = RUNS / "_cache"
LABEL = "Ref($close, -2)/Ref($close, -1) - 1"
SEGMENTS = {
    "train": ("2012-01-01", "2020-12-31"),
    "valid": ("2021-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2026-07-23"),
}
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fair-comparison pairing (Issue #1 acceptance #2): same order of magnitude, same depth.
MLP_HIDDEN = 224                       # [158, 224, 1]   -> 35,841 params (Linear+ReLU)
KAN_HIDDEN = 24                        # [158, 24, 1]    -> 38,160 params (10/edge: base+spline8+scaler)
KAN_GRID_RANGE = (-3.0, 3.0)           # inputs are RobustZScoreNorm-clipped to +/-3
TRAIN_CFG = {"lr": 1e-3, "weight_decay": 1e-4, "batch": 4096, "max_epoch": 200, "patience": 15}
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100]


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- data (cached)


def prepare_data():
    """Alpha158 features + labels as float32 numpy; parquet-cached under common/runs/kan/_cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    splits = {}
    if all((CACHE / f"{s}.parquet").exists() for s in ("train", "valid", "test_feat", "label_test")):
        for s in ("train", "valid", "test_feat", "label_test"):
            splits[s] = pd.read_parquet(CACHE / f"{s}.parquet")
            print(f"[cache] loaded {s}: {splits[s].shape}", flush=True)
        return splits

    qlib.init(provider_uri=DATA_DIR, region=REG_CN)
    handler = Alpha158(
        instruments="csi300",
        start_time=SEGMENTS["train"][0],
        end_time=SEGMENTS["test"][1],
        fit_start_time=SEGMENTS["train"][0],
        fit_end_time=SEGMENTS["train"][1],
        label=[LABEL],
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fit_start_time": SEGMENTS["train"][0], "fit_end_time": SEGMENTS["train"][1], "clip_outlier": True}},
            {"class": "Fillna"},
        ],
        learn_processors=[{"class": "DropnaLabel"}, {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}],
    )
    dataset = DatasetH(handler, SEGMENTS)  # 把 handler 里配置的数据根据 SEGMENTS 分割成 train, valid 和 test
    
    for seg in ("train", "valid"):
        df = dataset.prepare(seg, col_set=["feature", "label"], data_key=DataHandler.DK_L)
        df.to_parquet(CACHE / f"{seg}.parquet")
        splits[seg] = df
        print(f"[prep] {seg}: {df.shape}", flush=True)

    df = dataset.prepare("test", col_set="feature", data_key=DataHandler.DK_I)
    df.to_parquet(CACHE / "test_feat.parquet")
    splits["test_feat"] = df
    print(f"[prep] test_feat: {df.shape}", flush=True)

    codes = list(df.index.get_level_values("instrument").unique())
    label = D.features(codes, [LABEL], start_time=SEGMENTS["test"][0], end_time=SEGMENTS["test"][1])
    label.columns = ["label"]
    label.index = label.index.reorder_levels(["datetime", "instrument"])
    label.to_parquet(CACHE / "label_test.parquet")
    splits["label_test"] = label
    print(f"[prep] label_test: {label.shape}", flush=True)
    return splits


def xy(df):
    # train/valid frames carry qlib's two-level columns (group, name); test_feat is flat feature names
    groups = [c[0] if isinstance(c, tuple) else "feature" for c in df.columns]
    feat_cols = [c for c, g in zip(df.columns, groups) if g == "feature"]
    lab_cols = [c for c, g in zip(df.columns, groups) if g == "label"]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[lab_cols[0]].to_numpy(dtype=np.float32) if lab_cols else None
    names = [c[1] if isinstance(c, tuple) else c for c in feat_cols]
    return X, y, names


# ---------------------------------------------------------------- torch training


def valid_rankic(model, Xva, yva_np):
    """Global Spearman on valid — label SNR is too low for MSE to carry early-stop signal."""
    from scipy.stats import spearmanr
    model.eval()
    with torch.no_grad():
        p = model(Xva).squeeze(-1).cpu().numpy()
    r = float(spearmanr(p, yva_np).statistic)
    return -1.0 if np.isnan(r) else r


def train_torch(model, Xtr, ytr, Xva, yva_np, tag):
    """AdamW + MSE loss + early stopping / model selection on valid Spearman rank IC."""
    t0 = time.perf_counter()
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"])
    n, bs = Xtr.shape[0], TRAIN_CFG["batch"]
    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, TRAIN_CFG["max_epoch"] + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            loss = nn.functional.mse_loss(model(Xtr[idx]).squeeze(-1), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        vic = valid_rankic(model, Xva, yva_np)
        if vic > best + 1e-6:
            best, best_epoch, bad = vic, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if epoch % 10 == 0 or bad >= TRAIN_CFG["patience"]:
            print(f"[{tag}] epoch {epoch} valid_rankic {vic:.5f} (best {best:.5f} @ {best_epoch})", flush=True)
        if bad >= TRAIN_CFG["patience"]:
            break
    model.load_state_dict(best_state)
    return best_state, best_epoch, time.perf_counter() - t0


def predict_torch(model, X):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 65536):
            outs.append(model(X[i : i + 65536]).squeeze(-1).float().cpu().numpy())
    return np.concatenate(outs)


# ---------------------------------------------------------------- evaluation (mirrors S1b)


def evaluate_dump(pred: pd.Series, out_dir: Path):
    label = pd.read_parquet(CACHE / "label_test.parquet")["label"]
    ic = daily_ic(pred, label)
    ric = daily_ic(pred, label, method="spearman")
    pd.DataFrame({"ic": ic, "rankic": ric}).to_csv(out_dir / "ic.csv")
    metrics = {"ic": ic_summary(ic), "rankic": ic_summary(ric)}

    layered_returns(pred, label, n_groups=10).to_csv(out_dir / "layered.csv")

    strategy = TopkDropoutStrategy(topk=30, n_drop=1, signal=pred)
    report, _ = backtest_daily(
        start_time=SEGMENTS["test"][0],
        end_time=SEGMENTS["test"][1],
        strategy=strategy,
        benchmark="SH000300",
        exchange_kwargs={
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
            "trade_unit": 100,
        },
    )
    report.to_csv(out_dir / "backtest_report.csv")
    perf = {k: float(v) for k, v in risk_analysis(report["return"] - report["bench"]).iloc[:, 0].items()}
    perf["annualized_return"] = float(risk_analysis(report["return"]).iloc[:, 0]["annualized_return"])
    perf["daily_turnover_mean"] = float(report["turnover"].mean())
    metrics["backtest"] = perf
    return metrics


# ---------------------------------------------------------------- models


def run_linear(splits):
    out = RUNS / "b1_linear"
    out.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, feats = xy(splits["train"])
    Xva, yva, _ = xy(splits["valid"])
    Xte, _, _ = xy(splits["test_feat"])
    set_seed()
    t0 = time.perf_counter()
    best_a, best_ic, best_model = None, -1.0, None
    for a in RIDGE_ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        ic = float(spearmanr(m.predict(Xva), yva).statistic)
        print(f"[linear] alpha={a} valid_rankic={ic:.5f}", flush=True)
        if ic > best_ic:
            best_a, best_ic, best_model = a, ic, m
    train_secs = time.perf_counter() - t0
    pred = pd.Series(best_model.predict(Xte), index=splits["test_feat"].index).dropna()
    pred.index.names = ["datetime", "instrument"]
    pred.to_frame("score").to_parquet(out / "pred.parquet")
    metrics = evaluate_dump(pred, out)
    metrics["model"] = "linear_ridge"
    metrics["train"] = {"selected_alpha": best_a, "valid_rankic": best_ic, "selection": "valid_rankic",
                        "train_seconds": train_secs,
                        "n_params": int(Xtr.shape[1]) + 1, "seed": SEED}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"[linear] DONE RankIC={metrics['rankic']['ic_mean']:.4f} -> {out}", flush=True)


def _run_nn(splits, kind, out):
    out.mkdir(parents=True, exist_ok=True)
    Xtr_np, ytr_np, feats = xy(splits["train"])
    Xva_np, yva_np, _ = xy(splits["valid"])
    Xte_np, _, _ = xy(splits["test_feat"])
    Xtr = torch.from_numpy(Xtr_np).to(DEVICE)
    ytr = torch.from_numpy(ytr_np).to(DEVICE)
    Xva = torch.from_numpy(Xva_np).to(DEVICE)
    Xte = torch.from_numpy(Xte_np).to(DEVICE)
    d = Xtr.shape[1]
    set_seed()
    if kind == "mlp":
        model = nn.Sequential(nn.Linear(d, MLP_HIDDEN), nn.ReLU(), nn.Linear(MLP_HIDDEN, 1))
        cfg_extra = {"architecture": [d, MLP_HIDDEN, 1], "activation": "ReLU"}
    else:
        model = KAN(layers_hidden=[d, KAN_HIDDEN, 1], grid_size=5, spline_order=3,
                    grid_range=list(KAN_GRID_RANGE))
        cfg_extra = {"architecture": [d, KAN_HIDDEN, 1], "grid_size": 5, "spline_order": 3,
                     "grid_range": list(KAN_GRID_RANGE)}
    n_par = n_params(model)
    print(f"[{kind}] params={n_par} device={DEVICE} train={Xtr.shape[0]} valid={Xva.shape[0]}", flush=True)
    best_state, best_epoch, secs = train_torch(model, Xtr, ytr, Xva, yva_np, kind)
    torch.save(best_state, out / "model.pt")
    pred = pd.Series(predict_torch(model, Xte), index=splits["test_feat"].index).dropna()
    pred.index.names = ["datetime", "instrument"]
    pred.to_frame("score").to_parquet(out / "pred.parquet")
    metrics = evaluate_dump(pred, out)
    metrics["model"] = kind
    metrics["train"] = {"n_params": n_par, "train_seconds": secs, "best_epoch": best_epoch,
                        "optimizer": "AdamW", "seed": SEED, "selection": "valid_rankic",
                        **TRAIN_CFG, **cfg_extra}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    if kind == "kan":
        dump_spline_curves(model, feats, out)
    print(f"[{kind}] DONE RankIC={metrics['rankic']['ic_mean']:.4f} -> {out}", flush=True)
    return model


def run_mlp(splits):
    _run_nn(splits, "mlp", RUNS / "b1_mlp")


def run_kan(splits):
    out = RUNS / "b1_kan"
    if (out / "metrics.json").exists() and not (out / "spline_curves_top.parquet").exists():
        # model already trained; only redo the interpretability dump
        Xte_np, _, feats = xy(splits["test_feat"])
        d = Xte_np.shape[1]
        model = KAN(layers_hidden=[d, KAN_HIDDEN, 1], grid_size=5, spline_order=3,
                    grid_range=list(KAN_GRID_RANGE))
        model.load_state_dict(torch.load(out / "model.pt", map_location=DEVICE))
        model.to(DEVICE)
        dump_spline_curves(model, feats, out)
        print("[kan] metrics exist; spline curves regenerated", flush=True)
        return
    _run_nn(splits, "kan", out)


def dump_spline_curves(model, feat_names, out):
    """Learned first-layer edge functions phi_ij(x) = spline(x) + w_base*SiLU(x), top features by L1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    layer = model.layers[0]
    xs = torch.linspace(KAN_GRID_RANGE[0], KAN_GRID_RANGE[1], 201, device=DEVICE)
    xg = xs.unsqueeze(1).repeat(1, layer.in_features)
    with torch.no_grad():
        bases = layer.b_splines(xg)                 # (201, in, coeff)
        sw = layer.scaled_spline_weight             # (out, in, coeff)
        bw = layer.base_weight                       # (out, in)
        phi = torch.einsum("xjk,ijk->xji", bases, sw) + bw.T.unsqueeze(0) * F.silu(xs).unsqueeze(1).unsqueeze(2)
        imp = bw.abs().sum(0) + sw.abs().sum(2).sum(0)   # (in,)
    top = imp.argsort(descending=True)[:12]
    phi = phi.cpu().numpy()                          # (201, in, out)
    names = [feat_names[i] for i in top.tolist()]
    df = pd.DataFrame({f"{feat_names[j]}__h{i}": phi[:, j, i] for j, i in
                       zip(top.tolist(), range(len(top))) for i in range(phi.shape[2])},
                      index=xs.cpu().numpy())
    df.index.name = "x"
    df.to_parquet(out / "spline_curves_top.parquet")

    fig, axes = plt.subplots(4, 3, figsize=(13, 10))
    for ax, j, name in zip(axes.flat, top.tolist(), names):
        curves = phi[:, j, :]                        # (201, out)
        edge_l1 = np.abs(curves).max(axis=0)
        hi = int(np.argmax(edge_l1))
        ax.plot(xs.cpu().numpy(), curves, color="steelblue", alpha=0.18, lw=0.7)
        ax.plot(xs.cpu().numpy(), curves[:, hi], color="crimson", lw=1.6,
                label=f"strongest edge (h{hi})")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("normalized feature (RobustZScoreNorm, clipped ±3)", fontsize=7)
        ax.set_ylabel("activation", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    fig.suptitle("KAN first-layer learned spline transforms (top-12 features by edge L1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "spline_curves.png", dpi=160)
    plt.close(fig)
    print(f"[kan] spline curves -> {out}/spline_curves.png", flush=True)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["linear", "mlp", "kan"], default=None,
                    help="run a single model (default: all three)")
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    splits = prepare_data()
    qlib.init(provider_uri=DATA_DIR, region=REG_CN)  # needed by backtest_daily regardless of cache

    todo = [args.model] if args.model else ["linear", "mlp", "kan"]
    for m in todo:
        out = RUNS / f"b1_{m}"
        if (out / "metrics.json").exists():
            print(f"[skip] {m}: metrics.json exists", flush=True)
            if m == "kan" and not (out / "spline_curves_top.parquet").exists():
                run_kan(splits)
            continue
        {"linear": run_linear, "mlp": run_mlp, "kan": run_kan}[m](splits)
    print("run_b1 DONE", flush=True)


if __name__ == "__main__":
    main()
