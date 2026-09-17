#!/usr/bin/env python
# B15 (Issue #27): symbolic distillation of the B1 KAN — freeze B4's 158 strongest-edge
# closed-form formulas as the first layer (no training, no gradients), retrain ONLY the
# 24->1 second KAN layer, and quantify RankIC retention vs a paired control.
#
# Paired control: the original spline KAN KAN([158,24,1]) trained in the SAME script with
# the SAME b1 training path / seed / early stopping; the ONLY difference between the two
# runs is whether the first layer is the frozen symbolic formula layer or the trainable
# spline layer. Test is scored exactly once per model (selection happens on valid only).
#
# Formula source (read-only): common/runs/kan/explain/symbolic_snap_edges.csv JOIN
# shape_classification.csv on (feature, hidden=strongest_hidden) -> 158 edges, R2 >= 0.99.
# Edge function semantics follow B4: phi(x) fitted as a*f(b*x+c)+d where phi is the FULL
# efficient-kan first-layer edge (spline + base-weight*SiLU), grid x in [-3, 3].
#
# Outputs to common/runs/kan/symbolic-distill/: distilled/ + kan_control/ (model.pt,
# metrics.json, pred.parquet, ic.csv, backtest_report.csv, layered.csv each), formulas.csv,
# formulas.md, retention.json. Nothing under explain/ is modified.
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qlib  # noqa: E402
import run_b1 as b1  # noqa: E402  reuses data prep / training / eval code path unchanged
from efficient_kan import KAN  # noqa: E402
from qlib.constant import REG_CN  # noqa: E402

RUNS = b1.RUNS / "symbolic-distill"
EXPLAIN = b1.RUNS / "explain"

# same 8-family dictionary as B4's symbolic snap (keys must match best_cand strings)
CANDIDATES = {
    "x": lambda t: t,
    "x^2": lambda t: t * t,
    "x^3": lambda t: t * t * t,
    "|x|": lambda t: t.abs(),
    "sgn(x)*log(1+|x|)": lambda t: t.sign() * torch.log1p(t.abs()),
    "sin": lambda t: torch.sin(t),
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}
CAND_ORDER = list(CANDIDATES)


def load_strongest_edges() -> pd.DataFrame:
    """158 strongest edges: shape_classification (feature -> strongest hidden) JOIN snap params."""
    shape = pd.read_csv(EXPLAIN / "shape_classification.csv")
    snap = pd.read_csv(EXPLAIN / "symbolic_snap_edges.csv")
    best = shape[["feature", "strongest_hidden", "edge_norm"]].rename(
        columns={"strongest_hidden": "hidden"})
    edges = best.merge(
        snap[["feature", "hidden", "best_cand", "best_r2", "a", "b", "c", "d", "snap_status"]],
        on=["feature", "hidden"], how="left", validate="one_to_one")
    assert len(edges) == 158 and edges.best_cand.notna().all(), "missing snap params"
    assert edges.best_r2.min() >= 0.99, "B4 acceptance was R2 >= 0.99 on strongest edges"
    unknown = set(edges.best_cand) - set(CANDIDATES)
    assert not unknown, f"unknown candidate families: {unknown}"
    return edges


class SymbolicFirstLayer(nn.Module):
    """Frozen first layer: each of the 158 features has exactly ONE edge (its B4 strongest
    edge) into hidden node strongest_hidden, computed as a*f(b*x+c)+d. Buffers only —
    no parameters, no gradients; hidden-node input is the sum of incoming edge outputs."""

    def __init__(self, edges: pd.DataFrame, feat_names: list, hidden: int = b1.KAN_HIDDEN):
        super().__init__()
        col = [feat_names.index(f) for f in edges.feature]
        cand = torch.tensor([CAND_ORDER.index(c) for c in edges.best_cand], dtype=torch.long)
        self.register_buffer("col_idx", torch.tensor(col, dtype=torch.long))
        self.register_buffer("hidden_idx", torch.tensor(edges.hidden.to_numpy(), dtype=torch.long))
        self.register_buffer("cand_ids", cand)
        for p in ("a", "b", "c", "d"):
            self.register_buffer(p, torch.tensor(edges[p].to_numpy(dtype=np.float32)))
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, 158)
        z = self.b * x[:, self.col_idx] + self.c
        f = torch.empty_like(z)
        for k, fn in enumerate(CANDIDATES.values()):
            m = self.cand_ids == k
            if m.any():
                f[:, m] = fn(z[:, m])
        out = self.a * f + self.d
        h = torch.zeros(x.shape[0], self.hidden, device=x.device, dtype=x.dtype)
        h.index_add_(1, self.hidden_idx, out)
        return h


class SymbolicDistillKAN(nn.Module):
    """Second layer is the SAME KANLayer(24->1, grid 5, order 3) as the original [158,24,1]
    KAN's second layer; first layer is the frozen 158-edge symbolic layer."""

    def __init__(self, edges: pd.DataFrame, feat_names: list):
        super().__init__()
        self.first = SymbolicFirstLayer(edges, feat_names)
        self.second = KAN(layers_hidden=[b1.KAN_HIDDEN, 1], grid_size=5, spline_order=3,
                          grid_range=list(b1.KAN_GRID_RANGE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


def readable_formula(row) -> str:
    t = f"{row.b:.3f}*x{row.c:+.3f}"
    if row.best_cand == "sgn(x)*log(1+|x|)":
        body = f"sign({t})*log1p(|{t}|)"
    else:
        body = f"{row.best_cand}({t})"
    return f"{row.a:.3f}*{body}{row.d:+.3f}"


def write_formula_tables(edges: pd.DataFrame):
    edges = edges.sort_values("edge_norm", ascending=False).reset_index(drop=True)
    edges["formula"] = edges.apply(readable_formula, axis=1)
    cols = ["feature", "hidden", "edge_norm", "best_cand", "a", "b", "c", "d", "best_r2", "formula"]
    edges[cols].to_csv(RUNS / "formulas.csv", index=False)
    lines = [
        "# B15 蒸馏公式表：158 条最强边符号公式（第一层，冻结）",
        "",
        "来源：`explain/symbolic_snap_edges.csv` JOIN `shape_classification.csv`，边函数口径同 B4"
        "（φ = spline + base·SiLU 的完整边，定义域 x∈[-3,3]，RobustZScoreNorm clip 后）。",
        "每特征仅保留其最强边（边范数 L1 口径，同 B4），按 edge_norm 降序。",
        "",
        "| # | feature | hidden | 函数族 | R² | 公式 |",
        "|---|---------|--------|--------|-----|------|",
    ]
    for i, r in edges.iterrows():
        lines.append(f"| {i + 1} | {r.feature} | h{r.hidden} | {r.best_cand} | "
                     f"{r.best_r2:.4f} | `{r.formula}` |")
    (RUNS / "formulas.md").write_text("\n".join(lines) + "\n")
    print(f"[b15] formula tables -> {RUNS}/formulas.{{csv,md}}", flush=True)


def run_variant(splits, kind: str, edges: pd.DataFrame):
    out = RUNS / ("distilled" if kind == "distilled" else "kan_control")
    out.mkdir(parents=True, exist_ok=True)
    Xtr_np, ytr_np, feats = b1.xy(splits["train"])
    Xva_np, yva_np, _ = b1.xy(splits["valid"])
    Xte_np, _, _ = b1.xy(splits["test_feat"])
    Xtr = torch.from_numpy(Xtr_np).to(b1.DEVICE)
    ytr = torch.from_numpy(ytr_np).to(b1.DEVICE)
    Xva = torch.from_numpy(Xva_np).to(b1.DEVICE)
    Xte = torch.from_numpy(Xte_np).to(b1.DEVICE)
    d = Xtr.shape[1]

    b1.set_seed()
    if kind == "distilled":
        model = SymbolicDistillKAN(edges, feats)
        first_layer = "symbolic_158_edges_frozen"
    else:
        model = KAN(layers_hidden=[d, b1.KAN_HIDDEN, 1], grid_size=5, spline_order=3,
                    grid_range=list(b1.KAN_GRID_RANGE))
        first_layer = "kan_spline_trainable"
    n_par = b1.n_params(model)
    print(f"[b15-{kind}] params={n_par} device={b1.DEVICE}", flush=True)

    best_state, best_epoch, secs = b1.train_torch(model, Xtr, ytr, Xva, yva_np, f"b15-{kind}")
    torch.save(best_state, out / "model.pt")
    model.load_state_dict(best_state)
    valid_rankic = b1.valid_rankic(model, Xva, yva_np)

    # test scored exactly once (paired discipline: selection used valid only)
    pred = pd.Series(b1.predict_torch(model, Xte), index=splits["test_feat"].index).dropna()
    pred.index.names = ["datetime", "instrument"]
    pred.to_frame("score").to_parquet(out / "pred.parquet")
    metrics = b1.evaluate_dump(pred, out)
    metrics["model"] = f"b15_{kind}"
    metrics["train"] = {"n_params": n_par, "train_seconds": secs, "first_layer": first_layer,
                        "n_symbolic_edges": 158,
                        "formula_source": "explain/symbolic_snap_edges.csv x shape_classification.csv",
                        "valid_rankic_best": valid_rankic, "best_epoch": best_epoch,
                        "optimizer": "AdamW", "seed": b1.SEED, "selection": "valid_rankic",
                        **b1.TRAIN_CFG, "architecture": [d, b1.KAN_HIDDEN, 1],
                        "grid_size": 5, "spline_order": 3, "grid_range": list(b1.KAN_GRID_RANGE)}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"[b15-{kind}] DONE valid_rankic {valid_rankic:.5f} "
          f"test_rankic {metrics['rankic']['ic_mean']:.5f} -> {out}", flush=True)
    return metrics


def backtest_layer_stats(out_dir: Path) -> dict:
    """Read-only summary of one variant's PERSISTED products (no training, no re-scoring).

    Exists because the historical `metrics.json` backtest block mixes two series under one dict.
    run_b1.py:210 runs qlib `risk_analysis(r - bench)` with its defaults (`freq="day"` -> N=238,
    `mode="sum"`), while the pre-fix run_b1.py:211 OVERWROTE `annualized_return` with
    `risk_analysis(report["return"]).iloc[0, 0]` = the FIRST row, i.e. the mean of the ABSOLUTE
    daily return. So: mean/std/IR/max_drawdown are EXCESS-series stats, while that field (now
    renamed `abs_daily_return_mean` on disk, fix #32) was an absolute DAILY mean. Mixed-series
    has to be spelled out, hence the explicit absolute/excess split below.

    Equity curve: use the `account` column (= `value` + `cash`). `return` is qlib's
    `return_rate = (now_earning + now_cost) / last_account_value` (backtest/account.py:283) —
    a position-view earning rate, NOT the account compounding rate, so `(1+return).prod() - 1`
    does not reproduce account growth. Net value and percentage drawdown must come from
    `account`; using `value` alone (position market value, excludes cash) is wrong.
    """
    bt = pd.read_csv(out_dir / "backtest_report.csv", parse_dates=["datetime"]).set_index("datetime")
    lay = pd.read_csv(out_dir / "layered.csv", parse_dates=["datetime"]).set_index("datetime")
    pred = pd.read_parquet(out_dir / "pred.parquet")["score"]
    raw = json.load(open(out_dir / "metrics.json"))["backtest"]  # field names are wrong; see docstring
    groups = [c for c in lay.columns if c != "long_short"]
    gm, ls = lay[groups].mean(), lay["long_short"]
    r, b = bt["return"], bt["bench"]
    ex, acct, n = r - b, bt["account"], len(bt)
    ratio = acct.iloc[-1] / acct.iloc[0]
    return {
        "n_test_days": n,
        "interval": [str(bt.index.min().date()), str(bt.index.max().date())],
        "years_at_238_days_per_year": float(n / 238.0),
        "net_value": {  # account column = value + cash
            "account_start": float(acct.iloc[0]),
            "account_end": float(acct.iloc[-1]),
            "account_ratio": float(ratio),
            "account_validated_as_value_plus_cash": bool(np.allclose(bt["value"] + bt["cash"], acct)),
            "cagr_238": float(ratio ** (238 / n) - 1),
            "max_drawdown_pct_account": float((acct / acct.cummax() - 1).min()),
            "max_drawdown_pct_value_only": float((bt["value"] / bt["value"].cummax() - 1).min()),
            "compound_of_return_col": float((1 + r).prod() - 1),
        },
        "absolute": {
            "daily_mean": float(r.mean()),
            "annualized_sum_x238": float(r.mean() * 238),
            "daily_vol": float(r.std(ddof=1)),
            "ir_sqrt238": float(r.mean() / r.std(ddof=1) * np.sqrt(238)),
        },
        "excess": {
            "daily_mean": float(ex.mean()),
            "annualized_sum_x238": float(ex.mean() * 238),
            "daily_vol": float(ex.std(ddof=1)),
            "ir_sqrt238": float(ex.mean() / ex.std(ddof=1) * np.sqrt(238)),
            "arithmetic_cum_drawdown": float((ex.cumsum() - ex.cumsum().cummax()).min()),
        },
        "trading": {"daily_turnover_mean": float(bt["turnover"].mean()),
                    "daily_cost_mean": float(bt["cost"].mean())},
        "metrics_json_raw": {k: float(v) for k, v in raw.items()},
        "layered": {
            "group_mean_returns": {c: float(gm[c]) for c in groups},
            "monotonicity_spearman_groupidx_meanret": float(spearmanr(range(len(groups)), gm.values).statistic),
            "long_short_daily_mean": float(ls.mean()),
            "long_short_daily_std": float(ls.std()),
            "long_short_ir_daily": float(ls.mean() / ls.std()),
            "long_short_ir_sqrt238": float(ls.mean() / ls.std() * np.sqrt(238)),
            "long_short_pos_rate": float((ls > 0).mean()),
            "extreme_groups": {"g0": float(gm[groups[0]]), "g9": float(gm[groups[-1]])},
        },
        "pred": {
            "std": float(pred.std()),
            "iqr": float(pred.quantile(0.75) - pred.quantile(0.25)),
            "mean_daily_cross_sectional_range": float(pred.groupby(level=0).apply(lambda x: x.max() - x.min()).mean()),
        },
    }


def write_layer_compare() -> dict:
    compare = {k: backtest_layer_stats(RUNS / k) for k in ("distilled", "kan_control")}
    # interval equality is what rules out "the IR gap is just a different window" — assert it, don't assume
    d, c = compare["distilled"], compare["kan_control"]
    assert d["n_test_days"] == c["n_test_days"] and d["interval"] == c["interval"], (d["interval"], c["interval"])
    with open(RUNS / "backtest_layer_compare.json", "w") as f:
        json.dump(compare, f, indent=2, default=float)
    print(f"[b15] backtest/layered compare -> {RUNS}/backtest_layer_compare.json", flush=True)
    return compare


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true",
                    help="summarize persisted products only (no qlib init, no training)")
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        write_layer_compare()
        return
    qlib.init(provider_uri=b1.DATA_DIR, region=REG_CN)  # backtest_daily needs qlib data
    edges = load_strongest_edges()

    metrics_d = run_variant(splits=b1.prepare_data(), kind="distilled", edges=edges)
    metrics_c = run_variant(splits=b1.prepare_data(), kind="control", edges=edges)
    write_formula_tables(edges)
    write_layer_compare()

    ret_d, ret_c = metrics_d["rankic"]["ic_mean"], metrics_c["rankic"]["ic_mean"]
    retention = {
        "distilled_test_rankic": ret_d,
        "control_test_rankic": ret_c,
        "b1_kan_reference_test_rankic": 0.026425067147420527,
        "retention_vs_control": ret_d / ret_c,
        "retention_vs_b1_reference": ret_d / 0.026425067147420527,
        "ic": {"distilled": metrics_d["ic"]["ic_mean"], "control": metrics_c["ic"]["ic_mean"],
               "retention_vs_control": metrics_d["ic"]["ic_mean"] / metrics_c["ic"]["ic_mean"]},
        "note": "test scored once per model; retention = distilled / control (paired, same script); "
                "IC retention (Pearson) is far lower than RankIC retention by design — see report",
    }
    with open(RUNS / "retention.json", "w") as f:
        json.dump(retention, f, indent=2)
    print(f"[b15] retention vs control = {ret_d / ret_c:.1%} "
          f"(vs b1 reference {ret_d / 0.026425067147420527:.1%})", flush=True)


if __name__ == "__main__":
    main()
