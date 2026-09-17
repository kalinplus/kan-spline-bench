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


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=b1.DATA_DIR, region=REG_CN)  # backtest_daily needs qlib data
    edges = load_strongest_edges()

    metrics_d = run_variant(splits=b1.prepare_data(), kind="distilled", edges=edges)
    metrics_c = run_variant(splits=b1.prepare_data(), kind="control", edges=edges)
    write_formula_tables(edges)

    ret_d, ret_c = metrics_d["rankic"]["ic_mean"], metrics_c["rankic"]["ic_mean"]
    retention = {
        "distilled_test_rankic": ret_d,
        "control_test_rankic": ret_c,
        "b1_kan_reference_test_rankic": 0.026425067147420527,
        "retention_vs_control": ret_d / ret_c,
        "retention_vs_b1_reference": ret_d / 0.026425067147420527,
        "note": "test scored once per model; retention = distilled / control (paired, same script)",
    }
    with open(RUNS / "retention.json", "w") as f:
        json.dump(retention, f, indent=2)
    print(f"[b15] retention vs control = {ret_d / ret_c:.1%} "
          f"(vs b1 reference {ret_d / 0.026425067147420527:.1%})", flush=True)


if __name__ == "__main__":
    main()
