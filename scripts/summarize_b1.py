#!/usr/bin/env python
# B1 summary (Issue #1): merge b1_{linear,mlp,kan} metrics with the S1b LightGBM reference
# into results/B1_main_table.md — main comparison, yearly RankIC, layering, param/time cost.
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "common" / "runs"
MODELS = [
    ("linear", "Linear/Ridge", RUNS / "kan" / "b1_linear"),
    ("lgb", "LightGBM (S1b)", RUNS / "s1b_alpha158_lgb"),
    ("mlp", "MLP", RUNS / "kan" / "b1_mlp"),
    ("kan", "KAN", RUNS / "kan" / "b1_kan"),
]
OUT = ROOT / "results" / "B1_main_table.md"


def load(m):
    with open(m / "metrics.json") as f:
        met = json.load(f)
    bt = met["backtest"]
    rep = pd.read_csv(m / "backtest_report.csv", index_col=0, parse_dates=True)
    ic = pd.read_csv(m / "ic.csv", index_col=0, parse_dates=True)
    layered = pd.read_csv(m / "layered.csv", index_col=0, parse_dates=True)
    return met, bt, rep, ic, layered


def fmt(v, pct=False, digits=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v*100:.2f}%" if pct else f"{v:.{digits}f}"


def main():
    rows = {}
    for key, disp, path in MODELS:
        met, bt, rep, ic, layered = load(path)
        tr = met.get("train", {})
        groups = [f"g{i}" for i in range(10)]
        gm = layered[groups].mean().to_numpy()
        rows[key] = {
            "name": disp, "params": tr.get("n_params"), "secs": tr.get("train_seconds"),
            "rankic": met["rankic"]["ic_mean"], "ic": met["ic"]["ic_mean"],
            "rankicir": met["rankic"]["icir"], "icir": met["ic"]["icir"],
            "excess": bt["mean"] * 252, "ir": bt["information_ratio"], "mdd": bt["max_drawdown"],
            "turnover": rep["turnover"].mean(),
            "ls": layered["long_short"].mean() * 252,
            "mono": spearmanr(np.arange(10), gm).statistic,
            "ic_series": ic,
        }
        extra = tr.get("selected_alpha")
        if extra is not None:
            rows[key]["extra"] = f"α={extra}"
        ep = tr.get("best_epoch")
        if ep is not None:
            rows[key]["extra"] = (rows[key].get("extra", "") + f", best epoch {ep}").strip(", ")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# B1 主表：Alpha158 → Linear/Ridge · LightGBM · MLP · KAN（CSI300）")
    L.append("")
    L.append("口径：与 S1b 完全一致 — train 2012–2020 / valid 2021–2022（超参与早停仅用 valid）")
    L.append("/ test 2023–2026.07 仅最终评测一次；RobustZScoreNorm(±3)+Fillna 特征、")
    L.append("DropnaLabel+CSZScoreNorm(label)；回测 topk 30 / drop 1，成本 open 5bp / close 15bp。")
    L.append("MLP [158,224,1]-ReLU vs KAN [158,24,1]（grid 5，order 3，grid_range ±3）：参数量同量级。")
    L.append("")
    L.append("## 1. 主对比表（test 2023-01 ~ 2026-07）")
    L.append("")
    L.append("| Model | Params | Train time | RankIC | IC | RankICIR | ICIR | Ann. excess | IR | Excess MDD | Daily turnover |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("linear", "lgb", "mlp", "kan"):
        r = rows[k]
        L.append(
            f"| {r['name']} | {r['params'] if r['params'] is not None else 'n/a'} "
            f"| {fmt(r['secs']/60, digits=1) + ' min' if r['secs'] is not None else 'n/a'} "
            f"| {fmt(r['rankic'])} | {fmt(r['ic'])} | {fmt(r['rankicir'])} | {fmt(r['icir'])} "
            f"| {fmt(r['excess'], pct=True)} | {fmt(r['ir'], digits=2)} | {fmt(r['mdd'], pct=True)} "
            f"| {fmt(r['turnover'], pct=True)} |"
        )
    L.append("")
    L.append("注：LightGBM 参数量与训练时间为 S1b 历史运行，未记录（n/a）；其余三列同机同协议本次实测。")
    L.append("超参记录：Linear/Ridge 与 MLP/KAN 的 valid 选择细节见 `common/runs/kan/b1_*/metrics.json` 的 `train` 块。")
    L.append("")
    L.append("## 2. 分年度 RankIC（test 按自然年拆分）")
    L.append("")
    years = [2023, 2024, 2025, 2026]
    L.append("| Model | " + " | ".join(str(y) for y in years) + " | full test |")
    L.append("|---|" + "---|" * (len(years) + 1))
    for k in ("linear", "lgb", "mlp", "kan"):
        s = rows[k]["ic_series"]["rankic"]
        yr = {y: s[s.index.year == y].mean() for y in years}
        L.append(f"| {rows[k]['name']} | " + " | ".join(fmt(yr[y]) for y in years) + f" | {fmt(s.mean())} |")
    L.append("")
    L.append("注：2026 为 1–7 月（数据覆盖至 2026-07-23）。regime 稳健性深化（滚动重训等）在 #3。")
    L.append("")
    L.append("## 3. 分层回测与成本")
    L.append("")
    L.append("| Model | Long-short ann. (10-layer) | Monotonicity (Spearman) |")
    L.append("|---|---|---|")
    for k in ("linear", "lgb", "mlp", "kan"):
        r = rows[k]
        L.append(f"| {r['name']} | {fmt(r['ls'], pct=True)} | {fmt(r['mono'], digits=2)} |")
    L.append("")
    L.append("逐日分层明细：`common/runs/{s1b_alpha158_lgb,kan/b1_*}/layered.csv`；成本口径 main（open 5bp / close 15bp），")
    L.append("敏感性四象限扩展不在 B1 范围（见 #3）。")
    L.append("")
    L.append("## 4. 公平性对照（MLP vs KAN）")
    L.append("")
    L.append("| Model | Architecture | Params | Train time | Notes |")
    L.append("|---|---|---|---|---|")
    for k in ("mlp", "kan"):
        r = rows[k]
        arch = "[158, 224, 1] ReLU" if k == "mlp" else "[158, 24, 1] grid5/order3, grid_range ±3"
        L.append(f"| {r['name']} | {arch} | {r['params']} | {fmt(r['secs']/60, digits=1)} min | {r.get('extra','')} |")
    L.append("")
    L.append("参数比 KAN/MLP = "
             f"{rows['kan']['params']/rows['mlp']['params']:.2f}；同一优化协议（AdamW，lr 1e-3，wd 1e-4，"
             "batch 4096，max 200 epoch，valid 早停 patience 15，seed 42）。KAN 单 epoch 明显更慢是已知事实，"
             "如实报告训练时间。")
    L.append("")
    L.append("## 5. KAN 样条可视化")
    L.append("")
    L.append("`common/runs/kan/b1_kan/spline_curves.png`：第一层 top-12 特征（按边 L1 范数）的学习到的")
    L.append("样条变换曲线（每特征 24 条边曲线 + 最强边高亮），曲线数据 `spline_curves_top.parquet`。")
    L.append("可解释性深化（符号/形状归因）在 #4。")
    L.append("")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
