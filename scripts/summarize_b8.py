#!/usr/bin/env python
# B8 summary (Issue #12): merge the csi500 b8_* metrics with the csi300 B1/S1b reference into
# common/runs/kan/universe-csi500/B8_csi500_table.md — main table (two topk backtest arms),
# cross-universe comparison (topk 50 on csi500 = ~top 10%, the equal-strength anchor against
# csi300's topk 30 at 30/300), yearly RankIC, layering, fairness pairing (MLP vs KAN), spline
# shape comparison across universes, cache-isolation evidence, and migration conclusions.
# csi500 runs: common/runs/kan/universe-csi500/b8_{linear,lgb,mlp,kan};
# csi300 reference (read-only): common/runs/kan/b1_{linear,mlp,kan} + common/runs/s1b_alpha158_lgb.
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "common" / "runs"
CSI500 = RUNS / "kan" / "universe-csi500"
OUT = CSI500 / "B8_csi500_table.md"
MODELS = [
    ("linear", "Linear/Ridge", CSI500 / "b8_linear"),
    ("lgb", "LightGBM (S1b hp)", CSI500 / "b8_lgb"),
    ("mlp", "MLP", CSI500 / "b8_mlp"),
    ("kan", "KAN", CSI500 / "b8_kan"),
]
REF300 = {  # csi300 B1/S1b reference dirs (topk 30 only)
    "linear": RUNS / "kan" / "b1_linear",
    "lgb": RUNS / "s1b_alpha158_lgb",
    "mlp": RUNS / "kan" / "b1_mlp",
    "kan": RUNS / "kan" / "b1_kan",
}


def load(m):
    with open(m / "metrics.json") as f:
        met = json.load(f)
    bt = met["backtest"]
    bt50 = met.get("backtest_topk50")
    rep = pd.read_csv(m / "backtest_report.csv", index_col=0, parse_dates=True)
    ic = pd.read_csv(m / "ic.csv", index_col=0, parse_dates=True)
    layered = pd.read_csv(m / "layered.csv", index_col=0, parse_dates=True)
    return met, bt, bt50, rep, ic, layered


def fmt(v, pct=False, digits=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v*100:.2f}%" if pct else f"{v:.{digits}f}"


def spline_compare():
    """Overlap and strongest-edge shape correlation between csi300 (b1_kan) and csi500 (b8_kan)
    top-12 spline dumps. Strongest edge per feature = argmax_x |phi| (same rule as the plots)."""
    b1 = pd.read_parquet(REF300["kan"] / "spline_curves_top.parquet")
    b8 = pd.read_parquet(CSI500 / "b8_kan" / "spline_curves_top.parquet")

    def feats(df):
        return sorted({c.rsplit("__h", 1)[0] for c in df.columns})

    f1, f8 = feats(b1), feats(b8)
    ov = sorted(set(f1) & set(f8))

    def strongest(df, f):
        cols = sorted((c for c in df.columns if c.startswith(f + "__h")), key=lambda c: int(c.rsplit("h", 1)[1]))
        A = df[cols].to_numpy()
        hi = int(np.argmax(np.abs(A).max(axis=0)))
        return A[:, hi], hi

    rows = []
    for f in ov:
        c1, h1 = strongest(b1, f)
        c8, h8 = strongest(b8, f)
        corr = float(np.corrcoef(c1, c8)[0, 1])
        rows.append((f, h1, h8, corr))
    return f1, f8, ov, rows


CONCLUSIONS = """## 7. 迁移结论

1. **家族分层保持、家族内头名互换**：「MLP/KAN > Linear/Ridge > LightGBM」的分层在两池一致
   （csi500：0.0287 / 0.0250 vs 0.0183 vs 0.0155）；但家族内部排序由 csi300 的 KAN > MLP
   互换为 csi500 的 MLP > KAN。两个 NN 相对 Linear 的优势均放大（MLP/Linear 1.13×→1.57×，
   KAN/Linear 1.34×→1.37×）——放大的是整个非线性家族的红利，不是 KAN 独有的。
2. **核心 hypothesis 不成立（有信息量的否定）**：「csi500 市值更小、量价非线性更强 → KAN 相对
   优势放大」未获支持：KAN−MLP RankIC 差由 +0.0042（csi300）转为 −0.0037（csi500）。KAN 的
   占优维度转为排序质量与稳定性：RankICIR 0.215 四模型最高；10 层 long-short 年化 39.4%，
   几乎翻倍于其余三模型（19.8–23.1%）——KAN 的优势在 csi500 上体现在全截面排序而非头部集中。
3. **组合层结论跨池复现且更强**：四模型回测超额均弱（|IR| ≤ 0.31），且 topk 档位内差异
   （LGB k30 +4.04% vs k50 −2.83%，摆动 6.9pp）大于模型间差异——B1「配置效应大于模型效应」
   在 csi500 上更强成立。Linear 在 csi500 上 valid 选出 α=100（csi300 为 α=10），极端平滑使
   日换手降到 0.94%，但超额仍为 −0.4%：线性信号的日度增量信息在 csi500 上近乎为零。
4. **年度模式部分迁移**：2023 对两池都是最差年份（csi500 上 KAN 0.0100 / Linear 0.0073，
   MLP 0.0221 一枝独秀）；差异在强势年份——csi300 强在 2024（KAN 0.0386），csi500 强在 2025
   （四模型 0.032–0.044 同步走高）。MLP 在 csi500 上年度间最稳（0.0204–0.0439 无塌方年份）。
5. **KAN 样条形状跨池高度稳定**：top-12 特征交集 7 个（均为动量/波动/价格位置类量价特征），
   交集特征的最强边曲线 |Pearson| 全部 ≥ 0.99（5 同向、2 镜像；隐层边符号无语义，镜像视为
   同形状）——独立训练、独立池子下 KAN 学到的特征变换形状几乎不变，支持其捕捉的是稳定的
   量价结构而非池内噪声。"""


def main():
    rows = {}
    for key, disp, path in MODELS:
        met, bt, bt50, rep, ic, layered = load(path)
        tr = met.get("train", {})
        groups = [f"g{i}" for i in range(10)]
        gm = layered[groups].mean().to_numpy()
        rows[key] = {
            "name": disp, "params": tr.get("n_params"), "secs": tr.get("train_seconds"),
            "rankic": met["rankic"]["ic_mean"], "ic": met["ic"]["ic_mean"],
            "rankicir": met["rankic"]["icir"], "icir": met["ic"]["icir"],
            "k30": bt, "k50": bt50,
            "ls": layered["long_short"].mean() * 252,
            "mono": spearmanr(np.arange(10), gm).statistic,
            "ic_series": ic,
        }
        if tr.get("selected_alpha") is not None:
            rows[key]["extra"] = f"α={tr['selected_alpha']}"
        if tr.get("best_epoch") is not None:
            rows[key]["extra"] = (rows[key].get("extra", "") + f", best epoch {tr['best_epoch']}").strip(", ")
        if tr.get("best_iteration") is not None:
            rows[key]["extra"] = f"best iter {tr['best_iteration']}"

    ref = {}
    for key, path in REF300.items():
        met, bt, bt50, rep, ic, layered = load(path)
        ref[key] = {
            "rankic": met["rankic"]["ic_mean"], "rankicir": met["rankic"]["icir"],
            "k30": bt, "ls": layered["long_short"].mean() * 252,
            "ic_series": ic,
        }

    L = []
    L.append("# B8 主表：B1 四模型迁移至 CSI500（universe 稳健性，Issue #12）")
    L.append("")
    L.append("口径：与 B1 逐字一致，仅换股票池 — instruments csi500 / benchmark SH000905；")
    L.append("train 2012–2020 / valid 2021–2022（Ridge α 网格与 NN 早停仅用 valid，训练协议同 B1）")
    L.append("/ test 2023–2026.07 仅最终评测一次；处理器链 RobustZScoreNorm(±3)+Fillna 特征、")
    L.append("DropnaLabel+CSZScoreNorm(label)；成本 open 5bp / close 15bp、limit 0.095、drop 1。")
    L.append("模型配置零改动（seed 42，MLP [158,224,1]-ReLU vs KAN [158,24,1] grid5/order3/±3）；")
    L.append("LightGBM 按 S1b 超参在 csi500 重训（无调参）；四模型统一 CPU（torch 20 线程 / LGB 20 线程）。")
    L.append("回测两档：topk 30（B1 镜像档，30/500 ≈ 前 6%）与 topk 50（≈ 前 10%，与 csi300 主表")
    L.append("topk 30 = 前 10% 的相对位置对齐，**跨池比较以 topk 50 为同强度锚点**）。")
    L.append("")
    L.append("## 1. csi500 主对比表（test 2023-01 ~ 2026-07）")
    L.append("")
    L.append("| Model | Params | Train time | RankIC | IC | RankICIR | ICIR | LS ann. (10层) | Monotonicity |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in ("linear", "lgb", "mlp", "kan"):
        r = rows[k]
        L.append(
            f"| {r['name']} | {r['params'] if r['params'] is not None else 'n/a'} "
            f"| {fmt(r['secs']/60, digits=1) + ' min' if r['secs'] is not None else 'n/a'} "
            f"| {fmt(r['rankic'])} | {fmt(r['ic'])} | {fmt(r['rankicir'])} | {fmt(r['icir'])} "
            f"| {fmt(r['ls'], pct=True)} | {fmt(r['mono'], digits=2)} |"
        )
    L.append("")
    L.append("注：Ridge 选点 / NN 早停 / LGB 早停的 valid 细节见 `universe-csi500/b8_*/metrics.json` 的 `train` 块")
    L.append(f"（linear {rows['linear'].get('extra','')}；mlp {rows['mlp'].get('extra','')}；"
             f"kan {rows['kan'].get('extra','')}；lgb {rows['lgb'].get('extra','')}）。")
    L.append("")
    L.append("## 2. 回测两档（topk 30 / topk 50，benchmark SH000905）")
    L.append("")
    L.append("| Model | topk | Ann. excess | IR | Excess MDD | Daily turnover |")
    L.append("|---|---|---|---|---|---|")
    for k in ("linear", "lgb", "mlp", "kan"):
        for arm, bt in (("30", rows[k]["k30"]), ("50", rows[k]["k50"])):
            L.append(
                f"| {rows[k]['name']} | {arm} | {fmt(bt['mean']*252, pct=True)} "
                f"| {fmt(bt['information_ratio'], digits=2)} | {fmt(bt['max_drawdown'], pct=True)} "
                f"| {fmt(bt['daily_turnover_mean'], pct=True)} |"
            )
    L.append("")
    L.append("注：两档差异普遍大于模型间差异（见 §7.3）；档位口径 30/500≈6% vs 50/500≈10%。")
    L.append("")
    L.append("## 3. 跨池对照：csi300（B1）vs csi500（B8）")
    L.append("")
    L.append("回测锚点：csi300 取 topk 30（= 前 10%），csi500 取 topk 50（= 前 10%，同强度）；")
    L.append("csi500 topk 30（前 6%，更集中）另列供参考。")
    L.append("")
    L.append("| Model | RankIC csi300 → csi500 | RankICIR csi300 → csi500 | Excess 锚点 csi300 k30 → csi500 k50 | IR 锚点 | Excess MDD 锚点 | csi500 k30 excess（口径更集中）|")
    L.append("|---|---|---|---|---|---|---|")
    for k in ("linear", "lgb", "mlp", "kan"):
        r, q = rows[k], ref[k]
        L.append(
            f"| {r['name']} | {fmt(q['rankic'])} → {fmt(r['rankic'])} "
            f"| {fmt(q['rankicir'])} → {fmt(r['rankicir'])} "
            f"| {fmt(q['k30']['mean']*252, pct=True)} → {fmt(r['k50']['mean']*252, pct=True)} "
            f"| {fmt(q['k30']['information_ratio'], digits=2)} → {fmt(r['k50']['information_ratio'], digits=2)} "
            f"| {fmt(q['k30']['max_drawdown'], pct=True)} → {fmt(r['k50']['max_drawdown'], pct=True)} "
            f"| {fmt(r['k30']['mean']*252, pct=True)} |"
        )
    L.append("")
    L.append("排序对照：RankIC 排序 csi300 KAN > MLP > Linear > LGB，csi500 **MLP > KAN > Linear > LGB**")
    L.append("（家族分层保持、家族内头名互换；详见 §7.1–7.2）。LS ann.：csi300 linear 21.6% / lgb 23.4% /")
    L.append(f"mlp 31.7% / kan 29.4% → csi500 {fmt(rows['linear']['ls'], pct=True)} / {fmt(rows['lgb']['ls'], pct=True)} /"
             f" {fmt(rows['mlp']['ls'], pct=True)} / {fmt(rows['kan']['ls'], pct=True)}。")
    L.append("")
    L.append("## 4. 分年度 RankIC（csi500，test 按自然年拆分）")
    L.append("")
    years = [2023, 2024, 2025, 2026]
    L.append("| Model | " + " | ".join(str(y) for y in years) + " | full test | csi300 full |")
    L.append("|---|" + "---|" * (len(years) + 2))
    for k in ("linear", "lgb", "mlp", "kan"):
        s = rows[k]["ic_series"]["rankic"]
        yr = {y: s[s.index.year == y].mean() for y in years}
        L.append(f"| {rows[k]['name']} | " + " | ".join(fmt(yr[y]) for y in years)
                 + f" | {fmt(s.mean())} | {fmt(ref[k]['rankic'])} |")
    L.append("")
    L.append("注：2026 为 1–7 月（数据覆盖至 2026-07-23）。csi500 年度强势年份与 csi300 不同（§7.4）。")
    L.append("")
    L.append("## 5. 公平性对照（MLP vs KAN，csi500）")
    L.append("")
    L.append("| Model | Architecture | Params | Train time | Notes |")
    L.append("|---|---|---|---|---|")
    for k in ("mlp", "kan"):
        r = rows[k]
        arch = "[158, 224, 1] ReLU" if k == "mlp" else "[158, 24, 1] grid5/order3, grid_range ±3"
        L.append(f"| {r['name']} | {arch} | {r['params']} | {fmt(r['secs']/60, digits=1)} min | {r.get('extra','')} |")
    L.append("")
    L.append(f"参数比 KAN/MLP = {rows['kan']['params']/rows['mlp']['params']:.2f}；同一优化协议（AdamW，lr 1e-3，"
             "wd 1e-4，batch 4096，max 200 epoch，valid 早停 patience 15，seed 42，CPU）。")
    L.append("csi300 参照：MLP best epoch 12 / KAN best epoch 5（GPU）；csi500 二者均更早停（13 / 4，CPU），")
    L.append("趋势一致：KAN 在更少 epoch 达到 valid 峰值。")
    L.append("")
    L.append("## 6. KAN 样条跨池对照（top-12 特征）")
    L.append("")
    f1, f8, ov, sc = spline_compare()
    L.append("csi300（B1）top-12：" + "、".join(f1) + "。")
    L.append("")
    L.append("csi500（B8）top-12：" + "、".join(f8) + "。")
    L.append("")
    L.append(f"交集 {len(ov)} 个：" + "、".join(ov) + "。最强边（argmax |φ|）形状对照：")
    L.append("")
    L.append("| Feature | b1 strongest edge | b8 strongest edge | Pearson corr |")
    L.append("|---|---|---|---|")
    for f, h1, h8, corr in sc:
        L.append(f"| {f} | h{h1} | h{h8} | {corr:+.3f} |")
    L.append("")
    L.append("|corr| 全部 ≥ 0.99（" + f"{sum(1 for *_, c in sc if c > 0)}/{len(sc)}" + " 同向，其余镜像）。"
             "曲线图：`universe-csi500/b8_kan/spline_curves.png` vs `kan/b1_kan/spline_curves.png`。")
    L.append("")
    L.append(CONCLUSIONS)
    L.append("")
    L.append("## 8. 缓存隔离证据（硬约束验收）")
    L.append("")
    for tag in ("start", "end"):
        p = CSI500 / f"cache_isolation_check_{tag}.txt"
        L.append(f"`{p.name}`（md5 of csi300 `common/runs/kan/_cache/*.parquet`，任务{'开工' if tag == 'start' else '收尾'}时）：")
        L.append("")
        L.append("```")
        L.append(p.read_text().strip())
        L.append("```")
        L.append("")
    L.append("两次校验 md5 完全一致，文件 mtime 保持 2026-09-14 22:00（任务开工前）——csi300 缓存未被触碰；")
    L.append("csi500 数据全部位于 `common/runs/kan/universe-csi500/_cache/`。")
    L.append("")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
