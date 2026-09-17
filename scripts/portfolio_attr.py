#!/usr/bin/env python
# B6 (Issue #7): portfolio-layer attribution & backtest sensitivity on frozen B1 predictions.
# Zero training: reads test-segment predictions of b1_{linear,mlp,kan} + s1b LGB, reruns
# backtests / statistics only. Four stages, all idempotent:
#   stats  - pre-cost top-k returns, decile long-short spread w/ CI, paired excess CIs (no qlib)
#   attr   - limit-hit attribution (qlib closes) + drag decomposition cost vs friction
#   sweep  - backtest sensitivity: topk {10,30,50} x drop {1,5} + weekly/monthly resampled signals
#   summary- collect all CSVs into results-ready markdown
# Outputs land in common/runs/kan/portfolio-attr/ (physical files in agentic-feature-mining,
# never committed there). Backtest config (costs, benchmark, limits, deal price) mirrors
# scripts/run_b1.py evaluate_dump exactly; only topk / n_drop / signal frequency vary.
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

import qlib
from qlib.constant import REG_CN
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data import D

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "common" / "runs"
OUT = RUNS / "kan" / "portfolio-attr"
MODELS = {  # display order = B1 main table
    "linear": RUNS / "kan" / "b1_linear",
    "lgb": RUNS / "s1b_alpha158_lgb",
    "mlp": RUNS / "kan" / "b1_mlp",
    "kan": RUNS / "kan" / "b1_kan",
}
MODEL_NAMES = {"linear": "Linear/Ridge", "lgb": "LightGBM (S1b)", "mlp": "MLP", "kan": "KAN"}
LABEL_CACHE = RUNS / "kan" / "_cache" / "label_test.parquet"
DATA_DIR = "/data1/wujunxi/kailin/data/qlib_data/cn_data"
SEG = ("2023-01-01", "2026-07-23")
EXCHANGE_KWARGS = {
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
    "trade_unit": 100,
}
# sensitivity grid; (30, 1, "daily") is the B1 main config -> reuse frozen b1 reports, never rerun
SWEEP = [(k, d, "daily") for k in (10, 30, 50) for d in (1, 5)] + [(30, 1, "weekly"), (30, 1, "monthly")]
TOPK_ATTR = 30            # ideal-portfolio width for limit-hit / drag attribution (B1 main config)
LIMIT_TH = 0.095          # |daily change| counted as limit-hit (matches exchange limit_threshold)
BOOT_N = 10000
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- shared helpers


def load_preds():
    preds = {}
    for key, path in MODELS.items():
        s = pd.read_parquet(path / "pred.parquet")["score"].dropna()
        s.index = s.index.set_names(["datetime", "instrument"])
        preds[key] = s.sort_index()
    label = pd.read_parquet(LABEL_CACHE)["label"]
    label.index = label.index.set_names(["datetime", "instrument"])
    return preds, label


def ideal_topk_returns(pred: pd.Series, label: pd.Series, k: int) -> pd.Series:
    """Per signal-day equal-weight mean label of the score top-k (label-aligned universe)."""
    df = pd.concat([pred.rename("p"), label.rename("l")], axis=1).dropna()
    return df.groupby(level=0, group_keys=False).apply(
        lambda g: g.nlargest(k, "p")["l"].mean(), include_groups=False
    )


def per_year(series: pd.Series, fn) -> dict:
    out = {f"y{y}": float(fn(g)) for y, g in series.groupby(series.index.year)}
    out["full"] = float(fn(series))
    return out


def bootstrap_ci(x: np.ndarray, n: int = BOOT_N) -> tuple:
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def resample_signal(pred: pd.Series, freq: str) -> pd.Series:
    """Low-frequency rebalance: keep one signal cross-section per period (last trading day;
    first period uses its first day so the series starts without look-ahead), then carry that
    frozen cross-section forward on every trading day until the next signal date."""
    dates = pred.index.get_level_values("datetime").unique().sort_values()
    per = dates.to_period(freq)
    sig = pd.Series(dates.to_numpy(), index=per).groupby(level=0).max()
    sig.iloc[0] = dates[per == per[0]].min()  # bootstrap the very first period
    sig_sorted = np.sort(sig.to_numpy())
    pos = np.searchsorted(sig_sorted, dates.to_numpy(), side="right") - 1
    pos = np.clip(pos, 0, len(sig_sorted) - 1)
    by_day = {d: g.droplevel("datetime") for d, g in pred.groupby(level="datetime")}
    frames = []
    for day, sd in zip(dates, sig_sorted[pos]):
        g = by_day[pd.Timestamp(sd)]
        frames.append(pd.Series(g.to_numpy(),
                                index=pd.MultiIndex.from_product([[day], g.index],
                                                                 names=["datetime", "instrument"])))
    return pd.concat(frames).sort_index()


def run_backtest(pred: pd.Series, topk: int, n_drop: int):
    strategy = TopkDropoutStrategy(topk=topk, n_drop=n_drop, signal=pred)
    report, _ = backtest_daily(start_time=SEG[0], end_time=SEG[1], strategy=strategy,
                               benchmark="SH000300", exchange_kwargs=EXCHANGE_KWARGS)
    return report


def perf_from_report(report: pd.DataFrame) -> dict:
    perf = {k: float(v) for k, v in risk_analysis(report["return"] - report["bench"]).iloc[:, 0].items()}
    perf["annualized_return"] = float(risk_analysis(report["return"]).iloc[:, 0]["annualized_return"])
    perf["excess_mean_bp"] = float((report["return"] - report["bench"]).mean() * 1e4)
    perf["cost_bp"] = float(report["cost"].mean() * 1e4)
    perf["daily_turnover_mean"] = float(report["turnover"].mean())
    perf["n_days"] = int(len(report))
    return perf


# ---------------------------------------------------------------- stage: stats (no qlib)


def stage_stats():
    OUT.mkdir(parents=True, exist_ok=True)
    preds, label = load_preds()

    # 1) pre-cost top-k returns: equal-weight top-k by score, mean label, bp/d + annualized
    rows = []
    for key, pred in preds.items():
        for k in (10, 30, 50):
            r = ideal_topk_returns(pred, label, k) * 1e4
            rec = {"model": key, "model_name": MODEL_NAMES[key], "topk": k}
            rec.update({f"{c}_bp": v for c, v in per_year(r, np.mean).items()})
            rec["full_ann_pct"] = float(r.mean() * 252 / 100)
            rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "stats_precost_topk.csv", index=False)

    # 2) decile long-short spread (frozen B1 layered.csv): mean bp/d, t-stat, 95% CI
    rows = []
    for key, path in MODELS.items():
        ls = pd.read_csv(path / "layered.csv", index_col=0, parse_dates=True)["long_short"] * 1e4
        x = ls.to_numpy()
        t = float(sstats.ttest_1samp(x, 0).statistic)
        p = float(sstats.ttest_1samp(x, 0).pvalue)
        lo, hi = bootstrap_ci(x)
        rec = {"model": key, "model_name": MODEL_NAMES[key]}
        rec.update({f"{c}_bp": v for c, v in per_year(ls, np.mean).items()})
        rec.update({"full_ann_pct": float(ls.mean() * 252 / 100), "t_stat": t, "p_value": p,
                    "ci95_lo_bp": lo, "ci95_hi_bp": hi})
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "stats_decile_spread.csv", index=False)

    # 3) excess return CIs: per-model daily excess + paired differences (bootstrap + t)
    exc = {}
    for key, path in MODELS.items():
        rep = pd.read_csv(path / "backtest_report.csv", index_col=0, parse_dates=True)
        exc[key] = (rep["return"] - rep["bench"]) * 1e4  # bp/d
    rows = []
    for key, e in exc.items():
        x = e.to_numpy()
        lo, hi = bootstrap_ci(x)
        rec = {"pair": key, "model_name": MODEL_NAMES[key]}
        rec.update({f"{c}_bp": v for c, v in per_year(e, np.mean).items()})
        rec.update({"t_p": float(sstats.ttest_1samp(x, 0).pvalue),
                    "ci95_lo_bp": lo, "ci95_hi_bp": hi})
        rows.append(rec)
    pairs = [("kan", "linear"), ("kan", "mlp"), ("kan", "lgb"), ("linear", "mlp"),
             ("linear", "lgb"), ("mlp", "lgb")]
    for a, b in pairs:
        d = (exc[a] - exc[b]).dropna()
        x = d.to_numpy()
        lo, hi = bootstrap_ci(x)
        rec = {"pair": f"{a}-{b}", "model_name": f"{MODEL_NAMES[a]} − {MODEL_NAMES[b]}"}
        rec.update({f"{c}_bp": v for c, v in per_year(d, np.mean).items()})
        rec.update({"t_p": float(sstats.ttest_1samp(x, 0).pvalue),
                    "ci95_lo_bp": lo, "ci95_hi_bp": hi})
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "stats_excess_ci.csv", index=False)
    print("[stats] done -> stats_precost_topk.csv / stats_decile_spread.csv / stats_excess_ci.csv", flush=True)


# ---------------------------------------------------------------- stage: attr (limit-hit + drag)


def stage_attr():
    OUT.mkdir(parents=True, exist_ok=True)
    preds, label = load_preds()
    qlib.init(provider_uri=DATA_DIR, region=REG_CN)

    # ---- limit-hit attribution: ideal top30 holdings, |chg| on the deal day (s+1)
    codes = sorted({i for p in preds.values() for i in p.index.get_level_values("instrument").unique()})
    close = D.features(codes, ["$close"], start_time=SEG[0], end_time=SEG[1])
    close.columns = ["close"]
    if close.index.names[0] != "datetime":  # D.features order can vary with cache state
        close = close.swaplevel()
    close = close.sort_index()
    chg = close.groupby(level="instrument")["close"].pct_change(fill_method=None).rename("chg").sort_index()
    days = sorted(chg.index.get_level_values("datetime").unique())
    next_map = dict(zip(days[:-1], days[1:]))

    rows = []
    for key, pred in preds.items():
        tops = {d: pred.xs(d, level="datetime").nlargest(TOPK_ATTR).index.tolist()
                for d in sorted(pred.index.get_level_values("datetime").unique())}
        for s, ins in tops.items():
            d = next_map.get(s)
            if d is None:
                continue
            c = chg.reindex(pd.MultiIndex.from_product([[d], ins], names=["datetime", "instrument"]))
            rows.append({"model": key, "signal_day": s, "match": float(c.notna().mean()),
                         "hit": float((c.abs() >= LIMIT_TH).mean()),
                         "hit_up": float((c >= LIMIT_TH).mean()),
                         "hit_dn": float((c <= -LIMIT_TH).mean())})
    hit = pd.DataFrame(rows).set_index(["model", "signal_day"])
    assert hit["match"].mean() > 0.9, f"reindex match rate {hit['match'].mean():.2f} — index misaligned"
    agg = hit.groupby(level=0)[["hit", "hit_up", "hit_dn"]].mean() * 100
    agg.columns = [f"{c}_pct" for c in agg.columns]
    yearly = (hit.groupby([hit.index.get_level_values(0), hit.index.get_level_values(1).year])
              [["hit", "hit_up", "hit_dn"]].mean() * 100)
    yearly.columns = [f"{c}_pct" for c in yearly.columns]
    yearly.index = yearly.index.set_names(["model", "year"])
    yearly.reset_index().to_csv(OUT / "limit_hit_yearly.csv", index=False)
    agg.reset_index().to_csv(OUT / "limit_hit_full.csv", index=False)

    # ---- drag decomposition: ideal top30 (label mean, signal day s) vs report return at s+2
    # qlib timing: signal s -> deal close s+1 -> held to s+2 close, i.e. report.return(s+2)
    # equals the realized PnL of the portfolio selected by signal s. drag = ideal - actual,
    # cost from report cost column (deal-day cost charged at s+1 shows up in return(s+2)?
    # no: qlib books trade cost on the deal day's account value; we attribute report.cost(s+2)
    # which is the cost of the s+2 deals -- matching the s->s+2 holding window's own rebalance.
    rows = []
    for key, path in MODELS.items():
        rep = pd.read_csv(path / "backtest_report.csv", index_col=0, parse_dates=True)
        rdays = list(rep.index)
        plus2 = {d0: rdays[i + 2] for i, d0 in enumerate(rdays[:-2])}
        ideal = ideal_topk_returns(preds[key], label, TOPK_ATTR) * 1e4
        for s, iv in ideal.items():
            t = plus2.get(s)
            if t is None or t not in rep.index:
                continue
            r = rep.loc[t]
            rows.append({"model": key, "signal_day": s, "ideal_bp": float(iv),
                         "actual_bp": float(r["return"] * 1e4),
                         "cost_bp": float(r["cost"] * 1e4)})
    drag = pd.DataFrame(rows)
    drag["drag_bp"] = drag["ideal_bp"] - drag["actual_bp"]
    drag["friction_bp"] = drag["drag_bp"] - drag["cost_bp"]
    drag.to_csv(OUT / "drag_daily.csv", index=False)

    cols = ["ideal_bp", "actual_bp", "drag_bp", "cost_bp", "friction_bp"]
    full = drag.groupby("model")[cols].mean().reindex(MODELS)
    full.to_csv(OUT / "drag_full.csv")
    drag["year"] = pd.to_datetime(drag["signal_day"]).dt.year
    yr = drag.groupby(["model", "year"])[cols].mean()
    yr.reindex(pd.MultiIndex.from_product([MODELS, sorted(drag["year"].unique())])).to_csv(OUT / "drag_yearly.csv")
    print("[attr] done -> limit_hit_{full,yearly}.csv / drag_{daily,full,yearly}.csv", flush=True)


# ---------------------------------------------------------------- stage: sweep (backtests)


def stage_sweep():
    OUT.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=DATA_DIR, region=REG_CN)
    preds, _ = load_preds()
    rows = []
    for key, pred in preds.items():
        for topk, drop, freq in SWEEP:
            if (topk, drop, freq) == (30, 1, "daily"):
                # B1 main config: reuse the frozen b1/s1b report
                rep = pd.read_csv(MODELS[key] / "backtest_report.csv", index_col=0, parse_dates=True)
                perf = perf_from_report(rep)
                perf["source"] = "b1"
                rows.append({"model": key, "model_name": MODEL_NAMES[key], "topk": topk,
                             "n_drop": drop, "freq": freq, **perf})
                print(f"[sweep] {key} k30 d1 daily <- b1 (frozen)", flush=True)
                continue
            tag = f"{key}__k{topk}_d{drop}_{freq}"
            d = OUT / "sweep" / tag
            done = d / "summary.json"
            if done.exists():
                perf = json.loads(done.read_text())
                rows.append({"model": key, "model_name": MODEL_NAMES[key], "topk": topk,
                             "n_drop": drop, "freq": freq, **perf})
                print(f"[sweep] {tag} cached", flush=True)
                continue
            d.mkdir(parents=True, exist_ok=True)
            sig = pred if freq == "daily" else resample_signal(pred, "W" if freq == "weekly" else "M")
            if freq != "daily":
                sig.to_frame("score").to_parquet(d / "signal.parquet")
            t0 = time.perf_counter()
            rep = run_backtest(sig, topk, drop)
            rep.to_csv(d / "backtest_report.csv")
            perf = perf_from_report(rep)
            perf["source"] = "b6"
            perf["backtest_seconds"] = round(time.perf_counter() - t0, 1)
            done.write_text(json.dumps(perf, indent=2, default=float))
            rows.append({"model": key, "model_name": MODEL_NAMES[key], "topk": topk,
                         "n_drop": drop, "freq": freq, **perf})
            print(f"[sweep] {tag}: excess {perf['excess_mean_bp']:.2f} bp/d "
                  f"IR {perf['information_ratio']:.2f} turn {perf['daily_turnover_mean']*100:.2f}% "
                  f"({perf['backtest_seconds']}s)", flush=True)
            gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sens_matrix.csv", index=False)
    print("[sweep] done -> sens_matrix.csv", flush=True)


# ---------------------------------------------------------------- stage: summary


def stage_summary():
    pre = pd.read_csv(OUT / "stats_precost_topk.csv")
    dec = pd.read_csv(OUT / "stats_decile_spread.csv")
    ci = pd.read_csv(OUT / "stats_excess_ci.csv")
    hitf = pd.read_csv(OUT / "limit_hit_full.csv")
    hity = pd.read_csv(OUT / "limit_hit_yearly.csv")
    drf = pd.read_csv(OUT / "drag_full.csv", index_col=0)
    dry = pd.read_csv(OUT / "drag_yearly.csv", index_col=[0, 1])
    sens = pd.read_csv(OUT / "sens_matrix.csv")

    L = ["# B6 portfolio attribution summary (generated by scripts/portfolio_attr.py)", ""]

    L += ["## Backtest sensitivity matrix (annualized excess %, IR, turnover %)", "",
          "| Model | topk | drop | freq | Ann. exc % | IR | Exc MDD % | Turnover % | Cost bp/d |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in sens.iterrows():
        L.append(f"| {r['model_name']} | {int(r['topk'])} | {int(r['n_drop'])} | {r['freq']} | "
                 f"{r['mean']*252*100:.2f} | {r['information_ratio']:.2f} | {r['max_drawdown']*100:.1f} | "
                 f"{r['daily_turnover_mean']*100:.2f} | {r['cost_bp']:.2f} |")
    L.append("")

    L += ["## Limit-hit attribution (ideal top30, deal-day |chg| >= 9.5%, % of holdings)", "",
          "| Model | 2023 | 2024 | 2025 | 2026 | full |",
          "|---|---|---|---|---|---|"]
    for key in MODELS:
        cells = []
        for y in (2023, 2024, 2025, 2026):
            v = hity[(hity["model"] == key) & (hity["year"] == y)]["hit_pct"]
            cells.append(f"{v.iloc[0]:.2f}" if len(v) else "n/a")
        f = hitf[hitf["model"] == key]["hit_pct"].iloc[0]
        L.append(f"| {MODEL_NAMES[key]} | " + " | ".join(cells) + f" | {f:.2f} |")
    L.append("")

    L += ["## Drag decomposition (bp/d, ideal top30 -> actual backtest)", "",
          "| Model | ideal | actual | drag | cost | friction |",
          "|---|---|---|---|---|---|"]
    for key in MODELS:
        r = drf.loc[key]
        L.append(f"| {MODEL_NAMES[key]} | {r['ideal_bp']:.2f} | {r['actual_bp']:.2f} | "
                 f"{r['drag_bp']:.2f} | {r['cost_bp']:.2f} | {r['friction_bp']:.2f} |")
    L += ["", "Yearly drag (bp/d):", "",
          "| Model | year | ideal | actual | drag | cost | friction |", "|---|---|---|---|---|---|---|"]
    for key in MODELS:
        for y in dry.loc[key].dropna().index:
            r = dry.loc[(key, y)]
            L.append(f"| {MODEL_NAMES[key]} | {int(y)} | {r['ideal_bp']:.2f} | {r['actual_bp']:.2f} | "
                     f"{r['drag_bp']:.2f} | {r['cost_bp']:.2f} | {r['friction_bp']:.2f} |")
    L.append("")

    L += ["## Pre-cost top-k returns (bp/d, equal-weight, label-aligned)", "",
          "| Model | k=10 | k=30 | k=50 |", "|---|---|---|---|"]
    for key in MODELS:
        cells = []
        for k in (10, 30, 50):
            v = pre[(pre["model"] == key) & (pre["topk"] == k)]["full_bp"]
            cells.append(f"{v.iloc[0]:.2f}")
        L.append(f"| {MODEL_NAMES[key]} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Decile long-short spread (bp/d, 95% bootstrap CI, paired t p)", "",
          "| Model | full | t p | CI95 |", "|---|---|---|---|"]
    for _, r in dec.iterrows():
        L.append(f"| {r['model_name']} | {r['full_bp']:.2f} | {r['p_value']:.3g} | "
                 f"[{r['ci95_lo_bp']:.2f}, {r['ci95_hi_bp']:.2f}] |")
    L.append("")

    L += ["## Excess CIs (bp/d; per-model and paired differences)", "",
          "| Pair | full | t p | CI95 |", "|---|---|---|---|"]
    for _, r in ci.iterrows():
        L.append(f"| {r['pair']} | {r['full_bp']:.2f} | {r['t_p']:.3g} | "
                 f"[{r['ci95_lo_bp']:.2f}, {r['ci95_hi_bp']:.2f}] |")
    L.append("")
    (OUT / "attr_summary.md").write_text("\n".join(L))
    print(f"[summary] -> {OUT / 'attr_summary.md'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stats", "attr", "sweep", "summary"], required=True)
    args = ap.parse_args()
    {"stats": stage_stats, "attr": stage_attr, "sweep": stage_sweep, "summary": stage_summary}[args.stage]()


if __name__ == "__main__":
    main()
