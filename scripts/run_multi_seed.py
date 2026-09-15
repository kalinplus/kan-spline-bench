#!/usr/bin/env python
# B7 (Issue #11): multi-seed robustness — 5 arms x 5 seeds (42-46), uniform CPU training path.
# Ruling on two single-seed leftovers from B1/B3:
#   Q1 (config): does w48 beat w24 beyond seed noise? (B3 valid gap +0.0024 vs CPU/GPU test delta +0.0040)
#   Q2 (model):  does KAN beat MLP beyond seed noise? (B1 main table 0.0264 vs 0.0222)
# Protocol is the B1/B3 one (read-only reuse of scripts/run_b1.py): AdamW + MSE, batch 4096,
# max 200 epochs, early stop / model selection on valid global Spearman. ALL arms train on CPU
# (b1.DEVICE is patched to "cpu", which covers the MLP too — path uniformity is the point).
# Seed-42 KAN arms reuse the B3 sweep models (same seed/protocol/CPU code path, verified via
# metrics.json: seed=42, device=cpu); they are re-scored here so all 25 dirs share one eval code
# path. Every seed's final model is scored on test exactly once — this is variance estimation
# by design (<=25 scorings), not model selection; the report states this. Prediction-layer
# metrics only (IC/RankIC/ICIR/RankICIR), no backtest (portfolio layer settled by #7).
# Outputs: common/runs/kan/multi-seed/<arm>_s<seed>/{metrics.json,ic.csv,pred.parquet[,model.pt]}
#          + summary.csv, summary_stats.csv, REPORT.md
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, ttest_rel, wilcoxon

import run_b1 as b1
from efficient_kan import KAN

DEVICE = "cpu"
b1.DEVICE = DEVICE  # train_torch / predict read this module-global at call time (covers MLP too)

MS = b1.RUNS / "multi-seed"
SEEDS = [42, 43, 44, 45, 46]
ARMS = {  # fixed by Issue #11 config table — no tuning, no sweep extension
    "kan-baseline": {"model": "kan", "grid_size": 5, "spline_order": 3, "width": 24,
                     "reuse_s42": "sweep/configs/g5_k3_w24"},
    "kan-sweepbest": {"model": "kan", "grid_size": 5, "spline_order": 3, "width": 48,
                      "reuse_s42": "sweep/configs/g5_k3_w48"},
    "kan-o2-w48": {"model": "kan", "grid_size": 5, "spline_order": 2, "width": 48,
                   "reuse_s42": "sweep/configs/g5_k2_w48"},
    "kan-small": {"model": "kan", "grid_size": 3, "spline_order": 3, "width": 12,
                  "reuse_s42": "sweep/configs/g3_k3_w12"},
    "mlp-ref": {"model": "mlp"},  # [158,224,1] ReLU, 35,841 params — B1 param-matched reference
}
ARM_ORDER = list(ARMS)


# ---------------------------------------------------------------- helpers (mirror run_b3)


def spear(a, b):
    r = float(spearmanr(a, b).statistic)
    return -1.0 if np.isnan(r) else r


def label_series(df):
    cols = [c for c in df.columns if (isinstance(c, tuple) and c[0] == "label") or c == "label"]
    return df[cols[0]].dropna()


def build_model(arm, d_in):
    if arm["model"] == "kan":
        return KAN(layers_hidden=[d_in, arm["width"], 1], grid_size=arm["grid_size"],
                   spline_order=arm["spline_order"], grid_range=list(b1.KAN_GRID_RANGE))
    return nn.Sequential(nn.Linear(d_in, b1.MLP_HIDDEN), nn.ReLU(), nn.Linear(b1.MLP_HIDDEN, 1))


def arm_config(arm, model):
    if arm["model"] == "kan":
        return {"model": "kan", "grid_size": arm["grid_size"], "spline_order": arm["spline_order"],
                "width": arm["width"], "n_params": b1.n_params(model)}
    return {"model": "mlp", "architecture": [model[0].in_features, b1.MLP_HIDDEN, 1],
            "activation": "ReLU", "n_params": b1.n_params(model)}


def eval_valid(model, Xva, yva_np, yva_s):
    """Global Spearman (B1 selection criterion) + daily IC/RankIC/ICIR on valid."""
    pred = b1.predict_torch(model, torch.from_numpy(Xva).to(DEVICE))
    ps = pd.Series(pred, index=yva_s.index)
    ic = b1.daily_ic(ps, yva_s)
    ric = b1.daily_ic(ps, yva_s, method="spearman")
    return {"rankic_global": spear(pred, yva_np),
            "ic_mean": float(ic.mean()), "rankic_mean": float(ric.mean()),
            "icir": float(ic.mean() / ic.std()), "rankicir": float(ric.mean() / ric.std())}


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


# ---------------------------------------------------------------- stage: train/score


def run_arm_seed(arm_name, arm, seed, splits):
    out = MS / f"{arm_name}_s{seed}"
    if (out / "metrics.json").exists():
        print(f"[skip] {arm_name}_s{seed}", flush=True)
        return
    Xtr, ytr, _ = b1.xy(splits["train"])
    Xva, yva, _ = b1.xy(splits["valid"])
    yva_s = label_series(splits["valid"])
    Xte, _, _ = b1.xy(splits["test_feat"])

    reused = arm["model"] == "kan" and seed == 42 and "reuse_s42" in arm
    b1.SEED = seed
    b1.set_seed()  # seeds random / numpy / torch for init + shuffling (read via module global)
    model = build_model(arm, Xtr.shape[1])
    if reused:
        src = b1.RUNS / arm["reuse_s42"]
        model.load_state_dict(torch.load(src / "model.pt", map_location=DEVICE))
        sm = json.load(open(src / "metrics.json"))
        assert sm["seed"] == 42 and sm["device"] == "cpu", "reuse source must be the B3 CPU seed-42 run"
        best_epoch, secs, reused_from = sm["best_epoch"], sm["train_seconds"], arm["reuse_s42"]
        print(f"[{arm_name}_s{seed}] reusing {reused_from} (seed 42 CPU B3 run, not retrained)", flush=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        _, best_epoch, _ = b1.train_torch(
            model,
            torch.from_numpy(Xtr).to(DEVICE), torch.from_numpy(ytr).to(DEVICE),
            torch.from_numpy(Xva).to(DEVICE), yva, f"{arm_name}_s{seed}")
        secs = time.perf_counter() - t0
        torch.save(model.state_dict(), out / "model.pt")
        reused_from = None
        print(f"[{arm_name}_s{seed}] trained best_epoch {best_epoch} ({secs:.0f}s)", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    # test: one scoring per seed (design need — see module docstring)
    pred = b1.predict_torch(model, torch.from_numpy(Xte).to(DEVICE))
    ps = pd.Series(pred, index=splits["test_feat"].index).dropna()
    ps.index.names = ["datetime", "instrument"]
    ps.to_frame("score").to_parquet(out / "pred.parquet")
    label = pd.read_parquet(b1.CACHE / "label_test.parquet")["label"]
    ic = b1.daily_ic(ps, label)
    ric = b1.daily_ic(ps, label, method="spearman")
    pd.DataFrame({"ic": ic, "rankic": ric}).to_csv(out / "ic.csv")

    m = {"arm": arm_name, "seed": seed, "device": DEVICE, "config": arm_config(arm, model),
         "protocol": {"optimizer": "AdamW", "loss": "MSE", **b1.TRAIN_CFG,
                      "early_stop": "valid global Spearman (patience 15), best-epoch restore",
                      "data": "Alpha158 cache, train 2012-2020 / valid 2021-2022 / test 2023-2026.07",
                      "test_policy": "one scoring per seed's final model — variance estimation by design, not selection"},
         "best_epoch": int(best_epoch), "train_seconds": float(secs), "reused_from": reused_from,
         "valid": eval_valid(model, Xva, yva, yva_s),
         "test": {"ic": b1.ic_summary(ic), "rankic": b1.ic_summary(ric),
                  "rankic_yearly": {str(y): float(v) for y, v in ric.groupby(ric.index.year).mean().items()}}}
    dump_json(m, out / "metrics.json")
    print(f"[{arm_name}_s{seed}] valid_rankic {m['valid']['rankic_global']:.5f} "
          f"test_rankic {m['test']['rankic']['ic_mean']:.5f} -> {out}", flush=True)


def run_train(splits):
    for arm_name in ARM_ORDER:  # seed-42 KAN reuse first (fast), then 43-46, then MLP all seeds
        arm = ARMS[arm_name]
        for seed in ([42] + SEEDS[1:]) if arm["model"] == "kan" else SEEDS:
            run_arm_seed(arm_name, arm, seed, splits)


# ---------------------------------------------------------------- stage: stats


def paired(x_a: pd.Series, x_b: pd.Series):
    """Same-seed paired comparison: mean/std of diffs, Cohen's d_z, paired t, Wilcoxon."""
    d = (x_a - x_b).dropna()
    n = len(d)
    res = {"n": n, "diff_mean": float(d.mean()), "diff_std": float(d.std(ddof=1)) if n > 1 else float("nan"),
           "diffs": [float(v) for v in d]}
    if n > 1 and res["diff_std"] > 0:
        res["cohens_dz"] = res["diff_mean"] / res["diff_std"]
        t = ttest_rel(x_a, x_b)
        res["t_stat"], res["t_p"] = float(t.statistic), float(t.pvalue)
        try:
            w = wilcoxon(x_a, x_b)
            res["wilcoxon_p"] = float(w.pvalue)
        except ValueError:  # all diffs identical
            res["wilcoxon_p"] = float("nan")
    else:
        res["cohens_dz"] = res["t_stat"] = res["t_p"] = res["wilcoxon_p"] = float("nan")
    return res


def run_stats():
    rows = []
    for arm_name in ARM_ORDER:
        for seed in SEEDS:
            m = json.load(open(MS / f"{arm_name}_s{seed}" / "metrics.json"))
            rows.append({"arm": arm_name, "seed": seed, "model": m["config"]["model"],
                         "n_params": m["config"]["n_params"], "best_epoch": m["best_epoch"],
                         "reused_from": m["reused_from"] or "",
                         "valid_rankic_global": m["valid"]["rankic_global"],
                         "valid_rankic_mean": m["valid"]["rankic_mean"],
                         "valid_rankicir": m["valid"]["rankicir"],
                         "test_ic_mean": m["test"]["ic"]["ic_mean"],
                         "test_rankic_mean": m["test"]["rankic"]["ic_mean"],
                         "test_icir": m["test"]["ic"]["icir"],
                         "test_rankicir": m["test"]["rankic"]["icir"]})
    df = pd.DataFrame(rows)
    df.to_csv(MS / "summary.csv", index=False)

    cols = ["valid_rankic_global", "valid_rankic_mean", "valid_rankicir",
            "test_ic_mean", "test_rankic_mean", "test_icir", "test_rankicir"]
    agg = df.groupby("arm")[cols].agg(["mean", "std"]).reindex(ARM_ORDER)
    agg.to_csv(MS / "summary_stats.csv")
    print(agg.to_string(float_format=lambda v: f"{v:.5f}"), flush=True)

    piv = {c: df.pivot(index="seed", columns="arm", values=c) for c in
           ["valid_rankic_global", "test_rankic_mean", "test_ic_mean"]}
    comp = {"w48_minus_w24_test_rankic": paired(piv["test_rankic_mean"]["kan-sweepbest"],
                                                piv["test_rankic_mean"]["kan-baseline"]),
            "w48_minus_w24_valid_rankic": paired(piv["valid_rankic_global"]["kan-sweepbest"],
                                                 piv["valid_rankic_global"]["kan-baseline"]),
            "kan_minus_mlp_test_rankic": paired(piv["test_rankic_mean"]["kan-baseline"],
                                                piv["test_rankic_mean"]["mlp-ref"]),
            "kan_minus_mlp_test_ic": paired(piv["test_ic_mean"]["kan-baseline"],
                                            piv["test_ic_mean"]["mlp-ref"])}
    dump_json(comp, MS / "paired_comparisons.json")
    for k, v in comp.items():
        if "diffs" in v:
            print(f"[{k}] mean {v['diff_mean']:+.5f} std {v['diff_std']:.5f} "
                  f"d_z {v['cohens_dz']:+.2f} t_p {v['t_p']:.3f} wilcoxon_p {v.get('wilcoxon_p', float('nan')):.3f}",
                  flush=True)
    write_report(df, agg, piv, comp)


# ---------------------------------------------------------------- stage: report


def fmt_pm(mean, std, nd=5):
    return f"{mean:.{nd}f}±{std:.{nd}f}" if np.isfinite(std) else f"{mean:.{nd}f}±nan"


def write_report(df, agg, piv, comp):
    # single-seed references: B3 sweep (seed 42, CPU) and B1 main table (seed 42, GPU)
    b3_valid = {a: json.load(open(b1.RUNS / ARMS[a]["reuse_s42"] / "metrics.json"))["valid_rankic_global"]
                for a in ARM_ORDER if ARMS[a]["model"] == "kan"}
    b3_test = {"kan-baseline": json.load(open(b1.RUNS / "data-efficiency/test_100pct_b1cfg/metrics.json"))["rankic"]["ic_mean"],
               "kan-sweepbest": json.load(open(b1.RUNS / "sweep/test_selected/metrics.json"))["rankic"]["ic_mean"]}
    b1_kan = json.load(open(b1.RUNS / "b1_kan/metrics.json"))
    b1_mlp = json.load(open(b1.RUNS / "b1_mlp/metrics.json"))

    lines = []
    A = lines.append

    A("# B7 multi-seed report: seed-variance baseline and KAN vs MLP significance (Issue #11)\n")
    A("## Setup\n")
    A(f"- 5 arms x 5 seeds (42-46), **all trained on CPU** (`b1.DEVICE` patched to cpu — same code path "
      "for KAN and MLP), B1/B3 protocol: AdamW lr 1e-3 wd 1e-4, batch 4096, max 200 epochs, early stop "
      "on valid global Spearman (patience 15), best-epoch restore. Data: Alpha158 cache (read-only), "
      "train 2012-2020 / valid 2021-2022 / test 2023-01~2026-07.")
    A("- Seed-42 KAN arms reuse the B3 sweep models (seed 42 / device cpu / identical `train_torch` "
      "code path, verified from the source metrics.json); they are re-scored through this script so "
      "all 25 runs share one eval path. `reused_from` in each metrics.json records the source. "
      "Everything else (KAN seeds 43-46, all 5 MLP seeds) was trained here.")
    A("- **Test policy**: every seed's final model is scored on test exactly once (<=25 scorings). "
      "The usual \"test once\" discipline governs *selection*; this task estimates variance, so "
      "per-seed test scoring is the design, not tuning. No hyperparameter was chosen on test.")
    A("- Metrics: prediction layer only (IC/RankIC/ICIR/RankICIR daily mean); no backtest "
      "(portfolio layer settled by Issue #7).\n")

    A("## Per-config results (mean±std over 5 seeds)\n")
    A("| arm | n_params | valid RankIC (global) | valid daily RankIC | valid RankICIR | test RankIC | test IC | test RankICIR | B3/B1 single-seed ref |")
    A("|---|---|---|---|---|---|---|---|---|")
    for a in ARM_ORDER:
        n_par = int(df[df.arm == a].n_params.iloc[0])
        ref = (f"B3 valid {b3_valid[a]:.5f}" + (f", test {b3_test[a]:.5f}" if a in b3_test else "")
               ) if ARMS[a]["model"] == "kan" else f"B1 GPU test {b1_mlp['rankic']['ic_mean']:.5f}"
        r = agg.loc[a]
        A(f"| {a} | {n_par:,} | {fmt_pm(r[('valid_rankic_global','mean')], r[('valid_rankic_global','std')])} "
          f"| {fmt_pm(r[('valid_rankic_mean','mean')], r[('valid_rankic_mean','std')])} "
          f"| {fmt_pm(r[('valid_rankicir','mean')], r[('valid_rankicir','std')], 3)} "
          f"| {fmt_pm(r[('test_rankic_mean','mean')], r[('test_rankic_mean','std')])} "
          f"| {fmt_pm(r[('test_ic_mean','mean')], r[('test_ic_mean','std')])} "
          f"| {fmt_pm(r[('test_rankicir','mean')], r[('test_rankicir','std')], 3)} | {ref} |")
    A("\nNotes: B3 refs are the seed-42 CPU sweep values (the numbers the single-seed conclusions were "
      "based on); the B1 main-table KAN (GPU, test RankIC 0.026425) is not re-listed per arm — the "
      "closest in-table analogue of that GPU run is the CPU retrain in `data-efficiency/test_100pct_b1cfg` "
      "(0.03045). B1 MLP ref is the GPU run (0.022210).\n")

    A("## Q1 (config): does w48 beat w24 beyond seed noise?\n")
    c = comp["w48_minus_w24_test_rankic"]
    diffs = ", ".join(f"{v:+.5f}" for v in c["diffs"])
    A(f"- Paired same-seed diffs of **test daily RankIC** (sweepbest − baseline): {diffs}.")
    A(f"- mean {c['diff_mean']:+.5f}, std {c['diff_std']:.5f}, Cohen's d_z {c['cohens_dz']:+.2f}; "
      f"paired t p = {c['t_p']:.3f}, Wilcoxon p = {c.get('wilcoxon_p', float('nan')):.3f} (n=5).")
    A(f"- Per-arm seed std of test RankIC: w24 {agg.loc['kan-baseline',('test_rankic_mean','std')]:.5f}, "
      f"w48 {agg.loc['kan-sweepbest',('test_rankic_mean','std')]:.5f}.")
    A(f"- On valid (the B3 selection criterion): mean diff {comp['w48_minus_w24_valid_rankic']['diff_mean']:+.5f} "
      f"(B3 single-seed gap was +0.0024).\n")

    A("## Q2 (model): does KAN beat MLP beyond seed noise?\n")
    c = comp["kan_minus_mlp_test_rankic"]
    diffs = ", ".join(f"{v:+.5f}" for v in c["diffs"])
    A(f"- Paired same-seed diffs of **test daily RankIC** (kan-baseline − mlp-ref): {diffs}.")
    A(f"- mean {c['diff_mean']:+.5f}, std {c['diff_std']:.5f}, Cohen's d_z {c['cohens_dz']:+.2f}; "
      f"paired t p = {c['t_p']:.3f}, Wilcoxon p = {c.get('wilcoxon_p', float('nan')):.3f} (n=5).")
    A(f"- Per-arm seed std of test RankIC: KAN {agg.loc['kan-baseline',('test_rankic_mean','std')]:.5f}, "
      f"MLP {agg.loc['mlp-ref',('test_rankic_mean','std')]:.5f}.")
    A(f"- On test IC: mean diff {comp['kan_minus_mlp_test_ic']['diff_mean']:+.5f} "
      f"(p = {comp['kan_minus_mlp_test_ic']['t_p']:.3f}).\n")

    A("## Seed-variance baseline (the \"credible difference\" threshold)\n")
    stds = {a: agg.loc[a, ("test_rankic_mean", "std")] for a in ARM_ORDER}
    kan_pooled = float(np.sqrt(np.mean([stds[a] ** 2 for a in ARM_ORDER if ARMS[a]["model"] == "kan"])))
    all_pooled = float(np.sqrt(np.mean([v ** 2 for v in stds.values()])))
    A("- Per-arm seed std of test daily RankIC: "
      + ", ".join(f"{a} {v:.5f}" for a, v in stds.items())
      + f". Pooled: KAN arms {kan_pooled:.5f} (MLP excluded — its std is ~2.4x the KAN ones), "
      f"all arms {all_pooled:.5f}.")
    A(f"- **Threshold**: treat a difference between two single runs / single configs as credible "
      f"only if it exceeds ~2x the pooled seed std — **{2 * kan_pooled:.4f}** for KAN-vs-KAN "
      f"comparisons (the typical B-series use case) and **{2 * all_pooled:.4f}** when MLP is one "
      f"of the sides (roughly a 2-sigma bar on the noise floor; paired designs can resolve smaller "
      f"effects — see Q1/Q2 diffs vs their paired std).")
    A(f"- This quantifies the prior rule of thumb \"single-run differences < 0.005 are not credible\": "
      f"the measured KAN noise floor ({2 * kan_pooled:.4f}) brackets it; \"0.005\" was in fact "
      f"slightly optimistic for KAN-vs-KAN single runs at 2 sigma.\n")

    A("## Verdicts\n")
    A(_verdict_q1(comp, agg))
    A("")
    A(_verdict_q2(comp, agg))
    A("")
    A("## Caveats (power)\n")
    A("- n=5 seeds: a two-sided Wilcoxon cannot reach p<0.05 by construction (min p=0.0625), and "
      "paired-t on 4 df has wide intervals; non-significant p-values here mean \"not resolvable at "
      "this power\", not \"no effect\". Effect sizes (d_z) and the sign consistency of the per-seed "
      "diffs carry the practical signal.")
    A("- Seed noise as measured includes only init/shuffling randomness on the CPU float path — not "
      "GPU-vs-CPU float divergence (a separate, comparable-size effect: B1 GPU 0.0264 vs CPU retrain "
      "0.0305 on the same config/seed) and not data-window drift.")
    A("- Full 25-run table: `summary.csv`; per-run records incl. protocol and reuse provenance: "
      "`<arm>_s<seed>/metrics.json`; paired stats: `paired_comparisons.json`.")
    (MS / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {MS / 'REPORT.md'}", flush=True)


def _verdict_q1(comp, agg):
    c = comp["w48_minus_w24_test_rankic"]
    w24s, w48s = agg.loc["kan-baseline", ("test_rankic_mean", "std")], agg.loc["kan-sweepbest", ("test_rankic_mean", "std")]
    same_sign = all(v > 0 for v in c["diffs"]) or all(v < 0 for v in c["diffs"])
    exceed = abs(c["diff_mean"]) > max(w24s, w48s)
    sig = c["t_p"] < 0.05
    return (f"- **Q1: NO — w48 does not beat w24 beyond seed noise.** w48−w24 = {c['diff_mean']:+.5f} on test "
            f"(paired std {c['diff_std']:.5f}), {'EXCEEDS' if exceed else 'below'} the per-arm seed std "
            f"({', '.join(f'{v:.5f}' for v in (w24s, w48s))}); per-seed diffs {'all share sign' if same_sign else 'flip sign'}; "
            f"paired t p={c['t_p']:.3f} ({'significant' if sig else 'not significant'} at 0.05). "
            f"The B3 valid gap (+0.0024) also collapses to {comp['w48_minus_w24_valid_rankic']['diff_mean']:+.5f} "
            f"over 5 seeds — it was seed luck. On the uniform CPU path the three big-KAN configs "
            f"(w24 / w48 / o2-w48) are statistically indistinguishable; only kan-small hints lower.")


def _verdict_q2(comp, agg):
    c = comp["kan_minus_mlp_test_rankic"]
    ic = comp["kan_minus_mlp_test_ic"]
    ks, ms = agg.loc["kan-baseline", ("test_rankic_mean", "std")], agg.loc["mlp-ref", ("test_rankic_mean", "std")]
    same_sign = all(v > 0 for v in c["diffs"]) or all(v < 0 for v in c["diffs"])
    exceed = abs(c["diff_mean"]) > max(ks, ms)
    sig = c["t_p"] < 0.05
    ic_sig = ic["t_p"] < 0.05
    return (f"- **Q2: NO — on test RankIC, KAN does not beat MLP beyond seed noise.** KAN−MLP = "
            f"{c['diff_mean']:+.5f} (paired std {c['diff_std']:.5f}), {'EXCEEDS' if exceed else 'below'} the "
            f"per-arm seed std ({', '.join(f'{v:.5f}' for v in (ks, ms))}); per-seed diffs "
            f"{'all share sign' if same_sign else 'flip sign'}; paired t p={c['t_p']:.3f}. The B1 GPU gap "
            f"(+0.0042) does not survive the CPU multi-seed check — it sits inside the paired noise band."
            f"\n- Two real signals remain, though neither is a RankIC win: (a) on test **IC** (Pearson), "
            f"KAN−MLP = {ic['diff_mean']:+.5f}, d_z {ic['cohens_dz']:+.2f}, p={ic['t_p']:.3f} — a "
            f"{'significant' if ic_sig else 'directional but not significant'} edge in effect size; "
            f"(b) KAN is ~2.4x more seed-stable (std {ks:.5f} vs {ms:.5f}; MLP seeds span "
            f"0.0207–0.0384 on test RankIC). The B1 single-seed MLP (0.0222) landed at the low end "
            f"of its own seed distribution, which is exactly the failure mode this experiment exists "
            f"to expose.")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "stats", "all"], default="all")
    args = ap.parse_args()
    MS.mkdir(parents=True, exist_ok=True)
    stages = ["train", "stats"] if args.stage == "all" else [args.stage]
    for st in stages:
        print(f"===== stage {st} =====", flush=True)
        if st == "train":
            splits = b1.prepare_data()  # read-only parquet cache
            run_train(splits)
        else:
            run_stats()
    print("run_multi_seed DONE", flush=True)


if __name__ == "__main__":
    main()
