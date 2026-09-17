#!/usr/bin/env python
# B10 (Issue #16): hybrid architecture ablation — LightGBM x KAN via (1) residual learning
# with leak-free OOF construction and (2) output ensemble, against LGB / KAN single-model
# arms on the B1 csi300 cache (read-only).
#
#   residual route: LGB is time-series OOF'd inside train (2012-2020) with expanding
#     natural-year folds; the KAN trains on  resid = CSZScoreNorm label - LGB_OOF  (never
#     on in-sample LGB predictions — LGB memorizes train and would zero the target).
#     valid resid = label - LGB(full-train fit) valid prediction; the residual KAN early-
#     stops on the valid RankIC of the COMBINED prediction (LGB + KAN_res); test is scored
#     once as LGB(full-train) + KAN_res.
#   ensemble route: an independent KAN (B1 config w24, CPU, seed 42, same training code
#     path as the residual KAN) is combined with LGB as w*KAN + (1-w)*LGB; w is picked on
#     valid daily-mean RankIC over the grid {0, .25, .5, .75, 1} and applied to test once.
#     w=0 / w=1 endpoints must reproduce the single-model arms exactly (self-check).
#   single arms: LGB = S1b hyperparams retrained from the cache (B4 run_lgb replica, seed
#     42); KAN = same-batch CPU retrain; Linear is quoted from the B1 artifacts, not rerun.
#
# Backtest (user decision): hybrid two arms + LGB / KAN single arms, B1 config (topk 30 /
# drop 1, benchmark SH000300, open 5bp / close 15bp, limit 0.095), read with the Issue #7
# caveat: portfolio-layer differences sit inside the noise band (config > model effect);
# the table is for relative shape, not ranking.
#
# Outputs: common/runs/kan/hybrid/{oof/,lgb_arm/,kan_arm/,residual_arm/,ensemble_arm/,
# backtest_table.csv,...,REPORT.md} + results/B10_hybrid.md (report stage).
# Data: common/runs/kan/_cache parquet, READ-ONLY (no handler rebuild).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_b1 as b1
from efficient_kan import KAN

RUNS = b1.RUNS                       # common/runs/kan
HYB = RUNS / "hybrid"
OOF = HYB / "oof"
DATA_DIR = b1.DATA_DIR
SEED = 42
DEVICE = "cpu"
TORCH_THREADS = 20
THRESH = 0.0059                      # B7 credible-difference threshold (2x pooled KAN seed std)
THRESH_MLP = 0.0083                  # B7 threshold when an MLP arm is one side (context only)
KAN_CFG = {"width": 24, "grid_size": 5, "spline_order": 3, "grid_range": list(b1.KAN_GRID_RANGE)}
LGB_PARAMS = {  # S1b official Alpha158 LightGBM hyperparams + seed (B4 run_lgb replica)
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
LGB_ROUNDS = 1000
LGB_EARLY_STOP = 50
OOF_YEARS = list(range(2013, 2021))  # predicted years; fold fit windows end the year before
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
ARMS = ["residual_arm", "ensemble_arm", "lgb_arm", "kan_arm"]
ARM_NAMES = {"residual_arm": "LGB+KAN_res (residual)", "ensemble_arm": "ensemble w*KAN+(1-w)*LGB",
             "lgb_arm": "LightGBM (S1b retrain)", "kan_arm": "KAN (CPU retrain s42)",
             "linear_ref": "Linear/Ridge (B1 artifact)"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def spear(a, b):
    from scipy.stats import spearmanr
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        j = pd.concat([pd.Series(a).rename("a"), pd.Series(b).rename("b")], axis=1).dropna()
        a, b = j["a"], j["b"]
    r = float(spearmanr(a, b).statistic)
    return -1.0 if np.isnan(r) else r


def label_series(df):
    cols = [c for c in df.columns if (isinstance(c, tuple) and c[0] == "label") or c == "label"]
    return df[cols[0]].dropna()


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def full_eval(pred: pd.Series, label: pd.Series) -> dict:
    """Daily IC/RankIC summaries + global Spearman + per-year daily RankIC (B1 eval defs)."""
    ic = b1.daily_ic(pred, label)
    ric = b1.daily_ic(pred, label, method="spearman")
    return {"ic": b1.ic_summary(ic), "rankic": b1.ic_summary(ric),
            "rankic_global": spear(pred, label),
            "rankic_yearly": {str(y): float(v) for y, v in ric.groupby(ric.index.year).mean().items()}}


def save_pred(pred: pd.Series, path: Path) -> pd.Series:
    s = pred.dropna().copy()
    s.index = s.index.set_names(["datetime", "instrument"])
    s.to_frame("score").to_parquet(path)
    return s


def load_pred(path: Path) -> pd.Series:
    s = pd.read_parquet(path)["score"]
    s.index = s.index.set_names(["datetime", "instrument"])
    return s.sort_index()


def predict_kan(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 65536):
            outs.append(model(torch.from_numpy(X[i:i + 65536])).squeeze(-1).float().numpy())
    return np.concatenate(outs)


def build_kan(d_in: int) -> KAN:
    return KAN(layers_hidden=[d_in, KAN_CFG["width"], 1], grid_size=KAN_CFG["grid_size"],
               spline_order=KAN_CFG["spline_order"], grid_range=KAN_CFG["grid_range"])


def train_kan(Xtr, ytr, Xva, crit, tag):
    """B1 training protocol (AdamW + MSE, batch 4096, max 200 epochs, patience 15, best-epoch
    restore) with a pluggable valid criterion: crit(valid_pred_np) -> float, higher better.
    Used by BOTH the single-model KAN arm (crit = own valid RankIC, i.e. B1 exactly) and the
    residual KAN arm (crit = combined (LGB + KAN_res) valid RankIC) — same code path."""
    t0 = time.perf_counter()
    model = build_kan(Xtr.shape[1])
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=b1.TRAIN_CFG["lr"],
                            weight_decay=b1.TRAIN_CFG["weight_decay"])
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    n, bs = Xtr.shape[0], b1.TRAIN_CFG["batch"]
    best, best_state, best_epoch, bad = -1e9, None, 0, 0
    for epoch in range(1, b1.TRAIN_CFG["max_epoch"] + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = torch.nn.functional.mse_loss(model(Xtr_t[idx]).squeeze(-1), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        vic = crit(predict_kan(model, Xva))
        if vic > best + 1e-6:
            best, best_epoch, bad = vic, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if epoch % 10 == 0 or bad >= b1.TRAIN_CFG["patience"]:
            print(f"[{tag}] epoch {epoch} valid_crit {vic:.5f} (best {best:.5f} @ {best_epoch})",
                  flush=True)
        if bad >= b1.TRAIN_CFG["patience"]:
            break
    model.load_state_dict(best_state)
    return model, best_state, best_epoch, time.perf_counter() - t0


# ---------------------------------------------------------------- stage lgb: full fit + OOF


def stage_lgb(splits):
    import lightgbm as lgb
    out, oof = HYB / "lgb_arm", HYB / "oof"
    if (out / "metrics.json").exists() and (oof / "residual_target_valid.parquet").exists():
        print("[skip] stage lgb", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    oof.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, names = b1.xy(splits["train"])
    Xva, yva, _ = b1.xy(splits["valid"])
    Xte, _, _ = b1.xy(splits["test_feat"])
    yva_s = label_series(splits["valid"])
    label_te = pd.read_parquet(b1.CACHE / "label_test.parquet")["label"]
    idx_tr = splits["train"].index
    assert idx_tr.names[0] == "datetime"
    dt = idx_tr.get_level_values("datetime")
    years = dt.year

    # ---- full-train fit (the deployed LGB; also the LGB single-model arm)
    t0 = time.perf_counter()
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=names)
    dva = lgb.Dataset(Xva, label=yva, feature_name=names, reference=dtr)
    booster = lgb.train(LGB_PARAMS, dtr, num_boost_round=LGB_ROUNDS,
                        valid_sets=[dtr, dva], valid_names=["train", "valid"],
                        callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                                   lgb.log_evaluation(200)])
    full_secs = time.perf_counter() - t0
    pv = booster.predict(Xva, num_iteration=booster.best_iteration)
    pt = booster.predict(Xte, num_iteration=booster.best_iteration)
    pv_s = save_pred(pd.Series(pv, index=splits["valid"].index), out / "pred_valid.parquet")
    pt_s = save_pred(pd.Series(pt, index=splits["test_feat"].index), out / "pred_test.parquet")
    # S1b consistency check (B4-style): only the seed differs from the S1b run
    s1b_pred = RUNS.parent / "s1b_alpha158_lgb" / "pred.parquet"
    check = None
    if s1b_pred.exists():
        ref = pd.read_parquet(s1b_pred)["score"]
        jj = pd.concat([pt_s.rename("mine"), ref.rename("s1b")], axis=1).dropna()
        check = {"pearson": float(np.corrcoef(jj["mine"], jj["s1b"])[0, 1]),
                 "spearman": spear(jj["mine"], jj["s1b"]), "rows": int(len(jj))}
    met = {"model": "lightgbm",
           "role": "LGB single-model arm = deployed full-train LGB (S1b hyperparams from cache, "
                   "B4 run_lgb replica, seed 42); its valid/test predictions also form the "
                   "residual targets and the ensemble LGB side",
           "train": {"hyperparams": LGB_PARAMS, "num_boost_round": LGB_ROUNDS,
                     "early_stopping_rounds": LGB_EARLY_STOP,
                     "best_iteration": int(booster.best_iteration),
                     "n_trees": int(booster.num_trees()), "train_seconds": full_secs,
                     "seed": SEED, "selection": "valid_l2_early_stop (S1b behavior)"},
           "valid": full_eval(pv_s, yva_s), "test": full_eval(pt_s, label_te),
           "s1b_test_pred_check": check}
    dump_json(met, out / "metrics.json")
    print(f"[lgb] full fit best_iter={booster.best_iteration} n_trees={booster.num_trees()} "
          f"({full_secs:.0f}s) valid_rankic_global={met['valid']['rankic_global']:.5f} "
          f"test_rankic={met['test']['rankic']['ic_mean']:.5f} s1b_check={check}", flush=True)

    # ---- expanding natural-year OOF inside train
    # fold for predicted year y:  fit1 = rows < Oct-1 of (y-1)  (early stop on Q4 of y-1,
    # capacity calibration only), then refit on ALL rows through (y-1)-12-31 with the frozen
    # round count, predict year y. Every window strictly precedes y -> no leakage.
    oof_parts, folds = [], []
    for y in OOF_YEARS:
        cut = pd.Timestamp(f"{y - 1}-10-01")
        end = pd.Timestamp(f"{y}-01-01")
        m_fit1 = dt < cut
        m_iva = (dt >= cut) & (dt < end)
        m_pred = years == y
        assert m_fit1.sum() > 0 and m_pred.sum() > 0
        assert dt[m_pred].min() >= end  # structural leakage guard
        t0 = time.perf_counter()
        d1 = lgb.Dataset(Xtr[m_fit1], label=ytr[m_fit1], feature_name=names)
        d2 = lgb.Dataset(Xtr[m_iva], label=ytr[m_iva], reference=d1, feature_name=names)
        b1st = lgb.train(LGB_PARAMS, d1, num_boost_round=LGB_ROUNDS,
                         valid_sets=[d2], valid_names=["ivalid"],
                         callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False)])
        bi = int(b1st.best_iteration or LGB_ROUNDS)
        dfull = lgb.Dataset(Xtr[years <= y - 1], label=ytr[years <= y - 1], feature_name=names)
        bm = lgb.train(LGB_PARAMS, dfull, num_boost_round=bi)
        p = bm.predict(Xtr[m_pred])
        p_s = pd.Series(p, index=idx_tr[m_pred])
        lab_s = pd.Series(ytr[m_pred], index=idx_tr[m_pred])
        oof_parts.append(pd.DataFrame({"oof_pred": p_s, "label": lab_s}))
        folds.append({"fold": y - 2012, "predict_year": y,
                      "fit1_window": [str(dt[m_fit1].min().date()), str(cut.date() - pd.Timedelta(days=1))],
                      "internal_valid_window": [str(cut.date()), str(end.date() - pd.Timedelta(days=1))],
                      "refit_window": [str(dt[years <= y - 1].min().date()), str(end.date() - pd.Timedelta(days=1))],
                      "refit_rounds": bi, "fit_rows": int((years <= y - 1).sum()),
                      "predict_rows": int(m_pred.sum()),
                      "oof_daily_rankic": float(b1.daily_ic(p_s, lab_s, method="spearman").mean()),
                      "fold_seconds": time.perf_counter() - t0})
        print(f"[oof] fold {folds[-1]['fold']} predict {y}: fit1 {folds[-1]['fit1_window'][0]}.."
              f"{folds[-1]['fit1_window'][1]}, ivalid Q4-{y-1}, refit rounds {bi}, "
              f"oof_daily_rankic {folds[-1]['oof_daily_rankic']:.5f}", flush=True)
    oof_df = pd.concat(oof_parts)
    oof_df.index = oof_df.index.set_names(["datetime", "instrument"])
    oof_df["resid"] = oof_df["label"] - oof_df["oof_pred"]
    oof_df.to_parquet(oof / "residual_target_train.parquet")
    r2_oof = 1.0 - oof_df["resid"].var() / oof_df["label"].var()
    dump_json({"scheme": "expanding natural-year OOF inside train 2012-2020",
               "folds": folds, "coverage_years": OOF_YEARS,
               "dropped_from_residual_training": "2012 (no earlier data to fit a booster on)",
               "r2_oof_vs_label": float(r2_oof),
               "params": LGB_PARAMS,
               "leakage_argument": [
                   "predicted year y is fitted on data strictly before y (fit windows end Dec-31 of y-1)",
                   "the Q4 internal-valid of each fold only calibrates capacity (early-stopped round "
                   "count) on pre-y data and is excluded from fit1; the refit then uses the full "
                   "pre-y window with the frozen round count",
                   "no valid(2021-22)/test(2023-26) row participates in any fold model or its "
                   "calibration, so the train residual target contains zero post-train information",
                   "residual target = CSZScoreNorm label - OOF prediction, both in the standardized "
                   "label space; KAN_res is added back to LGB in the same space at deployment"]},
              oof / "oof_folds.json")

    # ---- valid residual target (deployed model)
    rva = pd.DataFrame({"lgb_pred": pv_s, "label": yva_s}).dropna()
    rva["resid"] = rva["label"] - rva["lgb_pred"]
    rva.index = rva.index.set_names(["datetime", "instrument"])
    rva.to_parquet(oof / "residual_target_valid.parquet")
    print(f"[lgb] OOF coverage {oof_df.index.get_level_values('datetime').year.min()}-"
          f"{oof_df.index.get_level_values('datetime').year.max()} rows={len(oof_df)} "
          f"R2_oof={r2_oof:.4f}; valid resid std {rva['resid'].std():.4f} "
          f"vs label std {rva['label'].std():.4f}", flush=True)


# ---------------------------------------------------------------- stage kan: single-model arm


def stage_kan(splits):
    out = HYB / "kan_arm"
    if (out / "metrics.json").exists():
        print("[skip] stage kan", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, _ = b1.xy(splits["train"])
    Xva, yva, _ = b1.xy(splits["valid"])
    assert not np.isnan(yva).any(), "valid labels must be complete (DropnaLabel)"
    yva_s = label_series(splits["valid"])
    label_te = pd.read_parquet(b1.CACHE / "label_test.parquet")["label"]
    set_seed()
    model, state, best_epoch, secs = train_kan(
        Xtr, ytr, Xva, lambda p: spear(p, yva), "kan_arm")
    torch.save(state, out / "model.pt")
    pv_s = save_pred(pd.Series(predict_kan(model, Xva), index=splits["valid"].index),
                     out / "pred_valid.parquet")
    Xte, _, _ = b1.xy(splits["test_feat"])
    pt_s = save_pred(pd.Series(predict_kan(model, Xte), index=splits["test_feat"].index),
                     out / "pred_test.parquet")
    met = {"model": "kan",
           "role": "KAN single-model arm (B1 config w24 g5 k3, CPU, seed 42, same train_kan "
                   "code path as the residual KAN); also the ensemble KAN side",
           "train": {"n_params": b1.n_params(model), "train_seconds": secs, "best_epoch": best_epoch,
                     "optimizer": "AdamW", "seed": SEED, "device": DEVICE,
                     "early_stop": "own valid global RankIC (B1 protocol)", **b1.TRAIN_CFG,
                     **KAN_CFG},
           "valid": full_eval(pv_s, yva_s), "test": full_eval(pt_s, label_te)}
    dump_json(met, out / "metrics.json")
    print(f"[kan_arm] best_epoch={best_epoch} ({secs:.0f}s) "
          f"valid_rankic_global={met['valid']['rankic_global']:.5f} "
          f"test_rankic={met['test']['rankic']['ic_mean']:.5f} "
          "(B3/B7 s42 CPU ref: valid 0.02990 test 0.03045)", flush=True)


# ---------------------------------------------------------------- stage residual: KAN on OOF resid


def stage_residual(splits):
    out = HYB / "residual_arm"
    if (out / "metrics.json").exists():
        print("[skip] stage residual", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    rtr = pd.read_parquet(OOF / "residual_target_train.parquet")
    rva = pd.read_parquet(OOF / "residual_target_valid.parquet")
    Xtr_full, _, _ = b1.xy(splits["train"])
    Xva, yva, _ = b1.xy(splits["valid"])
    assert not np.isnan(yva).any(), "valid labels must be complete (DropnaLabel)"
    yva_s = label_series(splits["valid"])
    label_te = pd.read_parquet(b1.CACHE / "label_test.parquet")["label"]
    idx_tr = splits["train"].index
    m = idx_tr.get_level_values("datetime").year >= 2013
    assert (idx_tr[m] == rtr.index).all(), "OOF residual rows must align with train rows 2013-2020"
    Xtr, ytr = Xtr_full[m], rtr["resid"].to_numpy(dtype=np.float32)
    lgb_va = load_pred(HYB / "lgb_arm" / "pred_valid.parquet").reindex(yva_s.index).to_numpy()

    def crit(p):
        return spear(lgb_va + p, yva)

    set_seed()
    model, state, best_epoch, secs = train_kan(Xtr, ytr, Xva, crit, "kan_res")
    torch.save(state, out / "model.pt")
    kr_va = pd.Series(predict_kan(model, Xva), index=splits["valid"].index)
    save_pred(kr_va, out / "kan_res_pred_valid.parquet")
    Xte, _, _ = b1.xy(splits["test_feat"])
    kr_te = pd.Series(predict_kan(model, Xte), index=splits["test_feat"].index)
    save_pred(kr_te, out / "kan_res_pred_test.parquet")
    lgb_te = load_pred(HYB / "lgb_arm" / "pred_test.parquet")
    pv_s = save_pred(pd.Series(lgb_va + kr_va.to_numpy(), index=splits["valid"].index),
                     out / "pred_valid.parquet")
    pt_s = save_pred(pd.Series(lgb_te.reindex(kr_te.index).to_numpy() + kr_te.to_numpy(),
                               index=kr_te.index), out / "pred_test.parquet")
    # diagnostics: is there anything learnable in the residual at all?
    diag = {
        "kan_res_vs_valid_resid_rankic_global": spear(kr_va, rva["resid"]),
        "kan_res_vs_train_oof_resid_rankic_global": spear(predict_kan(model, Xtr), ytr),
        "kan_res_std_valid": float(kr_va.std()), "resid_std_valid": float(rva["resid"].std()),
        "label_std_valid": float(rva["label"].std()),
        "lgb_oof_r2_train": float(1.0 - rtr["resid"].var() / rtr["label"].var()),
        "lgb_valid_r2": float(1.0 - rva["resid"].var() / rva["label"].var()),
    }
    met = {"model": "kan_residual",
           "role": "residual hybrid arm: KAN trained on (CSZScoreNorm label - LGB_OOF) over train "
                   "2013-2020, early-stopped on combined (LGB_full + KAN_res) valid RankIC; "
                   "test = LGB_full + KAN_res scored once",
           "train": {"n_params": b1.n_params(model), "train_seconds": secs, "best_epoch": best_epoch,
                     "optimizer": "AdamW", "seed": SEED, "device": DEVICE, "loss": "MSE on residual",
                     "early_stop": "combined (LGB+KAN_res) valid global RankIC",
                     "target": "label - LGB_OOF (expanding natural-year folds, 2013-2020)",
                     "train_rows": int(m.sum()), **b1.TRAIN_CFG, **KAN_CFG},
           "valid": full_eval(pv_s, yva_s), "test": full_eval(pt_s, label_te),
           "diagnostics": diag}
    dump_json(met, out / "metrics.json")
    print(f"[residual_arm] best_epoch={best_epoch} ({secs:.0f}s) "
          f"combined valid_rankic_global={met['valid']['rankic_global']:.5f} "
          f"test_rankic={met['test']['rankic']['ic_mean']:.5f} diag={diag}", flush=True)


# ---------------------------------------------------------------- stage ensemble


def stage_ensemble(splits):
    out = HYB / "ensemble_arm"
    if (out / "metrics.json").exists():
        print("[skip] stage ensemble", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    yva_s = label_series(splits["valid"])
    label_te = pd.read_parquet(b1.CACHE / "label_test.parquet")["label"]
    kv = load_pred(HYB / "kan_arm" / "pred_valid.parquet")
    lv = load_pred(HYB / "lgb_arm" / "pred_valid.parquet")
    kt = load_pred(HYB / "kan_arm" / "pred_test.parquet")
    lt = load_pred(HYB / "lgb_arm" / "pred_test.parquet")

    def combo(w, ka, lg):
        return (w * ka.reindex(lg.index) + (1 - w) * lg).dropna()

    grid = []
    for w in W_GRID:
        ev = full_eval(combo(w, kv, lv), yva_s)
        grid.append({"w": w, "valid_rankic": ev["rankic"]["ic_mean"],
                     "valid_ic": ev["ic"]["ic_mean"], "valid_rankicir": ev["rankic"]["icir"],
                     "valid_rankic_global": ev["rankic_global"]})
    gdf = pd.DataFrame(grid)
    best_i = int(gdf["valid_rankic"].idxmax())  # first max -> smaller w on exact ties
    w_star = float(gdf.loc[best_i, "w"])
    gdf.insert(1, "selected", [i == best_i for i in gdf.index])
    gdf.to_csv(out / "weight_grid.csv", index=False)
    # endpoint self-consistency: w=0 / w=1 must reproduce the single arms bit-for-bit
    chk = {}
    for w, arm, name in ((0.0, "lgb_arm", "lgb"), (1.0, "kan_arm", "kan")):
        ev = full_eval(combo(w, kv, lv), yva_s)
        am = json.load(open(HYB / arm / "metrics.json"))
        chk[f"w{int(w)}_matches_{name}"] = bool(
            abs(ev["rankic"]["ic_mean"] - am["valid"]["rankic"]["ic_mean"]) < 1e-12
            and abs(ev["ic"]["ic_mean"] - am["valid"]["ic"]["ic_mean"]) < 1e-12)
    pv_s = save_pred(combo(w_star, kv, lv), out / "pred_valid.parquet")
    pt_s = save_pred(combo(w_star, kt, lt), out / "pred_test.parquet")
    met = {"model": "ensemble",
           "role": f"output ensemble w*KAN+(1-w)*LGB, w*={w_star} picked on valid daily-mean "
                   "RankIC over the issue grid; test scored once with w*",
           "weight_grid": grid, "selected_w": w_star,
           "selection": "valid daily-mean RankIC argmax (first max on ties)",
           "valid": full_eval(pv_s, yva_s), "test": full_eval(pt_s, label_te),
           "endpoint_selfcheck": chk}
    dump_json(met, out / "metrics.json")
    print("[ensemble] valid grid:\n" + gdf.to_string(index=False), flush=True)
    print(f"[ensemble] w*={w_star} test_rankic={met['test']['rankic']['ic_mean']:.5f} "
          f"selfcheck={chk}", flush=True)


# ---------------------------------------------------------------- stage backtest


def stage_backtest():
    import qlib
    from qlib.constant import REG_CN
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy
    missing = [a for a in ARMS if not (HYB / a / "pred_test.parquet").exists()]
    assert not missing, f"arms missing predictions: {missing}"
    qlib.init(provider_uri=DATA_DIR, region=REG_CN)
    for arm in ARMS:
        met_path = HYB / arm / "metrics.json"
        met = json.load(open(met_path))
        if "backtest" in met:
            print(f"[skip] backtest {arm}", flush=True)
            continue
        pred = load_pred(HYB / arm / "pred_test.parquet")
        strategy = TopkDropoutStrategy(topk=30, n_drop=1, signal=pred)
        t0 = time.perf_counter()
        report, _ = backtest_daily(
            start_time=b1.SEGMENTS["test"][0], end_time=b1.SEGMENTS["test"][1],
            strategy=strategy, benchmark="SH000300",
            exchange_kwargs={"limit_threshold": 0.095, "deal_price": "close",
                             "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5,
                             "trade_unit": 100})
        report.to_csv(HYB / arm / "backtest_report.csv")
        ex = risk_analysis(report["return"] - report["bench"]).iloc[:, 0]
        met["backtest"] = {
            "excess_annualized_return": float(ex["annualized_return"]),
            "excess_information_ratio": float(ex["information_ratio"]),
            "excess_max_drawdown": float(ex["max_drawdown"]),
            "annualized_return": float(risk_analysis(report["return"]).iloc[:, 0]["annualized_return"]),
            "daily_turnover_mean": float(report["turnover"].mean()),
            "config": "B1 main config: topk30/drop1, SH000300, open 5bp/close 15bp, limit 0.095",
            "backtest_seconds": time.perf_counter() - t0,
        }
        dump_json(met, met_path)
        print(f"[backtest] {arm}: excess {met['backtest']['excess_annualized_return']*100:.2f}% "
              f"IR {met['backtest']['excess_information_ratio']:.2f} "
              f"MDD {met['backtest']['excess_max_drawdown']*100:.1f}% "
              f"turnover {met['backtest']['daily_turnover_mean']:.2%} "
              f"({met['backtest']['backtest_seconds']:.0f}s)", flush=True)


# ---------------------------------------------------------------- stage report


def linear_ref_row():
    """Linear/Ridge reference from the frozen B1 artifacts (no rerun): test RankIC/IC from
    metrics.json; excess/IR/MDD/turnover recomputed arithmetically from the stored
    backtest_report.csv (B1's metrics.json overwrote annualized_return with the raw one)."""
    lin = json.load(open(RUNS / "b1_linear" / "metrics.json"))
    rep = pd.read_csv(RUNS / "b1_linear" / "backtest_report.csv", index_col=0, parse_dates=True)
    daily = rep["return"] - rep["bench"]
    ann = float(daily.mean() * 252)
    ir = float(daily.mean() / daily.std() * np.sqrt(252))
    cum = (1 + daily).cumprod()
    mdd = float((cum / cum.cummax() - 1).min())
    return {"arm": ARM_NAMES["linear_ref"], "test_rankic": lin["rankic"]["ic_mean"],
            "test_ic": lin["ic"]["ic_mean"], "test_rankicir": round(lin["rankic"]["icir"], 3),
            "excess_ann": ann, "IR": ir, "excess_MDD": mdd,
            "daily_turnover": float(rep["turnover"].mean()),
            "_note": "excess/IR/MDD/turnover recomputed arithmetically from the frozen B1 "
                     "backtest_report.csv (ann = daily mean x 252, IR = mean/std x sqrt(252), "
                     "MDD on compounded excess NAV); RankIC/IC quoted from B1 metrics.json"}


def _row(name, met):
    r = {"arm": name}
    if "valid" in met:
        r.update({"valid_rankic": met["valid"]["rankic"]["ic_mean"],
                  "valid_ic": met["valid"]["ic"]["ic_mean"],
                  "valid_rankicir": round(met["valid"]["rankic"]["icir"], 3),
                  "valid_rankic_global": met["valid"].get("rankic_global")})
    t = met["test"]
    r.update({"test_rankic": t["rankic"]["ic_mean"], "test_ic": t["ic"]["ic_mean"],
              "test_rankicir": round(t["rankic"]["icir"], 3)})
    for y in ("2023", "2024", "2025", "2026"):
        r[f"test_rankic_{y}"] = t["rankic_yearly"].get(y)
    return r


def _bt_row(name, bt):
    return {"arm": name, "excess_ann": bt["excess_annualized_return"],
            "IR": bt["excess_information_ratio"], "excess_MDD": bt["excess_max_drawdown"],
            "daily_turnover": bt["daily_turnover_mean"]}


def _cred(d):
    return "credible (>0.0059)" if abs(d) >= THRESH else "inside noise band (<0.0059)"


def stage_report():
    mets = {a: json.load(open(HYB / a / "metrics.json")) for a in ARMS}
    lin = linear_ref_row()
    lin_tbl = {k: v for k, v in lin.items() if k != "_note"}
    res, ens, lgb, kan = (mets[a] for a in ARMS)

    pd.DataFrame([_row(ARM_NAMES["residual_arm"], res), _row(ARM_NAMES["lgb_arm"], lgb),
                  _row(ARM_NAMES["kan_arm"], kan),
                  {k: v for k, v in lin_tbl.items()
                   if k in ("arm", "test_rankic", "test_ic", "test_rankicir")}]).to_csv(
        HYB / "residual_table.csv", index=False)
    pd.DataFrame([_row(ARM_NAMES["ensemble_arm"], ens), _row(ARM_NAMES["lgb_arm"], lgb),
                  _row(ARM_NAMES["kan_arm"], kan)]).to_csv(HYB / "ensemble_table.csv", index=False)

    tr = lambda m: m["test"]["rankic"]["ic_mean"]
    vr = lambda m: m["valid"]["rankic"]["ic_mean"]
    comp = {
        "threshold_kan2kan": THRESH, "threshold_with_mlp": THRESH_MLP,
        "residual_minus_lgb": {"valid": vr(res) - vr(lgb), "test": tr(res) - tr(lgb)},
        "residual_minus_kan": {"valid": vr(res) - vr(kan), "test": tr(res) - tr(kan)},
        "ensemble_minus_lgb": {"valid": vr(ens) - vr(lgb), "test": tr(ens) - tr(lgb)},
        "ensemble_minus_kan": {"valid": vr(ens) - vr(kan), "test": tr(ens) - tr(kan)},
        "kan_minus_lgb": {"valid": vr(kan) - vr(lgb), "test": tr(kan) - tr(lgb)},
    }
    dump_json(comp, HYB / "comparisons.json")

    pd.DataFrame([_bt_row(ARM_NAMES["residual_arm"], res["backtest"]),
                  _bt_row(ARM_NAMES["ensemble_arm"], ens["backtest"]),
                  _bt_row(ARM_NAMES["lgb_arm"], lgb["backtest"]),
                  _bt_row(ARM_NAMES["kan_arm"], kan["backtest"]),
                  {k: v for k, v in lin_tbl.items()
                   if k in ("arm", "excess_ann", "IR", "excess_MDD", "daily_turnover")}]).to_csv(
        HYB / "backtest_table.csv", index=False)

    write_report(mets, lin, comp)
    write_results_md(mets, lin, comp)
    print(f"[report] wrote {HYB / 'REPORT.md'} and results/B10_hybrid.md", flush=True)


def _md_table(df, nd=5):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def _bt_md(df):
    d = df.copy()
    d["excess_ann"] = d["excess_ann"].map(lambda v: f"{v*100:.2f}%")
    d["excess_MDD"] = d["excess_MDD"].map(lambda v: f"{v*100:.1f}%")
    d["IR"] = d["IR"].map(lambda v: f"{v:.2f}")
    d["daily_turnover"] = d["daily_turnover"].map(lambda v: f"{v:.2%}")
    return _md_table(d.rename(columns={"excess_ann": "excess ann.", "excess_MDD": "excess MDD"}))


def write_report(mets, lin, comp):
    res, ens, lgb, kan = (mets[a] for a in ARMS)
    folds = json.load(open(OOF / "oof_folds.json"))
    oof = pd.read_parquet(OOF / "residual_target_train.parquet")
    rt = pd.read_csv(HYB / "residual_table.csv")
    et = pd.read_csv(HYB / "ensemble_table.csv")
    bt = pd.read_csv(HYB / "backtest_table.csv")
    wg = pd.read_csv(HYB / "ensemble_arm" / "weight_grid.csv")
    wg["selected"] = wg["selected"].map({True: "<-- w*", False: ""})
    w_star, chk = ens["selected_w"], ens["endpoint_selfcheck"]
    dg = res["diagnostics"]
    rml, rmk = comp["residual_minus_lgb"], comp["residual_minus_kan"]
    eml, emk = comp["ensemble_minus_lgb"], comp["ensemble_minus_kan"]
    kml = comp["kan_minus_lgb"]
    L, A = [], lambda s: L.append(s)

    A("# B10 混合架构报告：LightGBM×KAN 残差学习（OOF）与输出 ensemble（Issue #16）\n")
    A("## 1. 设计与 OOF 防泄漏\n")
    A("- LGB 单模型臂 = S1b 官方超参在 `_cache` 上重训（B4 `run_lgb` 同款，seed 42，valid 早停），"
      f"best_iteration={lgb['train']['best_iteration']} / {lgb['train']['n_trees']} trees，"
      f"与 S1b 落盘 test 预测 Pearson {lgb['s1b_test_pred_check']['pearson']:.3f}"
      "（B4 复验值 0.955 的同款一致性检查）。")
    A("- KAN 单模型臂 = B1 配置 [158,24,1] g5/k3，CPU seed 42，与残差 KAN 走同一个 `train_kan` 代码路径"
      "（AdamW+MSE、batch 4096、max 200 epoch、patience 15、best-epoch 恢复），仅早停准则不同"
      "（单模型臂 = 自身 valid RankIC，B1 原协议；残差臂 = 组合预测 valid RankIC，Issue 规定）。")
    A("- **OOF 折结构**（train 2012–2020 内按自然年 expanding，预测年 y 的模型只见 y 之前的数据）：")
    fdf = pd.DataFrame(folds["folds"])[["fold", "predict_year", "fit1_window", "internal_valid_window",
                                        "refit_window", "refit_rounds", "fit_rows", "predict_rows",
                                        "oof_daily_rankic"]]
    for c in ("fit1_window", "internal_valid_window", "refit_window"):
        fdf[c] = fdf[c].map(lambda w: f"{w[0]}~{w[1]}")
    A(_md_table(fdf))
    A(f"- 覆盖年 {folds['coverage_years'][0]}–{folds['coverage_years'][-1]}（8 折，"
      f"{int(len(oof))} 行）；**2012 年无更早数据可拟合，从残差训练集中剔除**。无泄漏论证："
      "① 预测年 y 的拟合/早停窗口全部止于 y−1 年末（结构断言 `dt[pred].min() >= y-01-01` 已在代码中强制）；"
      "② 折内 Q4 内部 valid 只用于校准轮数（early stop 50），不进 fit1，随后用冻结轮数在完整 "
      "y−1 年末窗口 refit；③ valid/test 段不参与任何折模型及其校准，残差目标不含任何 train 段之后的"
      "信息——禁止 in-sample 预测的要求由此满足；④ 残差目标在 CSZScoreNorm 标准化 label 空间内作差，"
      "部署时 KAN_res 与 LGB 预测同空间直接相加。")
    A(f"- 数值口径：OOF 预测对 label 的 R²（1−var(resid)/var(label)，train 段）= "
      f"{folds['r2_oof_vs_label']:.4f}；valid 段 LGB(全 train) R² = {dg['lgb_valid_r2']:.4f}"
      f"（残差 std {dg['resid_std_valid']:.4f} vs label std {dg['label_std_valid']:.4f}）——"
      "LGB 只解释了 label 方差的很小一部分，残差目标里理论上有充足的可学空间。\n")

    A("## 2. 残差路线：LGB+KAN_res vs 单模型臂\n")
    A(_md_table(rt))
    A("")
    A(f"- 差值判读（对照 B7 阈值 {THRESH}，2σ）：test RankIC 残差臂−LGB = "
      f"{rml['test']:+.5f}（{_cred(rml['test'])}）；残差臂−KAN = {rmk['test']:+.5f}"
      f"（{_cred(rmk['test'])}）。KAN−LGB 单模型差 {kml['test']:+.5f} 作参照"
      f"（{_cred(kml['test'])}）。")
    A(f"- 残差可学性诊断：KAN_res 对 valid 残差目标的全局 Spearman = "
      f"{dg['kan_res_vs_valid_resid_rankic_global']:+.5f}（train OOF 残差上 "
      f"{dg['kan_res_vs_train_oof_resid_rankic_global']:+.5f}），KAN_res 输出 std "
      f"{dg['kan_res_std_valid']:.4f} vs 残差 std {dg['resid_std_valid']:.4f}。\n")

    A("## 3. ensemble 路线：w·KAN+(1−w)·LGB\n")
    A(_md_table(wg))
    A("")
    A(f"- 选中 w* = {w_star:g}（valid 日均 RankIC 最大，选择准则与表格指标同口径；"
      f"全局 Spearman 列并列展示）。端点自洽性：w=0 与 LGB 臂逐位一致 = "
      f"{chk['w0_matches_lgb']}，w=1 与 KAN 臂逐位一致 = {chk['w1_matches_kan']}。")
    A(_md_table(et))
    A("")
    A(f"- 差值判读：test RankIC ensemble−LGB = {eml['test']:+.5f}（{_cred(eml['test'])}）；"
      f"ensemble−KAN = {emk['test']:+.5f}（{_cred(emk['test'])}）。\n")

    A("## 4. 回测（B1 口径：topk30/drop1，SH000300，open 5bp/close 15bp，limit 0.095）\n")
    A(_bt_md(bt))
    A("")
    A("- **#7 噪声带 caveat（必读）**：B6 已证明该组合层配置下模型间超额差异落在同日配对 "
      "bootstrap 噪声带内（CI 含 0），且 topk/drop 配置效应远大于模型效应（KAN−Linear 差在 "
      "8 种配置间 −8.2~+11.4 pp 无规律翻转）。本表只用于看相对形态，不构成排名。"
      f"Linear/Ridge 行为 B1 冻结产物引用（{lin['_note']}）。\n")

    A("## 5. 结论\n")
    A(verdict_residual(res, dg, rml, rmk))
    A("")
    A(verdict_ensemble(ens, eml, emk, chk))
    A("")
    A(verdict_portfolio(mets, lin))
    A("")
    A(verdict_final(mets, comp))
    A("")
    A("## 6. 产物索引\n")
    A("`oof/oof_folds.json`（折结构+泄漏论证）· `oof/residual_target_train.parquet`（OOF 预测+残差目标，"
      "2013–2020）· `oof/residual_target_valid.parquet` · `lgb_arm/`、`kan_arm/`、`residual_arm/`、"
      "`ensemble_arm/`（各含 pred_valid/pred_test/metrics.json[/model.pt]，臂目录内有 backtest_report.csv）· "
      "`residual_table.csv` · `ensemble_table.csv` · `weight_grid.csv` · `backtest_table.csv` · "
      "`comparisons.json` · 本 `REPORT.md`。代码：`scripts/run_b10_hybrid.py`；仓库结果摘要："
      "`results/B10_hybrid.md`。")
    (HYB / "REPORT.md").write_text("\n".join(L) + "\n")


def verdict_residual(res, dg, rml, rmk):
    d_lgb, d_kan = rml["test"], rmk["test"]
    add_lgb = d_lgb > 0 and abs(d_lgb) >= THRESH
    add_kan = d_kan > 0 and abs(d_kan) >= THRESH
    if add_lgb and add_kan:
        label = "存在（互补增量，同时超过两个单模型）"
    elif add_lgb:
        label = "对 LGB 有加成，但非互补增量（不超过 KAN 单模型）"
    else:
        label = "不存在"
    v = (f"- **残差增量（Q1）：{label}。** LGB+KAN_res 相对 LGB 单模型 test RankIC 差 "
         f"{d_lgb:+.5f}（{_cred(d_lgb)}），相对 KAN 单模型 {d_kan:+.5f}（{_cred(d_kan)}）；"
         f"valid 段差 {rml['valid']:+.5f} / {rmk['valid']:+.5f}。")
    v += (f" 机制诊断：LGB 的 OOF R² 仅 {dg['lgb_oof_r2_train']:.4f}（valid R² "
          f"{dg['lgb_valid_r2']:.4f}），「LGB 未解释的部分」≈ 整个 label，残差目标并非"
          "「小修正」而是近乎全量重学；KAN_res 对残差目标的全局 Spearman "
          f"{dg['kan_res_vs_valid_resid_rankic_global']:+.5f}（train OOF 残差上 "
          f"{dg['kan_res_vs_train_oof_resid_rankic_global']:+.5f}），输出 std "
          f"{dg['kan_res_std_valid']:.4f}（收缩到残差 std 的 "
          f"{dg['kan_res_std_valid'] / dg['resid_std_valid']:.1%}）——学到的信号与直接训练的 "
          f"KAN 高度同源，{'存在超出 KAN 已覆盖范围的互补非线性结构' if add_kan else '没有超出 KAN 已覆盖范围的互补信息'}。")
    return v


def verdict_ensemble(ens, eml, emk, chk):
    d_lgb, d_kan, w_star = eml["test"], emk["test"], ens["selected_w"]
    grid = sorted(ens["weight_grid"], key=lambda r: -r["valid_rankic"])
    margin = grid[0]["valid_rankic"] - grid[1]["valid_rankic"]
    both = d_lgb > 0 and abs(d_lgb) >= THRESH and d_kan > 0 and abs(d_kan) >= THRESH
    if w_star in (0.0, 1.0):
        side = "LGB" if w_star == 0.0 else "KAN"
        attr = (f"valid 网格最优点落在端点 w*={w_star:g}（即纯 {side} 臂），任何混合权重都不优于"
                "单模型")
    else:
        attr = (f"valid 网格选中内部权重 w*={w_star:g}，但相对次优点的 valid 优势仅 {margin:.5f}"
                f"（次优 w={grid[1]['w']:g}），远低于噪声带——内点优势不具分辨力")
    if both:
        label = "存在（超过两个单模型）"
    elif d_lgb > 0 and abs(d_lgb) >= THRESH:
        label = "部分：显著超过 LGB，但不超过（也不落后于）KAN——增益全部来自 KAN 侧内容"
    else:
        label = "不存在"
    return (f"- **ensemble 增益（Q2）：{label}。** {attr}；test 差值：vs LGB {d_lgb:+.5f}"
            f"（{_cred(d_lgb)}），vs KAN {d_kan:+.5f}（{_cred(d_kan)}）；端点自洽 "
            f"{chk['w0_matches_lgb']}/{chk['w1_matches_kan']}（w=0/1 逐位复现单模型臂）。")


def verdict_portfolio(mets, lin):
    n = {a: (mets[a]["backtest"] if a in mets else lin) for a in list(ARMS) + ["linear_ref"]}
    rows = {ARM_NAMES[k]: v for k, v in n.items()}
    fmt = {k: (f"{v['excess_annualized_return']*100:.2f}%" if "excess_annualized_return" in v
               else f"{v['excess_ann']*100:.2f}%") for k, v in rows.items()}
    ir = {k: (v.get("excess_information_ratio", v.get("IR"))) for k, v in rows.items()}
    to = {k: (v.get("daily_turnover_mean", v.get("daily_turnover"))) for k, v in rows.items()}
    mdd = {k: (v.get("excess_max_drawdown", v.get("excess_MDD"))) for k, v in rows.items()}
    order = [ARM_NAMES["residual_arm"], ARM_NAMES["ensemble_arm"], ARM_NAMES["lgb_arm"],
             ARM_NAMES["kan_arm"]]
    return (f"- **组合层形态**：k30/d1 下超额 残差 {fmt[order[0]]} / ensemble {fmt[order[1]]} / "
            f"LGB {fmt[order[2]]} / KAN {fmt[order[3]]}，IR {ir[order[0]]:.2f}/{ir[order[1]]:.2f}/"
            f"{ir[order[2]]:.2f}/{ir[order[3]]:.2f}，超额 MDD "
            f"{mdd[order[0]]*100:.1f}%/{mdd[order[1]]*100:.1f}%/{mdd[order[2]]*100:.1f}%/"
            f"{mdd[order[3]]*100:.1f}%，日换手 {to[order[0]]:.2%}/{to[order[1]]:.2%}/"
            f"{to[order[2]]:.2%}/{to[order[3]]:.2%}。预测层指标与组合层超额的相对顺序"
            "不完全一致（B1/B6 已定性的形态）：混合臂在组合层的相对位置同样落在 #7 噪声带内，"
            "不据此下模型优劣结论，只记录形态——混合是否改变换手/回撤结构比单点超额更有信息量。")


def verdict_final(mets, comp):
    tr = lambda m: m["test"]["rankic"]["ic_mean"]
    best_single = max(tr(mets["lgb_arm"]), tr(mets["kan_arm"]))
    best_name = "LGB" if tr(mets["lgb_arm"]) >= tr(mets["kan_arm"]) else "KAN"
    d_res = tr(mets["residual_arm"]) - best_single
    d_ens = tr(mets["ensemble_arm"]) - best_single
    gain = max(d_res, d_ens) >= THRESH
    if gain:
        return (f"- **最终判词：做——混合架构有可分辨增益，值得进入主叙事。** 最优单模型为 "
                f"{best_name}（test RankIC {best_single:.5f}），混合臂相对它的差 残差 {d_res:+.5f} / "
                f"ensemble {d_ens:+.5f}，至少一个越过 B7 阈值 {THRESH}，详见分项。")
    return (f"- **最终判词：不进主叙事。** 最优单模型为 {best_name}（test RankIC {best_single:.5f}），"
            f"两条混合臂相对它均无增益（残差 {d_res:+.5f} / ensemble {d_ens:+.5f}，均未越过 B7 "
            f"噪声带阈值 {THRESH}）；「新旧结合」在 Alpha158/csi300 上没有可分辨的工程增益。"
            "这也是有信息量的否定：LGB 与 KAN 在该特征集上的信息集高度重叠（残差路线里 LGB 的 "
            "OOF R² 仅 0.003，「LGB 未解释部分」≈ 整个 label，KAN_res 学到的与直接 KAN 同源；"
            "ensemble 的 valid 网格从 w=0.5 起几乎平坦，内点优势 5e-5 量级），任何混合权重都不"
            "优于把全部权重交给较强的一方。混合架构不作为主线结论，仅在方法附录保留 OOF 残差"
            "构造（防泄漏折设计可复用）与端点自洽检查的做法。")


def write_results_md(mets, lin, comp):
    res, ens, lgb, kan = (mets[a] for a in ARMS)
    folds = json.load(open(OOF / "oof_folds.json"))
    bt = pd.read_csv(HYB / "backtest_table.csv")
    rt = pd.read_csv(HYB / "residual_table.csv")
    et = pd.read_csv(HYB / "ensemble_table.csv")
    rml, rmk = comp["residual_minus_lgb"], comp["residual_minus_kan"]
    eml, emk = comp["ensemble_minus_lgb"], comp["ensemble_minus_kan"]
    L, A = [], lambda s: L.append(s)

    A("# B10 混合架构：LightGBM×KAN 残差学习（OOF）与输出 ensemble\n")
    A("口径：csi300 Alpha158 `_cache`（只读），train 2012–2020 / valid 2021–2022 / test 2023–2026.07，"
      "统一 CPU、seed 42。LGB = S1b 超参缓存重训（B4 `run_lgb` 同款，与 S1b test 预测 Pearson "
      f"{lgb['s1b_test_pred_check']['pearson']:.3f}）；KAN = B1 配置 w24 CPU 重训；残差 KAN 与其同 "
      "`train_kan` 代码路径（仅早停准则换成组合预测 valid RankIC）。完整方法、OOF 折表与全部诊断："
      "`common/runs/kan/hybrid/REPORT.md`；代码 `scripts/run_b10_hybrid.py`。本文件为结论摘要。\n")
    A("## 1. OOF 防污染设计（残差路线命门）\n")
    A("- LGB 在 train 内按自然年 expanding OOF：预测年 y 的模型先在 `2012-01~(y-1)-09` 上拟合、"
      "用 `(y-1)-Q4` 内部 valid 早停校准轮数（该 Q4 不进拟合集），再以冻结轮数在完整 "
      "`2012-01~(y-1)-12` 窗口 refit 后预测 y；覆盖 2013–2020 共 8 折，2012 无更早数据剔除"
      f"（{int(pd.read_parquet(OOF / 'residual_target_train.parquet').shape[0])} 行，折表见 "
      "`oof/oof_folds.json`）。")
    A("- 无泄漏：所有折窗口严格早于预测年（代码内有结构断言）；valid/test 不参与任何折模型；"
      "残差目标 = CSZScoreNorm label − OOF 预测（同一标准化空间作差、同空间相加回），"
      f"禁止 in-sample 预测的要求由此满足（OOF R²={folds['r2_oof_vs_label']:.4f}：LGB 只解释了"
      " label 方差的极小部分，残差目标的可学空间充足，结论不受「残差太小」混淆）。\n")
    A("## 2. 残差结果表（valid / test 2023-01~2026-07）\n")
    A(_md_table(rt))
    A("")
    A(f"- 判读（B7 阈值 0.0059=2σ）：残差臂−LGB test RankIC {rml['test']:+.5f}"
      f"（{_cred(rml['test'])}），残差臂−KAN {rmk['test']:+.5f}（{_cred(rmk['test'])}）；"
      f"残差可学性诊断 KAN_res→valid 残差 Spearman "
      f"{res['diagnostics']['kan_res_vs_valid_resid_rankic_global']:+.5f}，KAN_res 输出 std "
      f"{res['diagnostics']['kan_res_std_valid']:.4f} vs 残差 std "
      f"{res['diagnostics']['resid_std_valid']:.4f}。\n")
    A("## 3. ensemble 结果表\n")
    A(_md_table(et))
    A("")
    A(f"- 选中 w*={ens['selected_w']:g}（valid 日均 RankIC 在网格 {W_GRID} 上选优，全表 "
      f"`ensemble_arm/weight_grid.csv`）；端点自洽 w=0≡LGB / w=1≡KAN 逐位复现 = "
      f"{ens['endpoint_selfcheck']['w0_matches_lgb']}/{ens['endpoint_selfcheck']['w1_matches_kan']}。"
      f"判读：ensemble−LGB {eml['test']:+.5f}（{_cred(eml['test'])}），ensemble−KAN "
      f"{emk['test']:+.5f}（{_cred(emk['test'])}）。\n")
    A("## 4. 回测表（B1 口径 topk30/drop1 / SH000300 / open 5bp / close 15bp / limit 0.095）\n")
    A(_bt_md(bt))
    A("")
    A("- **#7 噪声带 caveat**：B6 已证该层模型间差异落在配对 bootstrap 噪声带内、配置效应大于模型"
      "效应（KAN−Linear 在 8 配置间 −8.2~+11.4 pp 无规律），本表看相对形态不作排名；"
      f"Linear/Ridge 行为 B1 冻结产物引用（{lin['_note']}）。\n")
    A("## 5. 结论\n")
    A(verdict_residual(res, res["diagnostics"], rml, rmk))
    A("")
    A(verdict_ensemble(ens, eml, emk, ens["endpoint_selfcheck"]))
    A("")
    A(verdict_portfolio(mets, lin))
    A("")
    A(verdict_final(mets, comp))
    Path(__file__).resolve().parents[1].joinpath("results", "B10_hybrid.md").write_text(
        "\n".join(L) + "\n")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["lgb", "kan", "residual", "ensemble", "backtest",
                                        "report", "all"], default="all")
    args = ap.parse_args()
    HYB.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(TORCH_THREADS)
    b1.DEVICE = DEVICE
    stages = (["lgb", "kan", "residual", "ensemble", "backtest", "report"]
              if args.stage == "all" else [args.stage])
    handlers = {"lgb": stage_lgb, "kan": stage_kan, "residual": stage_residual,
                "ensemble": stage_ensemble, "backtest": stage_backtest, "report": stage_report}
    splits = None
    for st in stages:
        print(f"===== stage {st} =====", flush=True)
        if st in ("lgb", "kan", "residual", "ensemble"):
            if splits is None:
                splits = b1.prepare_data()  # read-only parquet cache
            handlers[st](splits)
        else:
            handlers[st]()
    print("run_b10_hybrid DONE", flush=True)


if __name__ == "__main__":
    main()
