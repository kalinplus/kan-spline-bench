# B1 主表：Alpha158 → Linear/Ridge · LightGBM · MLP · KAN（CSI300）

口径：与 S1b 完全一致 — train 2012–2020 / valid 2021–2022（超参与早停仅用 valid）
/ test 2023–2026.07 仅最终评测一次；RobustZScoreNorm(±3)+Fillna 特征、
DropnaLabel+CSZScoreNorm(label)；回测 topk 30 / drop 1，成本 open 5bp / close 15bp。
MLP [158,224,1]-ReLU vs KAN [158,24,1]（grid 5，order 3，grid_range ±3）：参数量同量级。

## 1. 主对比表（test 2023-01 ~ 2026-07）

| Model | Params | Train time | RankIC | IC | RankICIR | ICIR | Ann. excess | IR | Excess MDD | Daily turnover |
|---|---|---|---|---|---|---|---|---|---|---|
| Linear/Ridge | 159 | 0.8 min | 0.0197 | 0.0157 | 0.1336 | 0.1083 | 6.11% | 0.56 | -20.76% | 6.67% |
| LightGBM (S1b) | n/a | n/a | 0.0152 | 0.0133 | 0.1073 | 0.0937 | 2.86% | 0.20 | -30.59% | 6.90% |
| MLP | 35841 | 0.1 min | 0.0222 | 0.0163 | 0.1775 | 0.1340 | 4.36% | 0.48 | -17.67% | 5.04% |
| KAN | 38160 | 0.2 min | 0.0264 | 0.0198 | 0.1846 | 0.1333 | 3.46% | 0.36 | -14.67% | 6.88% |

注：LightGBM 参数量与训练时间为 S1b 历史运行，未记录（n/a）；其余三列同机同协议本次实测。
超参记录：Linear/Ridge 与 MLP/KAN 的 valid 选择细节见 `common/runs/kan/b1_*/metrics.json` 的 `train` 块。

## 2. 分年度 RankIC（test 按自然年拆分）

| Model | 2023 | 2024 | 2025 | 2026 | full test |
|---|---|---|---|---|---|
| Linear/Ridge | 0.0153 | 0.0276 | 0.0195 | 0.0139 | 0.0197 |
| LightGBM (S1b) | 0.0034 | 0.0282 | 0.0156 | 0.0120 | 0.0152 |
| MLP | 0.0142 | 0.0231 | 0.0257 | 0.0288 | 0.0222 |
| KAN | 0.0177 | 0.0386 | 0.0218 | 0.0287 | 0.0264 |

注：2026 为 1–7 月（数据覆盖至 2026-07-23）。regime 稳健性深化（滚动重训等）在 #3。

## 3. 分层回测与成本

| Model | Long-short ann. (10-layer) | Monotonicity (Spearman) |
|---|---|---|
| Linear/Ridge | 21.64% | 0.93 |
| LightGBM (S1b) | 23.40% | 0.95 |
| MLP | 31.66% | 0.78 |
| KAN | 29.44% | 0.95 |

逐日分层明细：`common/runs/{s1b_alpha158_lgb,kan/b1_*}/layered.csv`；成本口径 main（open 5bp / close 15bp），
敏感性四象限扩展不在 B1 范围（见 #3）。

## 4. 公平性对照（MLP vs KAN）

| Model | Architecture | Params | Train time | Notes |
|---|---|---|---|---|
| MLP | [158, 224, 1] ReLU | 35841 | 0.1 min | best epoch 12 |
| KAN | [158, 24, 1] grid5/order3, grid_range ±3 | 38160 | 0.2 min | best epoch 5 |

参数比 KAN/MLP = 1.06；同一优化协议（AdamW，lr 1e-3，wd 1e-4，batch 4096，max 200 epoch，valid 早停 patience 15，seed 42）。KAN 单 epoch 明显更慢是已知事实，如实报告训练时间。

## 5. KAN 样条可视化

`common/runs/kan/b1_kan/spline_curves.png`：第一层 top-12 特征（按边 L1 范数）的学习到的
样条变换曲线（每特征 24 条边曲线 + 最强边高亮），曲线数据 `spline_curves_top.parquet`。
可解释性深化（符号/形状归因）在 #4。

