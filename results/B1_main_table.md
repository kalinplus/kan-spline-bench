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

## 6. B6 组合层归因：pre-cost 选股 / decile CI / 超额置信区间（Issue #7）

回测口径与 §1 完全一致（k30/d1）；归因细节（敏感性矩阵、触限、drag 分解）见
`common/runs/kan/portfolio-attr/`（README + attr_summary.md），代码 `scripts/portfolio_attr.py`。

### 6.1 Pre-cost top-k 选股收益（label 对齐、等权、无成本无执行约束）

| Model | top-10 (bp/d, ann) | top-30 (bp/d, ann) | top-50 (bp/d, ann) |
|---|---|---|---|
| Linear/Ridge | 10.78 / 27.2% | 7.36 / 18.5% | 6.58 / 16.6% |
| LightGBM (S1b) | 6.81 / 17.2% | 6.59 / 16.6% | 6.22 / 15.7% |
| MLP | 12.76 / 32.2% | 9.70 / 24.4% | 8.12 / 20.5% |
| KAN | 6.35 / 16.0% | 8.15 / 20.5% | 8.09 / 20.4% |

选股能力排序与 RankIC 一致的方向：KAN 的 top-30（8.15）与 top-50（8.09）pre-cost
收益均高于 Linear / LGB（MLP 头部集中更强，top-10/30 均最高，但其 RankIC 低于 KAN），
说明 B1 主表的「RankIC 占优但回测落后」反转不在选股层，而在执行层（drag 分解见 §7）。
注：KAN 的 top-10 收益偏低，其优势集中在 10 名之外、50 名之内的中头部票。

### 6.2 Long-short decile spread（带统计显著性）

| Model | LS bp/d | Ann. | t | p | 95% CI (bp/d) |
|---|---|---|---|---|---|
| Linear/Ridge | 8.59 | 21.6% | 2.11 | 0.035 | [0.48, 16.43] |
| LightGBM (S1b) | 9.29 | 23.4% | 2.29 | 0.023 | [1.42, 17.03] |
| MLP | 12.57 | 31.7% | 3.63 | <0.001 | [5.76, 19.37] |
| KAN | 11.68 | 29.4% | 2.77 | 0.006 | [3.23, 20.06] |

四模型 10 层多空价差均显著为正（CI 不含 0），预测层信号真实有效；KAN 幅度次于 MLP、
高于 Linear/LGB。

### 6.3 回测超额收益置信区间（同日配对，paired bootstrap 10k）

| Pair | 全期 bp/d | 95% CI | p (paired t) | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| KAN | 1.37 | [−2.45, 5.26] | 0.49 | −2.14 | 0.03 | +4.65 | +4.23 |
| Linear/Ridge | 2.42 | [−1.92, 6.86] | 0.28 | +2.89 | +4.31 | +2.22 | −1.48 |
| KAN − Linear | −1.05 | [−4.92, 2.76] | 0.59 | −5.03 | −4.28 | +2.43 | +5.70 |
| KAN − MLP | −0.35 | [−4.76, 4.07] | 0.87 | −4.51 | −1.91 | +1.58 | +6.51 |
| KAN − LGB | +0.24 | [−5.14, 5.58] | 0.93 | +3.07 | −1.25 | +3.41 | −7.99 |

四模型的绝对超额均不显著（CI 含 0）；模型两两配对差也均不显著。KAN 相对 Linear 的
超额差呈明显的时间结构：2023–2024 落后（−5.0 / −4.3 bp/d），2025–2026 反超
（+2.4 / +5.7 bp/d）。

## 7. 结论（B6 归因后改写，取代「RankIC 与回测反转」的表面误读）

1. **预测层 KAN 占优**：RankIC 0.0264 四模型最高；pre-cost top-30 选股收益 8.15 bp/d
   高于 Linear 7.36 / LGB 6.59（次于 MLP 9.70，但 MLP 的 RankIC 0.0222 低于 KAN，
   KAN 的排序优势更均匀）；decile LS spread 29.4% 年化（CI 显著为正）。选股信号真实有效。
2. **组合层差异在噪声范围内**：k30/d1 主口径下 KAN 年化超额 3.46% 落后 Linear 6.11%，
   但 KAN−Linear 配对差 CI [−4.92, 2.76] bp/d（p=0.59）含 0；回测敏感性矩阵
   （topk {10,30,50} × drop {1,5} + 周/月频信号，4 模型 × 8 配置）中该差在 −8.2 ~ +11.4 pp
   间无规律翻转——**配置效应远大于模型效应**，单看任一配置的模型排名不可靠。
3. **2023 执行摩擦主导 KAN 落后**：drag 分解（ideal top30 → 实际回测，bp/d）显示全期
   KAN drag 3.83（cost 0.68 + friction 3.15）vs Linear 1.92（0.66 + 1.26）；其中 2023 年
   KAN friction 4.77 vs Linear 0.52，2024 年 5.59 vs 4.11，而 2025–2026 收敛至 0.91 / −0.20。
   触限归因**不支持**「KAN 偏选触限票」假说：2023 年理想持仓买入端触限率 KAN 0.36% ≈
   Linear 0.33%（全期 0.80% vs 0.67%，差异微小）；2023 高摩擦来自执行连锁（换仓节流、
   现金残留与触限拦截的次生效应）而非选票本身。
4. **2025–2026 KAN 超额占优**：KAN 日超额 4.65 / 4.23 bp/d vs Linear 2.22 / −1.48
   （Linear 2026 转负，MLP 2026 亦转负 −2.28；LGB 年度间波动大 2023 −5.21 / 2026 +12.21）；
   与分年 RankIC（KAN 2025–26 持平高位 0.0218 / 0.0287，Linear 降至 0.0195 / 0.0139）方向一致。
5. **降频调仓不是免费午餐**：qlib TopkDropoutStrategy 的 n_drop 同时是单日换仓节流阀，
   月/周频重采样信号不会线性降低换手（monthly 6.3% ≈ daily 6.7%），只改变换手结构与
   信号新鲜度；降频组超额差异（KAN −3.1% ~ +12.7%）主要由调仓时点运气主导
   （机制细节见 `common/runs/kan/portfolio-attr/README.md`）。

