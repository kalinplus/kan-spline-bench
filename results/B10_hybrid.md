# B10 混合架构：LightGBM×KAN 残差学习（OOF）与输出 ensemble

口径：csi300 Alpha158 `_cache`（只读），train 2012–2020 / valid 2021–2022 / test 2023–2026.07，统一 CPU、seed 42。LGB = S1b 超参缓存重训（B4 `run_lgb` 同款，与 S1b test 预测 Pearson 0.955）；KAN = B1 配置 w24 CPU 重训；残差 KAN 与其同 `train_kan` 代码路径（仅早停准则换成组合预测 valid RankIC）。完整方法、OOF 折表与全部诊断：`common/runs/kan/hybrid/REPORT.md`；代码 `scripts/run_b10_hybrid.py`。本文件为结论摘要。

## 1. OOF 防污染设计（残差路线命门）

- LGB 在 train 内按自然年 expanding OOF：预测年 y 的模型先在 `2012-01~(y-1)-09` 上拟合、用 `(y-1)-Q4` 内部 valid 早停校准轮数（该 Q4 不进拟合集），再以冻结轮数在完整 `2012-01~(y-1)-12` 窗口 refit 后预测 y；覆盖 2013–2020 共 8 折，2012 无更早数据剔除（583360 行，折表见 `oof/oof_folds.json`）。
- 无泄漏：所有折窗口严格早于预测年（代码内有结构断言）；valid/test 不参与任何折模型；残差目标 = CSZScoreNorm label − OOF 预测（同一标准化空间作差、同空间相加回），禁止 in-sample 预测的要求由此满足（OOF R²=0.0029：LGB 只解释了 label 方差的极小部分，残差目标的可学空间充足，结论不受「残差太小」混淆）。

## 2. 残差结果表（valid / test 2023-01~2026-07）

| arm | valid_rankic | valid_ic | valid_rankicir | valid_rankic_global | test_rankic | test_ic | test_rankicir | test_rankic_2023 | test_rankic_2024 | test_rankic_2025 | test_rankic_2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LGB+KAN_res (residual) | 0.02542 | 0.02719 | 0.20600 | 0.02738 | 0.02689 | 0.01961 | 0.20300 | 0.01864 | 0.03796 | 0.02350 | 0.02796 |
| LightGBM (S1b retrain) | 0.01954 | 0.02284 | 0.12600 | 0.01945 | 0.01402 | 0.01187 | 0.09900 | 0.00255 | 0.02522 | 0.01643 | 0.01011 |
| KAN (CPU retrain s42) | 0.02884 | 0.03129 | 0.24200 | 0.02990 | 0.03045 | 0.02278 | 0.21500 | 0.02288 | 0.04314 | 0.02388 | 0.03314 |
| Linear/Ridge (B1 artifact) |  |  |  |  | 0.01973 | 0.01568 | 0.13400 |  |  |  |  |

- 判读（B7 阈值 0.0059=2σ）：残差臂−LGB test RankIC +0.01287（credible (>0.0059)），残差臂−KAN -0.00356（inside noise band (<0.0059)）；残差可学性诊断 KAN_res→valid 残差 Spearman +0.01704，KAN_res 输出 std 0.0568 vs 残差 std 0.9981。

## 3. ensemble 结果表

| arm | valid_rankic | valid_ic | valid_rankicir | valid_rankic_global | test_rankic | test_ic | test_rankicir | test_rankic_2023 | test_rankic_2024 | test_rankic_2025 | test_rankic_2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ensemble w*KAN+(1-w)*LGB | 0.02889 | 0.03152 | 0.23500 | 0.02976 | 0.02985 | 0.02255 | 0.21100 | 0.02156 | 0.04294 | 0.02401 | 0.03175 |
| LightGBM (S1b retrain) | 0.01954 | 0.02284 | 0.12600 | 0.01945 | 0.01402 | 0.01187 | 0.09900 | 0.00255 | 0.02522 | 0.01643 | 0.01011 |
| KAN (CPU retrain s42) | 0.02884 | 0.03129 | 0.24200 | 0.02990 | 0.03045 | 0.02278 | 0.21500 | 0.02288 | 0.04314 | 0.02388 | 0.03314 |

- 选中 w*=0.75（valid 日均 RankIC 在网格 [0.0, 0.25, 0.5, 0.75, 1.0] 上选优，全表 `ensemble_arm/weight_grid.csv`）；端点自洽 w=0≡LGB / w=1≡KAN 逐位复现 = True/True。判读：ensemble−LGB +0.01583（credible (>0.0059)），ensemble−KAN -0.00060（inside noise band (<0.0059)）。

## 4. 回测表（B1 口径 topk30/drop1 / SH000300 / open 5bp / close 15bp / limit 0.095）

| arm | excess ann. | IR | excess MDD | daily_turnover |
|---|---|---|---|---|
| LGB+KAN_res (residual) | 1.24% | 0.14 | -17.5% | 6.60% |
| ensemble w*KAN+(1-w)*LGB | 4.40% | 0.50 | -11.4% | 6.63% |
| LightGBM (S1b retrain) | 4.31% | 0.33 | -28.7% | 7.00% |
| KAN (CPU retrain s42) | 4.66% | 0.54 | -12.4% | 6.57% |
| Linear/Ridge (B1 artifact) | 6.11% | 0.58 | -19.3% | 6.67% |

- **#7 噪声带 caveat**：B6 已证该层模型间差异落在配对 bootstrap 噪声带内、配置效应大于模型效应（KAN−Linear 在 8 配置间 −8.2~+11.4 pp 无规律），本表看相对形态不作排名；Linear/Ridge 行为 B1 冻结产物引用（excess/IR/MDD/turnover recomputed arithmetically from the frozen B1 backtest_report.csv (ann = daily mean x 252, IR = mean/std x sqrt(252), MDD on compounded excess NAV); RankIC/IC quoted from B1 metrics.json）。

## 5. 结论

- **残差增量（Q1）：对 LGB 有加成，但非互补增量（不超过 KAN 单模型）。** LGB+KAN_res 相对 LGB 单模型 test RankIC 差 +0.01287（credible (>0.0059)），相对 KAN 单模型 -0.00356（inside noise band (<0.0059)）；valid 段差 +0.00589 / -0.00341。 机制诊断：LGB 的 OOF R² 仅 0.0029（valid R² 0.0005），「LGB 未解释的部分」≈ 整个 label，残差目标并非「小修正」而是近乎全量重学；KAN_res 对残差目标的全局 Spearman +0.01704（train OOF 残差上 +0.07110），输出 std 0.0568（收缩到残差 std 的 5.7%）——学到的信号与直接训练的 KAN 高度同源，没有超出 KAN 已覆盖范围的互补信息。

- **ensemble 增益（Q2）：部分：显著超过 LGB，但不超过（也不落后于）KAN——增益全部来自 KAN 侧内容。** valid 网格选中内部权重 w*=0.75，但相对次优点的 valid 优势仅 0.00005（次优 w=1），远低于噪声带——内点优势不具分辨力；test 差值：vs LGB +0.01583（credible (>0.0059)），vs KAN -0.00060（inside noise band (<0.0059)）；端点自洽 True/True（w=0/1 逐位复现单模型臂）。

- **组合层形态**：k30/d1 下超额 残差 1.24% / ensemble 4.40% / LGB 4.31% / KAN 4.66%，IR 0.14/0.50/0.33/0.54，超额 MDD -17.5%/-11.4%/-28.7%/-12.4%，日换手 6.60%/6.63%/7.00%/6.57%。预测层指标与组合层超额的相对顺序不完全一致（B1/B6 已定性的形态）：混合臂在组合层的相对位置同样落在 #7 噪声带内，不据此下模型优劣结论，只记录形态——混合是否改变换手/回撤结构比单点超额更有信息量。

- **最终判词：不进主叙事。** 最优单模型为 KAN（test RankIC 0.03045），两条混合臂相对它均无增益（残差 -0.00356 / ensemble -0.00060，均未越过 B7 噪声带阈值 0.0059）；「新旧结合」在 Alpha158/csi300 上没有可分辨的工程增益。这也是有信息量的否定：LGB 与 KAN 在该特征集上的信息集高度重叠（残差路线里 LGB 的 OOF R² 仅 0.003，「LGB 未解释部分」≈ 整个 label，KAN_res 学到的与直接 KAN 同源；ensemble 的 valid 网格从 w=0.5 起几乎平坦，内点优势 5e-5 量级），任何混合权重都不优于把全部权重交给较强的一方。混合架构不作为主线结论，仅在方法附录保留 OOF 残差构造（防泄漏折设计可复用）与端点自洽检查的做法。
