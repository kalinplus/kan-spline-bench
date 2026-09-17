# B9 滚动重训练：yearly walk-forward vs 全窗口（Issue #15）

口径：4 个 refit 点（2023-01 / 2024-01 / 2025-01 / 2026-01），每次 refit 只用该点之前的数据训练，训练窗尾部截 2 个交易日（label 为 T+1→T+2 收盘，严格无泄漏）；refit 前一年做 valid（早停 / Ridge α / LGB 早停均只用它）；每个 refit 模型只评对应 test 自然年，四段拼接成 2023-01~2026-07 全序列，与 B1 全窗口直接可比。两因素设计：窗口臂（rolling-5y = 回看 5 年，其中末年划做 valid，即 4 年训练；expanding = 2012-01 起）× 归一化臂（norm-global = 复用 2012–2020 全局 RobustZScoreNorm，即现有 `_cache`；norm-window = 每 refit 在各自截尾训练窗重拟合）。

实现与验证：未归一化 Alpha158 一次性提取到 `common/runs/kan/rolling/_raw/`，qlib 处理链（nanmedian/nanMAD×1.4826 → clip ±3 → Fillna(0)；label 另过 DropnaLabel(空操作)→CSZScoreNorm）用 pandas 复现；全局拟合版与现有 `_cache` 逐值对齐 **max|Δ| = 0**（train/valid/test_feat 三段全部为 0，门槛 1e-5，见 `align_check.json`），对齐门槛通过后才允许跑窗口归一化。四模型（KAN g5_k3_w24 / MLP [158,224,1] / Ridge α 网格每 refit valid 重选 / LightGBM S1b 超参重训早停），CPU、seed 42、B1/B3/B7 训练协议；norm-window 臂的归一化统计拟合在截尾后的训练窗上（label 的 median/MAD 也不触 valid 年价格）。归一化 caveat：norm-global 臂只滚模型权重、特征归一化层不滚（与 B1/B3/B7/B8 共用同一份归一化，实验内可比性完整）；norm-window 臂是归一化层彻底滚动的对照。边界说明：按 Issue 规定仅训练切片截尾 2 日；valid 年末尾 2 交易日的 label 用到下一年 test 段头两天的收盘（B1 的 valid 2021–22 与 test 2023 之间存在完全相同的 2 日重叠），早停/选 α 最多见到 2/242 天的重叠，且所有臂与 B1 参照口径一致，不影响比较。成本：64 次训练 + 灰区补种子（见下）。

## 1. 主表（拼接 test 2023-01 ~ 2026-07，RankIC 为日均值）

| Model | Window | Norm | RankIC | IC | RankICIR | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| Linear/Ridge | rolling-5y | global | 0.0228 | 0.0133 | 0.164 | 0.0072 | 0.0281 | 0.0235 | 0.0404 |
| Linear/Ridge | rolling-5y | window | 0.0227 | 0.0127 | 0.164 | 0.0076 | 0.0270 | 0.0232 | 0.0414 |
| Linear/Ridge | expanding | global | 0.0247 | 0.0174 | 0.164 | 0.0180 | 0.0319 | 0.0245 | 0.0243 |
| Linear/Ridge | expanding | window | 0.0247 | 0.0176 | 0.164 | 0.0179 | 0.0322 | 0.0246 | 0.0240 |
| LightGBM | rolling-5y | global | 0.0107 | 0.0034 | 0.075 | -0.0094 | 0.0067 | 0.0256 | 0.0274 |
| LightGBM | rolling-5y | window | 0.0135 | 0.0048 | 0.092 | -0.0122 | 0.0127 | 0.0272 | 0.0365 |
| LightGBM | expanding | global | 0.0207 | 0.0145 | 0.149 | 0.0061 | 0.0272 | 0.0257 | 0.0265 |
| LightGBM | expanding | window | 0.0186 | 0.0127 | 0.140 | 0.0059 | 0.0204 | 0.0247 | 0.0275 |
| MLP | rolling-5y | global | 0.0231 | 0.0110 | 0.141 | 0.0176 | 0.0338 | 0.0127 | 0.0328 |
| MLP | rolling-5y | window | 0.0229 | 0.0095 | 0.136 | 0.0179 | 0.0336 | 0.0137 | 0.0295 |
| MLP | expanding | global | 0.0282 | 0.0167 | 0.188 | 0.0196 | 0.0412 | 0.0289 | 0.0189 |
| MLP | expanding | window | 0.0277 | 0.0167 | 0.169 | 0.0212 | 0.0375 | 0.0298 | 0.0181 |
| KAN | rolling-5y | global | 0.0241 | 0.0142 | 0.155 | -0.0038 | 0.0328 | 0.0277 | 0.0523 |
| KAN | rolling-5y | window | 0.0237 | 0.0135 | 0.152 | -0.0048 | 0.0331 | 0.0270 | 0.0522 |
| KAN | expanding | global | 0.0320 | 0.0217 | 0.215 | 0.0156 | 0.0435 | 0.0296 | 0.0450 |
| KAN | expanding | window | 0.0318 | 0.0216 | 0.214 | 0.0154 | 0.0429 | 0.0300 | 0.0449 |
| *B1 全窗口参照* | full (2012-20) | global | 见 B1 | 见 B1 | 见 B1 | 见 B1 | 见 B1 | 见 B1 | 见 B1 |

B1 全窗口参照（GPU seed 42，test 2023-01~2026-07）：Ridge 0.0197（2023 0.0153 / 2024 0.0276 / 2025 0.0195 / 2026 0.0139）、LGB(S1b) 0.0152（0.0034 / 0.0282 / 0.0156 / 0.0120）、MLP 0.0222（0.0142 / 0.0231 / 0.0257 / 0.0288）、KAN 0.0264（0.0177 / 0.0386 / 0.0218 / 0.0287）。

## 2. 差值判读（对照 B7 种子噪声阈值：KAN 侧 0.0059、含非 KAN 侧 0.0083，2σ）

关键差值（拼接 test RankIC）：

| 对比 | KAN | MLP | Ridge | LGB |
|---|---|---|---|---|
| rolling − expanding（norm-global） | **-0.0079（可信变差）** | -0.0051（灰区） | -0.0019（噪声带内） | **-0.0100（可信变差）** |
| rolling − expanding（norm-window） | **-0.0082（可信变差）** | -0.0048（灰区） | -0.0020（噪声带内） | -0.0052（灰区） |
| norm-window − norm-global（rolling 臂） | -0.0004（噪声带内） | -0.0002（噪声带内） | -0.0001（噪声带内） | +0.0027（噪声带内） |
| norm-window − norm-global（expanding 臂） | -0.0001（噪声带内） | -0.0005（噪声带内） | +0.0000（噪声带内） | -0.0021（噪声带内） |
| rolling − B1 全窗口 | -0.0023（噪声带内） | +0.0009（噪声带内） | +0.0031（灰区） | -0.0044（灰区） |
| expanding − B1 全窗口 | +0.0055（灰区） | +0.0060（灰区） | +0.0050（灰区） | +0.0056（灰区） |

8 组窗口臂差值全部为负号（3 组可信变差、0 组可信变好、其余灰区/噪声带）；归一化臂 8 组差值全部远低于阈值。expanding−B1 四个模型同号为正且都落在灰区——一致的方向性提示（expanding 每 refit 用到了 2024-12 的数据，比 B1 的 2020-12 更新），但单看都不达 2σ。

灰区补种子复核（Issue #15 边界条款：MLP rolling−expanding 落 0.003–0.008 灰区 → KAN/MLP 两臂补 seeds 43–44）：seed 43 全 32 格完成；seed 44 按目录实际状态为 28/32 格完成（refit_2023–2025 各 8 格齐全，refit_2026 仅 4 格 rolling，另有 1 个无 metrics.json 的中断空目录），无任何完整拼接臂，故复核用 seeds 42+43。四臂（KAN/MLP × global/window）差值全部同号：KAN global −0.0079→−0.0051、KAN window −0.0082→−0.0057、MLP global −0.0051→−0.0062、MLP window −0.0048→−0.0066——rolling 的劣势是方向性稳定的，不是 seed luck；MLP 灰区差值在 s43 不缩反深。证据：`common/runs/kan/rolling/REPORT.md` §5b 与 `seeds_check.json`。

## 2b. LGB norm 双臂差异说明（B12 增补，回应外部评审 sanity check）

四模型中 LGB 是 norm 两臂差值稍大的一支（norm-window − norm-global：rolling 臂 +0.0027 / expanding 臂 −0.0021，其余三模型均在 ±0.0005 内），但仍全部远低于 B7 阈值、方向在两臂间不一致。幅度集中于树模型可由处理链的两处非线性解释：① RobustZScoreNorm 的 clip ±3 是非线性压缩，norm-window 重拟合出的 median/MAD 略移，会改变各分位落进 clip 区间的位置——被压到 ±3 边界的样本集合不同；② Fillna(0) 把缺失填在归一化尺度的 0 点，而 0 所对的分位位置随 median/MAD 漂移。树模型的分裂阈值直接作用在归一化值上，对输入的单调重参数化本应不变（这是树模型仅存的不变性）；正是 clip 的非线性和 Fillna 的分位位置漂移破坏了这种单调不变性，使归一化统计的微小漂移移动了样本跨越分裂阈值的命中集合，且效应方向随窗口/年份而异。这是预期行为而非管线异常：对齐门槛 max|Δ|=0 保证的是 norm-global 臂与 `_cache` 逐值一致，norm-window 臂本来就是有意不同分布的特征，两臂差值落在噪声带内说明全局归一化没有系统性吃亏。

## 3. 结论

- **① rolling ≥ expanding 不成立——方向相反**。8 组窗口差值全负，KAN（-0.0079/-0.0082）与 LGB（-0.0100）可信变差，MLP 灰区偏负，Ridge 噪声带内偏负。滚动 5 年窗不是升级是降级。
- **② 2023 塌方年没有被平滑——被放大**。KAN 2023 RankIC：rolling -0.0038 / expanding +0.0156 / B1 +0.0177；LGB：-0.0094 / +0.0061 / +0.0034——rolling 把 2023 直接打成负 RankIC，expanding 与 B1 保持正值。2018–2021 训练窗全是 regime 切换前的数据，恰在 regime 破裂处泛化最差。
- **③ recency hypothesis 在 test 上被否**。B3「50% ≈ 100%」成立于 2021–2022 valid（平静期，老数据是死重）；但 2023–2026 test 恰好证明老历史在 regime 切换处是保险——丢掉 2012–2017 是净损失。recency 设定的是「必须保留多新」的下限，不是「可以丢多老」的许可。
- **④ 部署口径：保留全窗口（accumulate, don't truncate）**。rolling-5y 可信差于 expanding 且不优于 B1；若需要周期性重训，expanding walk-forward（每年 refit、训练窗 2012 起不丢历史，尾部 2 日截尾同规）与 B1 灰区内同向略优，是安全配方。归一化层维持全局（窗口重拟合无增益，8 组差值全在噪声带内）。

## 4. 产物

逐 refit 格子、拼接序列、对齐证据、判读全表与补种子复核：`common/runs/kan/rolling/`（REPORT.md、summary.csv、comparisons.csv、seeds_check.json、align_check.json、`refit_<year>/`、`stitched/`、`_raw/`）；代码 `scripts/run_rolling.py`（本仓库）。组合层不评（#7 已定论）。
