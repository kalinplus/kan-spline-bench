# B11 expanding walk-forward 三种子复核：主发现升级评估（Issue #19）

口径：expanding × norm-global × 4 模型（KAN g5_k3_w24 / MLP [158,224,1] / Ridge / LightGBM）× 3 seeds（42/43/44），各 refit 年模型只评对应 test 自然年、四段拼接成 2023-01~2026-07 全序列（B9 walk-forward 纪律，无泄漏），对照 B1 静态全窗口（冻结产物，GPU seed-42 单点）。B1 的不确定性按 Issue #19 预钉规则引用 B7 五种子方差尺（2σ：KAN 0.0059、含非 KAN 模型 0.0083）。

实现与验证：先盘点后补训——开工时磁盘实测（以 `metrics.json` 完成为格标志）：s42 全 64 格、s43 KAN/MLP 全 32 格（Ridge/LGB 全缺）、s44 KAN/MLP 28/32（refit_2023–2025 各 8 格齐全，refit_2026 仅 4 格 rolling，另有 1 个无 metrics.json 的中断空目录 `refit_2026/kan_expanding_global_s44`，按缺失处理）。主表缺格恰 18 个：s43 Ridge/LGB 8 + s44 Ridge/LGB 8 + s44 KAN/MLP refit_2026 各 1，全部补齐（CPU 约 15 分钟，最贵为 refit_2026 KAN 一格）；**124 个已存在格一个未动**（训练前后两次盘点比对，见 `inventory_b11.csv`，仅缺格翻转为 `new_b11`）。norm-window 不新训（B9 已证归一化因子差值全在噪声带内），补充列直接引用 KAN/MLP s42/s43 已有数据。Ridge 逐位复现、S1b LightGBM 协议不注入 seed（s43/s44 与 s42 指标逐位一致），seed 变异只来自 KAN/MLP——如实报告，不作伪方差。B9 可复现性：`scripts/run_rolling.py` 的 `--stage all` 管线未动（新增 `inventory`/`b11` 为独立 stage、训练臂过滤参数默认关闭）；`--stage report` 再生成的 REPORT.md 与改前逐字节 diff 仅为 §5b 两句 seed-44 文案按目录实际状态修正（原 18/32 与 14/32 自相矛盾，实为 28/32 完成且无完整拼接臂）。

## 1. 主表：expanding × norm-global 拼接 test RankIC（3 seeds，逐 seed 值）

| Model | s42 | s43 | s44 | mean±std | 2023 (mean) | 2024 (mean) | 2025 (mean) | 2026 (mean) |
|---|---|---|---|---|---|---|---|---|
| Ridge | 0.02472 | 0.02472 | 0.02472 | 0.02472±0.00000 | +0.0180 | +0.0319 | +0.0245 | +0.0243 |
| LightGBM | 0.02073 | 0.02073 | 0.02073 | 0.02073±0.00000 | +0.0061 | +0.0272 | +0.0257 | +0.0265 |
| MLP | 0.02819 | 0.02651 | 0.02373 | 0.02614±0.00226 | +0.0200 | +0.0359 | +0.0243 | +0.0231 |
| KAN | 0.03196 | 0.02906 | 0.03019 | 0.03041±0.00146 | +0.0145 | +0.0454 | +0.0248 | +0.0423 |

n=3 诚实条款：std 为 ddof=1（自由度 2），只作散布参考；判读以逐 seed 值、符号一致性与阈值对照为准。逐 seed 分年值见 `common/runs/kan/rolling/REPORT.md` B11 小节与 `b11_main.csv`。补充列（不新训，引用已有格）：expanding × norm-window KAN s42 0.03183 / s43 0.02918；MLP s42 0.02773 / s43 0.02819（s44 无完整 norm-window 拼接臂）。

## 2. expanding − B1 逐 seed 差值与升级判定（预钉三分支规则）

| Model | d(s42) | d(s43) | d(s44) | mean±std | 2σ 阈值 |
|---|---|---|---|---|---|
| Ridge | +0.00499 | +0.00499 | +0.00499 | +0.00499±0.00000 | 0.0083 |
| LightGBM | +0.00558 | +0.00558 | +0.00558 | +0.00558±0.00000 | 0.0083 |
| MLP | +0.00598 | +0.00430 | +0.00152 | +0.00393±0.00226 | 0.0083 |
| KAN | +0.00554 | +0.00264 | +0.00377 | +0.00398±0.00146 | 0.0059 |

证据核对：4 模型 3-seed 均值全部同号为正 ✔；12 个 (model × seed) 差值全部为正 ✔；|mean diff| 超出对应 2σ 阈值的模型数 = **0**（最大为 LGB +0.00558 < 0.0083、KAN +0.00398 < 0.0059）。

**判定落字：分支 B——维持「方向一致灰区、建议采纳」**（升级分支 A 要求 ≥2 模型过阈值，未满足；降级分支 C 要求出现异号，未出现）。

## 3. 结论

- 主发现**不升级为「可信主发现」**，维持 B9 表述并加误差棒：expanding yearly refit 优于 B1 静态全窗口在 4 模型 × 3 seeds 共 12 个差值上方向完全一致（全部 +0.0015~+0.0060），但没有任何模型均值越过 2σ 阈值——是方向高度稳定的灰区效应，不是统计显著效应。B9 的「逐年累积」形态在三 seed 下保持（KAN 2023 均值 +0.0145 → 2026 均值 +0.0423）。
- 部署口径维持 B9 结论不变：**accumulate, don't truncate + 每年 expanding refit**，归一化层维持全局。证据强度限定：n=3（自由度 2），方向一致性证据，非推断统计。
- 产物：`common/runs/kan/rolling/{inventory_b11.csv, b11_main.csv, b11_3seed.json, REPORT.md B11 小节, stitched/<model>_expanding_global_s{43,44}/, stitched/{kan,mlp}_expanding_window_s43/}`；代码：`scripts/run_rolling.py`（`--window/--norm` 训练过滤 + `inventory`/`b11` stage），另修正 `results/B9_rolling.md` 两处 seed-44 计数自相矛盾与 §3④「不截断」表述（改为「不丢历史，尾部 2 日截尾同规」）。
