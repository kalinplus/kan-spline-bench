# 面试准备：KAN 因子合成项目阅读指南

目标：半天读完最小集合，能扛住对这个项目的代码级追问。配套的项目结论与面试知识清单见本文件末尾第 5 节与 results/ 各结论文档。

## 1. 阅读顺序（4 个文件，约 600 行）

### ① efficient-kan 源码（~150 行，最优先）

路径：`/data1/wujunxi/kailin/envs/py311/lib/python3.11/site-packages/efficient_kan/kan.py`（全文件仅 285 行）

| 位置 | 读什么 | 回答什么问题 |
|---|---|---|
| `KANLinear.__init__` (L7-55) | `base_weight` / `spline_weight` / `spline_scaler` 三个张量的形状 | "每边 10 参数"的出处：1 base + 8 spline 系数 + 1 scaler；手算 38160 靠它 |
| `b_splines` (L78-112) | 样条基函数怎么算：x 过 SiLU 得 base 项，B 样条基展开得 spline 项 | KAN 前向的核心，读懂这一个函数就通了大半 |
| `forward` (L153-168) | 三个张量怎么乘起来（10 行） | 前向完整链路 |
| `KAN.__init__` / `forward` (L240-280) | 层怎么堆叠 | [158,24,1] 结构落地 |

跳过：`update_grid`、`curve2coeff`、`regularization_loss`（本项目训练未用 update_grid，正则不是叙事重点）。

### ② scripts/run_b1.py（~250 行，全项目协议源头）

所有后续实验（B3/B7/B8/B9/B15）都复用这个文件的定义，读它等于读了半个项目。

| 位置 | 读什么 | 回答什么问题 |
|---|---|---|
| `prepare_data` + `xy` (L72-131) | Alpha158 提取、RobustZScoreNorm 处理链、parquet cache | 输入分布与 ±3 clip 的来历 |
| `train_torch` (L143-171) | AdamW + MSE + valid RankIC 早停 patience 15 | 所有模型共用的训练循环；B7 说"统一 CPU 码路径"就是它 |
| `evaluate_dump` (L185-219) | test 评分、IC 落盘、回测入口 | "test 只评一次"纪律在代码哪里体现 |
| `run_mlp` / `run_kan` (L288-307) | 两个模型定义 | [158,224,1] vs [158,24,1] 的公平配对 |

跳过：`dump_spline_curves`（画图）、`run_linear`、`main` 的参数解析。

### ③ common/eval/ic.py（~30 行，指标定义）

路径：`/data1/wujunxi/kailin/repeat/agentic-feature-mining/common/eval/ic.py`

- `daily_ic` (L4)：按交易日 groupby，当日 pred 与 label 的 spearman/pearson。
- `ic_summary` (L15)：日度 IC 序列取 mean（=IC/RankIC）与 mean/std（=ICIR/RankICIR）。

### ④ scripts/run_b4_explain.py 的两个函数（~120 行，可解释性叙事）

| 位置 | 读什么 | 回答什么问题 |
|---|---|---|
| `extract_edges` (L81-125) | 从训练好的 KAN 取每条边的样条曲线（网格 [-3,3]×401 点、边 L1 范数排重要性） | "你怎么知道 KAN 学到了什么" |
| `fit_candidate` (L202-233) | 符号 snap：8 个候选函数族做 `a·f(b·x+c)+d` 仿射拟合取 R² 最高 | R²≥0.99 这个数字的判定处 |

跳过：画图、密度、LGB 重训部分（结论在 `results/B4_explain.md`，面试不会问到代码级）。

## 2. 按需查阅（不进最小集合）

- `scripts/run_multi_seed.py` 的 `paired` (L164-183)：B7 配对差与 p 值的算法，只有被追问"配对检验具体怎么做"才看。
- `scripts/run_rolling.py`（1166 行）：只读文件头注释和 `common/runs/kan/rolling/REPORT.md`，refit/stitch 概念在报告里更清楚。
- `scripts/run_b10_hybrid.py` / `run_b12.py` / `run_b14_seed_shapes.py` / `run_b15_symbolic_distill.py`：都是复用上述路径的变体，results/ 结论文档足够支撑面试。

## 3. 读完自检（四条全过才算完）

1. 不看材料手算 KAN 参数量：每边 10 参数（1 base + 8 spline + 1 scaler），(158×24 + 24)×10 = 38160。
2. 说出早停准则和 patience：valid 全局 Spearman RankIC，patience 15，best epoch 恢复。
3. 写出 RankIC 计算式：按日 groupby 求 pred 与 label 的 Spearman，再对日度序列取均值。
4. 解释一条样条边如何 snap 成 sigmoid：8 候选族 × 仿射 wrap a·f(b·x+c)+d，最小二乘拟合取 R² 最高者，R²≥0.99 判成功。

## 4. 五道陷阱题（答不好会暴露项目不是自己做的）

1. **"主表说 KAN 最好，后面又说 KAN 不比 MLP 强，信哪个？"**
   点估计与统计可信度的区别：B1 是单种子抽样（KAN 0.0264 vs MLP 0.0222）；B7 五种子配对检验差值 −0.00068、p=0.803，在噪声带内。KAN 立得住的是种子稳定性（std 0.0027 vs 0.0072，2.4×）、参数效率和可解释性。B12 还证明 MLP 本地调参后 0.0265 追平 KAN。
2. **"KAN 38160 个参数怎么算的？"**
   见自检第 1 条；对侧 MLP [158,224,1] = 158×224+224+224×1+1 = 35841，参数比 1.06。
3. **"label 有没有前视偏差？"**
   `Ref($close,-2)/Ref($close,-1)-1`（T+1 收盘买、T+2 收盘卖）；walk-forward refit 训练窗尾部截 2 个交易日；B9 如实披露了 valid 年末与 test 年头 2 日的 label 重叠（与 B1 同规，不影响臂间比较）。
4. **"0.0264 vs 0.0222 怎么证明不显著，而不是你实验没做好？"**
   多种子方差尺（KAN 侧 2σ=0.0059、含 MLP 侧 0.0083）+ 同种子配对设计（配对 std 0.0057 消掉共同变异）+ MLP 低端抽签证据（B1 的 MLP 0.0222 落在自己 5 种子分布低端，均值 0.0286 高于 KAN 的 0.0279）。
5. **"你说 KAN 可解释，换个种子解释还一样吗？"**
   B14 三层结构：因子级响应曲线跨种子高度稳定（|corr|≥0.88，至多镜像翻转，镜像属规范对称性 φ→−φ 配二层权重变号）；top-12 名单半稳定（最少重叠 7/12）；形状命名与方向符号不稳定。引用时必须分层："核心因子存在稳定非线性响应曲线"可讲，具体读法只能归为"主模型的读法"。

## 5. 项目叙事速查（30 秒版本）

在 Qlib + Alpha158 的标准化评测下，用 KAN 做因子非线性合成，与 Linear/Ridge、LightGBM、MLP 同数据同口径对比。表面 RankIC KAN 最高（0.0264），但多种子复核证明 KAN−MLP 差异在种子噪声内（p=0.803）；KAN 真正的可信优势是 2.4× 种子稳定性、同量级参数（38k vs 36k）下的 ICIR，以及唯一能把学到的函数蒸馏成 158 条可读公式而 RankIC 保留 97.2%（IC 68.3%，幅度被压缩）。训练口径结论：不丢历史 + 每年 expanding refit + 全局归一化（rolling-5y 可信劣于 expanding）。组合层配置效应大于模型效应，不作排名依据。所有"不显著"一律不声称显著，判定共用 2σ 噪声尺（0.0059/0.0083）。

各实验完整数字与口径见 `results/`（B1 主表 / B4 可解释性 / B9 滚动 / B10 混合 / B11 三种子 / B12 基线公平性 / B14 seed 稳定性 / B15 符号蒸馏）与 `common/runs/kan/multi-seed/REPORT.md`（B7）。
