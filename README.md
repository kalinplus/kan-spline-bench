# kan-spline-bench

在 Qlib + Alpha158 的标准化股票预测评测下，检验「用 KAN 做因子非线性合成」相比线性模型、LightGBM、MLP 到底强不强、强在哪，并且给每个结论配上统计可信度检验。

## 1. 这个项目在做什么

- **输入**：Alpha158——Qlib 自带的 158 个技术因子（K 线形态、动量、波动率等），A 股 CSI300 股票池（另在 CSI500 上复刻验证），2012–2026.07 日频行情。
- **被检验的对象**：KAN（Kolmogorov–Arnold Network）——一种把神经网络「权重乘加后过固定激活函数」换成「每条连线上放一条可学习的 B 样条曲线」的架构，卖点是学完之后每条曲线能抄成公式、天然可解释。本项目用的是 efficient-kan 实现。要回答的问题是：它拿来做因子合成（158 个因子 → 每日个股打分）是否真比经典模型强。
- **怎么评**：四个模型（Linear/Ridge、LightGBM、MLP、KAN）用同一份数据、同一套切分（train 2012–2020 / valid 2021–2022 / test 2023–2026.07）、同一个训练循环。核心指标是 RankIC——模型每天给全市场打分，打分与实际收益的 Spearman 秩相关按日取平均，衡量排序预测能力。纪律：超参选择和早停只看验证集，测试集每个配置只在最后评一次。
- **公平配对**：KAN [158,24,1]（38,160 参数，每条边 10 个：1 base + 8 样条系数 + 1 scaler）对 MLP [158,224,1]（35,841 参数），单隐藏层、参数量同量级，堵住「参数多所以好」的质疑。

## 2. 主要结论

- 表面上：B1 主表里 KAN 的 test RankIC 0.0264 四模型最高（MLP 0.0222、Linear 0.0197、LGB 0.0152）。
- 但换个种子就没了：5 臂 × 5 种子复核（B7）显示 KAN−MLP 差异在种子噪声内（配对 p=0.803）；B12 让 MLP 本地调参后 test RankIC 0.0265，直接追平 KAN。
- 立得住的 KAN 优势：
  1. **种子稳定性**：RankIC 的跨种子 std 0.0027 vs MLP 0.0072，稳 2.4 倍；
  2. **参数效率**：38k 参数达到 72k 参数 LGB 拼不过的水平，ICIR 同量级最优；
  3. **可解释性**：158 条最强样条边全部能贴合符号公式（R²≥0.99，92% 是 hump/gating 形态），蒸馏成纯公式层后 RankIC 保留 97.2%（B15）——这是四个模型里唯一能把「学到了什么」写成可读公式的。
- 训练口径（B9/B11）：「新数据比老历史重要」被否——rolling-5y 滚动窗口可信地劣于 expanding 累积全历史（2023 regime 切换年被滚动放大）。部署口径定为：**不丢历史 + 每年 expanding refit + 全局归一化**。
- 其余阴性结果：LGB→KAN 混合架构无可信增益（两者在 Alpha158 上信息集高度重叠，B10）；组合回测层配置效应大于模型效应，不作排名依据（B6）。
- 判读纪律：B7 定下 2σ 噪声尺（KAN 侧 0.0059、含 MLP 侧 0.0083），之后所有「显著 / 不显著」共用这把尺，差值落在尺内就叫噪声内，灰区绝不说成显著。

各实验的完整数字与口径见 `results/` 下对应文档，实验之间的关系见 `docs/design.md`。

## 3. 环境准备

```bash
# conda 环境（PyTorch、pyqlib、lightgbm 等已装好）
conda activate /data1/wujunxi/kailin/envs/py311

# 数据：qlib 格式 A 股日频行情，已在本地
#   /data1/wujunxi/kailin/data/qlib_data/cn_data

# 本仓库通过 symlink 复用 agentic-feature-mining 的共享基建
# （数据管道、IC/分层/回测评测模块）。单独 clone 本仓库时需重建：
ln -s ../agentic-feature-mining/common common

# 重建环境时注意：efficient-kan 必须带 --no-deps 安装，
# 否则会拖动 torch 升级把环境搞坏
pip install efficient-kan --no-deps
```

## 4. 复现命令

最小复现路径只有一条命令——B1 主表是全项目的协议源头，其余实验要么 import 它，要么只读它的产物：

```bash
python scripts/run_b1.py        # 训练 + 评测 linear/mlp/kan 三个模型
# 或只跑单个模型：python scripts/run_b1.py --model kan
```

首次运行会先提取 Alpha158 特征并落 parquet 缓存（`common/runs/kan/_cache/`），之后所有实验只读缓存不再碰 Qlib handler。单个模型训练在分钟级（主表实测 KAN 0.2 min、MLP 0.1 min，GPU）。

产物分两层：过程产物（模型权重、预测、逐日 IC、metrics.json）写 `common/runs/kan/<实验名>/`，不进 git；带结论的文档写 `results/*.md`，进 git。看结论去 `results/`，复算细节去 `runs/`。

其余实验按需复现（均已跑完，产物在仓库里）：

| 实验 | 回答的问题 | 脚本 | 结论在哪 |
|---|---|---|---|
| B1 主表 | 四模型同台，谁排序准 | `scripts/run_b1.py`（汇总 `scripts/summarize_b1.py`） | `results/B1_main_table.md` |
| B3 | KAN 超参敏不敏感、要多少数据 | `scripts/run_b3.py` | `common/runs/kan/{sweep,data-efficiency}/` |
| B4 | KAN 到底学到了什么（可解释性） | `scripts/run_b4_explain.py` | `results/B4_explain.md` |
| B6 | 组合回测层差异大不大 | `scripts/portfolio_attr.py --stage <stats\|attr\|sweep\|summary>` | `common/runs/kan/portfolio-attr/` |
| B7 | 结论扛得住换种子吗（定噪声尺） | `scripts/run_multi_seed.py` | `common/runs/kan/multi-seed/REPORT.md` |
| B8 | 换到 CSI500 还成立吗 | `scripts/run_b8_csi500.py` + `scripts/summarize_b8.py` | `common/runs/kan/universe-csi500/` |
| B9/B11 | 部署该怎么滚动训练 | `scripts/run_rolling.py` | `results/B9_rolling.md`、`results/B11_expanding_3seed.md` |
| B10 | LGB+KAN 混合有没有增益 | `scripts/run_b10_hybrid.py` | `results/B10_hybrid.md` |
| B12 | 基线是不是没调好才输 | `scripts/run_b12.py <w1\|w1b\|w2\|w3>` | `results/B12_baseline_significance.md` |
| B14 | 样条解读换种子还一样吗 | `scripts/run_b14_seed_shapes.py` | `results/B14_seed_stability.md` |
| B15 | 样条换成符号公式掉多少分 | `scripts/run_b15_symbolic_distill.py` | `results/B15_symbolic_distill.md` |

带 stage 参数的脚本，完整参数用 `--help` 看。两个可复现性要点：

- **回测要逐位可复现必须 `PYTHONHASHSEED=0`**：qlib `TopkDropoutStrategy` 持仓用 set 迭代，集合序依赖进程哈希种子，预测分有并列时回测 IR 会漂移；RankIC 是相关系数、与哈希无关，不受影响。
- **数据口径固定不动**：切分、label（`Ref($close,-2)/Ref($close,-1)-1`，即 T+1 收盘买 T+2 收盘卖）、处理器链全部定义在 `scripts/run_b1.py` 一处，不要在别的脚本里另起口径。

## 5. 目录结构

```
kan-spline-bench/
├── scripts/          # 全部实验代码，run_b1.py 是协议源头，其余脚本 import 它
├── results/          # 结论文档（进 git），一个实验一个 md
├── docs/             # design.md（系统怎么搭的）、reading-guide.md（半天读懂代码）
├── common -> ../agentic-feature-mining/common
│                     # symlink 到共享基建；过程产物物理上落在那边，不进 git
└── AGENTS.md         # 仓库特有约定（symlink、分支、环境）
```
