# Issue 计划

按系统级 AGENTS.md 的 Issue/PR 工作流管理：本文档是待开 issue 的全文草稿，GitHub issue 创建后以 issue 正文为准，本文档仅作计划存档。

## B1 KAN 因子合成对比（status:ready）

- **目标**：Alpha158 因子 → LightGBM / MLP / KAN（efficient-kan）三模型合成对比，作为"新架构 vs 传统 GBDT"的严格对照。
- **边界**：复用共享基建（`common/eval`、S1b 管道、同一 `base.yaml` 区间与成本），只新增 MLP/KAN 模型实现；不做因子挖掘、不动回测口径。
- **验收条件**：
  1. 三模型在主配置区间（test 2023–2026.07）出 IC/RankIC/ICIR/分层/回测全套，并入基线对比表（与 S1b LightGBM 同列可比）；
  2. KAN 学到的样条变换曲线可视化（可解释性产出）；
  3. 训练配置、种子、产物按 `common/runs/` 规范落盘。
- **依赖与资料**：共享基建（已验收）；efficient-kan 实现；需额外安装 PyTorch。
- **当前交接**：待开工；LightGBM 列已由 S1b 填好。
- **关联实现**：分支 `feature/kan-synth`。

## B2 FI-2010 DeepLOB 复现与改进（status:ready）

- **目标**：FI-2010 公开订单簿数据上复现 DeepLOB 基线，再做结构改进（attention / KAN 层替换）。
- **边界**：只用 FI-2010 公开数据（A 股 L2 因获取限制不在范围，报告中主动说明）；改进以消融形式报告，若做不出正向差异则转"失效模式分析"产出。
- **验收条件**：
  1. DeepLOB 复现指标与论文对齐（F1 偏差 ≤2%）；
  2. 至少一组结构改进的消融表，或失效模式分析报告；
  3. 数据管道与训练脚本可复现。
- **依赖与资料**：DeepLOB 论文与官方实现；FI-2010 数据集。
- **当前交接**：待开工。
- **关联实现**：分支 `feature/deeplob`。
