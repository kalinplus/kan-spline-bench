# kan-spline-bench

KAN 因子合成 + 订单簿端到端预测（Track B）：B1 用 KAN 做因子非线性合成（与 LightGBM/MLP 同评测对比），B2 在 FI-2010 订单簿数据上做端到端收益预测（DeepLOB 复现与改进）。

## 共享基建

本仓库通过 symlink 引用 `agentic-feature-mining` 的共享基建：

```
common -> ../agentic-feature-mining/common
```

`common/` 提供 Qlib 数据管道、统一评测模块（IC/RankIC/分层/回测）、基线脚本与实验记录规范。单独 clone 本仓库时请按上述相对路径重建 symlink：

```bash
ln -s ../agentic-feature-mining/common common
```

数据与环境要求见 `agentic-feature-mining/common/README.md`。

## 当前状态

初始化完成（2026-09-14），共享基建已通过验收。下一步：B1（Alpha158 → LightGBM/MLP/KAN 三方对比）。
