# AGENTS.md

Track B 仓库：KAN 因子合成 + 订单簿端到端预测。系统级工作流规范（Issue/PR 流程、六部分正文格式、状态标签、draft PR、merge 需用户确认）见个人 AGENTS.md，本文件只写仓库特有约定。

## symlink 关键约定

- `common -> ../agentic-feature-mining/common`：在本仓库改 `common/` 下的任何文件，实际修改的是 **agentic-feature-mining 仓库的工作区**——commit/push 必须切到那边执行，本仓库只跟踪 symlink 本身。
- 单独 clone 本仓库时需重建 symlink：`ln -s ../agentic-feature-mining/common common`（见 README）。
- Track B 的模型代码（KAN、DeepLOB 等）放本仓库；评测与数据管道一律 import `common`，不复制代码。

## Issue 与 PR

- 待办 issue 的全文草稿在 `docs/ISSUES.md`；GitHub issue 创建后以 issue 正文为准。
- 分支命名 `feature/<topic>`；PR 一律 draft 提交、base `main`，merge 由用户执行。
- 实验产物写 `common/runs/<exp_name>/`（物理上落在 agentic-feature-mining 仓库），规范同那边。

## 环境

- 同 Track A：`conda activate /data1/wujunxi/kailin/envs/py311`、数据 `/data1/wujunxi/kailin/data/qlib_data/cn_data`。
- B1 需额外安装 PyTorch 与 efficient-kan；B2 需 FI-2010 公开数据（单独下载，不入库）。
