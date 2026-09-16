# kan-spline-bench

KAN 因子合成：用 KAN 做因子非线性合成（与 LightGBM/MLP 同评测对比）。

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

B 系列实验已收尾（2026-09-15，Issue #1–#19 全部关闭）。结论速览：

**主表与模型对比（B1/B7/B8）**：csi300 拼接 test RankIC KAN 0.0264 > MLP 0.0222 > Linear 0.0197 > LGB 0.0152（B1）。但多种子复核（B7，5 seeds × 5 臂）证明「KAN 的 RankIC 优势」相对 MLP 不成立（p=0.803，差值在种子噪声内）——KAN 的可持续优势是种子稳定性（std 0.0027 vs 0.0072，2.4×）、IC 与 RankICIR，以及跨股票池的样条形状稳定性（B8 csi500：家族分层保持，KAN−MLP 互换在噪声带内）。可信差异阈值（2σ）：KAN 侧 0.0059、含 MLP 侧 0.0083，后续所有判读共用此尺。

**训练口径（B9/B11）**：recency hypothesis 被否——rolling-5y 可信劣于 expanding（KAN −0.0079），2023 regime 切换年被滚动放大而非平滑（KAN rolling −0.004 vs expanding +0.016）；老历史要保留、但权重须更新。expanding yearly refit 对 B1 静态切分 12/12 (model×seed) 差值为正（+0.0015~+0.0060，全部灰区；3 seeds 下 H0 偶然概率 6%–15%）。部署口径：**accumulate, don't truncate + 每年 expanding refit + 全局归一化**（B9 证明归一化窗口重拟合无增益）。

**组合层（B6）**：模型间差异落在配对 bootstrap 噪声带内、配置效应大于模型效应——组合层不作为模型排名依据，仅记录形态。

**混合架构（B10）**：LGB→KAN 的 OOF 残差学习与输出 ensemble 相对 KAN 单模型均无可信增益——LGB 与 KAN 在 Alpha158 上信息集高度重叠，「新旧结合」无工程价值；OOF 防泄漏折设计与 ensemble 端点自洽检查作为方法资产保留。

**可解释性（B4）**：top-12 样条图谱 + 符号贴合（158 条最强边全部 snap，R²≥0.99，92% hump/gating 形态）；KAN/LGB 重要性交叉核对 Spearman 0.527。

各实验结论文档见 `results/`（B1_main_table / B4_explain / B9_rolling / B10_hybrid / B11_expanding_3seed），过程产物见 `common/runs/kan/`（B3 sweep、B6 portfolio-attr、B7 multi-seed、B8 universe-csi500、B9–B11 rolling）。