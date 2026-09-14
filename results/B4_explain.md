# B4：KAN 可解释性 — 样条形状 · 符号化 · 重要性交叉印证

分析对象：B1 主 KAN（`[158,24,1]`，grid 5 / order 3 / ±3），纯事后分析不重训；LightGBM 侧按
Issue 允许重训一次导 gain importance（B1 同一份缓存数据，S1b 官方超参，seed 42，与 S1b 落盘
test 预测 Pearson 0.955 验证一致性）。完整方法、口径与全部表格：`common/runs/kan/explain/explain_report.md`；
本文件为结论摘要。

## 1. 样条形状：KAN 把因子扭成了什么

图集 `explain/spline_curves_top12_labeled.png`（边范数 top-12，验收要求 ≥10）。最强边判读
（自动分类 + 人工复核，修正记录见完整报告）：

- **单调门控（延迟激活 S 型）**：RESI5/RESI30/ROC30 —— 标准化值 x>0~1 后才显著激活，
  snap 成 sigmoid/tanh（R² ≥ 0.997）；"模型自己发现了乖离率/动量的阈值门控"。
- **单调递减（端部饱和）**：RESI60、CORD30、STD60、WVMA30、BETA60 —— 左端饱和平台后
  近线性下坠；RESI60 为对数压缩型（snap sgn·log，R² 0.993）。
- **峰谷形（U / 倒 U）**：RESI20、MA5、KLOW、MIN60 —— 谷/峰底系统性偏左（≈ −1.5σ），
  snap 成单瓣 sin（R² ≥ 0.998），本质是端部平缓的饱和峰谷。
- 全量 158 特征最强边分布：峰谷形及饱和变体 145 条（92%）vs 纯单调 13 条。

## 2. 符号化：snap 结果与不可 snap 边

手动函数族拟合（8 候选 × affine wrap a·f(b·x+c)+d；未用 pykan——与 efficient-kan 权重布局
不兼容且有 torch 升级风险，如实报告）：

- 全量 3,792 边：**3,681 snapped（97.2%）、44 borderline、60 not snappeable、7 negligible 死边**；
  60 条不可 snap 边全部为弱边（max|φ| 中位 0.016 vs snapped 边 0.116），逐条标注不硬凑。
- **158 个特征的最强边全部 snap 成功，R² ≥ 0.99**。
- 解读口径：sigmoid ≡ tanh（affine 等价族）；snap 成 sin 的边 99.3% 在定义域内不足一个周期，
  sin 在此是"端部平缓峰谷"的基函数而非周期信号；纯线性 x 零条——没有一条边是线性的。

## 3. 重要性交叉印证：KAN 边范数 vs LightGBM gain

| 口径 | 结果 |
|------|------|
| 全量 158 特征 rank Spearman | **0.527** |
| top-20 交集（7） | RESI5、CORD30、STD60、MIN60、ROC30、KMID、VWAP0 |
| 仅 KAN top-20（13） | RESI{10,20,30,60}、MA5、KLOW、WVMA30、BETA60、QTLD10、OPEN0、MIN10、KLEN、BETA10 |
| 仅 LGB top-20（13） | STD30、KSFT、MIN30、IMXD5、MAX60、CORD60、QTLU5、IMIN5、CORD5、RANK10、LOW0、MA10、RANK5 |

分歧讨论：

1. **RESI 家族**：KAN 五个窗口乖离率全部进 top-20（份额合计 6.6%），LGB 只认 RESI5。树模型对
   共线特征只挑主分裂点；KAN 为每个窗口独立学出可读门控形状，重要性摊到家族。
2. **集中度**：LGB gain 集中（top-1 STD30 占 7.3%），KAN 边范数扁平（top-1 仅 1.57%）——参数化
   结构差异，跨模型比排名合理、比份额需谨慎。
3. **LGB 独有**：STD30/KSFT/IMXD5 等，树擅长少数高信息量切点（重尾形态量尤甚）；KAN 受 ±3 截断
   与样条平滑影响未将其排进 top-20，但整体 RankIC 反超（0.0264 vs 0.0152）——两模型在互补
   特征集上各有优势。
4. **一致处**：7 个交集因子是两种参数化收敛到的同一批核心量价信号，与 B1 分层回测稳健层同源。

## 4. 产物索引（`common/runs/kan/explain/`）

`explain_report.md`（完整报告）· `spline_curves_top12_labeled.png`（图集）·
`shape_classification.csv`（形状指标+分类）· `symbolic_snap_edges.csv`（全量边 snap 明细）·
`kan_edge_importance.csv` / `lgb_gain_importance.csv`（双模型重要性）·
`importance_compare.csv` / `compare_summary.json`（对比）· `lgb_retrain_meta.json`（LGB 协议与
S1b 一致性验证）· `edge_curves_all.npz`（全量曲线）。脚本：`scripts/run_b4_explain.py`。
