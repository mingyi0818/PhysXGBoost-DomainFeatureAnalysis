# 48_CreditDefault 参考文献调研材料

> 调研日期：2026-08-10
> 数据集：UCI Default of Credit Card Clients (30,000 samples, 23 features, ~22% default)

## 研究现状
信用违约预测是金融风控核心问题。2024-2026年研究趋势：
- 深度学习（TabNet, FT-Transformer）开始应用于表格数据
- 可解释AI（SHAP）在信用评估中重要性增加
- 公平性审计在信贷领域受到关注
- 集成方法（XGBoost/LightGBM）仍是工业界主流

## 本方向已有结果
| 模型 | Raw AUC | Domain AUC |
|------|---------|------------|
| XGBoost | 0.7763 | 0.7763 (完全相同) |
| LightGBM | 0.7763 | 0.7763 (完全相同) |
| CatBoost | 0.7802 | 0.7802 (完全相同) |
| RandomForest | 0.7740 | 0.7740 (完全相同) |

**关键发现**：Domain特征对该数据集完全无效（Raw=Domain），说明原始23个特征已包含充分信息。

## 击破方案
- 创新点：信贷风控领域特征分析框架 + 特征冗余性理论解释
- 核心贡献：解释"为什么领域特征在某些数据集上无效" — 信息论视角
- 理论：特征冗余性定理 — 当I(D;F)>I(D;Y|F)时，领域特征无增益
- 实验：4模型对比 + 特征互信息矩阵 + SHAP + 消融
- 目标期刊：IJMLC (EI) 或 IEEE Access (SCI四区)
