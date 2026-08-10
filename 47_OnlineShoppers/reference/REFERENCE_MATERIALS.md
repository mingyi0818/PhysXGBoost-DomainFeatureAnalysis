# 47_OnlineShoppers 参考文献调研材料

> 调研日期：2026-08-10
> 数据集：UCI Online Shoppers Intention (12,330 sessions, 18 features, ~15.5% positive)

## SOTA文献（2020-2026）

| 编号 | 文献 | 年份 | 方法 | AUC | 局限 |
|------|------|------|------|-----|------|
| S1 | Plasun et al. | 2020 | RF/KNN/LR/XGBoost | AUC~0.93 | 无统计检验，无深度学习 |
| S2 | RF vs XGBoost对比 | 2025 | RF(200 trees)+XGBoost | AUC~0.93 | 仅2模型，特征工程简单 |
| S3 | L1-Logistic Regression | 2019 | L1惩罚LR | 未披露完整指标 | 单一模型 |

## 研究空白
1. **2024-2026年无新学术论文**使用该数据集 — 巨大研究空白
2. 无LightGBM/CatBoost系统对比
3. 无SHAP可解释性分析
4. 无统计显著性检验
5. 无电商领域特征工程框架

## 本方向已有结果
| 模型 | Raw AUC | Domain AUC |
|------|---------|------------|
| XGBoost | 0.9038 | 0.9048 |
| LightGBM | 0.9033 | 0.9008 |
| CatBoost | 0.8992 | 0.8995 |
| RandomForest | 0.8993 | 0.8994 |

## 击破方案
- 创新点：电商行为领域特征工程框架（会话深度/用户活跃度/时间模式/页面交互）
- 理论：信息论特征交互分析 + 特征冗余度量化
- 实验：4模型公平对比 + SHAP + 统计检验 + 消融
- 目标期刊：IJMLC (EI) 或 IEEE Access (SCI四区)
