# 46_BankMarketing_PhysXGBoost 参考文献调研材料

> 调研日期：2026-08-10
> 方向：金融领域特征增强的银行定期存款预测（PhysXGBoost）
> 数据集：UCI Bank Marketing Dataset (~45,211 samples, 16 features, ~11% positive)

---

## 一、数据集概述

| 项目 | 内容 |
|------|------|
| 来源 | UCI Machine Learning Repository |
| 原始论文 | Moro et al. (2014), Decision Support Systems |
| 样本数 | 42,718条记录 |
| 特征数 | 16（客户画像+财务+营销+宏观+时间） |
| 正类比例 | ~11.27% (定期存款订阅) |
| 任务类型 | 二分类不平衡分类 |
| 评估指标 | AUC-ROC, F1, Accuracy, Precision, Recall |

---

## 二、关键SOTA文献（2023-2026）

| 编号 | 文献 | 年份 | 来源 | 方法 | 关键指标 | 局限性 |
|------|------|------|------|------|---------|--------|
| S1 | Yu et al., "神经网络银行长期存款预测" | 2023 | 辽宁石化大学学报 | 三层前馈NN | AUC=0.9777, Acc=99.06% | 样本量小，无F1，无多种子 |
| S2 | Wang, "Bank Marketing Prediction Based on XGBoost" | 2025 | AEMPS | XGBoost | AUC=0.90, Acc=0.89 | 仅报告AUC/Acc，无F1 |
| S3 | Lee et al., "Bank Direct Marketing Campaign Success" | 2024 | AMCI | RF(8特征) | 最优AUC/Acc | 仅8特征，指标不完整 |
| S4 | Hasnataeni et al., "Ensemble Methods Unbalanced Bank Marketing" | 2025 | Inferensi | RF+ROSE | Acc=91.00%, AUC~0.94 | 无F1，无SHAP |
| S5 | Kuravi, "XAI and Fairness Auditing Bank Marketing" | 2025 | Preprints | XGBoost+SHAP/LIME | 未披露具体数值 | 预印本，指标不足 |
| S6 | Du, "Gradient Boosting Bank Marketing" | 2025 | AEMPS | Gradient Boosting | 最大ROC-AUC | 无具体数值 |
| S7 | Prasad et al., "EBM Bank Marketing" | 2025 | IEEE GIEST | Explainable Boosting Machine | 未披露 | 可解释但性能未知 |
| S8 | Prasad et al., "Blending Approach Bank Marketing" | 2025 | IEEE SCEECS | 混合方法 | 未披露 | UCI数据集但无指标 |
| S9 | Apriadi & Bisri, "RF Term Deposit Prediction" | 2025 | JCNAHPC | Random Forest | 未披露 | 仅RF无对比 |
| S10 | Gupta et al., "Bank Marketing Campaign Prediction" | 2026 | IEEE ICICDS | 未明确 | 未披露 | 元数据级信息 |

---

## 三、现有实验结果（本方向已有）

| 模型 | Raw AUC | Domain AUC | Raw F1 | Domain F1 |
|------|---------|------------|--------|-----------|
| XGBoost | 0.9375 | 0.9375 | 0.6017 | 0.5999 |
| LightGBM | 0.9388 | 0.9377 | 0.5921 | 0.5894 |
| CatBoost | 0.9356 | 0.9371 | 0.5762 | 0.5819 |
| RandomForest | 0.9253 | 0.9277 | 0.5721 | 0.5855 |

**关键发现**：领域特征对AUC影响极小（±0.002），对F1影响不一致。

---

## 四、研究空白与机会

1. **无系统性的领域特征工程框架**：现有工作多为直接使用原始特征，缺乏系统的金融领域知识驱动特征设计
2. **评估指标不完整**：多数论文仅报告Accuracy/AUC，不报告F1/Precision/Recall
3. **无统计显著性检验**：几乎所有论文都是单次实验，无多种子+统计检验
4. **可解释性分析不足**：仅S5使用了SHAP/LIME，但未披露具体结果
5. **无公平性审计**：仅S5提到公平性，但无系统分析
6. **LightGBM/CatBoost在该数据集上研究极少**：2024-2026论文中未见作为核心方法
7. **无特征交互的理论分析**：缺乏对领域特征为什么有效/无效的理论解释
