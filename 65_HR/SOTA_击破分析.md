# 65_HR SOTA击破分析

> 方向：员工离职预测的人力资源领域特征分析
> 撰写日期：2026-08-10

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| 方法 | HRFeat: Workforce Domain Feature Analysis |
| 数据集 | IBM HR Analytics (1,470 samples, ~30 features, classification) |
| 已有结果 | Raw AUC: 0.737-0.744, Domain AUC: 0.737-0.743 (几乎无差异) |
| 结果文件 | results/summary.json |

### 实验结果详情

| 模型 | Raw AUC | Domain AUC | 差值 |
|------|---------|------------|------|
| XGBoost | 0.7417 | 0.7392 | -0.0025 |
| LightGBM | 0.7378 | 0.7395 | +0.0017 |
| CatBoost | 0.7431 | 0.7428 | -0.0003 |
| RandomForest | 0.7371 | 0.7378 | +0.0007 |

## 2. SOTA共同缺点

| 缺点 | 解决方案 |
|------|---------|
| 无HR领域特征理论 | 基于组织行为学的特征交互界分析 |
| 无特征冗余量化 | 互信息 + 条件互信息分析 |
| 无统计检验 | 5种子 + Wilcoxon + 95% CI |
| 无参数敏感性 | 弹性系数量化超参影响 |
| 无可解释性 | SHAP + 组织行为学关联 |
| 小样本统计不足 | Bootstrap CI + 5种子 |

## 3. 击破方案

**创新点1**：人力资源领域特征工程框架
- career_* (任期, 晋升): career_stagnation = YearsSinceLastPromotion / YearsAtCompany, promotion_rate, role_tenure_ratio
- comp_* (薪资比, 股权): income_vs_peers = MonthlyIncome / median(MonthlyIncome|JobLevel), stock_value_score, pay_growth_rate
- satis_* (满意度复合): satisfaction_composite = mean(EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction), satisfaction_variance
- worklife_* (平衡评分): worklife_score = f(OverTime, BusinessTravel, DistanceFromHome, WorkLifeBalance)

**创新点2**：特征信息饱和性分析
- 原始30+特征AUC约0.74，信息量未完全饱和但提升空间有限
- 推导：小样本(1,470)下领域特征的方差贡献被噪声淹没

**定理1**（特征交互界）：对于二分类任务，给定特征集F和新特征D，条件互信息 I(Y;D|F) <= H(Y) - I(Y;F)。在HR数据集中 AUC(F) approx 0.74，剩余信息量有限。同时，小样本下估计 I(Y;D|F) 的方差为 O(1/sqrt(n))，当 n=1,470 时方差较大，领域特征的边际贡献可能被统计噪声掩盖。

**命题1**（特征冗余判据）：若领域特征D与原始特征集F的互信息 I(D;F) > I(D;Y|F)，则D的边际贡献为负。career_*特征与YearsAtCompany/YearsSinceLastPromotion高度相关，comp_*特征与MonthlyIncome/JobLevel相关，故冗余度高。

**创新点3**：SHAP组织行为学可解释性
- SHAP特征重要性 + 组织行为学理论关联
- 验证加班/收入特征重要性是否符合员工离职理论
- 满意度复合特征对不同离职类型的差异化影响

## 4. 实验设计

| 实验 | 内容 |
|------|------|
| 主对比 | 4模型 x 2特征集(Raw/Domain) |
| SHAP组织行为可解释性 | 特征重要性 + HR理论关联 |
| 特征聚类 | 基于互信息的层次聚类 |
| 消融 | 4类领域特征逐一移除(career/comp/satis/worklife) |
| 统计 | 5种子 + Wilcoxon + 95% CI + Cohen's d |
| 参数敏感性 | 学习率/树深度/估计器数量, 弹性系数等级 |
| 复杂度分析 | 理论O(N*d) + 实际运行时间/内存/FLOPs |

## 5. 推荐期刊

IJMLC (EI) 或 IEEE Access (SCI四区)

## 6. 风险提示

- 领域特征几乎无效果(部分模型甚至下降)，核心贡献转向可解释性和理论分析
- 小样本(1,470)下统计功效不足，领域特征贡献可能被噪声掩盖
- 论文核心叙事：小样本高维场景下领域特征工程的局限性
- AUC=0.74相对较低，需分析数据本身的预测难度
- 如实报告，不编造数据
