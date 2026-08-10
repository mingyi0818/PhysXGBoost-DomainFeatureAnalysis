# 47_OnlineShoppers SOTA击破分析

> 方向：电商行为特征增强的在线购物意图预测
> 撰写日期：2026-08-10

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| 方法 | EcomFeat: E-commerce Domain Feature Augmentation for Purchase Intention Prediction |
| 数据集 | UCI Online Shoppers Intention (12,330 sessions, 18 features, 15.5% positive) |
| 已有结果 | Raw AUC: 0.899-0.904, Domain AUC: 0.899-0.905 |

## 2. SOTA共同缺点

| 缺点 | 解决方案 |
|------|---------|
| 2024-2026无新论文 | 首个系统性现代评估 |
| 无LightGBM/CatBoost | 4种树模型公平对比 |
| 无SHAP | SHAP全局/局部可解释性 |
| 无统计检验 | 5种子+Wilcoxon+95%CI |
| 无领域特征框架 | 5类电商行为特征 |

## 3. 击破方案

**创新点1**：电商行为领域特征工程框架
- 会话深度特征(session_depth_*)：页面浏览深度、停留时间模式
- 用户活跃度(activity_*)： Administrative/Information/Productive比率
- 时间模式(temporal_*)：周末/季节/时段编码
- 页面交互(interaction_*)：跳出率×退出率、页面值密度
- 综合评分(composite_*)：购买倾向综合评分

**创新点2**：信息论特征交互分析
- 互信息特征重要性 I(f_i; Y)
- 条件互信息增量贡献 I(d_j; Y|F)
- 特征冗余度量化 R(d_j, F)

**定理1**（特征交互上界）：|AUC(T,F∪D)-AUC(T,F)| ≤ O(√(H(Y|F)-H(Y|F∪D)))

**命题1**（特征冗余性）：若I(D;F)>I(D;Y|F)，则D的边际贡献为负。

## 4. 实验设计
| 实验 | 内容 |
|------|------|
| 主对比 | 4模型×2特征集(Raw/Domain) |
| 消融 | 5类特征逐一移除 |
| 统计 | 5种子+Wilcoxon+95%CI+Cohen's d |
| SHAP | 全局重要性+依赖图+局部解释 |
| 参数敏感性 | 学习率/树深度/估计器数量 |
| 复杂度 | 运行时间+内存+参数量 |

## 5. 推荐期刊
IJMLC (EI, ~$400) 或 IEEE Access (SCI四区, $0可选)

## 6. 风险提示
- 领域特征效果有限(±0.002)，需诚实报告并深入分析原因
- 该数据集2024-2026无新论文，既是机会也是风险（需充分引用2020-2023文献+相关领域最新进展）
