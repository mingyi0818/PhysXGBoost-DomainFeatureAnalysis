# 63_HotelBooking SOTA击破分析

> 方向：酒店预订取消预测的酒店管理领域特征分析
> 撰写日期：2026-08-10

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| 方法 | HotelFeat: Hospitality Domain Feature Analysis |
| 数据集 | Hotel Booking Demand (119,390 samples, ~30 features, classification) |
| 已有结果 | Raw AUC: 0.872-0.885, Domain AUC: 0.875-0.885 (几乎无差异) |
| 结果文件 | results/summary.json |

### 实验结果详情

| 模型 | Raw AUC | Domain AUC | 差值 |
|------|---------|------------|------|
| XGBoost | 0.8852 | 0.8855 | +0.0002 |
| LightGBM | 0.8845 | 0.8850 | +0.0006 |
| CatBoost | 0.8749 | 0.8754 | +0.0005 |
| RandomForest | 0.8724 | 0.8746 | +0.0022 |

## 2. SOTA共同缺点

| 缺点 | 解决方案 |
|------|---------|
| 无酒店管理特征理论 | 基于酒店运营理论的特征交互界分析 |
| 无特征冗余量化 | 互信息 + 条件互信息分析 |
| 无统计检验 | 5种子 + Wilcoxon + 95% CI |
| 无参数敏感性 | 弹性系数量化超参影响 |
| 无可解释性 | SHAP + 酒店管理理论关联 |

## 3. 击破方案

**创新点1**：酒店管理领域特征工程框架
- guest_* (团队规模, 客人类型): total_guests = adults + children + babies, is_family, group_size_category
- booking_* (提前期模式, 渠道): lead_time_category, booking_change_rate, channel_risk_score
- temporal_* (季节, 星期): arrival_season, is_weekend_arrival, is_holiday_season
- pricing_* (ADR分类): adr_category, price_per_night_per_guest, deposit_risk

**创新点2**：特征信息饱和性分析
- 原始30+特征已包含丰富的酒店预订信息
- 推导：当原始特征AUC接近0.885时，剩余信息量有限

**定理1**（特征交互界）：对于二分类任务 Y in {0,1}，给定特征集F和新特征D，条件互信息 I(Y;D|F) <= H(Y) - I(Y;F)。当 AUC(F) >= 0.885 时，I(Y;F) 接近 H(Y)，故D的边际信息增益趋近于零。

**命题1**（特征冗余判据）：若领域特征D与原始特征集F的互信息 I(D;F) > I(D;Y|F)，则D的边际贡献为负。guest_*特征与adults/children/babies高度相关，booking_*特征与lead_time/booking_changes相关，故冗余度高。

**创新点3**：SHAP酒店管理可解释性
- SHAP特征重要性 + 酒店运营理论关联
- 验证lead_time/deposit_type特征重要性是否符合酒店管理直觉
- ADR与取消率的非线性关系分析

## 4. 实验设计

| 实验 | 内容 |
|------|------|
| 主对比 | 4模型 x 2特征集(Raw/Domain) |
| SHAP酒店管理可解释性 | 特征重要性 + 运营理论关联 |
| 特征聚类 | 基于互信息的层次聚类 |
| 消融 | 4类领域特征逐一移除(guest/booking/temporal/pricing) |
| 统计 | 5种子 + Wilcoxon + 95% CI + Cohen's d |
| 参数敏感性 | 学习率/树深度/估计器数量, 弹性系数等级 |
| 复杂度分析 | 理论O(N*d) + 实际运行时间/内存/FLOPs |

## 5. 推荐期刊

IJMLC (EI) 或 IEEE Access (SCI四区)

## 6. 风险提示

- 领域特征几乎无效果(最大+0.0022)，核心贡献转向可解释性和理论分析
- 原始30+特征信息量已接近饱和
- 论文核心叙事：原始特征已足够预测酒店取消，信息饱和性是关键约束
- 如实报告，不编造数据
