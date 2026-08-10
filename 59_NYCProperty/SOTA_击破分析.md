# 59_NYCProperty SOTA击破分析

> 方向：纽约房产价格预测的房地产领域特征分析
> 撰写日期：2026-08-10
> **注意：目录为空，需要数据采集后才能开展实验**

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| 方法 | NYCPropFeat: Real Estate Domain Feature Analysis |
| 数据集 | NYC Property Sales (待获取, ~100,000+ samples, ~20 features, regression) |
| 已有结果 | 无 (目录为空, 需数据采集) |
| 结果文件 | 无 |
| **状态** | **需数据采集** |

## 2. SOTA共同缺点

| 缺点 | 解决方案 |
|------|---------|
| 无房地产特征理论 | 基于城市经济学的特征交互界分析 |
| 无特征冗余量化 | 互信息 + 条件互信息分析 |
| 无统计检验 | 5种子 + Wilcoxon + 95% CI |
| 无参数敏感性 | 弹性系数量化超参影响 |
| 无可解释性 | SHAP + 房地产经济学关联 |

## 3. 击破方案

**创新点1**：房地产领域特征工程框架
- location_* (位置特征): borough_encoding, neighborhood_cluster, distance_to_subway, distance_to_manhattan
- building_* (建筑特征): age = current_year - year_built, age_category, unit_density, floor_area_ratio
- market_* (市场特征): price_per_sqft, price_trend_3month, borough_price_index
- temporal_* (时间特征): sale_quarter, sale_year, market_cycle_phase

**创新点2**：特征信息增益分析（待数据获取后验证）
- 假设原始特征R2约0.70-0.76，有提升空间
- 地理位置特征捕获社区价格梯度信息

**定理1**（特征交互界）：对于回归任务 Y = f(X) + epsilon，给定特征集F和新特征D，R2增量 Delta(R2) <= [H(Y) - I(Y;F)] / H(Y) = 1 - R2(F)。当 R2(F) = 0.75 时，理论上界 Delta(R2) <= 0.25，实际增益受D与F的冗余度约束。

**命题1**（特征冗余判据）：若领域特征D与原始特征集F的互信息 I(D;F) > I(D;Y|F)，则D的边际贡献为负。location_*特征与borough/zip_code相关，但neighborhood_cluster编码了更细粒度的空间结构。

**创新点3**：SHAP房地产经济学可解释性
- SHAP特征重要性 + 城市经济学理论关联
- 验证位置/面积/房龄特征重要性是否符合房地产估值理论
- 社区聚类对房价预测的边际贡献分析

## 4. 实验设计

| 实验 | 内容 |
|------|------|
| **数据采集** | 从NYC Open Data / Kaggle下载NYC Property Sales数据 |
| 主对比 | 4模型 x 2特征集(Raw/Domain) |
| SHAP房地产可解释性 | 特征重要性 + 城市经济学关联 |
| 特征聚类 | 基于互信息的层次聚类 |
| 消融 | 4类领域特征逐一移除(location/building/market/temporal) |
| 统计 | 5种子 + Wilcoxon + 95% CI + Cohen's d |
| 参数敏感性 | 学习率/树深度/估计器数量, 弹性系数等级 |
| 复杂度分析 | 理论O(N*d) + 实际运行时间/内存/FLOPs |

## 5. 推荐期刊

IJMLC (EI) 或 IEEE Access (SCI四区)

## 6. 风险提示

- **目录为空，需先获取数据**
- 数据来源建议：NYC Open Data (https://opendata.cityofnewyork.us/) 或 Kaggle NYC Property Sales
- 获取数据后需进行数据清洗（去除异常价格如$0转移、缺失值处理）
- 论文核心叙事取决于实验结果：若领域特征有效则强调特征工程价值，若无效则转向信息饱和性分析
- 如实报告，不编造数据
