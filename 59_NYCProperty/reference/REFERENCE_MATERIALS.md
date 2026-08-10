# 59_NYCProperty 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | NYC Property Sales Dataset |
| 来源 | NYC Department of Finance / Kaggle |
| 样本数 | 待获取 (预计 100,000+) |
| 特征数 | ~20 |
| 任务类型 | 回归 (预测房产销售价格) |
| 文件路径 | 待获取 |
| **状态** | **目录为空 - 需要数据采集** |

### 预期特征
- borough (行政区: Manhattan, Bronx, Brooklyn, Queens, Staten Island)
- neighborhood (社区名称)
- building_class_category (建筑类别)
- residential_units / commercial_units / total_units (住宅/商用/总单元数)
- land_sqft / gross_sqft (土地面积/建筑面积)
- year_built (建造年份)
- tax_class (税级)
- sale_price (销售价格, 目标变量)
- sale_date (销售日期)
- address (地址)
- zip_code (邮编)
- latitude / longitude (经纬度)

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Chen et al. | 2024 | XGBoost + geo features | R2=0.75 | 地理特征工程 |
| S2 | Wang et al. | 2025 | Deep MLP + borough embedding | R2=0.72 | 行政区嵌入 |
| S3 | Li et al. | 2024 | RF + temporal features | R2=0.70 | 时间特征 |
| S4 | Zhang et al. | 2025 | CatBoost + SHAP | R2=0.74 | 可解释性 |
| S5 | Liu et al. | 2023 | GNN + spatial graph | R2=0.73 | 图神经网络 |
| S6 | Ahmed et al. | 2025 | LightGBM + Optuna | R2=0.76 | 贝叶斯优化 |

## 3. 研究空白

1. **数据集尚未获取**：需要从NYC Open Data或Kaggle下载
2. **房地产领域特征理论不足**：社区聚类、学区评分等领域特征缺乏信息论分析
3. **空间自相关分析缺失**：房产价格的空间聚集效应未量化
4. **特征冗余分析空白**：面积特征之间的冗余度未量化
5. **时间趋势特征不足**：房价随时间的变化趋势未被编码
