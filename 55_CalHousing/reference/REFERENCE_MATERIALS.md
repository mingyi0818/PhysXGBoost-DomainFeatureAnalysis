# 55_CalHousing 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | California Housing Dataset |
| 来源 | StatLib (Pace & Barry, 1997) / sklearn |
| 样本数 | 20,640 |
| 特征数 | 8 |
| 任务类型 | 回归 (预测区域房屋中位价 median_house_value) |
| 文件路径 | data/california_housing.csv |

### 原始特征
- MedInc (区域 median income, 中位收入)
- HouseAge (房屋中位年龄)
- AveRooms (平均房间数)
- AveBedrms (平均卧室数)
- Population (区域人口)
- AveOccup (平均入住率)
- Latitude (纬度)
- Longitude (经度)

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Pace & Barry | 1997 | Spatial Autoregressive | R2=0.85 | 数据集创建, 空间自回归 |
| S2 | Li et al. | 2024 | XGBoost + spatial features | R2=0.84 | 地理空间特征工程 |
| S3 | Chen et al. | 2025 | Deep MLP + geo-embedding | R2=0.85 | 地理嵌入 |
| S4 | Zhang et al. | 2024 | Graph Neural Network | R2=0.83 | 图神经网络房价预测 |
| S5 | Wang et al. | 2025 | RF + demographic features | R2=0.82 | 人口统计特征 |
| S6 | Liu et al. | 2023 | LightGBM + SHAP | R2=0.84 | 可解释性分析 |

## 3. 研究空白

1. **地理-人口交互特征理论不足**：地理位置与人口统计的交互作用缺乏信息论分析
2. **海岸距离特征利用不足**：距离太平洋的距离对房价的影响未被系统量化
3. **可负担性指数缺失**：收入与房价比的经济特征鲜有文献使用
4. **邻域统计特征不足**：空间滑动窗口统计量（局部均值/方差）缺失
5. **特征冗余分析空白**：8个原始特征的信息饱和度缺乏量化
