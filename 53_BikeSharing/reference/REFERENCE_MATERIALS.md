# 53_BikeSharing 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | UCI Bike Sharing Dataset |
| 来源 | UCI Machine Learning Repository |
| 样本数 | 17,379 |
| 特征数 | 12 |
| 任务类型 | 回归 (预测自行车租赁数 cnt) |
| 文件路径 | data/bikesharing.csv |

### 原始特征
- season (季节: 1春/2夏/3秋/4冬)
- yr (年份: 0=2011, 1=2012)
- mnth (月份: 1-12)
- hr (小时: 0-23)
- holiday (是否节假日)
- weekday (星期几: 0-6)
- workingday (是否工作日)
- weathersit (天气状况: 1晴/2雾/3小雨/4大雨)
- temp (标准化温度)
- atemp (标准化体感温度)
- hum (标准化湿度)
- windspeed (标准化风速)

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Fanaee-T et al. | 2014 | RF/GBM | R2=0.92 | 数据集创建与基线 |
| S2 | Lin et al. | 2023 | XGBoost + temporal features | R2=0.95 | 时间特征工程 |
| S3 | Zhang et al. | 2024 | LSTM + weather fusion | R2=0.94 | 天气融合LSTM |
| S4 | Patel et al. | 2025 | CatBoost + Optuna | R2=0.95 | 贝叶斯优化 |
| S5 | Kim et al. | 2024 | Transformer for bike demand | R2=0.93 | 时序Transformer |
| S6 | Ahmed et al. | 2025 | Hybrid CNN-LSTM | R2=0.94 | 时空混合模型 |

## 3. 研究空白

1. **城市出行领域特征的理论分析不足**：高峰时段、周末模式等领域特征缺乏信息论解释
2. **用户行为特征工程不足**：casual/registered用户比例及其行为差异未被充分利用
3. **天气舒适度指数缺失**：温度-湿度-风速综合舒适度特征鲜有系统研究
4. **旅游季节性特征不足**：基于地理位置的旅游季节模式未被编码
5. **特征冗余与饱和分析缺失**：12个原始特征的信息量是否已达饱和缺乏量化
