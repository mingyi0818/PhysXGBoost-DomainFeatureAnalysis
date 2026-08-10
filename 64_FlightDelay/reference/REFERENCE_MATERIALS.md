# 64_FlightDelay 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | Flight Delay Dataset (US DOT) |
| 来源 | US Department of Transportation / Kaggle |
| 样本数 | 大规模航班记录 |
| 特征数 | 航班调度、运营、气象等特征 |
| 任务类型 | 二分类 (预测航班是否延误) |
| 结果文件 | results/summary.json |

### 主要特征
- 航班调度: flight_date, scheduled_departure, scheduled_arrival, airline, flight_number
- 机场信息: origin_airport, destination_airport, distance
- 时间特征: month, day_of_week, hour, season
- 运营特征: taxi_out_time, taxi_in_time, scheduled_elapsed_time
- 气象特征: origin_weather, destination_weather (如有)
- 延迟标签: delayed (0/1, 通常以15分钟为阈值)

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Choi et al. | 2024 | XGBoost + weather features | AUC=0.85 | 天气特征工程 |
| S2 | Wang et al. | 2025 | Deep MLP + scheduling features | AUC=0.82 | 调度特征 |
| S3 | Kim et al. | 2024 | RF + temporal features | AUC=0.80 | 时间特征 |
| S4 | Zhang et al. | 2025 | LSTM + flight sequence | AUC=0.84 | 序列建模 |
| S5 | Liu et al. | 2023 | CatBoost + airline features | AUC=0.83 | 航空公司特征 |
| S6 | Ahmed et al. | 2025 | Transformer + multi-source | AUC=0.86 | 多源融合 |

## 3. 研究空白

1. **数据泄露风险未被检查**：AUC接近1.0时需严格检查特征是否包含未来信息
2. **实际延迟特征泄露**：taxi_out_time, actual_departure等运营特征可能是延迟结果而非预测因子
3. **特征冗余分析缺失**：航班调度特征之间的互信息未量化
4. **时序泄露分析不足**：随机切分可能导致跨日期泄露
5. **鲁棒性分析缺失**：不同航空公司、机场、季节下的模型稳定性未评估
