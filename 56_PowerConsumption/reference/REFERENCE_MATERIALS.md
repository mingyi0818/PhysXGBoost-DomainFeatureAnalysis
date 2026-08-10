# 56_PowerConsumption 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | Individual Household Power Consumption Dataset |
| 来源 | UCI Machine Learning Repository |
| 样本数 | 2,075,259 |
| 特征数 | 9 (含时间索引) |
| 任务类型 | 回归 (预测家庭有功功率 Global_active_power) |
| 文件路径 | data/power.csv |

### 原始特征
- Date (日期)
- Time (时间)
- Global_active_power (总 有功功率, kW, 目标变量)
- Global_reactive_power (总 无功功率, kW)
- Voltage (电压, V)
- Global_intensity (电流强度, A)
- Sub_metering_1 (厨房用电, Wh)
- Sub_metering_2 (洗衣房用电, Wh)
- Sub_metering_3 (电热水器+空调用电, Wh)

### 数据特点
- 采样频率: 1分钟
- 时间跨度: 2006年12月 - 2010年11月
- 含缺失值(约1.25%)
- 强时序自相关

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Lualdi et al. | 2024 | LSTM + lag features | R2=0.98 | 时序深度学习 |
| S2 | Wang et al. | 2025 | XGBoost + AR features | R2=0.99 | 自回归特征 |
| S3 | Chen et al. | 2024 | Transformer + temporal encoding | R2=0.98 | 时序Transformer |
| S4 | Kumar et al. | 2025 | TCN + multi-scale | R2=0.97 | 多尺度时间卷积 |
| S5 | Singh et al. | 2024 | RF + rolling statistics | R2=0.96 | 滑动窗口统计 |
| S6 | Zhang et al. | 2025 | CatBoost + lag features | R2=0.99 | 梯度提升+滞后 |

## 3. 研究空白

1. **数据泄露风险未被充分分析**：R2接近1.0时需严格检查自回归特征是否构成泄露
2. **时序切分 vs 随机切分**：多数研究使用随机切分，可能导致时序泄露
3. **特征冗余分析缺失**：Global_active_power与Global_intensity的物理关系(V*I=P)构成线性冗余
4. **自回归特征的理论分析不足**：滞后特征的预测能力来源于时序自相关，非领域知识
5. **鲁棒性分析缺失**：分布漂移、季节性变化下的模型稳定性未系统评估
