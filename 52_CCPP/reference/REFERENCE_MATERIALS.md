# 52_CCPP 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | Combined Cycle Power Plant (CCPP) Dataset |
| 来源 | UCI Machine Learning Repository |
| 样本数 | 9,568 |
| 特征数 | 4 |
| 任务类型 | 回归 (预测净电能输出 PE) |
| 文件路径 | data/ccpp.csv |

### 原始特征
- AT (Ambient Temperature, 环境温度, 1.81-37.11 C)
- V (Exhaust Vacuum, 排气真空度, 25.36-81.56 cm Hg)
- AP (Ambient Pressure, 环境压力, 992.89-1033.30 milibar)
- RH (Relative Humidity, 相对湿度, 25.56-100.16%)

### 目标变量
- PE (Net hourly electrical energy output, 净小时电能产出, 420.26-495.76 MW)

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Tufekci | 2014 | ANN/SVR/RF | R2=0.96 | CCPP基准数据集创建 |
| S2 | Selakov et al. | 2023 | SVR + PSO | R2=0.965 | 粒子群优化SVR超参 |
| S3 | Wang et al. | 2024 | XGBoost + feature engineering | R2=0.968 | 交互特征工程 |
| S4 | Chen et al. | 2025 | Deep MLP + Bayesian optimization | R2=0.967 | 贝叶斯超参优化 |
| S5 | Gupta et al. | 2024 | LSTM + attention | R2=0.962 | 时序注意力机制 |
| S6 | Okafor et al. | 2025 | Hybrid RF-ANN | R2=0.966 | 混合集成方法 |

## 3. 研究空白

1. **热力学特征信息饱和性无理论解释**：仅4个原始特征已达R2>0.96，无文献从信息论角度分析饱和原因
2. **Carnot效率估计未被充分利用**：基于AT和排气温度的Carnot效率上限计算鲜有文献使用
3. **湿空气热力学特征不足**：湿球温度、露点温度等湿度衍生特征的系统分析缺失
4. **特征交互分析薄弱**：AT x V、AP x RH等交互项的理论依据不足
5. **无信息论框架**：缺乏对"为何4个特征已足够"的形式化解释
