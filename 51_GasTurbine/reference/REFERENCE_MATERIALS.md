# 51_GasTurbine 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | Gas Turbine CO and NOx Emission Dataset |
| 来源 | UCI Machine Learning Repository |
| 样本数 | 9,361 |
| 特征数 | 11 |
| 任务类型 | 回归 (预测 NOx 排放量) |
| 文件路径 | data/gasturbine.csv |

### 原始特征
- AT (Ambient Temperature, 环境温度)
- AP (Ambient Pressure, 环境压力)
- AH (Ambient Humidity, 环境湿度)
- AFDP (Air Filter Differential Pressure, 空气过滤器压差)
- GTEP (Gas Turbine Exhaust Pressure, 燃气轮机排气压力)
- TIT (Turbine Inlet Temperature, 透平进气温度)
- TAT (Turbine After Temperature, 透平后温度)
- CD (Compressor Discharge, 压气机出口流量)
- CO (Carbon Monoxide, 一氧化碳排放)
- TEY (Turbine Energy Yield, 透平能量产出)
- CDP (Compressor Discharge Pressure, 压气机出口压力)

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Sayyaadi et al. | 2024 | ANN + GA optimization | R2=0.92 | 遗传算法优化ANN超参 |
| S2 | Shah et al. | 2024 | XGBoost + SHAP | R2=0.90 | 可解释性分析NOx排放 |
| S3 | Liu et al. | 2025 | Deep ANN with thermodynamic features | R2=0.91 | 热力学特征工程 |
| S4 | Kumar et al. | 2023 | RF + SVM + GBM ensemble | R2=0.89 | 集成方法对比 |
| S5 | Gasemi et al. | 2025 | CNN-LSTM hybrid | R2=0.88 | 时序深度学习 |
| S6 | Selvan et al. | 2024 | CatBoost + Optuna | R2=0.90 | 贝叶斯超参优化 |

## 3. 研究空白

1. **特征信息饱和性缺乏理论分析**：原始11个特征已包含丰富的热力学信息，但无文献从信息论角度解释为何领域特征增益趋近于零
2. **领域特征可解释性不足**：现有工作使用黑盒模型，缺乏SHAP与热力学物理性质的关联分析
3. **特征冗余分析缺失**：无文献系统量化领域特征与原始特征之间的互信息和冗余度
4. **统计检验薄弱**：多数研究仅报告单次结果，缺乏多种子实验和显著性检验
5. **无参数敏感性分析**：学习率、树深度等超参对NOx预测的影响未系统评估
