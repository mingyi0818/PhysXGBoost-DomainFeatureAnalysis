# 56_PowerConsumption SOTA击破分析

> 方向：家庭用电量预测的能耗模式分析
> 撰写日期：2026-08-10
> **关键警告：R2接近1.0，存在数据泄露嫌疑，论文应调查泄露来源**

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| 方法 | PowerConsFeat: Energy Consumption Pattern Analysis |
| 数据集 | Individual Household Power Consumption (2,075,259 samples, regression) |
| 已有结果 | Raw R2: 0.996-0.9997, Domain R2: 0.997-0.9998 (接近完美, 存疑) |
| 结果文件 | results/summary.json |

### 实验结果详情

| 模型 | Raw R2 | Domain R2 | 差值 |
|------|--------|-----------|------|
| XGBoost | 0.9963 | 0.9968 | +0.0005 |
| LightGBM | 0.9990 | 0.9990 | +0.0000 |
| CatBoost | 0.9996 | 0.9997 | +0.0000 |
| RandomForest | 0.9997 | 0.9998 | +0.0000 |

## 2. SOTA共同缺点

| 缺点 | 解决方案 |
|------|---------|
| 数据泄露未被检查 | 系统分析R2接近1.0的泄露来源 |
| 无时序切分验证 | 对比random split vs chronological split |
| 无特征冗余分析 | Global_intensity与Global_active_power的物理冗余(P=V*I) |
| 无统计检验 | 5种子 + Wilcoxon + 95% CI |
| 无参数敏感性 | 弹性系数量化超参影响 |
| 无鲁棒性分析 | 分布漂移/季节变化下的稳定性 |

## 3. 击破方案

**创新点1**：能耗模式领域特征工程框架
- temporal_* (小时/日模式): hour_of_day, day_of_week, is_weekend, is_peak_hour
- weather_* (季节性): month, season, is_summer, is_winter
- lag_* (自回归): lag_1min, lag_5min, lag_15min, lag_60min, rolling_mean_15min, rolling_std_15min

**创新点2**：数据泄露调查（核心贡献）
- R2接近1.0异常，需系统调查泄露来源
- 假设1: Global_intensity与Global_active_power存在物理关系 P = V * I，构成线性冗余
- 假设2: 随机切分导致时序泄露，相邻样本同时出现在训练集和测试集
- 假设3: lag_1min特征几乎等于目标变量本身

**定理1**（特征交互界）：对于回归任务，若特征集F中存在特征X_i使得 I(Y;X_i) / H(Y) approx 1（即X_i几乎完全确定Y），则对任意新特征D，Delta(R2) <= 1 - R2(F) approx 0。在Power Consumption数据集中，Global_intensity与Global_active_power通过P=V*I物理关联，故R2(F) approx 1，领域特征增益趋近于零。

**命题1**（特征冗余判据）：若领域特征D与原始特征集F的互信息 I(D;F) > I(D;Y|F)，则D的边际贡献为负。lag_*特征与Global_active_power的滞后值高度相关（时序自相关），且当lag_1min approx Y时，I(D;F)极大而I(D;Y|F) approx 0。

**创新点3**：数据泄露诊断框架
- 消融实验: 逐步移除Global_intensity, lag_1min等高泄露嫌疑特征
- 切分对比: random split vs chronological split的R2差异
- 物理冗余分析: P = V * I关系对R2的贡献量化
- 提出数据泄露检测清单供未来研究参考

## 4. 实验设计

| 实验 | 内容 |
|------|------|
| 主对比 | 4模型 x 2特征集(Raw/Domain) |
| **泄露诊断** | 逐步移除嫌疑特征, R2变化分析 |
| 切分对比 | random split vs chronological split |
| 物理冗余分析 | 移除Global_intensity后的R2 |
| 消融 | 3类领域特征逐一移除(temporal/weather/lag) |
| 统计 | 5种子 + Wilcoxon + 95% CI + Cohen's d |
| 参数敏感性 | 学习率/树深度/估计器数量, 弹性系数等级 |
| 复杂度分析 | 理论O(N*d) + 实际运行时间/内存/FLOPs |

## 5. 推荐期刊

IJMLC (EI) 或 IEEE Access (SCI四区)

## 6. 风险提示

- **CRITICAL: R2接近1.0，强烈怀疑数据泄露**
- 可能的泄露来源：(1) Global_intensity与目标变量物理冗余 (2) 随机切分导致时序泄露 (3) lag特征近似目标
- 论文核心叙事：高R2不等于好模型，数据泄露诊断比追求高R2更重要
- 核心贡献转向：(1)数据泄露诊断框架 (2)物理冗余分析 (3)正确的时序切分方案
- 如实报告，不编造数据
- 建议在论文中明确标注"Data Leakage Investigation"
