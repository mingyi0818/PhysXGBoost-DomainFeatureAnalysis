# 64_FlightDelay SOTA击破分析

> 方向：航班延误预测的航空领域特征分析
> 撰写日期：2026-08-10
> **关键警告：AUC接近1.0，存在数据泄露嫌疑，论文应调查泄露来源**

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| 方法 | FlightFeat: Aviation Domain Feature Analysis |
| 数据集 | Flight Delay (大规模航班记录, classification) |
| 已有结果 | Raw AUC: 0.9998-0.99999 (接近完美, 存疑) |
| 结果文件 | results/summary.json |

### 实验结果详情

| 模型 | Raw AUC | Domain AUC | 差值 |
|------|---------|------------|------|
| XGBoost | 0.999992 | 0.999992 | +0.000000 |
| LightGBM | 0.999994 | 0.999994 | +0.000000 |
| CatBoost | 0.999984 | 0.999984 | +0.000000 |
| RandomForest | 0.999818 | 0.999164 | -0.000653 |

## 2. SOTA共同缺点

| 缺点 | 解决方案 |
|------|---------|
| 数据泄露未被检查 | 系统分析AUC接近1.0的泄露来源 |
| 无时序切分验证 | 对比random split vs date-based split |
| 无特征冗余分析 | 运营特征之间的物理冗余量化 |
| 无统计检验 | 5种子 + Wilcoxon + 95% CI |
| 无参数敏感性 | 弹性系数量化超参影响 |
| 无鲁棒性分析 | 不同航空公司/机场/季节下的稳定性 |

## 3. 击破方案

**创新点1**：航空领域特征工程框架
- scheduling_* (调度模式): departure_hour_category, day_of_week_pattern, airline_historical_delay_rate
- airport_* (机场特征): origin_congestion_index, destination_congestion_index, route_delay_prior
- weather_* (气象影响): origin_weather_severity, destination_weather_severity, wind_impact_score
- temporal_* (时间模式): season, is_holiday_period, is_peak_travel

**创新点2**：数据泄露调查（核心贡献）
- AUC接近1.0异常，需系统调查泄露来源
- 假设1: taxi_out_time等运营特征实际是延迟结果而非预测因子
- 假设2: actual_departure/arrival_time等特征包含未来信息
- 假设3: 随机切分导致同一航班的相邻记录同时出现在训练/测试集

**定理1**（特征交互界）：对于二分类任务，若特征集F中存在特征X_i使得 I(Y;X_i) / H(Y) approx 1，则对任意新特征D，Delta(AUC) <= 1 - AUC(F) approx 0。在Flight Delay数据集中，若包含actual_departure等延迟结果特征，则AUC(F) approx 1，领域特征增益趋近于零。

**命题1**（特征冗余判据）：若领域特征D与原始特征集F的互信息 I(D;F) > I(D;Y|F)，则D的边际贡献为负。scheduling_*特征与flight_date/scheduled_departure相关，airport_*特征与origin/destination相关，故冗余度高。当泄露特征存在时，I(D;Y|F) approx 0。

**创新点3**：数据泄露诊断框架
- 消融实验: 逐步移除actual_departure, taxi_out_time等高泄露嫌疑特征
- 切分对比: random split vs date-based split的AUC差异
- 时间因果性分析: 每个特征的获取时间是否早于预测目标时间
- 提出航空延误预测的数据泄露检测清单

## 4. 实验设计

| 实验 | 内容 |
|------|------|
| 主对比 | 4模型 x 2特征集(Raw/Domain) |
| **泄露诊断** | 逐步移除嫌疑特征, AUC变化分析 |
| 切分对比 | random split vs date-based split |
| 时间因果性分析 | 每个特征的因果性评估 |
| 消融 | 4类领域特征逐一移除(scheduling/airport/weather/temporal) |
| 统计 | 5种子 + Wilcoxon + 95% CI + Cohen's d |
| 参数敏感性 | 学习率/树深度/估计器数量, 弹性系数等级 |
| 复杂度分析 | 理论O(N*d) + 实际运行时间/内存/FLOPs |

## 5. 推荐期刊

IJMLC (EI) 或 IEEE Access (SCI四区)

## 6. 风险提示

- **CRITICAL: AUC接近1.0，强烈怀疑数据泄露**
- 可能的泄露来源：(1)运营特征包含延迟结果 (2)随机切分导致时序泄露 (3)actual时间包含未来信息
- 论文核心叙事：高AUC不等于好模型，数据泄露诊断是航班延误预测的关键前提
- 核心贡献转向：(1)数据泄露诊断框架 (2)因果性特征分析 (3)正确的时序切分方案
- 如实报告，不编造数据
- 建议在论文中明确标注"Data Leakage Investigation"
