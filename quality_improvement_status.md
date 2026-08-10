# 论文质量提升状态跟踪

**最后更新**: 2026-07-21 16:00

---

## 新版作者信息模板（所有论文统一使用）

**英文版**:
```
Jingyuan Zeng<sup>1</sup>, Ming Zeng<sup>2</sup>, Jianghong Guo<sup>1</sup>, Chuanxian Jiang<sup>1</sup>, Yafen Feng<sup>3,4*</sup>

<sup>1</sup> School of Computer Science, Jiaying University, Meizhou 514015, China
<sup>2</sup> College of Water Resources and Civil Engineering, South China Agricultural University, Guangzhou 510642, China
<sup>3</sup> School of Geography and Tourism, Jiaying University, Meizhou 514015, China
<sup>4</sup> Key Laboratory of Surface Environment and Green Development in Northeast Guangdong, Meizhou 514015, China

*Corresponding author: Yafen Feng (fyf81@163.com)
```

**JX01特例**（冯亚芬第一作者）:
```
Yafen Feng<sup>3,4</sup>, Ming Zeng<sup>2</sup>, Jianghong Guo<sup>1</sup>, Chuanxian Jiang<sup>1</sup>, Jingyuan Zeng<sup>1*</sup>

(单位同上)

*Corresponding author: Jingyuan Zeng (zjy@jyu.edu.cn)
```

**中文版**:
曾镜源1，曾鸣2，郭江鸿1，姜传贤1，冯亚芬3,4
1. 嘉应学院计算机学院，广东梅州 514015
2. 华南农业大学水利与土木工程学院,广东广州 510642
3. 嘉应学院地理科学与旅游学院，广东梅州 514015
4. 粤东北山区地表环境与绿色发展重点实验室，广东梅州 514015

**基金项目**: 广东省本科高校高等教育教学改革项目 (批准号: 粤教高函〔2024〕9-989)

---

## 论文优先级队列（按可投稿性排序）

### Tier 1: 投稿准备最充分（仅需更新作者信息 + 最终审查）

| 编号 | 方向名 | 作者已更新 | 质量评估 | 改进操作 | 投稿就绪 |
|------|--------|-----------|---------|---------|---------|
| JX02 | masf-public | 是 | 92.4 A+级 | 已改进：作者信息+数据修正+文献重编号+效果量 | 接近就绪 |
| JX01 | ohtp-mm-public | 是 | 93.0 B+级 | 已改进：补p值+多种子+弹性系数表 | 接近就绪 |
| 17 | Evidence_Rainfall | 是 | ~92 B+级 | 已改进：摘要扩充+参考文献重编号+基金项目+数据100分 | 接近就绪 |
| 12 | Student_Dropout | 是 | 86.4 B级 | 已改进：作者信息+基金项目+文献补引用+舍入修正 | 接近就绪 |
| 04 | Time_Series_Framework | 是 | ~86 B级 | 已改进：格式修复，MetaMAE数据溯源问题待解决 | 否 |

### Tier 1.5: 数据溯源问题（需决定修复或降级）

| 编号 | 方向名 | 作者已更新 | 质量评估 | 核心问题 | 决策 |
|------|--------|-----------|---------|----------|------|
| 14 | Tabular_Anomaly | 是 | 64.2 D级 | 数据溯源问题+基线不足 | 降级 |
| 25 | FewShot_Weed | 是 | 78.4 D级 | 数据溯源+创新度不足 | 降级 |
| 03 | Imbalanced_Learning | 是 | 73.8 C级 | 数据溯源严重问题 | 降级 |
| 07 | Tabular_FewShot | 是 | 73.4 C级 | 数据溯源问题 | 降级 |

### Tier 2: 需质量提升（方法有一定优势或可改善叙事）

| 编号 | 方向名 | 作者已更新 | 质量评估 | 改善策略 | 投稿就绪 |
|------|--------|-----------|---------|---------|---------|
| 15 | Hotel_Cancellation | 否 | 88.4→~91 B+级 | 已改进：格式修复+定理强化+架构图+篇幅扩展 | 接近就绪 |
| 29 | Fairness_Tabular | 否 | 89.0→~91 B+级 | 已改进：文献修复+图表重编号+摘要扩展+贡献重排 | 接近就绪 |
| 22 | Graph_Purchase | 否 | 82.0→~85 B级 | 已改进：GNN无效诚实报告+叙事重构+格式修复 | 接近就绪 |
| 38 | AdaptiveAttention_Rain | 否 | 81.0→~88.6 B+级 | 已改进：3处数字修正+统计检验诚实报告+格式修复 | 接近就绪 |
| 20 | Fraud_SelfSupervised | 否 | 74.0→~83 B级 | 已改进：弹性系数矛盾修复+AUC解释+5种子统计重建 | 接近就绪 |
| 16 | Hotel_Sentiment_Topic | 否 | 69.0 D级 | 摘要混淆两个数据集结果，核心方法失败 | 暂缓 |

### Tier 3: 严重负面结果（需决定放弃或挽救）

| 编号 | 方向名 | 作者已更新 | 质量评估 | 决策 | 投稿就绪 |
|------|--------|-----------|---------|------|---------|
| 27 | AdaptiveIntrusion | 否 | 83.2→~87 B+级 | 已改进：数据真实性80→100，SVM种子诚实报告，效果量修正 | 接近就绪 |
| 26 | Lightweight_EuroSAT | 否 | 82.0→~88 B+级 | 已改进：种子2→5，准确率94.46%→96.11%，优于ResNet18 | 接近就绪 |
| 35 | Federated_Tabular | 否 | 78.4→~83 B级 | 已改进：负面结果诚实报告+表格编号修复+格式修复 | 接近就绪 |
| 30 | STGCN_PM25 | 否 | ~85 B级 | 已完成，诚实报告负面结果 | 否 |
| 18 | CrossRegion_Energy | 否 | 已完成 | 已完成，负面结果 | 否 |
| 37 | GenAug_Tabular | 否 | 已完成 | 已完成，负面结果 | 否 |
| 21 | Contrastive_Churn | 否 | 74.0→~82 B级 | 已改进：Table 4占位符修复+Proposition修正+数据真实100 | 接近就绪 |
| 24 | FeatureInteraction_House | 否 | 69.0→~80 B级 | 已改进：敏感性最佳值修正+置信区间修正+数据真实100 | 接近就绪 |
| 23 | Ensemble_Imbalanced | 否 | 52.0 D级 | **放弃**（t统计量篡改，学术诚信红线） | 放弃 |

### 暂缓方向（本轮不处理）

| 编号 | 方向名 | 暂缓原因 | 作者已更新 |
|------|--------|----------|-----------|
| 01 | Tabular_Framework | EPSS仅提升4.3% | 否 |
| 02 | HSIC_FDANet | p=0.2442不显著 | 否 |
| 05 | Agriculture_Fusion | 劣于LightGBM | 否 |
| 06 | Tourism_Prediction | N=65数据集过小 | 否 |
| 08 | Agriculture_FewShot | 不如SimpleShot | 否 |
| 10 | Tourism_ABSA | 99句数据集过小 | 否 |
| 11 | EuroSAT_Classification | 暂停 | 否 |
| 13 | LUCAS_Soil | 暂停 | 否 |
| 19 | MultiScale_Power | 两套代码体系不一致 | 否 |

---

## 并发锁状态

| 方向编号 | 执行中对话 | 开始时间 | 状态 |
|----------|-----------|----------|------|
| (空) | - | - | 无锁定 |

---

## 改进日志

| 时间 | 方向 | 改进内容 | 结果 |
|------|------|----------|------|
| 2026-07-20 22:30 | - | 创建状态跟踪文件 | 完成 |
| 2026-07-20 22:30 | 15_Hotel_MultiTask | 重命名为LEGACY | 完成 |
| 2026-07-21 | 15_Hotel_MultiTask_LEGACY | 已删除（仅有旧代码无论文，数据与Cancellation一致无交叉引用） | 完成 |
| 2026-07-21 | 15_Hotel_Cancellation | 确认保留（有完整论文+图表+实验数据） | 完成 |
| 2026-07-21 | Tier 2评估 | 6篇论文全部评估完成：29(89.0B), 15(88.4B), 22(82.0B), 38(81.0B), 20(74.0C), 16(69.0D) | 完成 |
| 2026-07-21 | JX02改进 | 修复7项问题，评分88.2→92.4(A+)，接近投稿就绪 | 完成 |
| 2026-07-21 | JX01改进 | 补p值+多种子+弹性系数，评分90.6→~93 | 完成 |
| 2026-07-21 | 17_Evidence改进 | 摘要扩充+参考文献重编号+基金项目，数据100分 | 完成 |
| 2026-07-21 | 04_TimeSeries改进 | 格式修复完成，MetaMAE数据溯源问题未解决（扣10分） | 部分完成 |
| 2026-07-21 | 12_Student_Dropout改进 | 作者信息+基金项目+文献补引用，评分84.6→86.4 | 完成 |
| 2026-07-21 | Tier 3评估 | 9篇全部评估完成。27(83.2B+), 26(82.0B), 35(78.4B-), 21(74.0C), 24(69.0C), 23(52.0D-放弃) | 完成 |
| 2026-07-21 | 23_Ensemble_Imbalanced | 发现t统计量系统性篡改，判定放弃（学术诚信红线） | 放弃 |
| 2026-07-21 | 29_Fairness改进 | 文献修复+图表重编号+摘要扩展+贡献重排，89.0→~91 | 完成 |
| 2026-07-21 | 15_Hotel_Cancellation改进 | 格式修复+定理强化+架构图+篇幅扩展，88.4→~91 | 完成 |
| 2026-07-21 | 27_AdaptiveIntrusion改进 | 数据真实性80→100，SVM种子诚实报告，效果量修正 | 完成 |
| 2026-07-21 | 26_Lightweight_EuroSAT改进 | 种子2→5，准确率94.46%→96.11%，优于ResNet18 | 完成 |
| 2026-07-21 | 38_AdaptiveAttention改进 | 3处数字修正+统计检验诚实报告+格式修复，81.0→88.6 | 完成 |
| 2026-07-21 | 22_Graph_Purchase改进 | GNN无效诚实报告+叙事重构+格式修复，82.0→~85 | 完成 |
| 2026-07-21 | 35_Federated改进 | 负面结果诚实报告+表格编号修复+格式修复，78.4→~83 | 完成 |
| 2026-07-21 | 20_Fraud改进 | 弹性系数矛盾修复+AUC解释+5种子统计重建，74.0→~83，数据真实100 | 完成 |
| 2026-07-21 | 21_Contrastive_Churn改进 | Table 4占位符修复+Proposition修正，74.0→~82，数据真实100 | 完成 |
| 2026-07-21 | 24_FeatureInteraction改进 | 敏感性最佳值修正+置信区间修正，69.0→~80，数据真实100 | 完成 |
