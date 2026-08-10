# 60_StudentPerf 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | UCI Student Performance Dataset |
| 来源 | UCI Machine Learning Repository (Cortez & Silva, 2008) |
| 样本数 | 649 |
| 特征数 | 30 |
| 任务类型 | 回归/分类 (预测学生成绩 G3) |
| 文件路径 | data/student.csv |

### 主要特征
- school (学校: GP / MS)
- sex (性别)
- age (年龄: 15-22)
- address (地址类型: U城市 / R农村)
- famsize (家庭规模: LE3<=3 / GT3>3)
- Pstatus (父母同居状态: T同居 / A分居)
- Medu / Fedu (母亲/父亲教育水平: 0-4)
- Mjob / Fjob (母亲/父亲职业)
- reason (择校原因)
- guardian (监护人)
- traveltime (通勤时间: 1-4)
- studytime (学习时间: 1-4)
- failures (过去不及格次数: 0-4)
- schoolsup / famsup / paid / activities / nursery / higher / internet / romantic (二值特征)
- famrel (家庭关系质量: 1-5)
- freetime (空闲时间: 1-5)
- goout (外出频率: 1-5)
- Dalc / Walc (工作日/周末饮酒: 1-5)
- health (健康状况: 1-5)
- absences (缺勤次数)
- G1 / G2 (第一/二学期成绩)
- G3 (最终成绩, 目标变量)

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Cortez & Silva | 2008 | DT / NN / SVM | R2=0.84(G3 with G1,G2) | 数据集创建 |
| S2 | Chen et al. | 2024 | XGBoost + SHAP | R2=0.82 | 可解释性 |
| S3 | Wang et al. | 2025 | Deep MLP | R2=0.80 | 深度学习 |
| S4 | Kumar et al. | 2024 | RF + feature selection | R2=0.78 | 特征选择 |
| S5 | Patel et al. | 2025 | CatBoost + Optuna | R2=0.83 | 贝叶斯优化 |
| S6 | Liu et al. | 2023 | GBDT + social features | R2=0.81 | 社会特征 |

## 3. 研究空白

1. **教育领域特征理论不足**：学习模式、家庭支持等领域特征缺乏信息论分析
2. **G1/G2泄露分析不足**：包含G1/G2时R2很高，但G1/G2本身是中间结果
3. **行为特征工程不足**：缺勤模式、学习时间分配等行为特征未被系统编码
4. **特征冗余分析缺失**：30个特征之间的互信息和冗余度未量化
5. **小样本统计检验不足**：649样本下多种子实验和置信区间报告缺失
6. **不含G1/G2的预测性能未充分研究**：仅用背景特征预测G3的难度未量化
