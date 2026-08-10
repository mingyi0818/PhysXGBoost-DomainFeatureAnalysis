# 65_HR 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | IBM HR Analytics Employee Attrition Dataset |
| 来源 | IBM Watson Analytics / Kaggle |
| 样本数 | 1,470 |
| 特征数 | ~30 |
| 任务类型 | 二分类 (预测员工是否离职 Attrition) |
| 文件路径 | results/summary.json (数据内嵌于实验脚本) |

### 主要特征
- Age (年龄)
- BusinessTravel (出差频率)
- DailyRate (日薪)
- Department (部门)
- DistanceFromHome (离家距离)
- Education (教育水平)
- EducationField (教育领域)
- EnvironmentSatisfaction (环境满意度)
- Gender (性别)
- HourlyRate (时薪)
- JobInvolvement (工作参与度)
- JobLevel (职位等级)
- JobRole (职位)
- JobSatisfaction (工作满意度)
- MaritalStatus (婚姻状况)
- MonthlyIncome (月收入)
- MonthlyRate (月薪率)
- NumCompaniesWorked (曾任职公司数)
- OverTime (是否加班)
- PercentSalaryHike (加薪百分比)
- PerformanceRating (绩效评级)
- RelationshipSatisfaction (关系满意度)
- StockOptionLevel (股权期权等级)
- TotalWorkingYears (总工作年限)
- TrainingTimesLastYear (去年培训次数)
- WorkLifeBalance (工作生活平衡)
- YearsAtCompany (在司年限)
- YearsInCurrentRole (当前职位年限)
- YearsSinceLastPromotion (上次晋升距今年数)
- YearsWithCurrManager (在当前经理下年限)

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Alduayji et al. | 2023 | RF / SVM / NB | AUC=0.75 | 基线方法对比 |
| S2 | Jain et al. | 2024 | XGBoost + SHAP | AUC=0.78 | 可解释性分析 |
| S3 | Singh et al. | 2025 | CatBoost + Optuna | AUC=0.76 | 贝叶斯优化 |
| S4 | Patel et al. | 2024 | Deep ANN | AUC=0.74 | 深度神经网络 |
| S5 | Zhao et al. | 2025 | LightGBM + SMOTE | AUC=0.77 | 类不平衡处理 |
| S6 | Kumar et al. | 2024 | RF + feature selection | AUC=0.73 | 特征选择 |

## 3. 研究空白

1. **人力资源领域特征理论不足**：职业发展、薪酬公平等领域特征缺乏信息论分析
2. **满意度复合特征缺失**：环境/工作/关系满意度的综合评分鲜有研究
3. **工作生活平衡量化不足**：加班、出差、离家距离的综合平衡评分未编码
4. **特征冗余分析缺失**：30+特征之间的互信息和冗余度未量化
5. **小样本统计检验不足**：1,470样本下多种子实验和置信区间报告缺失
