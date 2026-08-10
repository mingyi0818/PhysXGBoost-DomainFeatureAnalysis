# 十个研究方向专属指令

**依赖声明**：本文件依赖 `aicommand.md` 中定义的通用工作流、审查标准和作者信息。请先阅读并理解 `aicommand.md`。

---

## 十个研究方向

### 方向一：表格数据统一框架 (Unified Framework for Tabular Data)
- **文件夹路径**：`d:\ResearchPaperPrepare\01_Tabular_Framework`
- **核心创新**：自适应分位数校准（Adaptive Quantile Calibration, AQC）
- **数据集**：Telco-Customer-Churn + IBM-HR-Analytics + UCI-Adult + Bank-Marketing
- **目标期刊**：Expert Systems with Applications（IF=8.5）

### 方向二：时序预测统一框架 (Unified Framework for Time-Series Forecasting)
- **文件夹路径**：`d:\ResearchPaperPrepare\02_Time_Series_Framework`
- **核心创新**：时序预测不确定性量化对比研究（Uncertainty Quantification Comparative Study）
- **数据集**：Air Passengers + Daily Climate + Daily Births + Stock Price + Shampoo Sales
- **目标期刊**：Neurocomputing（IF=5.5）

### 方向三：文本 + LLM 框架 (Text + LLM Framework)
- **文件夹路径**：`d:\ResearchPaperPrepare\03_Text_LLM_Framework`
- **核心创新**：轻量级对比学习情感分析
- **数据集**：515K-Hotel-Reviews-Europe + TripAdvisor-Hotel-Reviews + SMS-Spam-Collection
- **目标期刊**：Information Processing & Management（IF=8.0）
- **状态**：数据集太大，原来的实验数据不真实，所以还是暂时不做这个方面，以后提示词里所有方面不包括方向三。

### 方向四：不平衡学习 (Imbalanced Learning)
- **文件夹路径**：`d:\ResearchPaperPrepare\04_Imbalanced_Learning`
- **核心创新**：对比学习引导的重加权损失（Contrastive-Guided Reweighting Loss, CGRL）
- **数据集**：Credit-Card-Fraud + Pima-Diabetes + Heart-Disease + Telco-Churn
- **目标期刊**：IEEE Transactions on Knowledge and Data Engineering（IF=8.0）

### 方向五：旅游数据智能预测 (Tourism Data Intelligence)
- **文件夹路径**：`d:\ResearchPaperPrepare\05_Tourism_Prediction`
- **核心创新**：多视图时空融合Transformer（Multi-View Spatio-Temporal Fusion Transformer, MVSTFT）
- **数据集**：Bike-Sharing
- **目标期刊**：Tourism Management Perspectives（IF=5.5）

### 方向六：农业数据融合决策支持 (Agricultural Data Fusion)
- **文件夹路径**：`d:\ResearchPaperPrepare\06_Agriculture_Fusion`
- **核心创新**：多模态农业知识图谱融合（Multi-Modal Agricultural Knowledge Graph Fusion, MMAKGF）
- **数据集**：Crop-Recommendation + EuroCrops + coffee-quality-database
- **目标期刊**：Computers and Electronics in Agriculture（IF=7.7）

### 方向七：农业小样本增量学习 (Few-shot Incremental Learning for Agriculture)
- **文件夹路径**：`d:\ResearchPaperPrepare\07_Agriculture_FewShot`
- **核心创新**：农业少样本增量学习方法的系统性比较研究（Systematic Comparison of FSIL Methods for Agricultural Image Classification）
- **数据集**：PlantVillage + DeepWeeds
- **目标期刊**：Computers and Electronics in Agriculture（IF=7.7）

### 方向八：表格数据小样本增量学习 (Few-shot Incremental Learning for Tabular Data)
- **文件夹路径**：`d:\ResearchPaperPrepare\08_Tabular_FewShot`
- **核心创新**：对比学习增强的原型记忆网络（Contrastive-Enhanced Prototype Memory Network, CE-PMN）
- **数据集**：Telco + Adult
- **目标期刊**：Expert Systems with Applications（IF=8.5）

### 方向九：HSIC解耦轻量原型网络 (HSIC-Disentangled Lightweight Prototypical Network)
- **文件夹路径**：`d:\ResearchPaperPrepare\09_HSIC_FDANet`
- **核心创新**：HSIC引导的特征解耦原型网络（HSIC-guided Feature Disentanglement Prototypical Network），面向边缘设备部署
- **数据集**：PlantVillage
- **目标期刊**：Computers and Electronics in Agriculture（IF=7.7）

### 方向十：AI增强时空Transformer旅游预测 (AI-Enhanced Spatial-Temporal Transformer for Tourism)
- **文件夹路径**：`d:\ResearchPaperPrepare\10_AI_Tourism_Forecast`
- **核心创新**：图注意力+时序Transformer的时空融合预测（ST-Transformer with Graph Attention）
- **数据集**：Guangdong Tourism（规划中）+ ALANA (Canada/Mexico/USA)
- **目标期刊**：Tourism Management（IF=11.3）
- **状态**：论文草稿完成，需要补充真实实验

---

## 十方向专属启动指令

请按照 `aicommand.md` 中定义的多智能体协作工作流（Phase 1-4），为上述十个研究方向逐一启动算法辩论与论文生成流程。

**执行规则**：
1. 采用轮次迭代制进行多模型辩论审查：每轮5个模型依次审查，后发言的模型基于前面的发言提出问题和建议；每轮结束后由DeepSeek-V4-Pro负责总结，合理的意见接受修改，不合理的提出反对意见；至少迭代5轮，确保每篇论文达到SCI一区论文质量要求。
2. 注意一次只处理一个方向，避免上下文不足的问题。
3. 默认不启动方向三（文本+LLM框架），除非明确要求。
4. 所有论文必须严格遵循 `aicommand.md` 中的审查标准和学术红线。

**启动顺序建议**：方向九 → 方向七 → 方向六 → 方向四 → 方向一 → 方向二 → 方向五 → 方向八 → 方向十 → 方向三（可选）

如果你已经完全理解 `aicommand.md` 的内容和本文件的研究方向，请开始为第一个方向启动流程。
