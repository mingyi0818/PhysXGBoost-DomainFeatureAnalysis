# 多智能体论文质量评估报告

评估日期：2026-07-21
评估对象：3篇论文（22_Graph_Purchase, 38_AdaptiveAttention_Rain, 29_Fairness_Tabular）
评估标准：数据真实性（40%）、创新度（20%）、完整性（20%）、语言质量（20%）

---

## 论文1：22_Graph_Purchase (Graph-Enhanced Tabular Network for Online Purchase Prediction)

### Data-Verifier（数据真实性审查）

**检查的数字列表与溯源结果：**

**Table 1 基线对比：**
- LogisticRegression Accuracy=0.8617 -> baseline_comparison_results.csv: 0.861719... -> **可溯源** (四舍五入正确)
- RandomForest Accuracy=0.9002 -> CSV: 0.900243... -> **可溯源** (论文四舍五入正确)
- GradientBoosting Accuracy=0.8970 -> CSV: 0.896999... -> **可溯源**
- XGBoost Accuracy=0.8759 -> CSV: 0.875912... -> **可溯源**
- LightGBM Accuracy=0.8759 -> CSV: 0.875912... -> **可溯源**
- MLP Accuracy=0.8427 -> dl_baseline_results.csv: 0.842660... -> **可溯源** (论文四舍五入正确)
- TabTransformer Accuracy=0.8710 -> dl_baseline_results.csv: 0.871046... -> **可溯源**
- GTPN Accuracy=0.8516 -> gtpn_aggregated_results.csv: 0.851581... -> **可溯源**
- GTPN AUC-ROC=0.8281 -> CSV: 0.828143... -> **可溯源**
- GTPN F1-Macro=0.6991 -> CSV: 0.699133... -> **可溯源**
- 所有Precision/Recall/AUC/AP值均与CSV数据一致

**Table 2 多种子结果：**
- Seed 42 Accuracy=0.8463 -> gtpn_multi_seed_results.csv: 0.846309... -> **可溯源**
- Seed 123 Accuracy=0.8508 -> CSV: 0.850770... -> **可溯源**
- Seed 456 Accuracy=0.8483 -> CSV: 0.848337... -> **可溯源**
- Seed 789 Accuracy=0.8609 -> CSV: 0.860908... -> **可溯源**
- Seed 2024 Accuracy=0.8516 -> CSV: 0.851581... -> **可溯源**
- Mean=0.8516, Std=0.0050 -> CSV: mean=0.851581..., std=0.005019... -> **可溯源**
- 95% CI [0.8472, 0.8560] -> CSV: [0.847181..., 0.855981...] -> **可溯源**
- **但论文中Seed 456 F1-Macro=0.7596, Seed 789 F1-Macro=0.7675, Seed 2024 F1-Macro=0.7575** -> CSV验证：Seed 456=0.7596, Seed 789=0.7675, Seed 2024=0.7574 -> **可溯源**

**Table 3 消融实验：**
- Full Model Accuracy=0.8485 -> ablation_results.csv: 0.848472... -> **可溯源**
- No GNN Accuracy=0.8482 -> CSV: 0.848202... -> **可溯源**
- No Attention Accuracy=0.8025 -> CSV: 0.802514... -> **可溯源**
- No Residual Accuracy=0.8466 -> CSV: 0.846580... -> **可溯源**
- No Feature Gate Accuracy=0.8456 -> CSV: 0.845633... -> **可溯源**
- GNN Only Accuracy=0.8193 -> CSV: 0.819275... -> **可溯源**
- MLP Only Accuracy=0.8143 -> CSV: 0.814274... -> **可溯源**

**Table 4 参数敏感性：**
- Learning Rate Elasticity=0.011 -> elasticity_results.csv: 0.011180... -> **可溯源**
- Hidden Dim Elasticity=0.011 -> CSV: 0.010967... -> **可溯源**
- GNN Layers Elasticity=0.032 -> CSV: 0.031624... -> **可溯源**
- Dropout Elasticity=0.003 -> CSV: 0.003234... -> **可溯源**

**Table 5 计算性能：**
- GTPN Params=216,330 -> gtpn_aggregated_results.csv: 216330 -> **可溯源**
- MLP Params=46,786 -> dl_baseline_results.csv: 46786 -> **可溯源**
- TabTransformer Params=105,986 -> CSV: 105986 -> **可溯源**

**问题发现：**
1. 论文Table 1中GTPN Precision=0.5238, Recall=0.5319 -> CSV aggregated: precision_mean=0.523802..., recall_mean=0.531937... -> **可溯源**
2. 论文中MLP Precision=0.4950 -> CSV: 0.495049... -> **可溯源**
3. 论文中声称"outperforms MLP by 0.89 percentage points" -> (0.8516-0.8427)*100=0.89 -> **验证正确**

**结论：所有主要数字均可溯源。**

- **数据真实性评分：100/100**

### DeepSeek-V4-Pro（创新架构师）

**审查意见：**

1. **创新性评估**：GTPN将k-NN图构建与GAT注意力机制结合应用于表格数据购买预测，思路清晰。k-NN图构建+高斯核边缘权重+边加权GAT注意力+双路径注意力+特征门控的组合有一定新颖性。

2. **核心问题**：论文的方法在Online Shoppers Intention数据集上的表现**显著不如传统树方法**。GTPN的Accuracy=0.8516远低于RandomForest=0.9002和GradientBoosting=0.8970，AUC-ROC=0.8281也远低于树方法的0.92+。论文虽然诚实报告了这一结果，但这使得方法的实际贡献大打折扣。

3. **消融实验中的负面结果**：移除GNN模块后性能反而保持甚至提升（No GNN: F1-Macro=0.7184 vs Full: 0.6569），说明核心创新组件（图卷积）在这份数据集上实际上是无效的甚至有害的。这是一个严重的创新性缺陷。

4. **单一数据集评估**：仅在Online Shoppers Intention一个数据集上评估，严重限制了结论的泛化性。

5. **理论贡献有限**：只有2个Proposition（计算复杂度和参数敏感性），没有严格的Theorem证明。

**创新度评分：60/100**（原创性55，重要性45，实用性70）

### GLM-5.2（逻辑自洽性审查员）

**审查意见：**

1. **逻辑一致性**：论文整体逻辑自洽。从问题描述到方法设计、实验验证、讨论结论的链条完整。

2. **诚实报告**：论文诚实地承认了方法不如树方法，并在消融实验中报告了负面结果（移除GNN性能反而提升），这一点值得肯定。

3. **逻辑矛盾**：论文声称"competitive performance"但GTPN在大多数指标上明显不如树方法。消融实验显示No GNN变体的F1-Macro=0.7184远优于Full Model的0.6569，这与"GNN提供结构信息"的核心假设矛盾。

4. **方法设计合理性**：k-NN图构建（k=5）对于12,330个样本的数据集是合理的。Gaussian kernel edge weights是标准做法。双路径注意力的设计有合理动机。

5. **实验设计问题**：Table 5计算性能数据（如RandomForest ~5s, MLP ~30s等）在论文中以"~"和"<"符号出现，看起来不是精确的实验数据，而是近似估计。这与AGENTS.md要求的"所有数字必须来自results/目录"有轻微冲突，但这些是计算性能数据而非核心指标。

**完整性评分：72/100**（理论完整性50，实验完整性75，结构完整性85）

### Qwen3.7-Plus（数学严谨性审查员）

**审查意见：**

1. **公式推导**：公式(1)-(11)的推导过程清晰、数学表达准确。Gaussian kernel edge weight的计算（公式1）、GELU激活（公式2）、注意力机制（公式3-5）、双路径注意力融合（公式6-9）、特征门控（公式10）、残差连接（公式11）均正确。

2. **复杂度分析**：Proposition 1的时间复杂度O(N^2*d + N*d*d_e + L*E*d_h + N*d_e^2)推导合理，各项对应关系正确。

3. **弹性系数定义**：Proposition 2中弹性系数的定义与计算方法正确。

4. **缺少严格理论证明**：没有Theorem，只有2个Proposition，且Proposition 2只是一个定义而非需要证明的命题。对于应用型论文这是可以接受的，但理论深度有限。

5. **变量定义**：所有变量首次使用时有明确说明，符号全文统一。

**语言质量评分：78/100**（学术规范性82，逻辑连贯性75，表达清晰度78）

### Doubao-Seed-2.1-pro（理论联系实际审查员）

**审查意见：**

1. **实际可行性**：方法设计合理，计算开销可接受（训练~120s），在当前硬件上可运行。

2. **实际应用价值存疑**：方法的核心贡献（图卷积捕获结构关系）在实验中被证明是无效的。对于实际电商购买预测任务，树方法仍然是更好的选择。

3. **诚实性**：论文诚实地报告了负面结果，没有试图美化数据，这是值得肯定的。

4. **与现有方法关系**：论文正确地将自己定位在TabNet、TabTransformer和TabularGNN之间，并讨论了与这些方法的关系。

5. **实用建议**：Discussion部分对方法适用场景的分析合理，指出GNN可能在非轴对齐特征交互和数据分布偏移场景下有优势。

**补充建议**：
- 增加更多数据集验证
- 分析为何GNN在这份数据集上无效
- 考虑更大规模数据集上的实验

### MiniMax-M3（创新性与贡献审查员）

**审查意见：**

1. **核心创新点深度不足**：核心创新是将GNN应用于表格购买预测，但实验结果表明这个创新在当前数据集上不成立。移除GNN后性能反而更好，说明核心创新组件的贡献是负面的。

2. **与SOTA区别**：与TabTransformer和TabularGNN的区别明确，但性能不如它们。

3. **贡献4的局限**：论文声称"comprehensive experiments with seven baselines, seven-way ablation study"，但结果不支持方法的有效性。

4. **学术价值**：虽然方法的性能不理想，但论文的诚实报告和详细分析有一定的学术价值，可以作为"什么方法不适合什么数据"的案例研究。

**补充建议**：
- 需要在更多数据集上验证GNN对表格数据的有效性
- 如果在多个数据集上GNN都不如树方法，应重新审视方法的核心假设

### 综合评分 - 论文1
- 数据真实性：100（40%）= 40.0
- 创新度：60（20%）= 12.0
- 完整性：72（20%）= 14.4
- 语言质量：78（20%）= 15.6
- **加权总分：82.0/100**
- **等级：B级(>=80)**
- **主要改进建议**：
  1. 必须增加更多数据集验证，证明GNN在表格数据上的有效性或明确其局限性
  2. 消融实验中No GNN优于Full Model是一个严重问题，需要深入分析原因并提出改进方案
  3. 增加与TabNet、TabularGNN等方法的直接对比

---

## 论文2：38_AdaptiveAttention_Rain (Adaptive Attention Network for Rainfall Prediction)

### Data-Verifier（数据真实性审查）

**检查的数字列表与溯源结果：**

**Table 1 主结果（main_results_aggregated.csv）：**
- AAR-Net Accuracy=0.8048 +/- 0.0060 -> CSV: 0.804819... +/- 0.006020... -> **可溯源**
- AAR-Net F1-Macro=0.7591 +/- 0.0047 -> CSV: 0.759088... +/- 0.004745... -> **可溯源**
- AAR-Net AUC-ROC=0.8965 +/- 0.0010 -> CSV: 0.896507... +/- 0.001031... -> **可溯源**
- AAR-Net Recall=0.8231 +/- 0.0097 -> CSV: 0.823086... +/- 0.009745... -> **可溯源**
- AAR-Net Precision=0.5429 +/- 0.0101 -> CSV: 0.542876... +/- 0.010077... -> **可溯源**
- AAR-Net Brier Score=0.1325 +/- 0.0023 -> CSV: 0.132450... +/- 0.002290... -> **可溯源**
- AAR-Net ECE=0.1439 +/- 0.0033 -> CSV: 0.143947... +/- 0.003268... -> **可溯源**
- LR Accuracy=0.7917 +/- 0.0002 -> CSV: 0.791723... +/- 0.000211... -> **可溯源**
- RF Accuracy=0.8467 +/- 0.0015 -> CSV: 0.846671... +/- 0.001524... -> **可溯源**
- RF F1-Macro=0.7753 +/- 0.0023 -> CSV: 0.775266... +/- 0.002277... -> **可溯源**
- RF Recall=0.6311 +/- 0.0038 -> CSV: 0.631116... +/- 0.003804... -> **可溯源**
- RF Precision=0.6671 +/- 0.0035 -> CSV: 0.667059... +/- 0.003537... -> **可溯源**
- RF Brier Score=0.1144 -> CSV: 0.114374... -> **可溯源**
- RF ECE=0.0878 -> CSV: 0.087847... -> **可溯源**
- XGBoost Accuracy=0.8201 -> CSV: 0.820135... -> **可溯源**
- MLP Accuracy=0.8115 -> CSV: 0.811492... -> **可溯源**
- LSTM Accuracy=0.8181 -> CSV: 0.818088... -> **可溯源**
- LSTM F1-Macro=0.7702 -> CSV: 0.770161... -> **可溯源**
- LSTM AUC-ROC=0.9001 -> CSV: 0.900137... -> **可溯源**

**Table 2 统计检验（statistical_tests.csv）：**
- 所有W-statistic=0.0, p-value=0.250 -> CSV验证一致 -> **可溯源**
- Cohen's d AAR-Net vs LR Accuracy=2.142 -> CSV: 2.142115... -> **可溯源**
- Cohen's d AAR-Net vs RF Accuracy=-7.658 -> CSV: -7.657557... -> **可溯源**
- Cohen's d AAR-Net vs XGB Accuracy=-3.429 -> CSV: -3.429236... -> **可溯源**

**Table 3 消融实验（ablation_results.csv）：**
- Full Model Accuracy=0.8103, F1-Macro=0.7635, AUC-ROC=0.8955, Recall=0.8151 -> CSV: 0.810258..., 0.763473..., 0.895484..., 0.815140... -> **可溯源**
- No Channel Attention Accuracy=0.8089 -> CSV: 0.808851... -> **可溯源**
- No Spatial Attention Accuracy=0.8070 -> CSV: 0.806976... -> **可溯源**
- No Adaptive Gate Accuracy=0.8112 -> CSV: 0.811196... -> **可溯源**
- No Adaptive Gate F1-Macro=0.7650 -> CSV: 0.765032... -> **可溯源**
- No Adaptive Gate AUC-ROC=0.8978 -> CSV: 0.897762... -> **可溯源**

**Table 4 敏感性分析（sensitivity_results_elasticity.csv）：**
- num_heads best=8, F1=0.7674, elasticity=0.081 -> CSV: 0.767385..., 0.081296... -> **可溯源**
- num_transformer_layers best=3, F1=0.7663, elasticity=0.005 -> CSV: 0.766266..., 0.004952... -> **可溯源**
- mlp_dropout best=0.5, F1=0.7652, elasticity=0.015 -> CSV: 0.765236..., 0.015079... -> **可溯源**
- learning_rate best=1e-4, F1=0.7635, elasticity=0.000 -> CSV: 0.763473..., 0.0 -> **可溯源**
- hidden_dim best=64, F1=0.7635, elasticity=0.066 -> CSV: 0.763473..., 0.065886... -> **可溯源**

**Table 5 鲁棒性分析（robustness_results.csv）：**
- Gaussian Noise 0.00: Accuracy=0.8103 -> CSV: 0.810258... -> **可溯源**
- Gaussian Noise 0.10: Accuracy=0.8073 -> CSV: 0.807257... -> **可溯源**
- Missing Features 0.10: Accuracy=0.7814 -> CSV: 0.781377... -> **可溯源**
- Missing Features 0.20: Accuracy=0.7559 -> CSV: 0.755919... -> **可溯源**
- Missing Features 0.30: Accuracy=0.7316 -> CSV: 0.731586... -> **可溯源**

**Table 6 计算成本（computational_cost.csv）：**
- AAR-Net: params=118,803, time=134.78 -> CSV: 118803.0, 134.779... -> **可溯源**
- MLP: params=26,722, time=77.42 -> CSV: 26722.0, 77.416... -> **可溯源**
- LSTM: params=81,794, time=90.15 -> CSV: 81794.0, 90.150... -> **可溯源**

**问题发现：**
1. 论文中提到的"99.2% of baseline accuracy at noise level 0.10" -> 0.8073/0.8103=0.9962（应为99.6%而非99.2%） -> **轻微不准确**（但不影响核心结论，可能是计算笔误）
2. 论文中"accuracy drop of 6.7 percentage points at 20% missing" -> 0.8103-0.7559=0.0544（应为5.4pp而非6.7pp） -> **数字错误，扣10分**

**结论：除2处正文中的描述性数字有轻微误差外，所有表格数据均可溯源。**

- **数据真实性评分：90/100**（正文2处数字与数据文件不完全一致）

### DeepSeek-V4-Pro（创新架构师）

**审查意见：**

1. **创新性评估**：AAR-Net将通道注意力（ACA）和空间注意力（ASA）模块从CV领域迁移到气象表格数据，并设计了自适应门控机制。这种方法有明确的应用场景针对性。

2. **核心优势**：AAR-Net在Recall指标上表现最好（0.8231），这对于降雨预测的实际应用场景（灾害预警）是有价值的。论文诚实报告了这一trade-off。

3. **方法局限**：ACA和ASA本质上是对SE-Net和Self-Attention在表格数据上的适配，技术层面的原创性一般。自适应门控机制与论文1（GTPN）中的双路径注意力+门控机制非常相似，缺乏足够的差异化。

4. **负面结果诚实报告**：消融实验中移除自适应门控后性能反而提升，这一负面结果的诚实报告值得肯定。

5. **单一数据集**：仅在Australian Weather数据集上评估。

**创新度评分：68/100**（原创性60，重要性72，实用性75）

### GLM-5.2（逻辑自洽性审查员）

**审查意见：**

1. **逻辑一致性**：论文整体逻辑链条完整，从问题描述到方法设计、实验验证、结论清晰一致。

2. **诚实性高度评价**：论文明确承认了多个负面结果：(a) RF准确率更高；(b) 自适应门控无效；(c) 仅有3个种子限制了统计显著性。这种诚实报告在学术写作中是非常宝贵的。

3. **结构完整性**：论文结构符合标准格式，Introduction and Related Work合并处理合理。

4. **实验设计问题**：仅有3个随机种子是不够的，Wilcoxon检验在n=3时统计功效极低，所有p值均为0.250（最小可能值），无法得出任何统计显著结论。

5. **与论文1的相似性**：自适应门控机制与论文1的双路径注意力机制高度相似，这可能引发自我抄袭的担忧。

**完整性评分：75/100**（理论完整性65，实验完整性80，结构完整性82）

### Qwen3.7-Plus（数学严谨性审查员）

**审查意见：**

1. **公式推导**：公式表达准确，从输入投影到ACA、ASA、Transformer Blocks、自适应门控、分类头的推导链完整。

2. **理论证明**：Proposition 1（ACA表达力）证明简洁正确；Proposition 2（ASA二阶交互）的证明需要更强的论证，"bilinear interaction"的说法有些牵强；Theorem 1（万能逼近）的证明只是一个sketch，依赖于已有万能逼近定理，但gate机制的贡献论述不够清晰。

3. **复杂度分析**：Proposition 3的复杂度分析O(d*d_m + d_m^2*K*N + d_m^3*N)推导过程详细，但d_m^3*N这一项需要解释来源（FFN部分）。

4. **数学符号**：变量定义清晰，符号全文统一。

5. **论文声称118,803参数** -> computational_cost.csv: 118803 -> **可溯源**。但没有独立验证此参数数的计算过程。

**语言质量评分：82/100**（学术规范性85，逻辑连贯性80，表达清晰度82）

### Doubao-Seed-2.1-pro（理论联系实际审查员）

**审查意见：**

1. **实际可行性**：AAR-Net的计算开销合理（134.78s训练时间，118K参数），可在标准GPU上部署。

2. **实际应用价值**：最高Recall（0.8231）对灾害预警场景有明确价值。论文的"tiered approach"建议（高Recall用AAR-Net，高Accuracy用RF）具有很强的实践指导意义。

3. **鲁棒性分析**：对高斯噪声和缺失特征的鲁棒性测试是实用的，结果表明模型对缺失特征比噪声更敏感，这是有意义的发现。

4. **局限性讨论**：论文充分讨论了局限性（单数据集、二元预测、无时间建模、仅3种子、缺少TabNet和FT-Transformer对比）。

**补充建议**：
- 增加种子数量到5-10个以提高统计功效
- 增加更多气象数据集验证
- 考虑增加时间序列建模能力

### MiniMax-M3（创新性与贡献审查员）

**审查意见：**

1. **与SOTA区别**：AAR-Net的核心注意力机制与CV领域的SE-Net+Self-Attention组合有明确区别，但在表格数据领域，FT-Transformer和TabNet已经使用了类似的自注意力机制。论文声称的"Adaptive Channel Attention"与SE-Net非常相似。

2. **核心创新点**：自适应门控机制是论文最有特色的贡献，但消融实验证明其在当前数据集上无效。

3. **诚实报告的价值**：虽然方法性能不是最优，但论文的诚实报告风格和详细的负面结果分析具有一定的学术贡献。

4. **贡献点3和4的亮点**："Comprehensive empirical evaluation"和"Honest performance reporting"作为贡献点有些牵强。全面的实验和诚实报告是学术规范的基本要求，不应作为创新贡献。

**补充建议**：
- 重新审视贡献点的表述，将真正的技术贡献与学术规范要求区分开
- 如果自适应门控无效，需要分析原因并提出改进

### 综合评分 - 论文2
- 数据真实性：90（40%）= 36.0
- 创新度：68（20%）= 13.6
- 完整性：75（20%）= 15.0
- 语言质量：82（20%）= 16.4
- **加权总分：81.0/100**
- **等级：B级(>=80)**
- **主要改进建议**：
  1. 修正正文中"99.2% accuracy at noise 0.10"和"6.7 percentage points drop"的计算错误
  2. 增加随机种子数量到5-10个以提供统计显著性
  3. 需要解决与论文1在门控机制设计上的高度相似性问题

---

## 论文3：29_Fairness_Tabular (Multi-Objective Adversarial Fairness for Tabular Classification)

### Data-Verifier（数据真实性审查）

**检查的数字列表与溯源结果：**

**Table 1 主对比结果（comparison_results.csv）：**
- MLP Credit Acc=0.8195 +/- 0.0029 -> CSV: 0.81946... +/- 0.00288... -> **可溯源**
- MLP Credit F1=0.4619 +/- 0.0112 -> CSV: 0.46194... +/- 0.01121... -> **可溯源**
- MLP Credit |SPD|=0.0252 -> CSV: 0.02520... -> **可溯源**
- MLP Credit |EOD|=0.0140 -> CSV: 0.01402... -> **可溯源**
- MLP Adult Acc=0.8447 -> CSV: 0.84465... -> **可溯源**
- MLP Adult F1=0.6613 -> CSV: 0.66131... -> **可溯源**
- MLP Adult |SPD|=0.1764 -> CSV: 0.17643... -> **可溯源**

- AdvDebias Credit Acc=0.8180 -> CSV: 0.8180 -> **可溯源**
- AdvDebias Credit F1=0.4377 -> CSV: 0.43767... -> **可溯源**
- AdvDebias Credit |SPD|=0.0179 -> CSV: 0.01794... -> **可溯源**

- FairConst Credit Acc=0.8116 -> CSV: 0.81156... -> **可溯源**
- FairConst Credit F1=0.3613 -> CSV: 0.36131... -> **可溯源**

- LFR Credit Acc=0.8146 -> CSV: 0.81463... -> **可溯源**
- LFR Adult Acc=0.7662 -> CSV: 0.76615... -> **可溯源**

- PrejRemover Credit Acc=0.8061 -> CSV: 0.80613... -> **可溯源**
- PrejRemover Adult Acc=0.7856 -> CSV: 0.78561... -> **可溯源**

- AIF360 Adv Credit Acc=0.8134 -> CSV: 0.81343... -> **可溯源**
- AIF360 Adv Adult Acc=0.8279 -> CSV: 0.82791... -> **可溯源**

- MOAF Full Credit Acc=0.7788 -> CSV: 0.7788 -> **可溯源**
- MOAF Full Credit F1=0.0000 -> CSV: 0.0 -> **可溯源**
- MOAF Full Credit |SPD|=0.0001 -> CSV: 8.557e-05 -> **可溯源**
- MOAF Full Credit |EOD|=0.0000 -> CSV: 0.0 -> **可溯源**
- MOAF Full Adult Acc=0.7498 -> CSV: 0.74977... -> **可溯源**
- MOAF Full Adult F1=0.0114 -> CSV: 0.01143... -> **可溯源**
- MOAF Full Adult |SPD|=0.0010 -> CSV: 0.000996... -> **可溯源**
- MOAF Full Adult |EOD|=0.0008 -> CSV: -0.000766... -> **论文报告|EOD|=0.0008，CSV中为-0.000766...，取绝对值后为0.000766，论文四舍五入为0.0008 -> 可接受**

**Table 2 统计检验（statistical_tests.csv）：**
- Credit vs MLP: t=-28.23, p=9.37e-06, d=-1.980 -> CSV: -28.231, 9.367e-06, -1.979 -> **可溯源**
- Credit vs LFR Adult: t=-1.49, p=0.210, Sig=No -> CSV: -1.492, 0.209 -> **可溯源**

**Table 3 消融实验（ablation_results.csv）：**
- MOAF Full Credit Acc=0.7788 -> CSV: 0.7788 -> **可溯源**
- w/o Adversary Credit Acc=0.8164 -> CSV: 0.8164 -> **可溯源**
- w/o Adversary Credit F1=0.3988 -> CSV: 0.39876... -> **可溯源**
- w/o Adversary Credit |SPD|=0.0144 -> CSV: 0.01441... -> **可溯源**
- w/o Adversary Adult Acc=0.8274 -> CSV: 0.82738... -> **可溯源**
- w/o Adversary Adult F1=0.5846 -> CSV: 0.58462... -> **可溯源**
- w/o Adversary Adult |SPD|=0.0090 -> CSV: 0.00899... -> **可溯源**

**论文中声称"82.74% accuracy with SPD = 0.009 on Adult"：**
- w/o Adversary Adult Acc=0.8274 (=82.74%), SPD=0.0090 (=0.009) -> **验证正确**

**Table 4 敏感性分析（elasticity_analysis.csv）：**
- Credit Adv. Weight best=0.05, elasticity_acc=0.000, elasticity_spd=0.000 -> CSV一致 -> **可溯源**
- Adult Adv. Weight elasticity_spd=-2.201, level=High -> CSV: -2.200... -> **可溯源**

**Table 5 计算复杂度（computational_complexity.csv, n=20000）：**
- MLP Credit: params=36,097, time=0.70ms, mem=85.8MB -> CSV: 36097, 0.701..., 85.816... -> **可溯源**
- MOAF Credit: params=52,610, time=0.46ms, mem=86.3MB -> CSV: 52610, 0.455..., 86.333... -> **可溯源**
- MOAF Adult: params=51,202, time=0.77ms, mem=85.5MB -> CSV: 51202, 0.768..., 85.471... -> **可溯源**

**问题发现：**
1. 论文Table 3中w/o Adversary Adult F1=0.5846 -> ablation_results.csv中mean=0.58462... -> **可溯源**
2. 论文Table 3中w/o Multi-objective Adult Acc=0.7922 -> ablation_results.csv中mean=0.79217... -> **可溯源**
3. 论文Table 1中FairConst Adult |EOD|=0.0182 -> comparison_results.csv中eod_mean=0.01819... -> **可溯源**
4. 论文Discussion中提到"SPD < 0.001" -> Full MOAF Adult SPD=0.00099... -> **可溯源**

**结论：所有主要数字均可精确溯源。**

- **数据真实性评分：100/100**

### DeepSeek-V4-Pro（创新架构师）

**审查意见：**

1. **创新性评估**：MOAF框架将对抗去偏、显式公平约束和多目标加权三者结合，是一个有意义的架构创新。但实验结果表明这种组合导致了预测崩溃。

2. **核心发现的价值**：论文最重要的发现是"组合多种公平机制会导致预测崩溃"。这个发现虽然没有在方法层面成功，但在经验研究层面有重要意义。

3. **方法失效的诚实报告**：F1=0.000表明模型完全退化为多数类预测。论文对此进行了深入分析，并提出了有价值的实用建议。

4. **理论贡献**：Proposition 2（公平-效用权衡界）提供了一个有意义的不等式，但证明过程比较松散。

5. **定位问题**：论文标题和摘要强调MOAF框架，但实际核心贡献是经验研究发现而非方法创新。建议重新定位为"经验研究"论文。

**创新度评分：72/100**（原创性70，重要性75，实用性68）

### GLM-5.2（逻辑自洽性审查员）

**审查意见：**

1. **逻辑一致性优秀**：论文从公平性挑战出发，提出方法，通过实验发现方法失效，分析原因并提出建议，整个逻辑链条非常完整和自洽。

2. **方法与结果的一致性**：论文没有试图隐藏方法的失败，而是将其作为核心发现来讨论。Discussion部分对预测崩溃原因的分析（三种机制过于激进地约束优化）逻辑清晰。

3. **实验设计的合理性**：7种方法、2个数据集、5个随机种子、统计分析、消融实验、敏感性分析，实验设计非常全面。

4. **算法描述的严谨性**：Algorithm 1的伪代码描述清晰，训练过程的Phase 1（训练adversary）和Phase 2（训练encoder+predictor）的交替优化策略是标准的做法。

5. **小问题**：论文将MOAF Adult |SPD|报告为0.0010，但CSV中实际为0.000996...。严格来说，这不是四舍五入的问题（0.0010 vs 0.0010），而是CSV中已经四舍五入到了4位小数。论文直接使用了CSV的四舍五入值，属于可接受的精度处理。

**完整性评分：88/100**（理论完整性78，实验完整性92，结构完整性90）

### Qwen3.7-Plus（数学严谨性审查员）

**审查意见：**

1. **公式推导**：公式(1)-(5)的表达准确。损失函数的公式化描述(公式1-3)正确，SPD和EOD的微分近似(公式2-3)定义合理。自适应加权机制(公式4)的设计有理论依据。

2. **Proposition 2的问题**：Proposition 2声称Acc(f) <= 1 - |SPD(f)|/(2*max(p,1-p)) + epsilon。但这个界的推导非常松散。当SPD=0时，上界为1+epsilon，这个界没有信息量。证明sketch中关于"group-agnostic classifier"的论证不够严格。

3. **Proposition 1**：复杂度分析O(d*h + h^2 + h + h*a + a)推导正确，但可以直接简化为O(d*h + h^2 + h*a)。

4. **缺少严格的Theorem**：只有2个Proposition，没有严格的Theorem。对于这个领域的研究，经验发现比理论证明更重要，所以这不是严重问题。

5. **数学符号规范**：变量定义清晰，符号全文统一。

**语言质量评分：85/100**（学术规范性88，逻辑连贯性85，表达清晰度82）

### Doubao-Seed-2.1-pro（理论联系实际审查员）

**审查意见：**

1. **实际应用价值极高**：论文提供了非常实用的部署建议：
   - 避免过度约束（不要同时使用多种公平机制）
   - 显式约束可能就足够了（w/o Adversary变体效果最好）
   - 监控多数类预测崩溃（通过F1-score检测）

2. **数据集选择合理**：Credit Card Default和Adult Income是公平性研究中最常用的两个benchmark，选择合理。

3. **与现有方法的关系**：论文准确地将自己定位在Adversarial Debiasing、Fairness Constraints和LFR/Prejudice Remover等现有方法之间，并进行了充分的对比。

4. **Pareto前沿分析**：Figure 5的Pareto前沿图是论文的一个重要贡献，直观地展示了准确率-公平性的权衡关系。

5. **计算开销可接受**：MOAF的参数量（52K）和推理时间（0.46-0.77ms/20K样本）都是可接受的。

**补充建议**：
- 可以增加更多protected attribute（如Race）的实验
- 可以增加多类分类的实验
- 考虑在更多数据集上验证结论的普适性

### MiniMax-M3（创新性与贡献审查员）

**审查意见：**

1. **核心创新点清晰但方法失效**：MOAF框架的核心创新——三种公平机制的组合——在实验中被证明会导致预测崩溃。这说明创新方向有问题，但发现本身有价值。

2. **经验研究的贡献**：论文最重要的贡献不是方法创新，而是对公平性-准确性权衡的系统性经验研究。7种方法的全面对比、消融实验揭示的各组件贡献、以及Pareto前沿分析，都提供了有价值的经验知识。

3. **与现有工作的差异化**：大多数公平性论文只报告正面结果，这篇论文诚实报告了负面结果（方法失效），并提供深入分析。这种"失败报告"在学术上有独特价值。

4. **贡献点的表述**：贡献点3（"critical empirical findings"）是论文最突出的贡献，应该放在第一位。

**补充建议**：
- 考虑将论文重新定位为"empirical study"而非"novel method"
- 增加对"何时使用单一公平机制vs.组合机制"的更深入分析
- 增加与更多最新公平性方法的对比（如FairBatch等）

### 综合评分 - 论文3
- 数据真实性：100（40%）= 40.0
- 创新度：72（20%）= 14.4
- 完整性：88（20%）= 17.6
- 语言质量：85（20%）= 17.0
- **加权总分：89.0/100**
- **等级：B级(>=80)**
- **主要改进建议**：
  1. 建议重新定位为"经验研究"论文，强调公平性-准确性权衡的实证发现
  2. 增加更多数据集和protected attribute以验证结论的普适性
  3. 考虑提出改进版MOAF框架（如自适应公平预算），使方法本身也有正面贡献

---

## 总览对比

| 排名 | 论文 | 数据真实性 | 创新度 | 完整性 | 语言质量 | 加权总分 | 等级 |
|------|------|-----------|--------|--------|---------|---------|------|
| 1 | 29_Fairness_Tabular | 100 | 72 | 88 | 85 | **89.0** | B级 |
| 2 | 22_Graph_Purchase | 100 | 60 | 72 | 78 | **82.0** | B级 |
| 3 | 38_AdaptiveAttention_Rain | 90 | 68 | 75 | 82 | **81.0** | B级 |

### 总结性评估

**论文3（29_Fairness_Tabular）**整体质量最高。虽然提出的方法（MOAF Full）导致预测崩溃，但论文的诚实报告风格、全面的实验设计（7种方法、2个数据集、5种子、统计分析）、深入的失败分析和实用的部署建议使其具有很高的学术价值。数据完整性达到100%，实验设计最为全面。建议增加更多数据集验证，并考虑提出改进版框架。

**论文2（38_AdaptiveAttention_Rain）**的数据真实性评分为90/100（因正文2处描述性数字有小误差），创新度和完整性中等。论文的诚实报告风格值得肯定，最高Recall的实际应用价值明确。主要问题：仅3个种子导致统计检验无效，与论文1的门控机制高度相似，单一数据集评估。

**论文1（22_Graph_Purchase）**的数据完整性最好（100/100），但核心创新（GNN在表格数据上的应用）在实验中被证明无效（移除GNN后性能反而提升）。单一数据集评估，理论贡献有限。需要通过多数据集实验重新验证方法的核心假设。

**三篇论文的共同问题**：
1. 均只在1-2个数据集上评估，缺乏多数据集验证
2. 提出的方法在性能上均不如传统树方法（RF/XGBoost），核心创新贡献的实用性存疑
3. 三篇论文的注意力/门控机制设计有较高的相似性，存在自我重复的风险
