# 论文投稿管理总览

> **文件定位**: 本文件为 `D:\ResearchPaperPrepare` 工作区唯一顶层管理文件。
> **最后更新**: 26-08-08 (No.66投稿材料按KBS期刊准备完成：Cover Letter+Highlights+6项KBS必需声明，备选期刊记录到看板第十节)
> **管理规则**:
> - 负面结果暂缓投稿，优先正面结果；尽量改进或换数据集
> - 新方向研发为主线，负面结果改进和超长实验为后台低优先级
> - 时间格式: YY-MM-DD HH:MM
> - 候选新方向需经用户确认后方可执行
> - 仅接受SCI/SSCI/EI期刊，ESCI不算
> - **GitHub规则**: 仓库名不带数字前缀；仅上传源代码+数据+README；上传前消除AI痕迹；README仅用于复现说明

---

## 总览表

### 阶段B — 代码已上传GitHub（19篇）

| 编号 | 方向             | 结果 | 评分 | 核心亮点                       | 目标期刊             | GitHub                                                                                              |
| ---- | ---------------- | ---- | ---- | ------------------------------ | -------------------- | --------------------------------------------------------------------------------------------------- |
| 17   | 证据降雨预测     | 正面 | 90.6 | Theorem 3诊断判据; 数据真实性100 | Applied Intelligence (SCI Q3) | [Evidence_Rainfall](https://github.com/mingyi0818/Evidence_Rainfall)                      |
| [25](#no25--少样本杂草分类)   | 少样本杂草分类   | 正面 | 78.4 | 5-way 1-shot 55.59% (p<0.0001) | CEA (SCI Q1)         | [FewShot_Weed](https://github.com/mingyi0818/FewShot_Weed)                                          |
| [07](#no07--表格少样本)   | 表格少样本       | 正面 | 73.4 | Telco 1-shot 74.82% (↑19.2%)   | PR (SCI Q1)          | [Tabular_FewShot](https://github.com/mingyi0818/Tabular_FewShot)                                    |
| [16](#no16--酒店情感主题)   | 酒店情感主题     | 正面 | 69.0 | Acc=0.9839/F1=0.9737           | TMP (SSCI Q1)        | [Hotel_Sentiment_Topic](https://github.com/mingyi0818/Hotel_Sentiment_Topic)                        |
| [14](#no14--表格异常检测)   | 表格异常检测     | 正面 | 64.2 | NSL-KDD AUC=0.9808             | ESWA (SCI Q1)        | [Tabular_Anomaly](https://github.com/mingyi0818/Tabular_Anomaly)                                    |
| JX01 | 教学研究预警     | 混合 | 93.0 | AUC=0.723                      | CAE (SCIE Q2, APC免费) | [ohtp-mm-public](https://github.com/mingyi0818/ohtp-mm-public)                                      |
| [JX02](#nojx02--教学研究kc) | 教学研究KC       | 混合 | 92.4 | SVM Acc=0.6269                 | CAEE (SCIE+EI Q2)    | [masf-public](https://github.com/mingyi0818/masf-public)                                            |
| [15](#no15--酒店取消)   | 酒店取消         | 混合 | 91.0 | AUC=0.9436                     | JHTT (SSCI)          | [Hotel_Cancellation](https://github.com/mingyi0818/Hotel_Cancellation)                              |
| [43](#no43--旅游poi推荐)   | 旅游POI推荐      | 混合 | 90.2 | Recall@10=0.0277               | JHTT (SSCI Q1)       | [Tourism_Recommend](https://github.com/mingyi0818/Tourism_Recommend)                                |
| [38](#no38--自适应注意力降雨)   | 自适应注意力降雨 | 混合 | 88.6 | Recall=0.8231                  | ASC (SCI Q1)         | [AdaptiveAttention_Rain](https://github.com/mingyi0818/AdaptiveAttention_Rain)                      |
| [27](#no27--自适应入侵检测)   | 自适应入侵检测   | 混合 | 87.0 | F1-Macro=0.7706                | SCN (SCI)            | [AdaptiveIntrusion](https://github.com/mingyi0818/AdaptiveIntrusion)                                |
| [12](#no12--学生辍学预测)   | 学生辍学预测     | 混合 | 86.4 | weeks 8-16优于LR/LSTM          | IEEE Access (SCI Q3) | [Student_Dropout](https://github.com/mingyi0818/Student_Dropout)                                    |
| [41](#no41--碳排放预测)   | 碳排放预测       | 混合 | 86.2 | CrossAttn贡献+46.9%MAE         | EMS/APEN (SCI Q1)    | [Carbon_Emission](https://github.com/mingyi0818/Carbon_Emission)                                    |
| [42](#no42--时序概率预测)   | 时序概率预测     | 正面 | 86.0 | SG-DER ETTm2 R²=0.817          | ASC (SCI Q1)         | [Probabilistic_TS](https://github.com/mingyi0818/Probabilistic_TS)                                  |
| [20](#no20--自监督欺诈检测)   | 自监督欺诈检测   | 混合 | 83.0 | AUC-ROC=0.9717                 | ASC (SCI Q1)         | [Fraud_SelfSupervised](https://github.com/mingyi0818/Fraud_SelfSupervised)                          |
| [24](#no24--特征交互房价预测)   | 特征交互房价     | 混合 | 80.0 | R²=0.835                       | APIN (SCI)           | [FeatureInteraction_House](https://github.com/mingyi0818/FeatureInteraction_House)                  |
| [39](#no39--钓鱼url检测)   | 钓鱼URL检测      | 混合 | 79.0 | MVTT F1=0.962<RF 0.970         | SCN (SCI Q2)         | [Phishing_URL](https://github.com/mingyi0818/Phishing_URL)                                          |
| [40](#no40--表格特征选择)   | 表格特征选择     | 混合 | 78.2 | K=50 Acc=0.9720                | APIN/SCN (SCI Q2/Q3) | [Tabular_FeatureSelection](https://github.com/mingyi0818/Tabular_FeatureSelection)                  |
| [03](#no03--不平衡学习)   | 不平衡学习       | 混合 | 73.8 | Credit Fraud F1=0.8865         | KBS (SCI Q1)         | [Imbalanced_Learning](https://github.com/mingyi0818/Imbalanced_Learning)                            |

> **提示**: 点击编号列的链接可跳转至该方向的详细条目及理论水平提升分析。

---

## 投稿决策仪表盘

### 阶段B详细条目（19篇）

#### No.14 — 表格异常检测
- **结果性质**: 正面
- **目标期刊**: ESWA (SCI Q1, IF≈8.5)
- **关键指标**: NSL-KDD AUC=0.9808, UNSW AUC=0.8724
- **论文路径**: `14_Tabular_Anomaly/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Tabular_Anomaly
- **质量评估**: Tier 1.5 (64.2 D级, 数据溯源问题+基线不足)
- **最后更新**: 26-07-23
- **下一步行动**: 改进数据溯源和基线数量

#### No.25 — 少样本杂草分类
- **结果性质**: 正面
- **目标期刊**: CEA (SCI Q1, IF≈8.3)
- **关键指标**: 5-way 1-shot 55.59% (p<0.0001, d>11)
- **论文路径**: `25_FewShot_Weed/paper/paper_draft.md`
- **数据集路径**: `D:\datasets\CropAndWeed\CropAndWeed_cropped\` (CropAndWeed裁剪数据，26-07-24从工作区迁出)
- **GitHub仓库**: https://github.com/mingyi0818/FewShot_Weed
- **质量评估**: Tier 1.5 (78.4 D级, 数据溯源+创新度不足)
- **最后更新**: 26-07-24
- **下一步行动**: 改进数据溯源和创新度

#### No.07 — 表格少样本
- **结果性质**: 正面
- **目标期刊**: PR (SCI Q1, IF≈7.6)
- **关键指标**: Telco 1-shot 74.82% (↑19.2%, p<0.001)
- **论文路径**: `07_Tabular_FewShot/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Tabular_FewShot
- **质量评估**: Tier 1.5 (73.4 C级, 数据溯源问题)
- **最后更新**: 26-07-23
- **下一步行动**: 改进数据溯源

#### No.16 — 酒店情感主题
- **结果性质**: 正面
- **目标期刊**: TMP (SSCI Q1, IF≈7.3)
- **关键指标**: Acc=0.9839/F1=0.9737; LDA Cv=0.2475
- **论文路径**: `16_Hotel_Sentiment_Topic/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Hotel_Sentiment_Topic
- **质量评估**: Tier 2 (69.0 D级, 摘要混淆两个数据集，核心方法失败)
- **最后更新**: 26-07-23
- **下一步行动**: 修复摘要中两个数据集结果混淆问题，重新评估核心方法

#### No.17 — 证据降雨预测
- **结果性质**: 正面（诊断工具定位，诚实报告负面结果）
- **目标期刊**: Applied Intelligence (SCI Q3, IF≈3.5, 订阅模式免版面费)
- **关键指标**: 数据真实性100/100; Theorem 3诊断判据H_E∝1/S; C4共形覆盖0.9499
- **论文路径**: `17_Evidence_Rainfall/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Evidence_Rainfall
- **阶段**: B（代码已上传GitHub，含verify_results.py验证脚本）
- **质量评估**: 四项评分达标（数据真实性100/创新度82/完整性85/语言质量86）
- **最后更新**: 26-07-26
- **下一步行动**: LaTeX格式转换+图片EPS/TIFF转换+DOI验证+代码注释人工化

#### No.03 — 不平衡学习
- **结果性质**: 混合
- **目标期刊**: KBS (SCI Q1, IF≈7.6)
- **关键指标**: Credit Fraud F1=0.8865
- **论文路径**: `03_Imbalanced_Learning/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Imbalanced_Learning
- **质量评估**: Tier 1.5 (73.8 C级, 数据溯源严重问题)
- **最后更新**: 26-07-23
- **下一步行动**: 修复数据溯源严重问题

#### No.38 — 自适应注意力降雨
- **结果性质**: 混合
- **目标期刊**: ASC (SCI Q1, IF≈7.2)
- **关键指标**: AUC=0.8965/Recall=0.8231
- **论文路径**: `38_AdaptiveAttention_Rain/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/AdaptiveAttention_Rain
- **质量评估**: Tier 2 (~88.6 B+级, 已改进3处数字修正+统计检验)
- **最后更新**: 26-07-23
- **下一步行动**: 接近就绪，可进入投稿准备

#### No.JX01 — 教学研究预警
- **结果性质**: 混合
- **目标期刊**: CAE (SCIE Q2, IF≈2.7, 订阅免费)
- **关键指标**: AUC=0.723, DeLong z=8.63, CodeBench三学期验证
- **论文路径**: `JX01_Teaching_Research_1/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/ohtp-mm-public
- **质量评估**: Tier 1 (93.0 B+级, 已改进补p值+多种子+弹性系数+Highlights+DeLong+CodeBench稳定性)
- **最后更新**: 26-07-26 15:00
- **下一步行动**: 统一摘要DeLong值后投稿CAE（降级至Q2提升命中率，APC免费）

#### No.JX02 — 教学研究KC
- **结果性质**: 混合
- **目标期刊**: CAEE (SCIE+EI Q2, IF≈2.75)
- **关键指标**: SVM Acc=0.6269; LOPO-CV均值0.4739
- **论文路径**: `JX02_Teaching_Research_2/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/masf-public
- **质量评估**: Tier 1 (92.4 A+级, 已改进作者信息+数据修正+文献重编号)
- **最后更新**: 26-07-23 14:15
- **下一步行动**: 接近就绪，可进入投稿准备

#### No.15 — 酒店取消
- **结果性质**: 混合
- **目标期刊**: JHTT (SSCI)
- **关键指标**: AUC=0.9436; ADR R²=0.8566
- **论文路径**: `15_Hotel_Cancellation/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Hotel_Cancellation
- **质量评估**: Tier 2 (~91 B+级, 已改进格式修复+定理强化+架构图)
- **最后更新**: 26-07-23
- **下一步行动**: 接近就绪，可进入投稿准备

#### No.20 — 自监督欺诈检测
- **结果性质**: 混合
- **目标期刊**: ASC (SCI Q1, IF≈7.2)
- **关键指标**: AUC-ROC=0.9717
- **论文路径**: `20_Fraud_SelfSupervised/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Fraud_SelfSupervised
- **质量评估**: Tier 2 (~83 B级, 已改进弹性系数矛盾修复+5种子统计重建)
- **最后更新**: 26-07-23
- **下一步行动**: 接近就绪，可进入投稿准备

#### No.24 — 特征交互房价预测
- **结果性质**: 负面结果（方法重新设计后TIGN未超越基线）
- **目标期刊**: APIN (SCI, IF≈3.4)
- **关键指标**: TIGN R²=0.902 vs XGBoost R²=0.904 vs LightGBM R²=0.904；GNN组件零效果
- **论文路径**: `24_FeatureInteraction_House/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/FeatureInteraction_House
- **质量评估**: 暂缓（诚实负面结果，GNN组件弹性系数=0.0，统计检验不显著）
- **最后更新**: 26-07-23
- **下一步行动**: 暂缓投稿，TIGN方法未超越XGBoost/LightGBM基线

#### No.27 — 自适应入侵检测
- **结果性质**: 混合
- **目标期刊**: SCN (SCI, IF≈1.9)
- **关键指标**: F1-Macro=0.7706
- **论文路径**: `27_AdaptiveIntrusion/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/AdaptiveIntrusion
- **质量评估**: Tier 3 (~87 B+级, 已改进数据真实性80→100，SVM种子诚实报告)
- **最后更新**: 26-07-23
- **下一步行动**: 接近就绪，可进入投稿准备

#### No.12 — 学生辍学预测
- **结果性质**: 混合
- **目标期刊**: IEEE Access (SCI Q3, IF≈3.4)
- **关键指标**: weeks 8-16优于LR/LSTM
- **论文路径**: `12_Student_Dropout/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Student_Dropout
- **质量评估**: Tier 1 (86.4 B级, 已改进作者信息+基金项目+文献补引用)
- **最后更新**: 26-07-23
- **下一步行动**: 接近就绪，可进入投稿准备

#### No.39 — 钓鱼URL检测
- **结果性质**: 混合（方法重设计后仍不如基线）
- **目标期刊**: SCN (SCI Q2)
- **关键指标**: MVTT F1=0.9618/AUC=0.9934（优于LR，显著劣于RF/LGB/XGB的~0.970）
- **论文路径**: `39_Phishing_URL/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Phishing_URL
- **质量评估**: Tier 2 (79.0 B级) — MVTT方法创新但无法超越树模型集成方法，诚实报告负面结果
- **最后更新**: 26-07-23
- **下一步行动**: 改进论文草稿（如实报告MVTT vs 树模型差距），投SCN或暂缓

#### No.40 — 表格特征选择
- **结果性质**: 混合
- **目标期刊**: APIN/SCN (SCI Q2/Q3)
- **关键指标**: K=50 Acc=0.9720（Multiple Features接近最优；Arcene不如RFE-RF）
- **论文路径**: `40_Tabular_FeatureSelection/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Tabular_FeatureSelection
- **质量评估**: Tier 2 (78.2 B-级) — MIB思路有新意(72/100)，但效果未超传统方法，消融设计有缺陷
- **最后更新**: 26-07-23
- **下一步行动**: 暂不建议投稿；建议改进方法或换用更大规模高维数据集后重新评估

#### No.41 — 碳排放预测
- **结果性质**: 混合
- **目标期刊**: EMS/APEN (SCI Q1/Q2)
- **关键指标**: MAE=0.0746±0.0062, R²=0.789±0.047（显著差于N-BEATS；CrossAttn贡献+46.9%MAE）
- **论文路径**: `41_Carbon_Emission/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Carbon_Emission
- **质量评估**: Tier 1 (86.2 B+级) — 应用价值高(85/100)，数据真实性100分，方法创新一般(73/100)
- **最后更新**: 26-07-23
- **下一步行动**: 可投APEN（SCI Q2）更稳妥；EMS需要更强的方法创新
- **下一步行动**: 需进行质量评估

#### No.42 — 时序概率预测
- **结果性质**: 正面（SG-DER-TSF方法重设计成功）
- **目标期刊**: ASC (SCI Q1)
- **关键指标**: SG-DER-TSF ETTm2 R²=0.817, ETTm1 R²=0.634, ETTh2 R²=0.546（均大幅优于原DE-TSF）
- **论文路径**: `42_Probabilistic_TS/paper/paper_draft.md`
- **GitHub仓库**: https://github.com/mingyi0818/Probabilistic_TS
- **质量评估**: Tier 1.5 (86.0 C级) — Stop-Gradient创新有效解决DER损害点预测问题，Data-Verifier 100分通过
- **最后更新**: 26-07-23
- **下一步行动**: 论文草稿已完成，准备投稿

#### No.43 — 旅游POI推荐
- **结果性质**: 混合
- **目标期刊**: JHTT (SSCI Q1)
- **关键指标**: Recall@10=0.0277/NDCG@10=0.0378（均值最优，与NGCF无显著差异p=0.742）
- **论文路径**: `43_Tourism_Recommend/paper/paper_draft.md`
- **数据集路径**: `D:\datasets\tourism\Yelp_POI_Recommend\` (Yelp原始数据，26-07-24从工作区迁出)
- **GitHub仓库**: https://github.com/mingyi0818/Tourism_Recommend
- **质量评估**: Tier 1 (90.2 A-级) — 方法创新强(83/100)，理论深度最高(88/100)，数据真实性100分
- **最后更新**: 26-07-24
- **下一步行动**: 可直接投JHTT（SSCI Q1），三图框架+3个Theorem支撑充分

### 理论水平提升分析（17个方向）

> 以下分析基于各方向论文草稿精读 + 2025-2026年最新研究联网检索，针对每个方向提出具体可执行的理论提升建议。所有引用论文均真实可查。

#### 方向25 — 少样本杂草分类 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 1定理+3命题，Theorem 1实质为Snell(2017)的复述非原创；Proposition 3"有利初始化"无严格证明；缺泛化界、收敛性、Neural Collapse分析；理论与实验脱节。

**最新研究对标(2025-2026)**:
1. Luthra等 "Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning" NeurIPS 2025 — SupCon损失的少样本误差界
2. Zhou等 "UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning" CVPR 2025 — EM收敛性分析
3. Nguyen等 "Provably Improving Generalization of Few-shot models with Synthetic Data" ICML 2025 — 分布差异泛化界
4. Kim等 "Model Merging is Secretly Certifiable" arXiv 2025 — PAC-Bayes低样本泛化界

**理论提升建议**:
- **高优先级**: 新增Theorem 2(少样本泛化界，基于Rademacher/PAC-Bayes，量化K=1 vs K=5差异)；新增Theorem 3(对比预训练几何结构定理，借鉴Luthra 2025的ETF结构)
- **中优先级**: Proposition 4(ECA置换等变性)；Proposition 5(两阶段收敛速率)；新增"特征空间几何分析"节
- **低优先级**: Theorem 1扩展到GMM

**预期效果**: 1定理3命题(0原创)→3定理5命题(2原创)，Methodology占比30%→≥40%，创新度75→85+，完整性78→85+。

---

#### 方向07 — 表格少样本 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 5定理+5命题但质量严重缺陷——Theorem 5收敛性假设根本性错误(凸假设用于非凸神经网络)；Proposition 2遗忘率界推导方向错误；Theorem 2互信息过于空泛；Theorem 4非真正定理。

**最新研究对标(2025-2026)**:
1. Suleymanov等 "SPRINT: Semi-supervised Prototypical Representation for FSCIL Tabular" arXiv 2026 — 表格FSCIL直接竞品
2. Shi等 "Latte: Transfering LLMs' Latent-level Knowledge" IJCAI 2025 — LLM增强少样本表格
3. Kang等 "TaRL/mTaRL" WWW 2026 — 原型元学习理论框架
4. Chen等 "A Closer Look at Training Strategy for Modern Meta-Learning" NeurIPS 2020(2025仍被引) — S/Q训练O(1/√n)泛化界
5. Ouyang等 "Projection Head is Secretly an Information Bottleneck" ICLR 2025 — 信息瓶颈理论

**理论提升建议**:
- **高优先级**: 修正Theorem 5(非凸SGD收敛O(1/√T))；修正Proposition 2推导方向；Theorem 2升级为定量互信息下界；删除/重写Theorem 4
- **中优先级**: 新增PAC-Bayes泛化界；新增任务相似度自适应界；补充信息瓶颈分析章节
- **低优先级**: 数据缩放定律(Fukuchi 2026)

**预期效果**: 修复致命数学错误避免审稿人一眼识破；理论占比25%→35%+；完整性+创新度各提升10-15分。

---

#### 方向16 — 酒店情感主题 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 0定理+3命题(均平庸)；Proposition 3证明有数学瑕疵；**摘要数据混淆问题(0.9839/0.9737实为Datafiniti上TSTE-DM而非TSTE-VI，学术诚信红线)**；核心Finding 1(trade-off)无理论解释；posterior collapse无理论分析。

**最新研究对标(2025-2026)**:
1. Bohara & Esposito "Geometric Convergence Analysis of VI via Bregman Divergences" arXiv 2025 — ELBO收敛性
2. Sun等 "Natural Gradient VI: Guarantees for Non-Conjugate Models" NeurIPS 2025 — 非共轭变分推断
3. "Posterior Collapse as a Phase Transition in VAEs" arXiv 2025 — 相变理论
4. Zhang & Zhang "Historical Consensus" arXiv 2026 — 崩溃解排除定理
5. Chen等 "Learning Topic Models: Identifiability" JASA 2023 — 主题可辨识性

**理论提升建议**:
- **高优先级**: 修正摘要数据混淆(诚信红线)；新增Theorem 2(后验崩溃相变定理，解释Table 3)；新增Theorem 3(Topic-Sentiment互信息trade-off定理，支撑Finding 1)；修正/删除Proposition 3
- **中优先级**: Theorem 1(ELBO收敛性)；Proposition 4(主题可辨识性)；新增2.6理论分析节
- **低优先级**: Bregman几何视角；sentiment embedding表达能力定理

**预期效果**: 0定理→3定理+2命题；trade-off定理为核心发现提供数学根基；创新度75→85+，完整性70→82+。

---

#### 方向14 — 表格异常检测 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 0定理+2命题；Proposition 2证明有逻辑漏洞("≈"模糊表述)；核心发现"重建误差优于潜空间距离"无形式化界；对比学习理论缺失；基线仅5个且均为2018年前，0个近3年SOTA。

**最新研究对标(2025-2026)**:
1. Chen & Liu "Two-Layer ConvAE Provably Detect Unseen Anomalies" ICLR 2026 — 锥集理论
2. Hirth等 "Denoising without Diffusion" ICML 2026 — 固定噪声去噪器
3. Li等 "OFA-TAD: One-for-All Anomaly Detection for Tabular" ICML 2026 — 通用表格异常检测
4. Li等 "CLIMB: Contrastive Learning via Variational Information Bottleneck" TPAMI 2025 — 对比学习信息瓶颈
5. SpiCAE IEEE IEMCON 2025 — NSL-KDD同任务SOTA(F1=0.97)

**理论提升建议**:
- **高优先级**: 重写Proposition 2(基于Rademacher复杂度)；新增Theorem 1(重建误差可分性界，借鉴ICLR 2026锥集理论)；补充≥3个2024-2026基线(SpiCAE/ICL/GraphACGAN)；新增Proposition 3(评分函数单调性，解释α高弹性)
- **中优先级**: Theorem 2(对比正则化泛化界)；增加第3个数据集
- **低优先级**: 对比损失λ闭式解；互信息估计实验

**预期效果**: 0定理→2定理+3命题；Methodology 1.5页→3-4页；基线不足问题解决；数据真实性风险消除。

---

#### 方向JX02 — 教学研究KC [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 0定理+4命题(均"可检验声明"非严格证明)；构造循环性核心创新仅文字描述无形式化；缺泛化误差界；理论与实验呼应不严密。

**最新研究对标(2025-2026)**:
1. Freiesleben & Zezulka "Benchmarking Epistemology: Construct Validity" arXiv 2025 — 构造效度四步框架
2. Hafner等 "Measuring what Matters: Construct Validity in LLM Benchmarks" NeurIPS 2025 — 基准失效模式分类
3. Zhan等 "TSDR: Temporal Smoothness Doubly Robust Learning for KT" arXiv 2026 — Rademacher泛化界
4. Cheng等 "Uncertainty-aware Knowledge Tracing" AAAI 2025 — 不确定性量化
5. Shimada & Okada "Reliability Coefficient for BKT" TKL 2025 — 可靠性分析

**理论提升建议**:
- **高优先级**: 新增Theorem 1(构造循环性形式化——种子恢复精度上界)；新增Theorem 2(泛化误差界，解释LOPO-CV下降)；P3升级为Theorem 3(字符级特征互信息证明)
- **中优先级**: Proposition 5(反馈有效性信息论下界)；可辨识性分析小节
- **低优先级**: 不确定性量化；可靠性系数

**预期效果**: 0定理→3定理；核心创新从"发现循环性"升级为"形式化循环性并给出泛化界"；理论占比1/4→1/3+。

---

#### 方向15 — 酒店取消 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 1定理+2命题；Proposition 2近平凡(平方和最小)；**Theorem 1在T=2下退化为空(Remark 1自爆)**；缺泛化界、收敛性、负迁移形式化；AGMTL不如XGBoost无理论解释。

**最新研究对标(2025-2026)**:
1. Shao & Wu "Sharper Risk Bound for MTL with Multi-Graph" arXiv 2025 — O(√(log T/n))风险界
2. Zakerinia等 "Low Intrinsic Dimensionality to Non-Vacuous Generalization in MTL" ICLR 2026 — PAC-Bayes MTL界
3. Tu等 "MTIF: Measuring Fine-Grained Relatedness in MTL" TMLR 2026 — 跨任务影响函数
4. Jeong & Yoon "Selective Task Group Updates" ICLR 2025 — proximal inter-task affinity
5. Katz "Coupled Supply and Demand Forecasting" arXiv 2026 — 住宿市场供需耦合

**理论提升建议**:
- **高优先级**: 新增Theorem 2(MTL泛化界，基于Shao 2025)；修正Theorem 1退化(推广到T≥3或改为对齐误差界)；新增Proposition 3(负迁移量化界，呼应λ敏感性)
- **中优先级**: Proposition 4(图正则收敛性O(1/√t))；理论与实验对照小节
- **低优先级**: 篇幅调整

**预期效果**: 1定理(退化)+2命题→2定理+4命题；修复Theorem 1退化漏洞避免直接拒稿；"不如XGBoost"转化为"MTL联合泛化优势"理论论证。

---

#### 方向43 — 旅游POI推荐 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 3定理+1命题但缺乏架构特异性——Theorem 1(置换不变性)为GNN标准性质；Theorem 2(表达力优于LightGCN)仅"特例归约"论证过浅；Theorem 3(O(1/T)收敛)为通用结论非原创；**理论声称优于但实验对NGCF p=0.742不显著，矛盾未解释**。

**最新研究对标(2025-2026)**:
1. Yang等 "Your Graph Recommenders are Provably Doing Graph Contrastive Learning" KDD 2025 — BPR-GCL等价性
2. He等 "Collaborative Filtering Meets Spectrum Shift" KDD 2025 — 谱偏移理论(直接解释p=0.742)
3. Qin等 "GeoMamba: Multi-granular POI Recommendation" AAAI 2025 — 地理SSM理论
4. Zhang等 "LightCCF: Contrastive Learning's Capability of Neighborhood Aggregation" arXiv 2025 — InfoNCE-图卷积等价
5. He等 "Revisiting LightGCN: Unexpected Inflexibility" TOIS 2025 — LightGCN理论缺陷

**理论提升建议**:
- **高优先级**: 新增Theorem 4(谱偏移分析，直接解释p=0.742)；新增Proposition 2(Time2Vec失效理论解释)；Theorem 2对比基准扩展到NGCF+修正LightGCN
- **中优先级**: 泛化界(Rademacher)；谱滤波视角小节；cross-graph attention理论保证
- **低优先级**: BPR-GCL等价性定理；WL测试表达力

**预期效果**: 3通用定理→架构特异性定理+谱分析+泛化界+负结果理论解释；化解p=0.742致命短板；创新度70→85+。

---

#### 方向38 — 自适应注意力降雨 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: Definition 1+Proposition 1-3+Theorem 1-2+Corollary 1-2；Theorem 2(S=1退化)证明较严谨；但Theorem 1(通用逼近)仅证明草图；Corollary 2(死权重)无梯度范数界且论文自认C5未验证；无信息论分析、无泛化界。

**最新研究对标(2025-2026)**:
1. Liang等 "Why Attention Fails: Degeneration of Transformers into MLPs in TS" arXiv 2025 — 注意力退化谱系
2. "LimiX-2M: Mitigating Low-Rank Collapse in Tabular Foundation Models" arXiv 2026 — 值敏感坍缩证明
3. Sanyal等 "When Attention Collapses" TMLR 2026 — lazy layers/rank-1退化
4. Giorlandino & Goldt "Two failure modes of deep transformers" ICLR 2026 — 信号传播统一理论
5. Harris & Chen "SaTformer: Space-Time Transformer for Precipitation" NeurIPS 2025 Workshop — 降雨SOTA

**理论提升建议**:
- **高优先级**: 补全Theorem 1严格证明(引用Cybenko UAT+Stone-Weierstrass)；新增Proposition 4(梯度范数界O(1/d_m))并补做C5验证；新增"退化谱系统一表述"小节
- **中优先级**: Theorem 3(表达能力鸿沟S=1 vs S=d)；Proposition 5(有效秩下界)
- **低优先级**: Rademacher泛化界；优化景观分析

**预期效果**: 从"单一案例诊断报告"升级为"注意力退化理论的表格数据特例研究"；嵌入2025-2026 rank collapse理论谱系；创新度75→85+。

---

#### 方向27 — 自适应入侵检测 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 0定理+2命题；Proposition 1(FSG梯度可流)仅链式法则；Proposition 2(复杂度)工程推导；**消融显示移除FSG反而F1提升(0.7720 vs 0.7659)与Proposition 1矛盾**；GNN不如RF无理论解释。

**最新研究对标(2025-2026)**:
1. El Mahdaouy等 "Deep Learning for Contextualized NetFlow-Based NIDS" arXiv 2026 — 图建模在IDS中的理论基础
2. Keriven "Backward Oversmoothing: Why Hard to Train Deep GNNs" arXiv 2025 — 反向传播过平滑
3. Wu等 "Demystifying Oversmoothing in Attention-Based GNNs" NeurIPS 2024-2025 — GAT过平滑指数速率
4. Chen等 "Expressive Power of Subgraph GNNs" ICML 2025 — k-hop子图逼近定理
5. Wang等 "FairFS: Deep Feature Selection Biases" WWW 2026 — 层偏差理论

**理论提升建议**:
- **高优先级**: Theorem 2(过平滑速率O(ρ^l)，解释为何2层即饱和)；Proposition 3(FSG失效理论解释，基于FairFS层偏差)；Proposition 4(kNN图vs决策树本质差异，拉普拉斯谱角度)
- **中优先级**: Theorem 1(GGNN表达能力上界，1-WL框架)
- **低优先级**: 复杂度分析扩展(通信/显存)

**预期效果**: 0定理→2定理+4命题；"GNN不如RF"从经验讨论升级为定理化结论；创新度70→85+，完整性75→85+。

---

#### 方向12 — 学生辍学预测 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 0定理+5命题；仅Proposition 1有完整证明，Proposition 2为Proof Sketch，P3-5仅为Justification；4.2.1节"为何树方法优于深度学习"五视角分析(最具理论深度)被错放Discussion而非Methodology；缺泛化界、因果推断、收敛性。

**最新研究对标(2025-2026)**:
1. Mihoubi等 "Transformative AI Framework for Student Dropout" arXiv 2025 — RAG+跨模态融合
2. Jiang & Peng "Weekly Prediction with TFT" AICSS 2025 — EWG/IEI/RSS教育指标
3. Buñay-Guisñan等 "Group Counterfactual Explanations" Electronics 2026 — 反事实解释(OULAD)
4. Susnjak等 "Doubly Robust Evaluation of AI-Guided Student Support" arXiv 2025 — 因果推断
5. da Silva等 "Auditable Policy-Simulation for Dropout" arXiv 2026 — 生存分析框架
6. Hollmann等 "TabPFN: tabular foundation model" Nature 2025 — 挑战树模型统治

**理论提升建议**:
- **高优先级**: Proposition 2提升为Theorem 1(跨模态注意力表达能力，构造性证明)；新增Theorem 2(Rademacher泛化界，解释树模型优势)；新增Theorem 3(时间风险单调性，信息论证明)；4.2.1五视角分析迁移至Methodology
- **中优先级**: Proposition 6(干预因果效应界)；完善P3-5为正式证明；时间风险建模
- **低优先级**: 活动理论；反事实解释；TabPFN对比

**预期效果**: 0定理→3定理+2命题；"MSFN不如树模型"从负结果转化为"深度学习表格泛化瓶颈"理论洞察；创新度75→82+，完整性70→85+。

---

#### 方向41 — 碳排放预测 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 0定理+2命题；Proposition 1(表达能力)证明有漏洞(W_Q=W_K=0退化为均匀分布≠任意线性组合)；Proposition 2(复杂度)基础大O；缺收敛性、direction-consistency loss梯度理论、趋势季节可分性保证；**消融显示移除DirLoss MAE反而下降(0.0746→0.0660)无理论解释**；不如N-BEATS无归纳偏置对比。

**最新研究对标(2025-2026)**:
1. Lee等 "CosDir: Scale-Invariant Cosine Direction Loss" arXiv 2026 — MSE小波动梯度消失证明
2. Chen等 "PMLF: Physics-guided Multi-scale Loss" NeurIPS 2025 — 结构异质时序损失理论
3. Li等 "FTimeXer" arXiv 2026 — FFT频域分支+门控时频融合
4. TwinsFormer ICLR 2025 — 趋势-季节双流相加守恒证明
5. Wu等 "Transformers as TS Foundation Models" arXiv 2025 — Dobrushin条件approximation+generalization界

**理论提升建议**:
- **高优先级**: 新增Theorem 1(CrossAttn非线性近似能力，基于UAT)；新增Theorem 2(DirLoss梯度病态分析，引用CosDir理论解释消融反常)
- **中优先级**: Proposition 3(分解可分性，借鉴TwinsFormer守恒)；Proposition 4(N-BEATS归纳偏置对比)
- **低优先级**: 频域分析；物理引导损失

**预期效果**: 0定理→2定理+2命题；CrossAttn +46.9%贡献获理论支撑；DirLoss失效从"实验失败"转化为"理论发现"；完整性75→85+，创新度→82+。

---

#### 方向42 — 时序概率预测 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 0定理+2命题；Proposition 1(梯度解耦)近同义反复(stop-gradient定义本身)；Proposition 2(复杂度)"lightweight encoder"表述含糊；**核心创新"梯度衰减"(式7)未形式化为定理**；不确定性校准理论空白；缺泛化界与收敛性。

**最新研究对标(2025-2026)**:
1. Chinta等 "ProbFM: Probabilistic TS Foundation Model" arXiv 2026 — DER+NIG基础模型
2. Li等 "PPM: Parametric Prior Mapping" ICML 2026 — 一致性/表达能力/梯度稳定化三定理范式
3. Ponce等 "Dual Perspectives on Non-Contrastive SSL" ICLR 2026 — stop-gradient动力学系统严格证明
4. Huang等 "DEMR: Adaptive Evidential Learning" arXiv 2025 — 独立发现DER梯度病态
5. Hu等 "COP: Conformal Optimistic Prediction" ICLR 2026 — 分布无关有限样本覆盖保证
6. Brigato等 "No Champions in Supervised Long-Term TS Forecasting" TMLR 2026 — ETT基准统计显著性质疑

**理论提升建议**:
- **高优先级**: 式7梯度衰减形式化为Theorem 1(本文最核心原创理论)；新增Theorem 2(SG-DER平衡点稳定性，借鉴ICLR 2026动力学方法)；清理Proposition 2"lightweight encoder"含糊表述
- **中优先级**: 新增Section 2.7理论性质汇总(参照PPM结构)；Theorem 3(conformal覆盖有限样本界)；实验验证梯度衰减理论预测
- **低优先级**: PAC-Bayes泛化界；引用ETT基准批评文献讨论局限性

**预期效果**: 0定理→3定理+命题；梯度衰减从文字洞察提升为可引用数学结论；稳定性分析回答"stop-gradient是否破坏不确定性学习"审稿必问。

---

#### 方向20 — 自监督欺诈检测 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 0定理+2命题；Proposition 1(参数计数)工程描述；Proposition 2(弹性系数)为度量定义非可证明命题；**消融显示λ_c=0反而AUC-PR最高("对比学习悖论")无理论解释**；冻结编码器策略无表示漂移界；类加权CE无Bayes一致性分析。

**最新研究对标(2025-2026)**:
1. Lee等 "On the Similarities of Embeddings in Contrastive Learning" ICML 2025 — 对齐-分离权衡不可达性
2. Ochieng "Diversity Is All You Need: Spectral Bounds on Gradient" NeurIPS 2025 — InfoNCE梯度谱界
3. Cortes等 "Balancing the Scales: Learning from Imbalanced Data" ICML 2025 — 代价敏感非Bayes一致+类敏感Rademacher
4. Lyu等 "Statistical Theory of Overfitting for Imbalanced Classification" arXiv 2025 — 少数类logit分布偏移
5. Esser等 "Theoretical Foundations of Representation Learning from Unlabeled Data" arXiv 2025 — DAE学习动力学

**理论提升建议**:
- **高优先级**: Theorem 1(DAE流形重建性质，基于Esser 2025)；Theorem 3(冻结编码器表示漂移界O(η·‖∇L‖))；Proposition 2修正为基于AUC-PR的弹性系数
- **中优先级**: Theorem 2(NT-Xent对齐-分离权衡界，解释对比学习悖论)；Proposition 3(类加权CE的Bayes不一致性)；新增2.6理论差异对比节
- **低优先级**: τ=0.5谱理论解释；logit分布偏移分析

**预期效果**: 0定理→3定理+2命题；DAE流形性质+表示漂移界+对齐-分离权衡构成完整理论链；创新度65→85+，完整性70→88+。

---

#### 方向24 — 特征交互房价 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 1定理+2命题；**Theorem 1仅Proof Sketch且over-claim("可表达≠可学习")**；**GNN组件零效果(w/o GAT R²=0.8501 > Full 0.8279)无理论解释**；TIGNN不如XGBoost/LightGBM(R²=0.835 vs 0.898)仅一笔带过Grinsztajn 2022。

**最新研究对标(2025-2026)**:
1. Dubbeldam等 "Graph-based Tabular DL Should Learn Feature Interactions" arXiv 2025 — GTDL无法恢复真实交互图
2. Charuthamrong等 "Edge-updating GNN for Feature Interactions" Neural Networks 2026 — 动态边属性+残差缓解小图过平滑
3. Deng等 "Interaction Bottleneck of DNN: Discovery, Proof, Modulation" 2025 — 中阶交互瓶颈理论证明
4. Chakraborty等 "Dynamical Systems Pruning for Oversmoothing in GAT" ICML 2025 — 动力系统视角解释GAT退化
5. Grinsztajn等 "Why tree-based models still outperform DL on tabular" NeurIPS 2022 — 三条归纳偏置准则

**理论提升建议**:
- **高优先级**: 新增Theorem 3(小图过度平滑现象，Dirichlet能量分析) + 新增2.9节"GAT组件零效果理论解释"；严格化Theorem 1(构造性证明+明确"可表达≠可学习"，引用Deng 2025)；Proposition 4(参数-样本比失衡)
- **中优先级**: Theorem 2(互信息初始化一致性)；Proposition 5(可识别性)；Discussion对标Grinsztajn三条归纳偏置
- **低优先级**: Proposition 3(MDL/PAC-Bayes稀疏正则)；Proposition 6(Rademacher泛化界)

**预期效果**: 1定理(sketch)+2命题→3定理+6命题；"GNN零效果"从敷衍解释升级为独特"理论解释负面结果"贡献(符合AGENTS.md诚实报告原则)；创新度75→85+，完整性78→85+。

---

#### 方向39 — 钓鱼URL检测 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 0定理+1命题(复杂度)；**贡献点4声称"prove expressiveness proposition"但正文未给出——贡献与正文不一致硬伤**；**relation-aware bias消融零下降(0.9385→0.9385)失效无理论解释**；MV-HGA显著低于RF/LGB(p<0.01, d≈-4)仅"树模型更优"一笔带过；k零弹性未解释。

**最新研究对标(2025-2026)**:
1. Tian等 "URL2Graph++" arXiv 2025 — 双粒度URL子词/字符图
2. Guo等 "Graph-based Phishing with Loopy Belief Propagation" arXiv 2025 — F1=98.77%概率图模型
3. Wu等 "OMIB: Optimal Multimodal Information Bottleneck" ICML 2025 — 多模态IB五性质理论
4. Banerjee等 "Co-Hub Node Multiview Graph Learning" arXiv 2025 — 多视图层可识别性+估计误差界
5. Chen等 "LDC-GAT: Lyapunov-Stable GAT" Axioms 2025 — Dirichlet能量+Lyapunov稳定GAT
6. Somvanshi等 "Tree vs DL on Tabular Survey" ACM CSUR 2026 — 树模型三大优势理论

**理论提升建议**:
- **高优先级**: 补Theorem 1(多视图表达能力，1-WL框架，修复贡献-正文不一致)；补Theorem 2(多视图融合泛化界，基于OMIB)；补Proposition 3(树模型优势归纳偏置解释，化解负面结果)
- **中优先级**: Proposition 4(k-NN图零敏感性的图规则性解释)；Lemma 1(关系感知注意力置换等变性)；Proposition 2(View Gating Lipschitz稳定性)
- **低优先级**: 收敛性/雅可比谱分析；Dirichlet能量分析

**预期效果**: 0定理→2定理+4命题；Methodology 25%→40%+；贡献-正文一致消除立即拒稿风险；负面结果转化为"理论预测的必然"。

---

#### 方向40 — 表格特征选择 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 0定理+2命题(复杂度)；标题声明基于MIB(互信息瓶颈)但未证明与Tishby IB目标的等价性；未分析与mRMR的理论联系；MINE估计器偏差/方差未分析(正是Arcene效果不佳根因)；稀疏门温度τ无梯度理论；小K次优性无理论解释。

**最新研究对标(2025-2026)**:
1. Muvunza等 "MINERVA: MI Neural Estimation for Supervised Feature Selection" arXiv 2025 — MINE两阶段特征选择
2. Pad等 "SAND: One-Shot Feature Selection with Additive Noise" arXiv 2025 — 线性回归下严格有效性证明
3. Weingarten等 "Supervised Information Bottleneck" Entropy 2025 — IB有监督扩展+变分上界
4. Ryu等 "Contrastive Predictive Coding Done Right" ICLR 2026 — InfoNCE不一致MI估计
5. Song & Ermon "Understanding Limitations of Variational MI Estimators" ICLR 2020 — MINE方差指数增长
6. Jana等 "Support Recovery with Projected Stochastic Gates" arXiv 2022 — STG收敛性证明
7. Wang等 "FairFS: Deep Feature Selection Biases" WWW 2026 — 三类偏差形式化

**理论提升建议**:
- **高优先级**: Theorem 1(信息瓶颈等价性，证明SMIB-FS目标可改写为有监督IB拉格朗日形式)；Theorem 3(MINE估计误差界O(exp(I)/√n)，解释Arcene d/N≈11失效)；Proposition 3(稀疏门梯度消失分析，解释小K次优)
- **中优先级**: Theorem 2(mRMR连续松弛等价性)；Proposition 4(复杂度对比形式化)；新增Section 2.6理论与实验呼应
- **低优先级**: Discussion理论化升级

**预期效果**: 0定理→3定理+2命题；"MIB有新意但效果未超传统方法"转化为"高维小样本MINE估计失效的理论必然"；创新度75→85+，完整性78→85+。

---

#### 方向03 — 不平衡学习 [↩返回总览表](#阶段b--代码已上传github19篇)

**当前理论现状**: 6定理+7命题但质量严重问题——**几乎所有公式编号为(24)(一致性检查硬伤)**；Theorem 1-6证明流于形式；**Table 3出现无法溯源的"Margin-CGRL/HFD-CGRL"术语(Data-Verifier红线)**；单一种子seed=42违反多种子要求；**核心负结果"对比学习失效"仅4条经验猜测无定理支撑**；Theorem 5、6声称对比重加权改善校准/间隔但实验alpha=0更优，理论-实验矛盾。

**最新研究对标(2025-2026)**:
1. Nguyen等 "Neural and Minority Collapse in Contrastive Learning with Imbalanced" 2025 — **严格证明极端不平衡下少数类坍缩为单一向量(直接解释SupCon在578:1下失效)**
2. Cortes等 "Balancing the Scales" ICML 2025 — 代价敏感非Bayes一致+类敏感Rademacher复杂度
3. Cortes等 "Improved Balanced Classification" NeurIPS 2025 — GCA的H-一致性界1/√p_min优于GLA的1/p_min
4. "Rethinking Loss Reweighting as Inverse Problem: NC View" arXiv 2026 — Neural Collapse逆重加权
5. Bachoc等 "When majority rules, minority loses: bias amplification of GD" NeurIPS 2025 — 梯度下降偏差放大下界

**理论提升建议**:
- **高优先级**: 修复公式编号(全为(24)硬伤)；清理Table 3无法溯源术语(Data-Verifier红线)；新增Theorem A(少数类坍缩定理，引用Nguyen 2025解释核心负结果)；补充≥5种子实验+统计检验
- **中优先级**: Theorem 4重写为类敏感Rademacher复杂度版本；引用Cortes 2025 H-一致性理论；修正Theorem 6符号漏洞
- **低优先级**: NC理论重写Proposition 2；Proposition B(梯度偏差放大下界)

**预期效果**: 修复公式编号+Table 3溯源两大硬伤使Data-Verifier达100分；Minority Collapse定理将"对比学习失效"从经验猜测升级为理论定理；理论-实验自洽；创新度显著提升。

---

### 优先投稿建议

根据Tier分级和质量评估排序：

**第一梯队（Tier 1，质量>=86分，最优先投稿）**
- **JX01** 教学研究预警 (~93 B+级) — CAE (SCIE Q2, 订阅免费)
- **JX02** 教学研究KC (92.4 A+级) — CAEE (SCIE+EI Q2)
- **17** 证据降雨预测 (90.6 B+级, 诊断工具定位) — Applied Intelligence (SCI Q3, 免版面费)
- **15** 酒店取消 (91.0 B+级) — JHTT (SSCI)
- **43** 旅游POI推荐 (90.2 A-级) — JHTT (SSCI Q1)
- **42** 时序概率预测 (86.0 B+级) — ASC (SCI Q1)

**第二梯队（Tier 2，质量~80-91分，次优先投稿）**
- **38** 自适应注意力降雨 (~88.6 B+级) — ASC (SCI Q1, 已转型诊断论文)
- **12** 学生辍学预测 (86.4 B级) — IEEE Access (SCI Q3)
- **41** 碳排放预测 (86.2 B+级) — APEN (SCI Q2)
- **27** 自适应入侵检测 (~87 B+级) — SCN (SCI)
- **20** 自监督欺诈检测 (~83 B级) — ASC (SCI Q1)
- **39** 钓鱼URL检测 (79.5 B级) — SCN/ASC (SCI Q2/Q1)
- **24** 特征交互房价 (~80 B级) — APIN (SCI)

**第三梯队（需额外改进，数据溯源或核心方法问题）**
- **40** 表格特征选择 (78.2 B-级) — 暂不建议投稿，需改进方法
- **25** 少样本杂草分类 (78.4 D级) — CEA (SCI Q1) — 数据溯源+创新度不足
- **03** 不平衡学习 (73.8 C级) — KBS (SCI Q1) — 数据溯源严重问题
- **07** 表格少样本 (73.4 C级) — PR (SCI Q1) — 数据溯源问题
- **16** 酒店情感主题 (69.0 D级) — TMP (SSCI Q1) — 核心方法失败需修复
- **14** 表格异常检测 (64.2 D级) — ESWA (SCI Q1) — 数据溯源+基线不足

---

## 暂缓投稿（负面结果，19篇）

> **注：** 删除任何方向文件夹必须由用户明示指令执行，无自动截止日期。

#### No.01 — Tabular_Framework
- **暂缓原因**: EPSS仅提升4.3%，效果不显著
- **改进策略**: 更换方法或数据集
- **论文路径**: `01_Tabular_Framework/paper/paper_draft.md`


#### No.02 — HSIC_FDANet
- **暂缓原因**: p=0.2442不显著，核心方法无统计显著性
- **改进策略**: 核心方法不显著，暂不投入
- **论文路径**: `02_HSIC_FDANet/paper/paper_draft.md`


#### No.04 — Time_Series_Framework
- **暂缓原因**: 数据溯源问题未解决（MetaMAE）
- **改进策略**: 修复数据溯源（Tier 1 ~86B级，格式修复已完成，MetaMAE溯源待解决）
- **论文路径**: `04_Time_Series_Framework/paper/paper_draft.md`


#### No.05 — Agriculture_Fusion
- **暂缓原因**: 劣于LightGBM基线
- **改进策略**: 更换数据集或调整方法
- **论文路径**: `05_Agriculture_Fusion/paper/paper_draft.md`


#### No.06 — Tourism_Prediction
- **暂缓原因**: N=65数据集过小
- **改进策略**: 换用UCI Bike-Sharing 17379行数据集
- **论文路径**: `06_Tourism_Prediction/paper/paper_draft.md`


#### No.08 — Agriculture_FewShot
- **暂缓原因**: 不如SimpleShot基线
- **改进策略**: 换用PlantVillage数据集
- **论文路径**: `08_Agriculture_FewShot/paper/paper_draft.md`


#### No.09 — AI_Tourism_Forecast
- **暂缓原因**: C+级效果一般
- **改进策略**: 待评估
- **论文路径**: `09_AI_Tourism_Forecast/paper/paper_draft.md`


#### No.10 — Tourism_ABSA
- **暂缓原因**: 99句数据集过小
- **改进策略**: 换用SemEval-2014 4728条数据集
- **论文路径**: `10_Tourism_ABSA/paper/paper_draft.md`


#### No.18 — CrossRegion_Energy
- **暂缓原因**: 负面结果，跨区域GAT损害性能
- **改进策略**: 简化跨区域图结构
- **论文路径**: `18_CrossRegion_Energy/paper/paper_draft.md`


#### No.19 — MultiScale_Power
- **暂缓原因**: MSDT劣于Transformer
- **改进策略**: 已作诊断分析论文
- **论文路径**: `19_MultiScale_Power/paper/paper_draft.md`


#### No.21 — Contrastive_Churn
- **暂缓原因**: 混合结果，需改进（Tier 3 ~82B级）
- **改进策略**: 已改进Table 4占位符修复+Proposition修正+数据真实100分
- **论文路径**: `21_Contrastive_Churn/paper/paper_draft.md`


#### No.22 — Graph_Purchase
- **暂缓原因**: GNN无效（Tier 2 ~85B级）
- **改进策略**: 已改进GNN无效诚实报告+叙事重构+格式修复
- **论文路径**: `22_Graph_Purchase/paper/paper_draft.md`


#### No.23 — Ensemble_Imbalanced
- **暂缓原因**: t统计量系统性篡改，学术诚信红线
- **改进策略**: **已放弃**
- **论文路径**: `23_Ensemble_Imbalanced/paper/paper_draft.md`


#### No.26 — Lightweight_EuroSAT
- **暂缓原因**: 低于MobileNetV2（Tier 3 ~88B+级）
- **改进策略**: 已改进种子2→5，准确率94.46%→96.11%，优于ResNet18
- **论文路径**: `26_Lightweight_EuroSAT/paper/paper_draft.md`


#### No.29 — Fairness_Tabular
- **暂缓原因**: 预测完全坍缩（Tier 2 ~91B+级）
- **改进策略**: 已改进文献修复+图表重编号+摘要扩展+贡献重排
- **论文路径**: `29_Fairness_Tabular/paper/paper_draft.md`


#### No.30 — STGCN_PM25
- **暂缓原因**: 负面结果
- **改进策略**: 已完成，诚实报告负面结果
- **论文路径**: `30_STGCN_PM25/paper/paper_draft.md`


#### No.35 — Federated_Tabular
- **暂缓原因**: 混合结果（Tier 3 ~83B级）
- **改进策略**: 已改进负面结果诚实报告+表格编号修复+格式修复
- **论文路径**: `35_Federated_Tabular/paper/paper_draft.md`


#### No.36 — Interpretable_Tabular
- **暂缓原因**: 低于XGBoost（80.49%）
- **改进策略**: 可解释性-准确率权衡分析
- **论文路径**: `36_Interpretable_Tabular/paper/paper_draft.md`


#### No.37 — GenAug_Tabular
- **暂缓原因**: 负面结果，不如RF
- **改进策略**: 尝试扩散模型或换数据集
- **论文路径**: `37_GenAug_Tabular/paper/paper_draft.md`


---

## 暂停方向（2篇，非删除）

| 方向 | 暂停原因 | 恢复条件 |
|------|----------|----------|
| 11_EuroSAT | 训练时长过长，当前硬件无法按时完成 | 需在适当硬件上完成训练 |
| 13_LUCAS_Soil | 使用合成数据，未获取真实LUCAS土壤光谱数据 | 需获取真实数据 |

> **注：** 暂停≠删除，如需恢复或删除，请用户明示指令。

## 阶段A — 实验完成，待上传GitHub（1篇）

| 编号 | 方向 | 结果 | 评分 | 核心亮点 | 目标期刊 | GitHub |
| ---- | ---- | ---- | ---- | -------- | -------- | ------ |
| 66 | NYC出租车域特征 | 正面 | 93.0 | Domain特征R²提升10.2-30.6pp; 理论框架+SHAP+消融+敏感性+部署+分布偏移+TabPFN+噪声鲁棒性+公平性; 数据真实性100 | KBS (SCI Q2, 订阅制免APC) | 已上传 |

#### No.66 — NYC出租车域特征增强
- **结果性质**: 正面（域特征增强理论与应用研究，含理论框架）
- **目标期刊**: Knowledge-Based Systems (Elsevier, SCI Q2, IF≈8.0, 订阅制免APC)
- **备选期刊**: ①Engineering Applications of Artificial Intelligence (Elsevier, SCI Q2, 订阅制免APC) ②Applied Intelligence (Springer, SCI Q3, 订阅制免APC) ③Neural Computing and Applications (Springer, SCI Q3, 订阅制免APC) ④IET Intelligent Transport Systems (IET, SCI Q4, 免APC)
- **关键指标**: CatBoost+Domain R²=0.767±0.045; XGBoost+Domain R²=0.764±0.044; RF提升最大(+30.6pp); Wilcoxon p=0.016; Cohen's d=7.72-17.39
- **理论贡献**: Theorem 1(偏差-方差分解) + Theorem 2(Rademacher泛化界) + Proposition 1(特征互补性) + Proposition 2(样本复杂度)
- **论文路径**: `66_NYC_Taxi_DomainFeat/paper/paper_draft.md`
- **理论框架**: `66_NYC_Taxi_DomainFeat/paper/theoretical_framework.md`
- **数据集路径**: `66_NYC_Taxi_DomainFeat/data/nyc_green_taxi_50k.csv` (50K行NYC Green Taxi)
- **GitHub仓库**: https://github.com/mingyi0818/NYC_Taxi_DomainFeat (已上传，公开，含源代码+数据+结果+README.md)
- **审计状态**: 实验真实性通过（56个实验复现验证，42/56完全可复现，14个RF微小非确定性）
- **SOTA声称**: 未声称超越SOTA（与Kaggle竞赛不可直接比较：目标变量/样本量/特征集不同）
- **审计修复**: P1(引用不当)✓ P2(无来源声称)✓ M1-M5✓ L1/L2/L3/L4✓ 全部修复完成
- **改进方案执行**: 5个突破点全部完成（理论框架+消融实验+SHAP+敏感性+部署成本）
- **4个提升方向实验**: 分布偏移分析(Section 3.8) + TabPFN/MLP/线性基线对比(Section 3.9) + 噪声鲁棒性(Section 3.10) + 公平性分析(Section 3.11) 全部完成
- **数据真实性**: 100/100（210项+34项=244项自动化验证全部通过，误差<0.002）
- **论文质量评分**: 数据真实性100 + 创新度93 + 完整性95 + 语言质量87 = 综合评分93.0，迭代终止条件满足
- **实验数据**: 11个JSON文件 + 9个CSV文件 + 11幅300dpi图片
- **参考文献**: 48篇（近5年>50%）
- **投稿材料**: Cover Letter ✅ (KBS版) + Highlights ✅ (5条，每条≤85字符) + Data Availability Statement ✅ + CRediT ✅ + Declaration of Competing Interests ✅ + Declaration of Generative AI Use ✅ (均位于paper/目录)
- **最后更新**: 26-08-08
- **下一步行动**: 1)通过KBS投稿系统提交论文; 2)选择订阅制（非Open Access）避免APC; 3)如被拒转投备选期刊
- **看板详情页**: `66_NYC_Taxi_DomainFeat/dashboard.html`

## 候选新方向（7篇，待用户确认）

| 编号 | 方向           | 方法    | 目标期刊            | 优先级 | 数据集                     | 核心创新点                          | 预估工时 |
| ---- | -------------- | ------- | ------------------- | ------ | -------------------------- | ----------------------------------- | -------- |
| 44   | 能源异常检测   | TCR-AD  | Applied Energy (Q1) | ★★★★   | SGCC电力盗窃(42K客户)      | 时域+频域对比重构异常检测           | 7 days   |
| 45   | 多变量时序融合 | ADCVI   | ASC (Q1)            | ★★★    | 多个UCR/ETT数据集          | 自适应分解卷积多变量融合            | 7 days   |
| 46   | 农业目标检测   | LA-FPN  | CEA (Q1)            | ★★★    | CropAndWeed/VOC            | 轻量注意力特征金字塔网络            | 5 days   |
| 47   | 农业语义分割   | MCA-Net | CEA (Q1)            | ★★★    | 农田遥感分割数据集          | 多尺度上下文注意力分割              | 6 days   |
| 48   | 因果效应估计   | AGBRL   | NCA (Q2)            | ★★★    | 合成/半合成因果数据集       | 图贝叶斯强化学习因果推断            | 7 days   |
| 49   | 水质预测       | LSTA-WQ | EMS (Q1)            | ★★★★   | 水质监测时序数据集          | 长短时注意力水质预测                | 6 days   |
| 50   | AI恶意代码检测 | MGFF-MC | ESWA (Q1)           | ★★★★   | 恶意代码数据集              | 多粒度特征融合恶意代码检测          | 7 days   |

---

## 后台低优先级任务

| 方向 | 任务               | 预估 |
| ---- | ------------------ | ---- |
| 06   | 换用更大旅游数据集 | ~2h  |
| 08   | 换用PlantVillage   | ~3h  |
| 10   | 换用SemEval数据集  | ~2h  |
| 18   | 简化跨区域图结构   | ~3h  |
| 26   | 换用UC Merced      | ~3h  |

---

## 更新记录

| 日期     | 内容                                                                                                       |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| 26-08-08 | No.66改进方案执行完成：5个突破点全部完成（①理论框架Theorem 1-2+Proposition 1-2+复杂度分析 ②特征组消融4×4×7=112组 ③SHAP全局+局部+交互分析 ④超参数敏感性3参数×4配置×3种子=36组 ⑤部署成本8组测量）。生成7幅300dpi图片。论文重写整合理论+实验+图表，43篇参考文献。210项数据真实性自动化验证全部通过(100/100)。四维质量评分：数据真实性100+创新度88+完整性90+语言质量85，迭代终止条件满足。看板第九节已更新 |
| 26-08-06 | 新增17个方向(25,07,16,14,JX02,15,43,38,27,12,41,42,20,24,39,40,03)理论水平提升分析：基于各论文草稿精读+2025-2026最新研究联网检索，为每个方向提供当前理论现状评估、最新研究对标、具体可执行的理论提升建议(按高/中/低优先级)、预期效果。总览表编号列已添加可点击链接，各分析末尾添加"返回总览表"按钮 |
| 26-07-24 16:00 | 数据集路径迁移：方向43 Yelp数据集→`D:\datasets\tourism\Yelp_POI_Recommend\`；CropAndWeed(3文件夹)→`D:\datasets\CropAndWeed\`；DeepWeeds(5文件夹)→`D:\datasets\DeepWeeds\`；WeedSense→`D:\datasets\WeedSense\`。相关源代码(12个文件)路径已同步更新 |
| 26-07-23 11:32 | papersubmission.md恢复重建：投稿决策仪表盘+阶段B详细条目+暂缓详情+候选方向详情+优先投稿建议 |
| 26-07-23 | 19篇阶段A论文代码全部上传GitHub（仓库名无数字前缀），每篇含README.md复现说明。文件因PowerShell编码问题重建 |
| 26-07-22 | No.41/42/43 完成                                                                                           |
| 26-07-21 | 文件初始创建                                                                                               |
