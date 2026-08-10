# 论文包综合审计与整改建议

> 审计范围：`D:\ResearchPaperPrepare\toexpert` 下 8 个论文包  
> 审计日期：2026-07-26  
> 审计目标：按 SCI/SSCI 期刊审稿标准，对论文、结果表、代码、统计检验、参考文献和复现材料进行交叉核验。  
> 重要说明：本文档记录的是投稿前内部技术审计意见，不应直接作为作者回复信使用。涉及“伪造”“学术诚信”等表述时，应先保全原文件、提交记录和运行日志，再由作者团队正式核查。

---

## 1. 总体结论

8 个论文包目前均不适合直接投稿。建议等级如下：

| 论文包 | 当前建议 | 新颖性估计 | 最关键问题 |
|---|---|---:|---|
| `JX01_Teaching_Research_1` | Major Revision / Reject & Resubmit | 3.5/10 | 数据真实，但核心显著性在学生层聚类 bootstrap 下消失 |
| `JX02_Teaching_Research_2` | Reject | 2.5/10 | 合成代码与规则标签形成循环评测，反馈评分也是关键词同义反复 |
| `15_Hotel_Cancellation` | Reject，重做实验 | 2.5/10 | `total_guests` 是输入列的精确和，构成恒等映射级标签泄漏 |
| `17_Evidence_Rainfall` | Reject，统一实验版本后重投 | 3.0/10 | Table 1–6 与 Table 7–15 来自两套不同模型/结果版本 |
| `38_AdaptiveAttention_Rain` | Reject，修复模型后重投 | 2.0/10 | 注意力序列长度为 1，softmax 恒为 1，核心注意力机制退化 |
| `41_Carbon_Emission` | Reject，全部重跑 | 2.0/10 | 所谓 cross-attention 未实现，方向损失梯度为零，时序切分实为随机切分 |
| `42_Probabilistic_TS` | Reject，但诚实重做后有救 | 2.5/10 | 概率区间严重欠覆盖却被隐去，基线预算不公平，理论动机推导错误 |
| `43_Tourism_Recommend` | Reject，返回模型实现阶段 | 2.0/10 | 时空模块为死代码，消融基准硬编码，随机数被当作学习嵌入绘图 |

整体问题不是八篇文章各自偶发的小错误，而是生产流程存在重复性缺陷：

1. 论文方法描述与代码实现不一致；
2. 结果目录中保留多版互不兼容实验，却在正文中混用；
3. 超参数敏感性大量使用错误的“弹性系数”公式；
4. 测试集被用于调参或预处理统计量在切分前拟合；
5. 负面结果、校准失败或更强基线被保留在磁盘中，却未进入论文；
6. 消融实验只跑单种子，差异小于随机初始化噪声；
7. README、reproduce、cover letter 大量来自模板，和实际任务不符；
8. 存在错误、张冠李戴或疑似生成的参考文献；
9. 多个包包含硬编码或模拟图表，必须立即隔离，禁止投稿。

---

## 2. 投稿前必须执行的全局 P0 整改

### 2.1 冻结当前证据

在修改任何文件前：

- 对 8 个目录生成 SHA256 清单；
- 备份 `paper/`、`results/`、`code/`；
- 保存 Git commit、文件修改时间、Python/包版本、硬件和运行命令；
- 不要覆盖现有 CSV/JSON，将旧结果移入 `results/deprecated/<date>/`；
- 每次新实验必须记录：
  - `git_hash`
  - `seed`
  - 数据版本/hash
  - split protocol
  - 超参数
  - epoch/patience
  - 训练与测试时间
  - 生成该结果的脚本名。

### 2.2 删除或隔离模拟、硬编码和无源图表

以下文件不得随稿提交，除非以真实实验重新生成：

- `43_Tourism_Recommend/code/visualize.py` 中以 `np.random.randn` 模拟模型嵌入的 t-SNE；
- `43_Tourism_Recommend/code/fix_figures.py` 中手工插入的 `Full STGC-CF` 消融基准行；
- `15_Hotel_Cancellation/code/visualize.py` 中以指数函数加随机噪声模拟的训练曲线；
- 任何使用 fallback 常量而不是结果文件生成的图；
- 任何 results 中没有对应原始预测、逐种子结果或生成脚本的表格数字。

### 2.3 禁止继续使用当前“低弹性”模板

至少 6 个包使用了错误或不可解释的弹性公式。统一改为：

连续正参数与正指标可使用对数回归：

$$
\log Y_i = a + \varepsilon \log X_i + e_i,
$$

其中 $\varepsilon$ 才是局部/平均弹性。要求：

- 使用全部网格点，不得只挑选 `[0.1,10]` 等有利子区间；
- 报告斜率、标准误和置信区间；
- 指标含零或负值时不要强行取对数；
- 分类超参数（如 n-gram 类型）不叫弹性，可报告 ANOVA、Sobol 指数或范围/方差；
- 敏感性实验不得在测试集选最优值，应在 validation 或 nested CV 内完成；
- 每个配置至少 5 个种子，并报告误差棒。

### 2.4 建立单一事实来源

每篇论文只允许一套活动结果：

```text
results/
  raw/          # 每个 seed/fold 的原始预测和指标
  aggregated/   # 由 raw 自动聚合
  tables/       # 从 aggregated 自动生成
  figures/      # 只从 tables/raw 生成
  deprecated/   # 旧版，正文不得引用
```

论文 Markdown 中的表格应由脚本自动生成或自动核对。验证脚本不得硬编码所谓“paper value”，而应解析当前 `paper_draft.md` 后与 CSV/JSON 比较，任何不一致退出非零状态。

### 2.5 全量文献核验

对每条参考文献执行 DOI/Crossref/DBLP/OpenAlex 核验：

- 标题、作者、年份、期刊/会议、卷期、页码或文章号必须一致；
- 正文未引用条目删除；
- 不能用无关论文支撑方法、数据或阈值；
- 参考文献最新性不能靠未引用条目凑比例；
- 每篇稿件单独生成 `reference_audit.csv`：
  `id,title,doi,resolves,metadata_match,cited_in_text,status`。

---

## 3. `JX01_Teaching_Research_1`

### 3.1 可保留的基础

- ProgFeed 数据是真实学生数据，不是合成数据；
- 统计规模与原数据来源一致：
  - 6,693 次提交；
  - 215 名学生；
  - 17 个 lab；
  - feedback 分布 `no_feedback=4067 / tc=1605 / nl=1021`；
- CodeBench 也是真实公开数据；
- 决策曲线净收益公式实现正确；
- 论文主动避免直接宣称因果学习增益，这一态度应保留。

### 3.2 当前阻断项

#### （1）核心显著性在正确聚类后消失

论文以 i.i.d. bootstrap 对 3,323 条尝试独立重采样，忽略尝试聚集在 215 名学生内。学生层 cluster bootstrap 的复算结果约为：

- 点差：约 `-0.028`；
- 论文 i.i.d. 95% CI：约 `[-0.055,-0.003]`；
- 学生层 cluster 95% CI：约 `[-0.070,+0.012]`。

正确区间包含 0，因此“简单启发式显著优于 OHTP，p=0.027”不能保留。摘要、Highlights、结论必须改成：

> 启发式规则点估计更高，但在学生层聚类不确定性下差异不显著。

另有 winner’s curse：从约 13 个策略候选中选择最佳启发式后再检验，没有做多重比较校正。

#### （2）Cohen’s d=3.93 不可复现

该值无法由标准 AUC 效应量或配对差公式得到。应删除并改报：

- AUC 差；
- cluster bootstrap CI；
- 或清楚定义的配对 $d_z$。

#### （3）ECE 口径错误

代码中 Table 7 对应结果明确使用 `n_bins=8`，但论文称 “10-bin ECE”。

- 报告值 process ECE ≈ 0.076 是 8-bin；
- 正确 10-bin 复算约 0.0426；
- 论文对校准差距的叙述被明显夸大。

应统一报告 equal-width 10-bin、equal-mass 10-bin，并给可靠性图。

#### （4）构念效度偏弱

`late_high_struggle` 由后期提交次数中位数二分，而主要预测信号是早期提交次数。单一 `early_mean_attempts` 已接近完整模型 AUC，说明模型主要学到“早期多提交者后期仍多提交”的行为持续性，不一定是学习风险。

必须并列验证：

- late mean attempts 连续回归；
- late low pass；
- 最终掌握；
- wheel-spinning；
- 控制 early attempts 后的增量 $R^2$。

#### （5）实际随机化未被用于因果分析

ProgFeed 在 student×lab 层具有反馈条件随机化结构。现稿只把 feedback type 当特征，浪费了最有价值的识别条件。

推荐用 cross-fitted AIPW 估计策略价值：

$$
\hat V(\pi)=\frac1n\sum_i\sum_a I\{\pi(X_i)=a\}
\left[
\frac{I(A_i=a)}{\hat e_a(X_i)}(Y_i-\hat m_a(X_i))
+\hat m_a(X_i)
\right].
$$

标准误按学生聚类。结果应从“预测标签 precision”升级为“支持策略对真实后续结果的价值”。

### 3.3 其他必须修复

- `figures/architecture.svg` 不存在；
- `data/` 和若干关键统计生成脚本缺失或为空；
- `ohtp_top40` 同名指标有多套数值；
- nl 条件提升存在正负号冲突；
- 早/晚 lab 应按真实时间戳核验；
- CodeBench pooled 与 leave-term 标签阈值口径不一致；
- 正文需加入二次分析伦理声明及原研究 IRB 信息；
- 补 Beck & Gong wheel-spinning、Hattie & Timperley、Narciss、Vickers & Elkin、Nadeau & Bengio 等文献。

### 3.4 最低投稿门槛

1. 全部策略 CI 改为 student-cluster；
2. 删除 d=3.93 和错误 ECE；
3. 补齐可复现脚本；
4. 使用随机化反馈条件做 AIPW/ITT 分析；
5. CodeBench 做跨学期 rolling-origin 验证；
6. 摘要不再使用“显著优于”。

---

## 4. `JX02_Teaching_Research_2`

### 4.1 顶层问题：标签循环

数据链条是：

```text
misconception_id --由生成器按描述生成--> synthetic code
misconception_id --related_constructs + KC_MAP--> KC label
```

KC 标签并非人工从代码判断，而是从同一 `misconception_id` 确定性映射。模型学习的是从合成代码还原生成种子，而不是识别真实学生误概念。因此：

- 0.6269 accuracy 不能称作真实误概念诊断准确率；
- “code-only 避免标签泄漏”并不能消除生成过程的构念循环；
- 这是 construct validity 失败，不是普通 limitation。

### 4.2 反馈质量评测是同义反复

L1/L2/L3 都逐字包含 misconception description，因此 relevance 恒接近 1；只有 L2 包含代码行，因此按“与代码 token 重叠”定义的 specificity 必然最高；动作词又被模板预先植入，actionability 也由模板决定。

三套 rubric 还得出互相冲突排序：

- 一套：L2 > L3 > L1；
- 一套：L3 > L2 > L1；
- 一套：L2 = L3 > L1。

论文只报告支持命题的一套，属于选择性报告风险。

所谓三个评分者实际上是三个关键词协议，不是人类。Krippendorff α≈0.819 不能替代 human IRR，因为模板直接包含协议关键词。

### 4.3 已确认的硬错误

- `e1_confusion_matrix.csv` 是完美对角阵，相当于 100% accuracy，与正文 0.6269 冲突；
- E6 LOPO 实际是 CodeBERT embedding + LinearSVC、63 类 misconception、n=1055；
- 正文却把 E6 写成 10 类 KC 分类并与 char-TFIDF 0.6269 直接比较；
- 正文又声称 CodeBERT “remains untested”，但磁盘已有：
  - 8 类 KC CodeBERT-SVM accuracy ≈ 0.7037；
  - 63 类 fine-grained accuracy ≈ 0.4825；
- t=7.369 的实际对比是 Word-SVM vs RandomForest，不是正文声称的 Char-SVM 对比；
- Cohen’s d=3.685 用了 $\sqrt{df}=\sqrt4$，正确应约 3.298；
- 多种子 CI 用 z=1.96 而 n=5 应使用 $t_4$；
- majority baseline 是 315/1021=0.3085，不是图中的随机 0.10；
- AST-only 0.3164 和 NB 0.3154 都只是多数类水平；
- 参数 C 全区间 `[0.001,100]` 的弹性约 0.2037，应为 medium，而非选择 `[0.1,10]` 后的 low；
- KC “Top 5” 共现并非真实 Top 5；
- 引言称 9 个实验、正文 E1–E8、结论称 7 个实验；
- Figure 1 指向不存在的 `architecture.svg`；
- 缺 data、CodeBERT embedding、若干结果生成脚本。

### 4.4 必须重做

#### 人工标注

建议至少 1,500–1,800 份真实学生提交，10 个 KC 每类约 150：

- 3 名独立标注者；
- 20% 三人重复标注；
- nominal Krippendorff α 目标 ≥0.80；
- 盲于规则标签和模型预测；
- 第 4 名专家裁决；
- 允许多标签；
- 发布 codebook 和裁决规则。

#### 真实数据

优先：

- CodeBench UFAM；
- FalconCode；
- CSEDM/CodeWorkout；
- Blackbox/BlueJ；
- Project CodeNet。

McMiner 只能作为 synthetic upper-bound，不得作为真实学生结论的唯一依据。

#### 学习结果

需 RCT、准实验或至少随机化反馈试验，主结局应为：

- 误概念后续复发；
- 修复所需提交次数；
- 延迟测验对应 KC 得分；
- time-to-correct。

### 4.5 最低投稿门槛

1. 人工标注打破循环；
2. 删除规则评分作为主证据；
3. 人类盲评并报告 IRR；
4. 统一 CodeBERT 与 LOPO 任务口径；
5. 删除完美混淆矩阵并重生成；
6. 重新做 GroupKFold(problem) 与近重复去重；
7. 诚实披露三套 rubric 和 CodeBERT 结果。

---

## 5. `15_Hotel_Cancellation`

### 5.1 恒等映射级标签泄漏

代码明确：

```python
df["total_guests"] = df["adults"] + df["children"] + df["babies"]
target_cols = ["is_canceled", "adr", "total_guests"]
feature_cols = [c for c in df.columns if c not in target_cols]
```

因此 `adults`、`children`、`babies` 仍在输入特征中，而第三个任务是它们的精确和。R²≈0.9985、树模型 MAE≈1e-4 是恒等关系，不是模型能力。

必须：

- 删除 `total_guests` 任务；或
- 从输入中删除构成它的三列，但这仍缺乏教育/业务意义；
- 更推荐改为二任务 cancellation + ADR，或改成 time-to-cancellation 生存任务。

### 5.2 消融实验无效

“w/o Graph Regularization”和“w/o Task Graph”在当前实现中目标函数与有效梯度相同，却得到约 0.0019 AUC 差异。这等于直接测出随机噪声地板。

所有消融差异：

- 多数小于或接近 0.0019；
- 所谓最大贡献约 0.004，也缺少多种子确认；
- 模型是在设置 seed 前构造，初始化不可比；
- full 行来自另一条实验路径。

消融需重新定义并运行 5–10 seeds，加入 identical-config repeat 作为噪声基线。

### 5.3 容量与调参不公平

实测参数量约：

- AGMTL：614k；
- SB-MLP：239k；
- ST-MLP：121k。

AGMTL 是 SB-MLP 的 2.57 倍，但论文表格将其写成约 0.5M vs 0.4M，掩盖容量差。+0.0021 AUC 不能归因于多任务结构。

必须增加：

- 参数量匹配的 SB-MLP-Large；
- 参数量匹配的 ST-MLP-Large；
- 同等 Optuna 搜索预算；
- tuned XGBoost/LightGBM/CatBoost。

### 5.4 图和表的诚信风险

- `Figure5_training_curves.png` 来自模拟指数曲线加随机噪声；
- 敏感性图包含硬编码或与 CSV 冲突数据；
- 邻接矩阵在公式中要求非对角归一化和为 1，代码却先对含对角线矩阵 softmax，再把对角线清零，实测非对角和约 0.658；
- Theorem 1 因此证明的不是代码实现；
- Table 5 参数量、时间、吞吐等多项无结果文件支撑；
- AGMTL 训练时间被除以 3 后与单模型基线比较，不合理。

### 5.5 数据协议

做对的一点：`reservation_status` 和 `reservation_status_date` 已删除。

仍需修复：

- 原数据大量重复行，当前未 `drop_duplicates()`；
- 随机切分导致重复记录跨 train/test；
- scaler/encoder 在切分前 fit；
- `assigned_room_type`、`booking_changes` 等预测时点可得性不清；
- 应按 arrival date 做时间切分或 rolling-origin；
- 增加 PR-AUC、MCC、balanced accuracy、Brier、ECE；
- 进行超售成本和 decision-curve 分析。

### 5.6 推荐重构

最有业务意义的升级是离散时间生存模型：

$$
h(t|x)=P(\text{cancel at }t\mid \text{not cancelled before }t,x),
$$

$$
\mathcal L=-\sum_i\sum_t
\left[d_{it}\log h_\theta(t|x_i)+(1-d_{it})\log(1-h_\theta(t|x_i))\right].
$$

报告 time-dependent AUC、integrated Brier、C-index 和超售决策收益。

---

## 6. `17_Evidence_Rainfall`

### 6.1 两套实验版本被混入一篇论文

同一个 EDL-Fixed seed 42 在论文中出现两套结果：

| 结果族 | Accuracy | F1 | Recall | Evidence S | 论文用途 |
|---|---:|---:|---:|---:|---|
| A：20 epoch fixed run | 约 0.8546 | 0.7720 | 0.5518 | 69.82 | Table 1–6 |
| B：123-dim checkpoint | 约 0.8697 | 0.7930 | 0.5701 | 101.97 | Table 7–15、摘要 |

因此论文并非在一套模型上完成全链路分析。最严重的后果是：

- 论文用 A 族 RF accuracy 0.8426；
- 同时用 B 族 RF skill scores；
- 在 B 族中 RF accuracy 实际约 0.9061 且全面领先；
- “accuracy 具有误导性”的核心叙事由跨版本拼接人为制造。

必须统一数据、特征维度、超参数、checkpoint 和 seed，从头生成全部 Table 1–15。

### 6.2 “100/100 验证器”验证的是另一版稿件

`verify_results.py` 硬编码并验证：

- EDL accuracy 0.8684；
- F1 0.7898；
- ECE 0.0221；
- RF accuracy 0.9061。

当前稿件却写：

- EDL accuracy 0.8559；
- F1 0.7742；
- ECE 0.0089；
- RF accuracy 0.8426。

验证脚本依然输出 135/135 passed 和 Data Authenticity Score 100/100。该工具不能作为数据真实性证据，必须重写为解析当前稿件的自动核验器。

### 6.3 头条相关系数无计算来源

`ρ(H_E,1/S)>0.999` 在摘要、贡献、结论等多处出现，但代码和结果中没有 Spearman/Kendall 计算。唯一记录只是验证报告将其标注为“directional claim, consistent with monotone relation”，即主动跳过验证。

必须：

- 实际计算 Spearman ρ；
- 1000 次 bootstrap CI；
- 保存原始 pair 数据或最少保存统计脚本与结果 JSON；
- 提供能让 ρ 显著低于 1 的反例，否则该“诊断”不可证伪。

### 6.4 理论与代码错位

- Theorem 3 渐近式漏掉 $1/S$ 因子；
- 该诊断对 binary Dirichlet EDL 近乎恒真，不能区分好坏；
- 论文 C2 写成 m=1 单观测 Beta 边际，实际代码实现 m=5 邻域 Beta-Binomial；
- m=1 会退化为普通 CE，不含二阶信息；
- 真正可能有新意的是 m≥2 邻域计数使 evidence strength 可辨识，但论文反而写错了；
- conformal 分组代码是 season×climate zone，正文却称 station groups；
- calibration/evaluation 对测试期随机对半切，coverage≈0.95 是交换性构造下的必然 sanity check，不是业务时间外推证据。

### 6.5 OOD 设计问题

- 空间 OOD 是随机留一半站，不是气候区隔离；
- temporal OOD 只是同一 test 内 2016/2017 分层；
- extreme event 使用当天 `Rainfall`，标签却是 `RainTomorrow`；
- 极端层又限制为正类后计算 macro-F1，指标无意义；
- OOD 预处理和主流程不一致，部分 imputer 在全量数据 fit。

### 6.6 推荐救援路线

1. 统一为单一 123-dim 或重新设计后的特征版本；
2. 所有表来自同一 checkpoint 家族；
3. 将 m=5 邻域 Beta-Binomial 明确为核心创新；
4. 做 $m\in\{1,2,3,5,9\}$ 消融；
5. 比较 ρ、OOD-AUROC、S、ECE；
6. rolling-origin 时间 conformal；
7. 真正的 climate-zone OOD；
8. Deep Ensemble、temperature scaling、MSP/Mahalanobis 基线；
9. 删除空 data/checkpoints 目录声明或真正提供文件。

---

## 7. `38_AdaptiveAttention_Rain`

### 7.1 注意力机制在代码中退化

`AdaptiveSpatialAttention.forward()`：

```python
x_seq = x.unsqueeze(1)  # (batch, 1, num_features)
B, S, D = x_seq.shape   # S=1
attn_weights = softmax(QK^T, dim=-1)
```

当 S=1 时，softmax 对单元素计算，权重恒为 1。因此：

$$
\mathrm{ASA}(x)=x+W_O W_Vx,
$$

与 Q/K 无关。`q_proj` 和 `k_proj` 永远收不到有效梯度。

Tabular Transformer 也把向量 `unsqueeze(1)` 成长度 1，存在同样退化。约 24,832/118,803=20.9% 参数属于死参数。

这直接否定：

- “cross-feature attention”；
- 多头提供多视角；
- Proposition 2；
- 将模型称为 adaptive attention network 的核心依据。

### 7.2 学习率敏感性是 no-op

五个 learning rate 配置在九个指标上逐位完全相同。原因：

- override 被写入 `cfg`；
- optimizer 却读取全局 `TRAIN['learning_rate']`；
- 扫描值从未进入 Adam。

论文将这个 bug 解释为 scheduler 与 early stopping 抵消不同学习率，必须删除并重跑。

### 7.3 时间泄漏

Rain in Australia 是同站连续日数据，当前使用随机分层 70/15/15：

- 相邻日期可能分布在 train/test；
- `RainToday(t+1)` 与 `RainTomorrow(t)` 存在直接时序联系；
- 缺失值填补和 rare-category 处理在切分前完成。

必须改为：

- train 2007–2014；
- val 2015；
- test 2016–2017；
- 另做 leave-station-out。

### 7.4 气象指标揭示主结论反转

模型头条是最高 recall/POD≈0.8231，但 Frequency Bias≈1.52，表示预报降雨次数多约 52%。按 CSI、ETS、HSS、BSS 和 AP，AAR-Net 仅居中后位；RF 频率偏差接近 1 且 BSS/HSS 更好。

必须报告：

- POD；
- FAR；
- CSI；
- ETS；
- HSS；
- Frequency Bias；
- Brier Skill Score；
- reliability diagram；
- climatology 与 persistence。

阈值应在 validation 按 CSI 或业务损失选择，不能固定 0.5 后把 recall 当模型能力。

### 7.5 其他问题

- 消融和敏感性只有 seed 42，差异均小于一个种子 std；
- 弹性公式分母硬编码 0.1，不是 $\Delta X/X$；
- Figure 6 实际是 robustness 图却被描述成 reliability diagram；
- Figure 5 对原始输入做 t-SNE，却被说成 learned representations；
- 所有图路径失效；
- 引文 [18] 高风险伪造/张冠李戴；
- README 称方法包含 graph-based modeling，但代码无图模型。

### 7.6 推荐重构

必须把每个气象变量变成 token，使序列长度等于特征数而非 1，再做真正特征注意力。进一步可引入天气型条件偏置：

$$
A^{(h)}=\mathrm{softmax}
\left(
\frac{Q^{(h)}K^{(h)\top}}{\sqrt{d_h}}
+\sum_k r_k B^{(h,k)}
\right),
$$

其中 $r_k$ 是样本的天气型后验。新模型需与 FT-Transformer、SAINT、TabNet、CatBoost、LightGBM 和真正的时序 GRU/TCN/PatchTST 比较。

---

## 8. `41_Carbon_Emission`

### 8.1 核心贡献未实现

论文称 CO₂ history 为 Query、辅助特征为 Key/Value。代码实际：

```python
h = self.input_proj(x)
attn_out = self.cross_attn(h, h, h)
```

Q=K=V=h，是普通 self-attention。输入投影已混合全部 9 个特征，源身份消失，因此：

- 不存在 cross-source attention；
- “Multi-Source Feature Fusion”标题不成立；
- 无法解释源级 attention weight；
- 贡献 1 必须重新实现或删除。

### 8.2 Direction loss 梯度为零

论文声称用 $\tanh(10z)$ 近似 sign 以保持可导。代码却直接：

```python
dir_loss = mse(sign(pred_diff), sign(target_diff))
```

`torch.sign` 几乎处处梯度为零，方向损失不能训练模型，只会污染验证损失、scheduler 和 early stopping。

应改为：

$$
\mathcal L_{\text{dir}}=
\frac1{H-1}\sum_h
\left[
\tanh(\gamma\Delta\hat y_h)
-\tanh(\gamma\Delta y_h)
\right]^2,
\quad \gamma=10,
$$

并加入梯度非零单元测试和 Directional Accuracy。

### 8.3 “Temporal split”实际是随机切分

代码在 `split_data()` 前执行 `np.random.permutation`。窗口长度 10、预测 5 年，相邻窗口高度重叠。复核显示约：

- 47/48 测试窗口与同国训练窗口目标年份重叠；
- train/test 目标年份范围相同；
- scaler 在全时间段 fit；
- 20 国中多数归一化端点落在测试期。

诚实时序留出后，简单 Ridge/RandomForest 性能显著恶化并可能输给 persistence。这意味着现有全部主结果作废。

### 8.4 小样本与参数比

- 20 国×30 年；
- 320 个窗口；
- 训练窗口约 224；
- MSFCE 参数约 501,136；
- 参数/训练窗口约 2,237:1。

该规模不支持 Transformer 方法优越性的结论。建议：

- 扩展至更多国家或省级面板；
- 使用 rolling-origin 和 leave-one-country-out；
- 增加 STIRPAT、LMDI、Panel FE、ARIMAX、XGBoost/LightGBM、DLinear/PatchTST、persistence。

### 8.5 基线和选择性报告

- N-BEATS、TCN、Informer 等多个基线优于 MSFCE；
- 所提方法在深度模型中接近末位；
- 论文只报告部分显著劣势，遗漏 Informer 对比中更小的 p 值；
- Informer 与 Transformer 实现实际相同；
- ARIMA 在长度 5 的平均目标序列上拟合，是无效稻草人；
- sensitivity 为单种子且在 test 选优；
- Table 4 多项 FLOPs、推理时间无生成代码或结果文件。

### 8.6 文献红线

已通过 Crossref 复核：

- `[5]` 所给 DOI 实际对应 Yuan 等人的区域供热论文，不是所引碳排放预测论文；
- `[10]` 所给 DOI 返回 404，无法解析。

必须逐条核验全部参考文献，删除错误条目。

### 8.7 推荐路线

最短可发表路径不是继续包装 MSFCE，而是改写为：

> 随机窗口切分如何在小样本国家碳排放面板中虚构 65%–100% 的预测增益：统计、机器学习与深度模型基准研究。

这比当前方法论文更诚实，也更有方法学价值。

---

## 9. `42_Probabilistic_TS`

### 9.1 概率预测主张被自身 PICP 否定

名义 95% 预测区间的实际 PICP：

- 全局范围约 0.0917–0.4842；
- SG-DER 平均约 0.287；
- 覆盖误差约 -47 至 -86 个百分点。

论文却只报告 MAE/RMSE/MAPE/SMAPE/R²，完全不报告已经存在于 CSV 的 PICP、MPIW、NLL，同时声称 “well-calibrated uncertainty”。

这是当前稿件最严重的选择性报告问题。必须：

- 报告 CRPS、NLL、pinball、PIT、PICP/ACE、Winkler；
- 加 Deep Ensemble、MC-Dropout、CQR、EnbPI/时序 conformal；
- 删除“well-calibrated”直到校准证据成立。

### 9.2 理论动机推导错误

论文称 NIG NLL 对 $\gamma$ 的梯度可能把预测均值推离 $y$。实际：

$$
\frac{\partial\mathcal L_{\text{NLL}}}{\partial\gamma}
=
\frac{(2\alpha+1)v(\gamma-y)}
{v(y-\gamma)^2+2\beta(1+v)}.
$$

分母为正，梯度与 $\gamma-y$ 同号，梯度下降始终把 $\gamma$ 推向 $y$。真实问题是大残差时梯度幅值衰减，不是方向冲突。

理论应改为“梯度衰减/条件数解耦”，并正面讨论 Meinert、Bengs、β-NLL 和 faithful heteroscedastic regression。

### 9.3 基线预算严重不公平

- SG-DER：100 epochs、patience 15、batch 32；
- 基线：25 epochs、patience 7、batch 64；
- SG-DER 训练时间约 288 s；
- 基线约 32–119 s。

不能据此宣称方法优越。所有方法必须同预算、同代码 commit、同 split、同评估脚本重跑。

### 9.4 Informer 与 Transformer 是同一模型

已独立复核：

- 两者各 80 行；
- index 完全对齐；
- MAE/RMSE/SMAPE/R² 的最大绝对差为 0；
- 80/80 行完全相同；
- 各配置参数量也相同。

代码没有 Informer 的 ProbSparse attention、distilling 和 decoder。不得将同一实现作为两个基线。

DeepAR 和 N-BEATS 也只是简化网络，不应使用正式算法名称，除非换官方实现。

### 9.5 表格拼接和统计错误

- Table 4 的 Range、Best、Default 来自三套不同实验；
- sensitivity 文本称 H=96、5 seeds，代码却是 H=24、seed 42；
- t 列在结果文件中不存在；
- p 值按四个 horizon 算术平均，统计上无效；
- 多项结果、弹性和复杂度无法溯源；
- ETTm1 vs Naive 差值为负却被写成 SG-DER 显著获胜；
- 消融中移除 trend branch 反而全面更好；
- scaler 在全序列 fit，存在归一化泄漏。

### 9.6 可救的方向

现有数据做正确 Friedman 检验，SG-DER 平均秩可能仍为第一，说明论文不是完全没有正面结果。推荐：

1. 等预算重跑；
2. 用 4 datasets×4 horizons 做 Friedman+Nemenyi；
3. 逐 horizon 报告；
4. 增加时序 conformal：

$$
s_i^{(h)}=
\frac{|y_i^{(h)}-\hat\gamma_i^{(h)}|}
{\hat\sigma_i^{(h)}},
\quad
\hat q_{1-\alpha}^{(h)}
=s_{(\lceil(n+1)(1-\alpha)\rceil)}^{(h)};
$$

$$
C^{(h)}(x)=
[\hat\gamma^{(h)}-\hat q^{(h)}\hat\sigma^{(h)},
\hat\gamma^{(h)}+\hat q^{(h)}\hat\sigma^{(h)}].
$$

5. 将核心故事改为：

> Stop-gradient 改善部分点预测，但原始 NIG 区间严重欠覆盖；必须使用独立共形校准才能获得可防御的覆盖保证。

---

## 10. `43_Tourism_Recommend`

### 10.1 论文描述的模型与代码不是同一模型

主要差异：

- 论文 GeoHash，代码 KMeans cluster + coordinate MLP；
- 论文 social graph，代码是共同访问 Jaccard kNN；
- 论文带权重与非线性的图卷积，代码只有无参数 `sparse.mm`；
- 论文层级多跳传播，代码每层都从初始 embedding 重算；
- 论文 GAT 式跨图注意力，代码是普通点积注意力且作用于时空特征；
- 论文内积打分，代码两层 MLP；
- 论文加自环，代码未加；
- Theorem 2 的归约前提不适用于实际实现。

### 10.2 “Spatial-Temporal”模块是死代码

训练调用：

```python
model(users, pos_items)
```

没有传 temporal features、geo cluster 或 coords。评估的 `get_all_item_scores()` 甚至没有相应参数。因此 Time2Vec、GeoEmbedding、time/spatial attention 均不进入前向，不收到梯度。

`w/o Temporal` 与 Full 全指标相同不是负面发现，而是删除死代码后的恒等结果。标题、摘要、贡献、Highlights 与 cover letter 必须撤回或实现真实时空前向。

### 10.3 消融基准硬编码

`fix_figures.py` 明确手写：

```python
# Full STGC-CF results (seed=42, from sensitivity analysis embed_dim=64 run)
full_row = {...}
```

该行来自另一组敏感性实验，不是与消融变体同一运行。所有消融百分比都以这个外来数字为基准。

另外 `wo_user_user`/`wo_poi_poi` 用单位矩阵而不是零矩阵，结果是把 embedding 放大，而不是移除图。

### 10.4 t-SNE 图是随机数

`visualize.py` 明确：

```python
# Create random embeddings (simulating what the model would learn)
user_emb = np.random.randn(...)
item_emb = np.random.randn(...)
```

该图已输出到 `paper/figures/fig5_tsne.png`。必须删除，不得用于任何投稿或汇报。

### 10.5 统计检验描述虚假

论文称：

- paired t-test；
- 5-fold CV；
- df=4。

实际代码：

- 从 3 个 seed 的四位小数均值/std 反推；
- 使用非配对标准误；
- p 值用正态分布而非 t 分布；
- 全库没有 5-fold CV；
- 原始 per-seed 结果未保存。

正确用 $t_4$ 后，BPR-MF、NeuMF 的 p 值不满足论文所称 `<0.001`。

### 10.6 RecSys 专项问题

可保留：

- 采用全物品排序，而非 99 个采样负例；
- Recall/NDCG 定义基本标准；
- 对 NGCF 不显著有部分诚实披露。

必须修复：

- LightGCN 弱于 BPR，显示基线欠调；
- SASRec 全零来自异常捕获，不是有效实验；
- 缺 ItemKNN、EASE、SLIM、POI 专用模型；
- 只有所提方法享受约 19 次 test 调参；
- val 正例未从 test 候选中屏蔽；
- Yelp review 被当作 check-in；
- 1 星评价也当正反馈；
- POI category 被丢弃；
- 欧氏经纬度距离应改 Haversine；
- 没有冷启动、长尾和 exposure-bias 分析；
- 与 NGCF 差 0.0003，小于自身 std 0.0015。

### 10.7 推荐重构

若保留旅游主题，应加入通用推荐论文无法声称的约束：

1. 个性化距离衰减；
2. POI 级季节 Fourier bias；
3. 多日行程时间窗/预算约束；
4. IPS/SNIPS/DR 暴露偏差评估；
5. Feasible Recall、总旅行距离、约束违约率等旅游指标。

否则该工作只是较弱的 LightGCN/NGCF 变体。

---

## 11. 跨论文整改优先级

### 第一阶段：立即停止投稿（1–2 天）

- 删除/隔离所有模拟图；
- 标记硬编码表格；
- 冻结旧结果；
- 全量 DOI 核验；
- 明确每篇活动结果版本；
- 禁止继续使用当前 verifier 的“100/100”结论。

### 第二阶段：修复代码与评估协议（1–2 周）

- 38：真正特征 token attention；
- 41：真正 cross-source attention + 可导 direction loss；
- 43：让时空分支进入 forward；
- 15：删除 total_guests 泄漏任务；
- JX02：人工标签；
- 42：等预算基线与 proper scoring rules；
- 17：单一 checkpoint 家族；
- JX01：cluster bootstrap 与随机化因果评估。

### 第三阶段：统一重跑（预计 GPU 资源）

| 论文 | 建议最低重跑成本 |
|---|---:|
| JX01 | CPU 为主，数小时；跨学期数据抽取另计 |
| JX02 | 约 4–10 GPU 小时，人工标注为主要成本 |
| 15 | 约 20–35 GPU 小时 |
| 17 | 约 15–20 GPU 小时 |
| 38 | 约 20–30 GPU 小时 |
| 41 | 约 8–12 GPU 小时 |
| 42 | 约 40–80 GPU 小时，取决于 ensemble/rolling-origin |
| 43 | 约 12–16 GPU 小时 |

### 第四阶段：重写论文

重写时必须遵循：

- 摘要只使用当前活动结果；
- 不得使用 “first”“significantly”“well-calibrated”“robust” 等词，除非有直接、完整证据；
- 每个理论命题必须对应实际实现；
- 每个图必须能从原始结果一键重建；
- 所有主表必须包含 baseline、样本量、split、seed、CI；
- 负面结果必须完整报告；
- Limitations 不能替代正确实验；
- Cover letter、README、reproduce 与正文保持同一标题、方法、数据集和指标。

---

## 12. 最终建议

### 可优先挽救

1. **JX01**：数据真实，主要问题是推断层级和因果价值不足；最接近可发表。
2. **42**：虽然当前概率校准失败且理论动机错误，但正确的多数据集秩检验可能仍支持温和正结论。
3. **17**：代码中的邻域 Beta-Binomial 比论文写出的 m=1 版本更有创新潜力，统一版本后可重构。

### 适合转为“评估陷阱/负面结果”论文

1. **41**：随机切分如何夸大碳排放预测；
2. **38**：S=1 注意力退化与气象表格模型审计；
3. **JX02**：合成误概念基准与真实人工标签之间的构念差距。

### 需要从方法层重建

1. **15**：删除泄漏任务，最好改为取消时间生存模型；
2. **43**：实现真实时空推荐和旅游行程约束，否则标题与贡献无法成立。

当前最重要的不是继续润色文字或增加图表，而是先建立一条可靠的证据链：

```text
真实数据
→ 无泄漏切分
→ 与论文一致的模型实现
→ 公平基线
→ 原始逐种子结果
→ 正确统计
→ 自动生成表图
→ 当前稿件自动核验
```

在这条链闭合前，任何一篇都不应进入正式投稿流程。
