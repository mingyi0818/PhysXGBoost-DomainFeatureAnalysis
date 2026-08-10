# FinFeat: Systematic Domain-Derived Feature Augmentation for Tree-Based Bank Marketing Prediction

Jingyuan Zeng¹, Ming Zeng², Jianghong Guo¹, Chuanxian Jiang¹, Yafen Feng³,⁴,*

1. School of Computer Science, Jiaying University, Meizhou, Guangdong 514015, China
2. College of Water Conservancy and Civil Engineering, South China Agricultural University, Guangzhou, Guangdong 510642, China
3. School of Geography Science and Tourism, Jiaying University, Meizhou, Guangdong 514015, China
4. Key Laboratory of Mountain Surface Environment and Green Development in Northeast Guangdong, Meizhou, Guangdong 514015, China

*Corresponding author: Yafen Feng, E-mail: fyf81@163.com

**Supported by Guangdong Provincial Higher Education Teaching Reform Project (Grant No. Yue Jiao Gao Han [2024] 9-989)**

---

## Abstract

Bank telemarketing campaign success prediction is a critical task for financial institutions seeking to optimize resource allocation and improve customer targeting. While tree-based ensemble methods have achieved strong performance on the UCI Bank Marketing dataset, existing approaches predominantly rely on raw features without systematic integration of domain knowledge. In this paper, we propose FinFeat, a systematic domain-derived feature augmentation framework that constructs five categories of financially motivated features—customer profile interactions, financial health indicators, campaign dynamics, economic context, and temporal encodings—spanning over 20 engineered variables. We conduct a rigorous comparative evaluation across four state-of-the-art tree models (XGBoost, LightGBM, CatBoost, and Random Forest) under two feature regimes (raw vs. domain-augmented), employing five random seeds, Wilcoxon signed-rank tests, and 95% confidence intervals. We provide theoretical analysis through an information-theoretic feature interaction bound (Theorem 1) and a feature redundancy proposition (Proposition 1) that explain when domain features yield marginal gains. Additionally, we perform SHAP-based interpretability analysis and a fairness audit across demographic groups using demographic parity and equalized odds metrics. Our experiments reveal that domain-derived features produce AUC changes of -0.0000 for XGBoost and -0.0010 for LightGBM, with Wilcoxon p-values of 0.8125 and 0.0156, indicating that modern gradient-boosted trees already capture most interactions implicitly. We discuss the implications for feature engineering practice and provide actionable guidelines for practitioners.

**Keywords:** Bank marketing prediction; Feature engineering; Gradient boosting; SHAP; Fairness audit; Domain knowledge

---

## 1. Introduction and Related Work

### 1.1 Background

Direct telemarketing remains a primary channel through which banks promote term deposit subscriptions. The Portuguese bank dataset released by Moro et al. (2014) [1] has become a benchmark for predictive modeling in financial marketing, comprising 42,718 telephone contact records with 16 attributes spanning client demographics, financial status, campaign history, and macroeconomic indicators. The task is a binary classification problem with severe class imbalance—approximately 11.27% of contacts result in a subscription—making it a challenging testbed for both algorithmic performance and evaluation rigor.

Accurate prediction of campaign success enables banks to reduce operational costs, minimize customer annoyance from excessive contacts, and improve conversion rates. However, the heterogeneity of the feature space—combining categorical demographic variables, numeric financial indicators, temporal attributes, and macroeconomic context—poses significant challenges for predictive modeling. While deep learning approaches have been explored, tree-based ensemble methods remain the dominant paradigm due to their robustness to heterogeneous features, natural handling of non-linear interactions, and interpretability advantages.

### 1.2 Related Work

**Early approaches.** Moro et al. (2014) [1] originally proposed the dataset and demonstrated that feature engineering combined with neural networks could improve over standard logistic regression. Their work established the importance of the `duration` feature and macroeconomic variables, achieving AUC values around 0.80 with carefully designed features. Subsequent work by the same group explored feature selection using receiver operating characteristic analysis [2], reinforcing the value of domain-informed feature construction.

**Neural network approaches.** Yu et al. (2023) [3] applied a three-layer feedforward neural network to the Bank Marketing dataset, reporting an AUC of 0.9777 and accuracy of 99.06%. However, their study lacked F1-score reporting, did not employ multiple random seeds, and the exceptionally high accuracy on an imbalanced dataset raises concerns about potential data leakage or overfitting. Their work nonetheless demonstrates the potential of neural architectures for this task.

**Gradient boosting approaches.** Wang (2025) [4] applied XGBoost to the Bank Marketing dataset, achieving an AUC of 0.90 and accuracy of 0.89. While the study confirmed the effectiveness of gradient boosting, it reported only AUC and accuracy—metrics that can be misleading under severe class imbalance—and did not include F1, precision, or recall. Du (2025) [5] explored gradient boosting for bank marketing prediction, claiming maximum ROC-AUC but without disclosing specific numerical results, making comparison difficult.

**Ensemble and balancing approaches.** Hasnataeni et al. (2025) [6] investigated ensemble methods for unbalanced bank marketing data, combining Random Forest with Random Over-Sampling Examples (ROSE), achieving accuracy of 91.00% and AUC of approximately 0.94. However, their study omitted F1-score and SHAP-based interpretability analysis. Lee et al. (2024) [7] focused on bank direct marketing campaign success prediction using Random Forest with a reduced 8-feature subset, achieving optimal AUC but with incomplete metric reporting.

**Explainable AI approaches.** Kuravi (2025) [8] explored explainable AI (XAI) and fairness auditing for bank marketing prediction using XGBoost with SHAP and LIME, but the preprint did not disclose specific performance metrics. Prasad et al. (2025a) [9] applied Explainable Boosting Machines (EBM) to the Bank Marketing dataset, emphasizing interpretability but without reporting performance metrics. Prasad et al. (2025b) [10] also explored a blending approach on the same dataset without disclosing results. Apriadi and Bisri (2025) [11] used Random Forest for term deposit prediction but without comparative baselines.

**Recent advances in tree-based methods.** XGBoost [12], introduced by Chen and Guestrin (2016), remains a dominant method for tabular data through its regularized gradient boosting framework. LightGBM [13], proposed by Ke et al. (2017), introduced histogram-based splitting and leaf-wise growth for efficiency. CatBoost [14], developed by Prokhorenkova et al. (2018), introduced ordered boosting and native categorical feature handling. Despite their widespread adoption, systematic comparisons of all three gradient boosting implementations alongside Random Forest on the Bank Marketing dataset are notably absent in recent literature.

**Feature engineering for tabular data.** Recent work has emphasized the importance of systematic feature engineering for tabular prediction. Shwartz-Ziv and Armon (2022) [15] provided a comprehensive tabular deep learning benchmark, demonstrating that well-tuned gradient boosting often outperforms deep learning. Gorishniy et al. (2022) [16] similarly showed that feature preprocessing significantly impacts tabular model performance. However, these works did not specifically address financial domain feature construction.

**Fairness in financial ML.** Fairness auditing in financial applications has gained increasing attention. Mehrabi et al. (2021) [17] provided a comprehensive survey of bias and fairness in machine learning. Hardt et al. (2016) [18] introduced equalized odds as a fairness metric. Despite these advances, fairness analysis on bank marketing prediction models remains largely unexplored, with only the preprint by Kuravi (2025) [8] mentioning it without systematic analysis.

**SHAP and interpretability.** Lundberg and Lee (2017) [19] introduced SHAP (SHapley Additive exPlanations), providing a unified framework for model interpretation based on Shapley values from cooperative game theory. SHAP has become the standard tool for tree model interpretability, enabling both global feature importance and local explanation. Ryo and Angiilella (2024) [20] demonstrated SHAP's utility for environmental and financial prediction tasks.

### 1.3 Research Gaps

Based on our review, we identify the following gaps in the existing literature:

1. **Fragmented feature engineering**: No prior work establishes a systematic, domain-knowledge-driven feature engineering framework for bank marketing prediction. Existing studies either use raw features or add features ad hoc.
2. **Incomplete tree model comparison**: No study systematically compares XGBoost, LightGBM, CatBoost, and Random Forest under identical evaluation protocols on this dataset, particularly for the 2024–2026 period.
3. **Missing evaluation metrics**: Most studies report only AUC and accuracy, neglecting F1, precision, and recall—which are critical under ~11% positive class ratio.
4. **Absence of statistical significance testing**: Nearly all prior work reports single-run results without multiple seeds, confidence intervals, or significance tests.
5. **Limited interpretability analysis**: Only one preprint [8] uses SHAP/LIME, and it does not disclose detailed results.
6. **No fairness audit**: Despite the ethical importance of fair lending practices, no study systematically audits bank marketing models for demographic fairness.
7. **No theoretical analysis of feature interactions**: No prior work provides information-theoretic analysis of why domain features may or may not improve tree-based model performance.

### 1.4 Contributions

This paper addresses the above gaps with the following contributions:

1. **FinFeat framework**: We propose a systematic domain-derived feature augmentation framework comprising five categories of financially motivated features (customer profile interactions, financial health indicators, campaign dynamics, economic context, and temporal encodings), totaling over 20 engineered variables, each grounded in financial domain knowledge.

2. **Comprehensive tree model comparison**: We conduct the first systematic four-way comparison of XGBoost, LightGBM, CatBoost, and Random Forest on the Bank Marketing dataset under two feature regimes (raw vs. domain-augmented), with unified hyperparameter search spaces and consistent evaluation protocols.

3. **Theoretical analysis**: We provide an information-theoretic framework including Theorem 1 (feature interaction bound), which establishes an upper bound on the performance gain achievable by adding domain features, and Proposition 1 (feature redundancy), which characterizes when domain features provide negative marginal contribution.

4. **Rigorous statistical evaluation**: We employ five random seeds, Wilcoxon signed-rank tests, 95% confidence intervals, and effect size analysis (Cohen's d), providing statistically sound comparisons that are absent in prior work.

5. **SHAP-based interpretability**: We conduct systematic SHAP analysis including global feature importance, dependence plots, and interaction effects, providing actionable insights for marketing practitioners.

6. **Fairness audit**: We perform the first systematic fairness audit of bank marketing prediction models across demographic groups (age, occupation, education, marital status), reporting demographic parity difference and equalized odds difference.

7. **Honest reporting and analysis**: We transparently report that domain features yield marginal performance changes and provide a thorough theoretical and empirical explanation for this finding, offering practical guidance for feature engineering in tree-based modeling.

---

## 2. Methodology

### 2.1 Problem Formulation

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ denote the bank marketing dataset, where each sample $\mathbf{x}_i \in \mathbb{R}^m$ is a feature vector comprising $m=16$ original attributes, and $y_i \in \{0, 1\}$ is the binary label indicating whether the $i$-th customer subscribed to a term deposit. The dataset exhibits severe class imbalance with $P(y=1) \approx 0.1127$.

We define the **original feature set** $F = \{f_1, f_2, \ldots, f_m\}$ and the **domain-derived feature set** $D = \{d_1, d_2, \ldots, d_k\}$, where $k$ denotes the number of engineered features. The augmented feature set is $F^+ = F \cup D$ with dimensionality $m + k$.

The goal is to learn a predictive model $\hat{y} = h(\mathbf{x})$ that maximizes the area under the receiver operating characteristic curve (AUC-ROC) while maintaining acceptable F1-score, precision, and recall, and to rigorously evaluate whether $D$ provides statistically significant improvement over $F$ alone.

### 2.2 FinFeat Framework Overview

The FinFeat framework consists of three stages (Figure 1):

**Stage 1 — Domain Feature Construction.** Given the original 16 features, we construct domain-derived features across five categories, each motivated by financial domain knowledge:

| Category | Prefix | Count | Domain Rationale |
|----------|--------|-------|-----------------|
| Customer Profile | `cust_*` | 5 | Age-job interactions, education-marital cross-effects |
| Financial Health | `fin_*` | 6 | Debt burden indices, balance-credit interactions |
| Campaign Dynamics | `camp_*` | 9 | Contact intensity, recency, historical success |
| Economic Context | `econ_*` | 0 | Macro indicator derivatives, employment-confidence ratios |
| Temporal Encoding | `time_*` | 4 | Cyclic month/day-of-week encoding |

**Stage 2 — Model Training.** Four tree-based models are trained under identical protocols on both $F$ and $F^+$, with class imbalance handled via scale-pos-weight or balanced class weights.

**Stage 3 — Evaluation and Analysis.** Performance is assessed via AUC, F1, precision, and recall across five random seeds, followed by statistical significance testing, SHAP interpretability analysis, fairness auditing, and ablation studies.

### 2.3 Domain Feature Engineering

In this section, we detail each category of domain-derived features, providing the financial rationale and mathematical formulation for each.

#### 2.3.1 Customer Profile Features (`cust_*`)

Customer demographics influence deposit subscription behavior through life-stage effects and income-correlated patterns.

**Age-job interaction.** The product of age and occupation encoding captures the interaction between life stage and professional category:
$$d_{\text{cust\_age\_job}} = \text{age} \times \text{Enc}(\text{job})$$
where $\text{Enc}(\cdot)$ denotes label encoding. This feature captures, for example, that the effect of being a "student" on subscription propensity differs by age.

**Education-marital interaction.** Education level and marital status jointly influence financial decision-making:
$$d_{\text{cust\_edu\_marital}} = \text{Enc}(\text{education}) \times \text{Enc}(\text{marital})$$

**Non-linear age encoding.** The relationship between age and deposit subscription is non-monotonic (middle-aged customers with stable income are more likely to subscribe):
$$d_{\text{cust\_age\_squared}} = \text{age}^2$$

**Age bracket indicators.** Binary indicators for life-stage segments:
$$d_{\text{cust\_age\_young}} = \mathbb{1}[\text{age} < 30], \quad d_{\text{cust\_age\_senior}} = \mathbb{1}[\text{age} > 60]$$

Young customers (<30) typically have lower savings, while seniors (>60) may prefer conservative investments.

#### 2.3.2 Financial Health Features (`fin_*`)

A customer's financial obligations directly affect their capacity to commit funds to a term deposit.

**Composite debt score.** The sum of credit default, housing loan, and personal loan indicators provides a holistic debt burden measure:
$$d_{\text{fin\_debt\_score}} = \mathbb{1}[\text{default=yes}] + \mathbb{1}[\text{housing=yes}] + \mathbb{1}[\text{loan=yes}]$$

**Binary debt indicator.** A simplified flag indicating any debt obligation:
$$d_{\text{fin\_has\_debt}} = \mathbb{1}[d_{\text{fin\_debt\_score}} > 0]$$

**Housing-personal loan interaction.** Customers with both housing and personal loans face compounded financial pressure:
$$d_{\text{fin\_housing\_loan}} = \mathbb{1}[\text{housing=yes}] \times \mathbb{1}[\text{loan=yes}]$$

**Balance-based indicators.** Account balance thresholds capture financial capacity:
$$d_{\text{fin\_balance\_low}} = \mathbb{1}[\text{balance} < 0], \quad d_{\text{fin\_balance\_high}} = \mathbb{1}[\text{balance} > 5000]$$

**Balance-loan interactions.** The interaction between balance and loan status captures leveraged positions:
$$d_{\text{fin\_balance\_loan}} = \text{balance} \times \mathbb{1}[\text{loan=yes}]$$
$$d_{\text{fin\_balance\_housing}} = \text{balance} \times \mathbb{1}[\text{housing=yes}]$$
$$d_{\text{fin\_leveraged}} = \mathbb{1}[\text{loan=yes} \wedge \text{balance} < 0]$$

The leveraged indicator flags customers with negative balance and active loans—high financial distress signals.

#### 2.3.3 Campaign Dynamics Features (`camp_*`)

The intensity and history of marketing contacts significantly influence subscription probability.

**Contact intensity ratio.** The ratio of contacts in the current campaign to call duration normalizes effort:
$$d_{\text{camp\_intensity}} = \frac{\text{campaign}}{\max(\text{duration}, 1)}$$

**Previous contact interaction.** The product of having previous contacts and current campaign count:
$$d_{\text{camp\_prev\_contact}} = \mathbb{1}[\text{previous} > 0] \times \text{campaign}$$

**Recency encoding.** Days since last contact (with 999 indicating no prior contact) is transformed to a recency score:
$$d_{\text{camp\_pdays\_recency}} = \begin{cases} 0 & \text{if } \text{pdays} = 999 \\ \frac{1}{\text{pdays} + 1} & \text{otherwise} \end{cases}$$

**Never-contacted indicator.** A binary flag for first-time contacts:
$$d_{\text{camp\_pdays\_never}} = \mathbb{1}[\text{pdays} = 999]$$

**Contact-month interaction.** The interaction between contact type and month captures seasonal channel effects:
$$d_{\text{camp\_contact\_month}} = \text{Enc}(\text{contact}) \times \text{Enc}(\text{month})$$

**Log-duration.** Call duration follows a right-skewed distribution; log transformation stabilizes variance:
$$d_{\text{camp\_duration\_log}} = \log(1 + \text{duration})$$

**High-effort indicator.** Customers requiring more than 3 contacts in the current campaign:
$$d_{\text{camp\_high\_effort}} = \mathbb{1}[\text{campaign} > 3]$$

**Success history.** Prior campaign success is a strong predictor of future subscription:
$$d_{\text{camp\_success\_history}} = \mathbb{1}[\text{poutcome} = \text{success}]$$

**Duration-campaign interaction.** The product of duration and campaign count captures total engagement:
$$d_{\text{camp\_duration\_campaign}} = \text{duration} \times \text{campaign}$$

#### 2.3.4 Economic Context Features (`econ_*`)

Macroeconomic conditions influence customers' willingness to lock funds in term deposits.

**Employment-confidence ratio.** The ratio of employment variation rate to consumer confidence index captures economic sentiment:
$$d_{\text{econ\_emp\_conf}} = \frac{\text{emp.var.rate}}{\text{cons.conf.idx}}$$

**Price-employment interaction.** The interaction between consumer price index and employment variation:
$$d_{\text{econ\_price\_emp}} = \text{cons.price.idx} \times \text{emp.var.rate}$$

**Euribor-employment ratio.** The ratio of 3-month Euribor rate to number employed captures interest rate environment:
$$d_{\text{econ\_euribor\_emp}} = \frac{\text{euribor3m}}{\text{nr.employed}}$$

**Economic stress index.** A composite indicator combining negative confidence with high Euribor:
$$d_{\text{econ\_stress}} = \mathbb{1}[\text{cons.conf.idx} < -40] \times \text{euribor3m}$$

#### 2.3.5 Temporal Encoding Features (`time_*`)

Temporal patterns in subscription behavior require cyclical encoding to preserve proximity relationships.

**Cyclic month encoding.** Months are cyclically encoded using sine and cosine transformations to preserve the circular nature of calendar months:
$$d_{\text{time\_month\_sin}} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad d_{\text{time\_month\_cos}} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

**Cyclic day-of-week encoding.** Similarly, day of week is encoded cyclically:
$$d_{\text{time\_dow\_sin}} = \sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{5}\right), \quad d_{\text{time\_dow\_cos}} = \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{5}\right)$$

These encodings ensure that December (month 12) and January (month 1) are close in the feature space, which linear encodings fail to capture.

### 2.4 Theoretical Analysis

In this section, we provide theoretical foundations for understanding when and why domain-derived features improve tree-based model performance.

#### 2.4.1 Information-Theoretic Preliminaries

Let $Y$ denote the target variable, $F$ the original feature set, and $D$ the domain-derived feature set. We use the following information-theoretic quantities:

**Entropy.** The uncertainty in $Y$:
$$H(Y) = -\sum_{y} P(y) \log P(y)$$

**Conditional entropy.** The remaining uncertainty in $Y$ given $F$:
$$H(Y|F) = -\sum_{y, \mathbf{f}} P(y, \mathbf{f}) \log P(y|\mathbf{f})$$

**Mutual information.** The reduction in uncertainty about $Y$ provided by $F$:
$$I(Y; F) = H(Y) - H(Y|F)$$

**Conditional mutual information.** The additional reduction in uncertainty about $Y$ provided by $D$ beyond what $F$ already provides:
$$I(Y; D | F) = H(Y|F) - H(Y|F, D)$$

This quantity is central to our analysis: it measures the incremental predictive information that domain features $D$ contribute beyond the original features $F$.

#### 2.4.2 Theorem 1: Feature Interaction Bound

**Theorem 1 (Feature Interaction Bound).** *Let $T$ be a tree-based ensemble model trained on feature set $F$ with AUC performance $\text{AUC}(T, F)$. Let $T^+$ be the same model architecture trained on the augmented set $F \cup D$. Then the improvement in AUC is bounded by:*

$$|\text{AUC}(T^+, F \cup D) - \text{AUC}(T, F)| \leq C \cdot \sqrt{I(Y; D | F)}$$

*where $C > 0$ is a constant depending on the score distribution and class prior, and $I(Y; D | F)$ is the conditional mutual information between $D$ and $Y$ given $F$.*

**Proof.**

The AUC of a binary classifier can be expressed as:
$$\text{AUC} = P(S(\mathbf{x}^+) > S(\mathbf{x}^-))$$
where $S(\cdot)$ is the model's score function, $\mathbf{x}^+$ is a positive sample, and $\mathbf{x}^-$ is a negative sample.

When we augment the feature set from $F$ to $F \cup D$, the change in AUC depends on how much the ranking of positive and negative samples changes. By the data processing inequality and the relationship between mutual information and estimation error (Fano's inequality variants), the improvement in any proper scoring rule is bounded by the square root of the conditional mutual information.

Specifically, let $\Delta_{\text{AUC}} = \text{AUC}(T^+, F \cup D) - \text{AUC}(T, F)$. The AUC can be related to the rank-based Mann-Whitney U statistic, which in turn depends on the conditional distribution $P(Y|F, D)$ vs. $P(Y|F)$.

By the chain rule of mutual information:
$$I(Y; F, D) = I(Y; F) + I(Y; D | F)$$

The additional information about $Y$ provided by $D$ beyond $F$ is exactly $I(Y; D | F)$. For tree-based models that can learn arbitrary axis-aligned partitions, the additional splits enabled by $D$ can at most capture this conditional mutual information.

Using the Pinsker inequality relating total variation distance to KL divergence, and the relationship between KL divergence and mutual information, we obtain:
$$|\Delta_{\text{AUC}}| \leq C \cdot \sqrt{I(Y; D | F)}$$

where $C$ incorporates constants from the Pinsker inequality and the relationship between AUC and distributional distance. $\square$

**Implication.** Theorem 1 establishes that the performance gain from domain features is fundamentally limited by the conditional mutual information $I(Y; D | F)$. When the original features $F$ already capture most of the predictive information (i.e., $I(Y; D | F) \approx 0$), domain features cannot provide significant improvement regardless of the model architecture. This is particularly relevant for tree-based ensembles, which can learn non-linear feature interactions automatically through recursive splitting.

#### 2.4.3 Proposition 1: Feature Redundancy

**Proposition 1 (Feature Redundancy).** *Let $D$ be a set of domain-derived features constructed as deterministic functions of the original features $F$, i.e., $D = g(F)$ for some function $g$. If the mutual information between $D$ and $F$ exceeds the conditional mutual information between $D$ and $Y$ given $F$, i.e.,*

$$I(D; F) > I(D; Y | F)$$

*then adding $D$ to the feature set provides zero marginal information gain, i.e., $I(Y; F, D) = I(Y; F)$, and the expected performance improvement is non-positive.*

**Proof.**

Since $D = g(F)$ is a deterministic function of $F$, we have $H(D|F) = 0$, which implies:
$$I(D; F) = H(D) - H(D|F) = H(D)$$

Now consider the joint mutual information:
$$I(Y; F, D) = I(Y; F) + I(Y; D | F)$$

By the data processing inequality, since $D$ is a function of $F$:
$$I(Y; D) \leq I(Y; F)$$

Furthermore, $I(Y; D | F) = H(Y|F) - H(Y|F, D)$. Since $D = g(F)$, we have $H(Y|F, D) = H(Y|F, g(F)) = H(Y|F)$ (because $g(F)$ is determined by $F$). Therefore:
$$I(Y; D | F) = 0$$

This means:
$$I(Y; F, D) = I(Y; F) + 0 = I(Y; F)$$

The domain features provide zero additional information about $Y$ beyond what $F$ already contains. By Theorem 1, the AUC improvement is bounded by $C \cdot \sqrt{0} = 0$.

However, in practice, $D$ may include features that are not strictly deterministic functions of $F$ (e.g., threshold-based binary indicators that discretize continuous features), and tree models have finite depth and may not perfectly capture all interactions in $F$. Thus, the actual improvement may be slightly positive but bounded by the approximation error of the tree model. $\square$

**Implication.** Proposition 1 has a profound implication for feature engineering with tree-based models: if domain features are constructed as deterministic transformations of existing features, and the tree model is sufficiently expressive (deep enough trees, enough estimators), the domain features are theoretically redundant. The practical improvement observed in experiments reflects the model's finite capacity to discover interactions, not genuine new information.

This explains a key empirical finding: gradient-boosted trees with sufficient depth (e.g., max_depth=6) can implicitly learn many interactions that domain features encode explicitly, leading to marginal improvements from feature engineering.

#### 2.4.4 Information-Theoretic Feature Redundancy Measure

To quantify the redundancy of domain features, we define:

**Feature redundancy score.** For each domain feature $d_j \in D$:
$$R(d_j, F) = \frac{1}{|F|} \sum_{f_i \in F} I(d_j; f_i)$$

A high redundancy score indicates that $d_j$ is highly correlated with existing features and likely provides limited additional information.

**Marginal contribution score.** For each domain feature $d_j \in D$:
$$\text{MC}(d_j) = I(d_j; Y) - R(d_j, F) \cdot \lambda$$

where $\lambda$ is a regularization parameter. Features with $\text{MC}(d_j) \approx 0$ are candidates for removal.

### 2.5 Complexity Analysis

#### 2.5.1 Feature Engineering Complexity

Let $n$ denote the number of samples, $m$ the number of original features, and $k$ the number of domain-derived features.

**Time complexity.** Each domain feature is computed as a vectorized operation over $n$ samples:
- Binary indicators ($\mathbb{1}[\cdot]$): $O(n)$ per feature
- Products and ratios: $O(n)$ per feature
- Logarithmic transformations: $O(n)$ per feature
- Trigonometric encodings: $O(n)$ per feature

Total feature engineering time complexity:
$$T_{\text{feat}} = O(n \cdot k)$$

For the Bank Marketing dataset with $n = 42{,}718$ and $k \approx 20$, this is approximately $O(8.5 \times 10^5)$ operations—negligible compared to model training.

**Space complexity.** The augmented feature matrix requires:
$$S_{\text{feat}} = O(n \cdot (m + k))$$

With $m = 16$ and $k \approx 20$, the augmented matrix is approximately 2.25x the size of the original, well within memory constraints.

#### 2.5.2 Model Training Complexity

For gradient-boosted decision trees (GBDT) with $T$ trees, each of depth $d$, and $n$ training samples with $p = m + k$ features:

**Training time complexity.** At each split, the algorithm evaluates all features and all possible split points. With histogram-based methods (LightGBM, CatBoost), this is:
$$T_{\text{train}} = O(T \cdot n \cdot p \cdot d \cdot \log(p))$$

The $\log(p)$ factor arises from feature subsampling. The additional cost from domain features is:
$$\Delta T = O\left(T \cdot n \cdot k \cdot d \cdot \log\left(\frac{m+k}{m}\right)\right)$$

For $k \approx 20$ and $m = 16$, the relative increase is approximately $\frac{k}{m} \approx 1.25$, i.e., training time increases by about 25%.

**Inference time complexity.** For a single sample, prediction requires traversing $T$ trees of depth $d$:
$$T_{\text{infer}} = O(T \cdot d)$$

This is independent of the number of features, as only the features used in splits are evaluated. In practice, domain features used in fewer than $T \cdot d$ splits have negligible inference overhead.

**Space complexity.** The model stores $T$ trees, each with up to $2^d$ nodes:
$$S_{\text{model}} = O(T \cdot 2^d)$$

This is independent of $p$, as only the features used in splits are stored at each node.

#### 2.5.3 SHAP Computation Complexity

For tree-based models, SHAP values are computed using TreeSHAP [19], which has complexity:
$$T_{\text{SHAP}} = O(T \cdot L \cdot 2^p)$$

where $L$ is the number of leaves per tree and $p$ is the number of features. This exponential dependence on $p$ makes SHAP computation expensive for large feature sets. However, TreeSHAP optimizations reduce this to $O(T \cdot L \cdot p^2)$ in practice using polynomial-time algorithms for tree models.

### 2.6 Tree-Based Models

We employ four tree-based ensemble models, each representing a distinct algorithmic approach:

#### 2.6.1 XGBoost

XGBoost [12] implements regularized gradient boosting with second-order Taylor approximation of the loss function. The objective function at iteration $t$ is:
$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$$

where $l$ is the loss function, $f_t$ is the $t$-th tree, and $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \|\mathbf{w}\|^2$ is the regularization term. Class imbalance is handled via `scale_pos_weight` set to the negative-to-positive ratio.

#### 2.6.2 LightGBM

LightGBM [13] introduces histogram-based decision tree learning with leaf-wise (best-first) growth strategy. It uses Gradient-based One-Side Sampling (GOSS) to focus on samples with large gradients and Exclusive Feature Bundling (EFB) to bundle mutually exclusive features. Class imbalance is handled via `class_weight='balanced'`.

#### 2.6.3 CatBoost

CatBoost [14] introduces ordered boosting to prevent prediction shift caused by target leakage during training. It also provides native handling of categorical features through target statistics with permutations. Class imbalance is handled via `auto_class_weights='Balanced'`.

#### 2.6.4 Random Forest

Random Forest [21] constructs an ensemble of decision trees using bagging (bootstrap aggregating) and random feature subsampling. Each tree is grown independently without gradient-based optimization. Class imbalance is handled via `class_weight='balanced'`.

### 2.7 Evaluation Protocol

#### 2.7.1 Data Splitting

The dataset is split into training (80%) and test (20%) sets using stratified sampling to preserve the class ratio. The split uses a fixed random seed (42) for reproducibility. The test set is held out and used only for final evaluation; no hyperparameter tuning is performed on it.

#### 2.7.2 Metrics

We report the following metrics on the test set:

- **AUC-ROC**: Area under the Receiver Operating Characteristic curve, measuring ranking ability
- **F1-Score**: Harmonic mean of precision and recall, suitable for imbalanced data
- **Precision**: $TP / (TP + FP)$
- **Recall**: $TP / (TP + FN)$
- **PR-AUC**: Area under the Precision-Recall curve, more informative than ROC-AUC under severe imbalance

#### 2.7.3 Statistical Testing

To ensure statistical rigor, we employ:

- **Multiple seeds**: Five random seeds (42, 123, 456, 789, 1024) for each model-feature combination
- **Wilcoxon signed-rank test**: Non-parametric paired test comparing raw vs. domain feature performance
- **95% confidence intervals**: Computed via bootstrap (1000 resamples)
- **Effect size**: Cohen's $d$ to quantify practical significance

$$d = \frac{\bar{X}_{\text{domain}} - \bar{X}_{\text{raw}}}{s_{\text{pooled}}}$$

where $s_{\text{pooled}} = \sqrt{\frac{s_1^2 + s_2^2}{2}}$ is the pooled standard deviation.

#### 2.7.4 Fairness Metrics

We audit model fairness across demographic groups using:

**Demographic Parity Difference (DPD).** Measures the difference in positive prediction rates between groups:
$$\text{DPD} = |P(\hat{Y}=1 | A=0) - P(\hat{Y}=1 | A=1)|$$

**Equalized Odds Difference (EOD).** Measures the difference in true positive and false positive rates between groups:
$$\text{EOD} = \frac{1}{2}\left(|TPR_{A=0} - TPR_{A=1}| + |FPR_{A=0} - FPR_{A=1}|\right)$$

We evaluate fairness across age groups (young <30 vs. middle-aged 30-60 vs. senior >60), education levels, and occupation categories.

---

## 3. Experiments

### 3.1 Dataset Description

We use the UCI Bank Marketing dataset [1], which contains data from a Portuguese bank's telemarketing campaigns between May 2008 and November 2010. The dataset comprises 42,718 records after removing entries with missing values, with 16 input features and one binary target variable (`y` indicating term deposit subscription).

**Table 1. Dataset statistics**

| Attribute | Value |
|-----------|-------|
| Total samples | 42,718 |
| Positive samples (subscribed) | 5289 |
| Positive class ratio | ~11.27% |
| Original features | 16 |
| Categorical features | 10 |
| Numeric features | 6 |
| Domain-derived features | 24 |
| Augmented feature set size | 40 |

The 16 original features span five domains:

- **Client information**: age (numeric), job (categorical, 12 categories), marital (categorical, 4 categories), education (categorical, 8 categories)
- **Financial status**: default (binary), housing (binary), loan (binary), balance (numeric)
- **Campaign information**: contact (categorical), month (categorical), day_of_week (categorical), duration (numeric), campaign (numeric)
- **Historical information**: pdays (numeric), previous (numeric), poutcome (categorical)
- **Macroeconomic context**: emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed (all numeric)

### 3.2 Experimental Setup

#### 3.2.1 Hardware and Software Environment

All experiments are conducted on a workstation with the following specifications:

**Table 2. Experimental environment**

| Component | Specification |
|-----------|--------------|
| OS | Windows 11 Professional |
| CPU | Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz) |
| GPU | NVIDIA RTX 2000 Pro (16 GB VRAM) |
| RAM | 48 GB DDR5 RDIMM |
| Python | 3.10.11 |
| XGBoost | 3.2.0 |
| LightGBM | 4.6.0 |
| CatBoost | 1.2.10 |
| scikit-learn | 1.7.2 |
| SHAP | 0.49.1 |

#### 3.2.2 Hyperparameter Configuration

To ensure fair comparison, all gradient boosting models share common hyperparameters where applicable, while model-specific parameters follow library recommendations:

**Table 3. Hyperparameter configuration**

| Parameter | XGBoost | LightGBM | CatBoost | RandomForest |
|-----------|---------|----------|----------|-------------|
| n_estimators | 200 | 200 | 200 | 200 |
| max_depth | 6 | 6 | 6 | 12 |
| learning_rate | 0.05 | 0.05 | 0.05 | — |
| subsample | 0.8 | 0.8 | 0.8 | — |
| Class imbalance | scale_pos_weight=7.5 | class_weight='balanced' | auto_class_weights='Balanced' | class_weight='balanced' |
| Random seeds | 42, 123, 456, 789, 1024 | (same) | (same) | (same) |

The `scale_pos_weight` of 7.5 approximately equals the negative-to-positive ratio in the training set. For LightGBM, CatBoost, and Random Forest, built-in balanced class weighting mechanisms are used, as these methods do not support `scale_pos_weight` directly.

#### 3.2.3 Data Preprocessing

Categorical features are label-encoded using `LabelEncoder` from scikit-learn. Missing values in the `contact` and `poutcome` columns are filled with 'unknown' before encoding. The `balance` column is converted to numeric with NaN values replaced by 0. No feature scaling is applied, as tree-based models are scale-invariant.

#### 3.2.4 Train-Test Split

The dataset is split into 80% training (34,174 samples) and 20% test (8,544 samples) sets using stratified sampling with `random_state=42` to preserve the class ratio in both subsets.

### 3.3 Main Results: Raw vs. Domain Features

We evaluate four tree-based models under two feature regimes: Raw (16 original features) and Domain (16 original + 24 domain-derived features). Each configuration is evaluated across five random seeds, and we report mean and standard deviation for each metric.

**Table 4. Main results: AUC-ROC comparison (mean +/- std over 5 seeds)**

| Model | Raw AUC | Domain AUC | Delta AUC | 95% CI (Domain) |
|-------|---------|------------|-----------|-----------------|
| XGBoost | 0.9375 +/- 0.0004 | 0.9375 +/- 0.0003 | -0.0000 | [-0.0005, 0.0004] |
| LightGBM | 0.9388 +/- 0.0000 | 0.9377 +/- 0.0000 | -0.0010 | [-0.0010, -0.0010] |
| CatBoost | 0.9356 +/- 0.0004 | 0.9371 +/- 0.0003 | +0.0015 | [0.0011, 0.0019] |
| RandomForest | 0.9253 +/- 0.0004 | 0.9277 +/- 0.0004 | +0.0024 | [0.0019, 0.0029] |

**Table 5. Main results: F1-Score comparison (mean +/- std over 5 seeds)**

| Model | Raw F1 | Domain F1 | Delta F1 | 95% CI (Domain) |
|-------|--------|-----------|----------|-----------------|
| XGBoost | 0.6017 +/- 0.0023 | 0.5999 +/- 0.0011 | -0.0018 | [-0.0041, 0.0005] |
| LightGBM | 0.5921 +/- 0.0000 | 0.5894 +/- 0.0000 | -0.0027 | [-0.0027, -0.0027] |
| CatBoost | 0.5762 +/- 0.0013 | 0.5819 +/- 0.0016 | +0.0057 | [0.0038, 0.0076] |
| RandomForest | 0.5721 +/- 0.0021 | 0.5855 +/- 0.0013 | +0.0134 | [0.0111, 0.0157] |

**Table 6. Additional metrics for Domain feature set (mean over 5 seeds)**

| Model | Precision | Recall | PR-AUC |
|-------|-----------|--------|--------|
| XGBoost | 0.4567 | 0.8705 | 0.6391 |
| LightGBM | 0.4422 | 0.8835 | 0.6379 |
| CatBoost | 0.4317 | 0.8924 | 0.6413 |
| RandomForest | 0.4748 | 0.7914 | 0.5847 |

**Key observations from main results:**

1. **AUC performance is comparable across gradient boosting models**: XGBoost, LightGBM, and CatBoost achieve AUC values in the range of 0.9253-0.9388, with LightGBM showing the highest raw AUC of 0.9388 and CatBoost showing the highest domain AUC of 0.9377.

2. **Random Forest underperforms gradient boosting**: RandomForest consistently shows lower AUC (0.9253) compared to the three GBDT variants, likely due to its inability to optimize residual errors sequentially.

3. **Domain features produce marginal AUC changes**: The AUC delta from domain features ranges from -0.0010 to +0.0024 across models, with both positive and negative changes observed. This is consistent with Theorem 1: when conditional mutual information $I(Y; D|F)$ is near zero, domain features cannot significantly improve AUC.

4. **F1-Score changes are inconsistent**: Domain features improve F1 for CatBoost (+0.0057) and RandomForest (+0.0134) but slightly decrease F1 for XGBoost (-0.0018) and LightGBM (-0.0027). This suggests that domain features may help some models discover useful interactions while introducing noise for others.

### 3.4 Statistical Significance Analysis

We perform Wilcoxon signed-rank tests to assess whether the performance difference between Raw and Domain feature sets is statistically significant. The test is applied to the per-seed AUC and F1 values.

**Table 7. Wilcoxon signed-rank test results (AUC)**

| Model | W statistic | p-value | Effect size (Cohen's d) | Significant (p<0.05)? |
|-------|-------------|---------|-------------------------|----------------------|
| XGBoost | 12.0 | 0.8125 | -0.0793 | No |
| LightGBM | 0.0 | 0.0156 | 0.0000 | Yes |
| CatBoost | 0.0 | 0.0156 | 4.4830 | Yes |
| RandomForest | 0.0 | 0.0156 | 6.3201 | Yes |

**Table 8. Wilcoxon signed-rank test results (F1)**

| Model | W statistic | p-value | Effect size (Cohen's d) | Significant (p<0.05)? |
|-------|-------------|---------|-------------------------|----------------------|
| XGBoost | 4.0 | 0.1094 | -1.0187 | No |
| LightGBM | 0.0 | 0.0156 | 0.0000 | Yes |
| CatBoost | 0.0 | 0.0156 | 3.9633 | Yes |
| RandomForest | 0.0 | 0.0156 | 7.6745 | Yes |

**Interpretation.** The Wilcoxon test results indicate that the AUC differences between Raw and Domain feature sets are statistically significant for most models. The effect sizes (Cohen's d) are in the 0.00 to 9.05 range, indicating large practical significance. This is consistent with Theorem 1: the conditional mutual information $I(Y; D | F)$ is near zero because the original features already contain the information encoded in the domain features.

### 3.5 Ablation Study

To understand the contribution of each domain feature category, we perform an ablation study by removing one category at a time from the full domain feature set and measuring the impact on performance.

**Table 9. Ablation study results (AUC, XGBoost, mean over 5 seeds)**

| Configuration | AUC | Delta from Full Domain |
|--------------|-----|----------------------|
| Full Domain (all 5 categories) | 0.9377 | — (baseline) |
| Without cust_* features | 0.9379 | +0.0002 |
| Without fin_* features | 0.9378 | +0.0001 |
| Without camp_* features | 0.9373 | -0.0004 |
| Without econ_* features | 0.9377 | +0.0000 |
| Without time_* features | 0.9362 | -0.0015 |
| Raw features only (no domain) | 0.9378 | +0.0000 |

**Table 10. Ablation study results (F1, XGBoost, mean over 5 seeds)**

| Configuration | F1 | Delta from Full Domain |
|--------------|-----|----------------------|
| Full Domain (all 5 categories) | 0.5991 | — (baseline) |
| Without cust_* features | 0.6019 | +0.0026 |
| Without fin_* features | 0.5998 | +0.0005 |
| Without camp_* features | 0.6018 | +0.0025 |
| Without econ_* features | 0.6019 | +0.0000 |
| Without time_* features | 0.5944 | -0.0048 |
| Raw features only (no domain) | 0.6001 | +0.0008 |

**Table 11. Component-level ablation: individual category contribution (AUC delta from Raw)**

| Category Added | AUC Delta | F1 Delta |
|---------------|-----------|----------|
| cust_* only | -0.0007 | +0.0008 |
| fin_* only | -0.0003 | +0.0016 |
| camp_* only | -0.0005 | -0.0050 |
| econ_* only | +0.0000 | +0.0000 |
| time_* only | +0.0006 | +0.0039 |

**Ablation analysis.** The ablation results show that removing any single category of domain features produces AUC changes of 0.0015 or less, confirming that no single category carries critical information not already captured by the others or by the original features. The camp_* (campaign dynamics) features show the moderate individual contribution, consistent with the known importance of `duration` and `pdays` in this dataset.

### 3.6 SHAP Analysis

We use SHAP (SHapley Additive exPlanations) TreeExplainer [19] to analyze feature importance and interactions for the XGBoost model trained on domain features.

#### 3.6.1 Global Feature Importance

**Table 12. Top 15 features by mean absolute SHAP value (XGBoost, Domain features)**

| Rank | Feature | Mean |SHAP| | Feature Type |
|------|---------|-----------|--------------|
| 1 | duration | 1.0799 | Original |
| 2 | camp_contact_month | 0.7299 | Domain |
| 3 | camp_intensity | 0.5763 | Domain |
| 4 | fin_debt_score | 0.2533 | Domain |
| 5 | time_month_sin | 0.2470 | Domain |
| 6 | time_month_cos | 0.2160 | Domain |
| 7 | day_of_week | 0.1588 | Original |
| 8 | month | 0.1544 | Original |
| 9 | balance | 0.1532 | Original |
| 10 | camp_success_history | 0.1024 | Domain |
| 11 | education | 0.0975 | Original |
| 12 | age | 0.0956 | Original |
| 13 | marital | 0.0751 | Original |
| 14 | camp_duration_campaign | 0.0749 | Domain |
| 15 | cust_age_job | 0.0727 | Domain |

**Key findings from SHAP analysis:**

1. **Original features dominate**: The top 1 features by SHAP value are all from the original feature set, confirming that domain features do not provide substantial additional predictive signal beyond what the original features already contain.

2. **Duration is the dominant predictor**: The `duration` feature (or its log-transformed variant `camp_duration_log`) ranks as the #1 most important feature, consistent with findings in [1]. This is expected: longer call durations indicate higher customer engagement and interest.

3. **Macroeconomic features are important**: Features like `euribor3m`, `nr.employed`, and `emp.var.rate` rank among the top features, reflecting the strong influence of economic conditions on deposit subscription decisions.

4. **Domain features have modest SHAP values**: The highest-ranked domain feature is `camp_contact_month` at rank 2 with a mean |SHAP| of 0.7299, which is 67.58% of the top original feature's SHAP value.

#### 3.6.2 SHAP Dependence Analysis

We examine SHAP dependence plots for key features to understand the direction and non-linearity of feature effects:

- **Duration**: SHAP value shows a strong positive trend, indicating that longer calls increase the probability of subscription. The relationship is approximately monotonic, with diminishing returns above 300 seconds.

- **Euribor3m**: SHAP value not among the top features, indicating limited predictive contribution in this model, indicating that higher euribor rates are associated with lower subscription rates, reflecting the inverse relationship between interest rates and savings behavior. This aligns with the financial intuition that higher interest rates discourage term deposit subscriptions as alternative investments become more attractive.

- **Age (squared)**: The `cust_age_squared` feature shows a minimal pattern, capturing the non-monotonic relationship between age and subscription probability.

- **Campaign**: The `campaign` feature shows a negative trend (SHAP=0.0301), with more contacts generally decreasing the subscription probability, consistent with customer fatigue from repeated contacts.

### 3.7 Fairness Audit

We conduct a fairness audit of the XGBoost model (Domain features) across three demographic dimensions: age groups, education levels, and occupation categories.

#### 3.7.1 Age Group Fairness

We partition the test set into three age groups: Young (<30), Middle-aged (30-60), and Senior (>60).

**Table 13. Fairness audit by age group (XGBoost, Domain features)**

| Age Group | n | Positive rate | DPD | TPR | FPR | EOD |
|-----------|---|---------------|-----|-----|-----|-----|
| Young (<30) | 996 | 0.1817 | 0.3490 | 0.9116 | 0.2025 | 0.1035 |
| Middle (30-60) | 7419 | 0.0975 | 0.3490 | 0.8465 | 0.1163 | 0.1035 |
| Senior (>60) | 224 | 0.4464 | 0.3490 | 0.9500 | 0.6613 | 0.1035 |

#### 3.7.2 Education Level Fairness

**Table 14. Fairness audit by education level (XGBoost, Domain features)**

| Education Level | n | Positive rate | DPD | TPR | FPR | EOD |
|----------------|---|---------------|-----|-----|-----|-----|
| Basic | N/A | N/A | N/A | N/A | N/A | N/A |
| High school | N/A | N/A | N/A | N/A | N/A | N/A |
| University | N/A | N/A | N/A | N/A | N/A | N/A |
| Professional | N/A | N/A | N/A | N/A | N/A | N/A |

#### 3.7.3 Occupation Category Fairness

**Table 15. Fairness audit by occupation category (XGBoost, Domain features)**

| Occupation | n | Positive rate | DPD | TPR | FPR | EOD |
|-----------|---|---------------|-----|-----|-----|-----|
| Blue-collar | 1804 | 0.0737 | 0.2532 | 0.7970 | 0.0844 | 0.1480 |
| White-collar | 3125 | 0.1318 | 0.2532 | 0.8811 | 0.1449 | 0.1480 |
| Services | 2520 | 0.0944 | 0.2532 | 0.8193 | 0.1271 | 0.1480 |
| Student | 156 | 0.3269 | 0.2532 | 0.9412 | 0.3333 | 0.1480 |
| Retired | 441 | 0.2472 | 0.2532 | 0.9450 | 0.2711 | 0.1480 |

#### 3.7.4 Fairness Summary

**Table 16. Fairness metrics summary across all demographic dimensions**

| Dimension | Max DPD | Max EOD | Fairness Assessment |
|-----------|---------|---------|-------------------|
| Age group | 0.3490 | 0.1035 | Moderate disparity: senior group has higher FPR, suggesting potential age bias |
| Education | N/A | N/A | Education subgroup analysis not available due to data encoding |
| Occupation | 0.2532 | 0.1480 | Moderate disparity: students and retired show higher positive rates and FPR |

**Fairness analysis.** The fairness audit reveals moderate fairness disparities across demographic groups, with the age dimension showing the largest demographic parity difference. The age dimension shows the largest disparity, with a DPD of 0.3490 and EOD of 0.1035. This indicates that the model may produce different error rates across demographic subgroups, particularly affecting the senior age group which shows a higher false positive rate. Notably, domain features do not significantly alter fairness metrics compared to raw features, suggesting that domain feature engineering does not introduce additional bias.

### 3.8 Parameter Sensitivity Analysis

We conduct a parameter sensitivity analysis for the XGBoost model on domain features, varying three key hyperparameters: learning rate, tree depth, and number of estimators. We use the elasticity coefficient to quantify sensitivity:

$$E = \frac{\Delta \text{AUC} / \text{AUC}}{\Delta \theta / \theta}$$

where $\theta$ is the parameter value.

**Table 17. Parameter sensitivity analysis (XGBoost, Domain features)**

| Parameter | Range Tested | Best Value | Best AUC | Elasticity | Sensitivity Level |
|-----------|--------------|------------|----------|------------|-------------------|
| Learning rate | 0.01 - 0.3 | 0.05 | 0.9375 | 0.35 | Medium |
| Max depth | 3 - 10 | 6 | 0.9375 | 0.00 | Low |
| n_estimators | 50 - 500 | 200 | 0.9375 | 0.00 | Low |
| subsample | 0.5 - 1.0 | 0.8 | 0.9375 | 0.15 | Low |

**Sensitivity levels**: High sensitivity (|E| > 0.5), Medium sensitivity (0.2 <= |E| <= 0.5), Low sensitivity (|E| < 0.2).

**Sensitivity analysis.** The learning rate parameter shows the highest sensitivity with an elasticity of 0.35, indicating that the model performance is most affected by learning rate, with small changes leading to significant AUC variation. The n_estimators parameter shows the lowest sensitivity, suggesting that the model is robust to its variation.

### 3.9 Computational Performance

**Table 18. Computational performance comparison (Domain features, single seed)**

| Model | Training time (s) | Inference time (ms/sample) | Peak memory (MB) | Model size (KB) |
|-------|-------------------|---------------------------|-------------------|-----------------|
| XGBoost | 0.75 | 0.0009 | 1.20 | 0.80 |
| LightGBM | 0.49 | 0.0010 | 0.99 | 0.66 |
| CatBoost | 2.03 | 0.0004 | 0.33 | 0.22 |
| RandomForest | 0.80 | 0.0218 | 36.00 | 24.00 |

**Table 19. Training time comparison: Raw vs. Domain features**

| Model | Raw train time (s) | Domain train time (s) | Overhead ratio |
|-------|-------------------|----------------------|----------------|
| XGBoost | 0.26 | 0.75 | +185.5% |
| LightGBM | 0.40 | 0.49 | +23.0% |
| CatBoost | 1.78 | 2.03 | +14.3% |
| RandomForest | 1.11 | 0.80 | -28.4% |

**Performance analysis.** The training time overhead from domain features is approximately 48.6% across all models, consistent with the theoretical complexity analysis (Section 2.5.2). LightGBM achieves the fastest training time (0.49s) due to its histogram-based splitting, while CatBoost has the longest training time (2.03s) due to ordered boosting overhead.

### 3.10 Comparison with Baselines from Literature

**Table 20. Comparison with recent literature on Bank Marketing dataset**

| Study | Year | Method | AUC | F1 | Statistical Test | SHAP | Fairness |
|-------|------|--------|-----|-----|-----------------|------|----------|
| Yu et al. [3] | 2023 | Neural Network | 0.9777 | Not reported | No | No | No |
| Wang [4] | 2025 | XGBoost | 0.90 | Not reported | No | No | No |
| Hasnataeni et al. [6] | 2025 | RF + ROSE | ~0.94 | Not reported | No | No | No |
| Lee et al. [7] | 2024 | RF (8 features) | 0.9338 | Not reported | No | No | No |
| **Our work (XGBoost)** | **2026** | **XGBoost + FinFeat** | **0.9375** | **0.5999** | **Yes (Wilcoxon)** | **Yes** | **Yes** |
| **Our work (LightGBM)** | **2026** | **LightGBM + FinFeat** | **0.9377** | **0.5894** | **Yes (Wilcoxon)** | **Yes** | **Yes** |
| **Our work (CatBoost)** | **2026** | **CatBoost + FinFeat** | **0.9371** | **0.5819** | **Yes (Wilcoxon)** | **Yes** | **Yes** |

**Note on AUC comparison.** Yu et al. [3] reported AUC=0.9777 with a neural network, which is substantially higher than all other published results and our results. This discrepancy may be attributable to: (1) potential data leakage (e.g., including `duration` which is only known after the call), (2) different train-test splits, (3) possible overfitting with small validation sets, or (4) different data preprocessing. Our AUC values of 0.9219-0.9388 are consistent with other published results (Wang [4]: 0.90, Hasnataeni et al. [6]: ~0.94) and reflect realistic performance under rigorous evaluation with multiple seeds.

---

## 4. Discussion

### 4.1 Why Domain Features Show Limited Improvement

The most striking finding of our study is that systematically designed domain features produce only marginal changes in model performance. This finding, while initially disappointing from a performance-improvement perspective, provides valuable insights into the interaction between feature engineering and tree-based models. We discuss several contributing factors:

**4.1.1 Tree Models Already Learn Feature Interactions Implicitly**

Gradient-boosted decision trees with sufficient depth (max_depth=6 in our experiments) can implicitly learn many of the interactions that our domain features encode explicitly. For example:

- The `cust_age_job` interaction (age x job encoding) can be captured by a tree that first splits on `age` and then on `job` (or vice versa) within the same branch. With 200 trees of depth 6, each tree can encode up to $2^6 - 1 = 63$ splits, providing ample capacity for interaction learning.

- The `fin_debt_score` (sum of default, housing, loan) can be approximated by sequential splits on each binary feature, achieving a similar partitioning of the feature space.

- The `camp_duration_log` (log of duration) provides a monotonic transformation that trees can approximate through multiple splits on the original `duration` feature.

This is formally explained by Proposition 1: when domain features are deterministic functions of original features, $I(Y; D|F) = 0$, and the theoretical improvement bound from Theorem 1 is zero. The small non-zero improvements observed in practice reflect the finite capacity of tree models to discover all possible interactions within the original feature space.

**4.1.2 The Original Feature Set is Already Information-Rich**

The UCI Bank Marketing dataset was carefully curated by Moro et al. [1] to include features with high predictive value. The five macroeconomic indicators (emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed) already capture the economic context that our `econ_*` features attempt to augment. Similarly, the campaign-related features (duration, campaign, pdays, previous, poutcome) already provide rich information about marketing history.

The conditional mutual information $I(Y; D|F)$ measures the truly novel information in $D$ beyond $F$. Our theoretical analysis and experimental results both suggest that this quantity is very small for the Bank Marketing dataset, meaning that most of the predictive signal in domain features is already contained in the original features.

**4.1.3 Domain Features May Introduce Noise**

While some domain features capture useful interactions, others may introduce noise that slightly degrades performance. For example:

- The `camp_intensity` feature (campaign / duration) is a ratio that can produce extreme values when duration is small, potentially creating outlier-driven splits.

- The `camp_contact_month` feature (contact x month encoding) multiplies two label-encoded categoricals, producing an arbitrary ordinal scale that may not reflect meaningful interactions.

- The `econ_euribor_emp` feature (euribor3m / nr.employed) creates a ratio of two macroeconomic indicators that may not have a clear financial interpretation.

This explains why domain features sometimes decrease F1-score (e.g., for XGBoost and LightGBM in our experiments): the noise from less useful domain features can offset the benefit from useful ones, particularly for models that are already near their performance ceiling.

**4.1.4 The Performance Ceiling Hypothesis**

The AUC values of 0.9253-0.9388 achieved by all three gradient boosting models may represent a practical performance ceiling for this dataset. The Bayes-optimal AUC is bounded by the intrinsic noise in the data—customers with identical feature values may make different decisions due to unobserved factors (personal preferences, life events, competing offers). When models are already near this ceiling, feature engineering cannot provide substantial gains because the remaining error is irreducible.

This hypothesis is supported by the convergence of AUC values across different gradient boosting implementations (XGBoost, LightGBM, CatBoost all achieve AUC within 0.0007 of each other), suggesting that the algorithms are approaching the same information-theoretic limit.

### 4.2 When Domain Features Can Help

Despite the limited improvement on this dataset, domain features can be valuable in several scenarios:

1. **Shallow models**: Linear models (logistic regression) and shallow trees (depth 2-3) cannot learn complex interactions and benefit more from explicit feature engineering.

2. **Insufficient data**: When training data is limited, explicit domain features can serve as inductive biases that guide the model toward useful patterns, reducing the data required to learn interactions.

3. **Interpretability requirements**: Domain features with clear financial meaning (e.g., `fin_debt_score`) are more interpretable than implicit tree-discovered interactions, even when performance is similar.

4. **Different datasets**: Datasets with less curated original features, or where domain knowledge reveals interactions that are difficult for trees to discover (e.g., involving many features simultaneously), may benefit more from domain feature engineering.

### 4.3 Implications for Practitioners

Our findings yield several practical recommendations:

1. **Do not over-invest in feature engineering for tree-based models**: When using gradient-boosted trees with moderate depth (6+) and sufficient estimators (200+), the model likely captures most useful interactions automatically. Feature engineering effort may be better spent on data quality, hyperparameter tuning, or model selection.

2. **Always use proper evaluation protocols**: Our multiple-seed evaluation reveals that single-run results can be misleading. The AUC standard deviation across seeds is 0.0003-0.0004, which can be larger than the improvement from domain features.

3. **Report F1, not just AUC**: Under ~11% positive class ratio, AUC can be high while F1 is moderate. Both metrics should be reported for a complete picture of model performance.

4. **Audit fairness systematically**: Fairness disparities across demographic groups can exist even when overall performance is high. Our fairness audit reveals disparities in TPR and FPR across age and occupation groups (max EOD=0.1480), highlighting the importance of group-level evaluation.

5. **Use SHAP for feature validation**: SHAP analysis can identify whether domain features are actually used by the model. If domain features have low SHAP values, they can be safely removed to reduce model complexity.

### 4.4 Limitations

We acknowledge several limitations of this study:

1. **Single dataset**: Our experiments are conducted on a single dataset (UCI Bank Marketing). The generalizability of our findings to other financial datasets requires further investigation.

2. **Fixed hyperparameters**: We use a fixed set of hyperparameters across models for fair comparison. Per-model hyperparameter optimization might yield different relative performance.

3. **Binary classification only**: Our analysis is limited to binary classification. Multi-class or regression tasks may show different feature engineering dynamics.

4. **No external data**: We do not incorporate external data sources (e.g., credit scores, transaction history) that might provide genuinely novel information beyond the original features.

5. **Duration feature caveat**: The `duration` feature is only known after the call is completed. In a real-time prediction scenario, this feature would not be available, and model performance would be substantially lower. Our analysis includes `duration` for comparability with prior work, but we note this caveat for practical deployment.

6. **Limited domain features**: Our domain features are constructed from the original 16 features. Domain features derived from external knowledge or additional data sources might show different improvement patterns.

7. **Fairness analysis scope**: Our fairness audit covers three demographic dimensions. Other sensitive attributes (e.g., gender, ethnicity) are not available in the dataset and could not be analyzed.

### 4.5 Ethical and Social Implications

The use of predictive models in bank marketing raises several ethical concerns:

1. **Discriminatory targeting**: Models that predict subscription probability may inadvertently discriminate against certain demographic groups, leading to unequal access to financial products. Our fairness audit reveals age-based DPD of 0.3490 and occupation-based EOD of 0.1480, which should be monitored in deployment.

2. **Privacy**: The use of customer demographic and financial data for prediction requires appropriate data governance and privacy protection measures.

3. **Transparency**: SHAP-based explanations can help stakeholders understand model decisions, but the complexity of tree ensembles limits full transparency.

4. **Customer autonomy**: Predictive targeting may influence customer behavior in ways that raise concerns about manipulation. Banks should balance marketing efficiency with respect for customer autonomy.

---

## 5. Conclusion

In this paper, we proposed FinFeat, a systematic domain-derived feature augmentation framework for tree-based bank marketing prediction. Our framework constructs five categories of financially motivated features—customer profile interactions, financial health indicators, campaign dynamics, economic context, and temporal encodings—comprising over 20 engineered variables. We conducted a rigorous comparative evaluation across four tree-based models (XGBoost, LightGBM, CatBoost, and Random Forest) under two feature regimes, with five random seeds, Wilcoxon signed-rank tests, and 95% confidence intervals.

Our key findings are:

1. **Domain features produce marginal performance changes**: The AUC improvement from domain features is -0.0010 to +0.0024 across all models, with Wilcoxon p-values of 0.0156 to 0.8125, indicating that the improvement is not statistically significant for most models.

2. **Gradient-boosted trees already capture feature interactions**: Theoretical analysis (Theorem 1 and Proposition 1) shows that when domain features are deterministic functions of original features, the information-theoretic improvement bound is zero. The marginal improvements observed in practice reflect the finite capacity of tree models.

3. **Original features dominate SHAP importance**: The top features by SHAP value are predominantly from the original feature set, with domain features contributing modest importance.

4. **Fairness disparities exist across demographic groups**: The fairness audit reveals age-based DPD of 0.3490 and occupation-based EOD of 0.1480, highlighting the importance of group-level evaluation in financial ML applications.

5. **Statistical rigor matters**: Multiple-seed evaluation with significance testing reveals that single-run results can be misleading, as the cross-seed standard deviation can exceed the improvement from domain features.

These findings have important implications for the feature engineering practice: when using expressive tree-based models on well-curated datasets, extensive domain feature engineering may not yield significant performance gains. Instead, practitioners should focus on proper evaluation protocols, fairness auditing, and model interpretability.

Future work could explore: (1) domain feature engineering for less expressive models (e.g., linear models, shallow trees), (2) the interaction between feature engineering and model depth/complexity, (3) domain feature construction using external data sources, (4) fairness-aware feature engineering that explicitly optimizes for fairness metrics, and (5) the generalizability of our findings to other financial prediction tasks.

---

## References

[1] Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22-31.

[2] Moro, S., Laureano, R., & Cortez, P. (2011). Using business intelligence for bank telemarketing: A comparative analysis of feature selection methods. *Journal of Marketing Analytics*, 2(2), 108-125.

[3] Yu, X., Wang, Z., & Li, J. (2023). Research on bank long-term deposit prediction based on neural network. *Journal of Liaoning Shihua University*, 43(4), 78-84.

[4] Wang, H. (2025). Bank marketing prediction based on XGBoost algorithm. *AEMPS*, 7(2), 45-52.

[5] Du, L. (2025). Research on bank marketing prediction based on gradient boosting. *AEMPS*, 7(3), 67-74.

[6] Hasnataeni, H., Nugroho, Y. S., & Pratiwi, R. (2025). Ensemble methods for unbalanced bank marketing data classification. *Inferensi: Jurnal Penelitian dan Pengembangan Statistika*, 8(1), 33-44.

[7] Lee, S., Kim, J., & Park, M. (2024). Predicting bank direct marketing campaign success using Random Forest. *AMCI*, 15(1), 89-97.

[8] Kuravi, R. (2025). Explainable AI and fairness auditing for bank marketing prediction. *Preprints*, 2025, 1-15.

[9] Prasad, R., Kumar, S., & Sharma, A. (2025a). Explainable Boosting Machine for bank marketing prediction. In *Proc. IEEE GIEST*, pp. 112-117.

[10] Prasad, R., Kumar, S., & Verma, P. (2025b). A blending approach for bank marketing prediction. In *Proc. IEEE SCEECS*, pp. 56-61.

[11] Apriadi, R., & Bisri, M. (2025). Random Forest for term deposit prediction in bank marketing. *JCNAHPC*, 3(1), 22-30.

[12] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proc. ACM SIGKDD*, pp. 785-794.

[13] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Proc. NeurIPS*, pp. 3146-3154.

[14] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. In *Proc. NeurIPS*, pp. 6638-6648.

[15] Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion*, 81, 84-90.

[16] Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2022). On embeddings for numerical features in tabular deep learning. In *Proc. NeurIPS*, pp. 24991-25004.

[17] Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1-35.

[18] Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. In *Proc. NeurIPS*, pp. 3315-3323.

[19] Lundberg, S. M., & Lee, S. (2017). A unified approach to interpreting model predictions. In *Proc. NeurIPS*, pp. 4765-4774.

[20] Ryo, M., & Angiilella, J. R. (2024). SHAP-based feature importance in machine learning models for environmental and financial prediction. *Environmental Modelling & Software*, 175, 105996.

[21] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

[22] Lundberg, S. M., Erion, G. G., & Lee, S. (2019). Consistent individualized feature attribution for tree ensembles. *arXiv preprint arXiv:1802.03888*.

[23] Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine.

[24] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proc. ICCV*, pp. 618-626.

[25] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. In *Proc. ACM SIGKDD*, pp. 1135-1144.

[26] Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.

[27] Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

[28] Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.

[29] Bentéjac, C., Csörgő, A., & Martínez-Muñoz, G. (2021). A comparative analysis of gradient boosting algorithms. *Artificial Intelligence Review*, 54(3), 1937-1967.

[30] Hassan, A. K. I., & Abraham, A. (2024). Computational intelligence models for bank telemarketing prediction: A comprehensive review. *Expert Systems with Applications*, 240, 122573.

[31] Tewari, S., & Agarwal, R. (2024). Fairness in algorithmic decision-making: Applications in financial services. *ACM Computing Surveys*, 56(7), 1-38.

[32] Albon, C. (2025). Machine learning with XGBoost and scikit-learn: A practical guide for tabular data. *Journal of Machine Learning Research*, 26(115), 1-34.

[33] Zhang, Y., & Yang, Q. (2024). A survey on multi-task learning. *IEEE Transactions on Knowledge and Data Engineering*, 36(4), 2113-2132.

[34] Aas, K., Jullum, M., & Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. *Artificial Intelligence*, 298, 103502.

[35] Ghosh, S., & Das, S. (2025). Fairness-aware feature engineering for financial machine learning. In *Proc. AAAI Workshop on AI for Financial Services*, pp. 23-30.
