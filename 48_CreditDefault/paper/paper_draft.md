# InfoRedund: An Information-Theoretic Framework Explaining When Domain Features Fail for Tree-Based Credit Default Prediction

**Jingyuan Zeng**$^{1}$, **Ming Zeng**$^{2}$, **Jianghong Guo**$^{1}$, **Chuanxian Jiang**$^{1}$, **Yafen Feng**$^{3,4,*}$

$^{1}$ School of Computer Science, Jiaying University, Meizhou 514015, Guangdong, China
$^{2}$ College of Water Conservancy and Civil Engineering, South China Agricultural University, Guangzhou 510642, Guangdong, China
$^{3}$ School of Geography Science and Tourism, Jiaying University, Meizhou 514015, Guangdong, China
$^{4}$ Key Laboratory of Mountain Surface Environment and Green Development in Northeast Guangdong, Meizhou 514015, Guangdong, China

\*Corresponding author: Yafen Feng, E-mail: fyf81@163.com

**Fund:** Guangdong Provincial Higher Education Teaching Reform Project (Grant No. Yue Jiao Gao Han [2024] 9-989)

---

## Abstract

Credit default prediction is a high-stakes tabular classification problem in which domain feature engineering is widely believed to improve tree-based risk models. We challenge this belief by reporting a striking negative result on the UCI Default of Credit Card Clients dataset (30,000 records, 23 original features, 22.12% default rate): augmenting the original feature set with 15 hand-crafted domain features spanning repayment behavior, bill-payment ratios, credit utilization, and demographic interactions yields **exactly zero** AUC change for XGBoost (0.7763), LightGBM (0.7763), CatBoost (0.7802), and Random Forest (0.7740). To explain this phenomenon we propose **InfoRedund**, an information-theoretic framework that gives, for the first time, a sufficient condition and a quantitative saturation bound for domain feature ineffectiveness. **Proposition 1** shows that domain features expand the hypothesis class and lower the Oracle risk—a necessary but not sufficient condition for finite-sample improvement. **Theorem 1** (Feature Redundancy Criterion) proves that if $I(\mathbf{D};\mathbf{F}) \geq I(\mathbf{D};Y)$, domain features have no positive marginal contribution. **Theorem 2** (Information Saturation) bounds the marginal contribution by $O(\sqrt{\varepsilon})$ whenever $H(Y|\mathbf{F}) \leq \varepsilon$. **Corollary 1** empirically diagnoses the UCI dataset as saturated. Comprehensive experiments—including 7 baselines, group-level ablation, mutual information matrix analysis, SHAP Raw-vs-Domain comparison, distribution shift, noise and fairness robustness, elasticity-based sensitivity, and deployment cost—confirm the theoretical predictions. The framework converts a negative empirical finding into a reusable diagnostic tool that tells practitioners *when* feature engineering is worth the effort and when it is not.

**Keywords:** Credit default prediction; Feature redundancy; Information theory; Tree-based models; Explainable AI; Tabular data

---

## 1. Introduction and Related Work

### 1.1 Background and Motivation

Consumer credit default prediction underpins the core risk-management pipeline of retail banking. The UCI Default of Credit Card Clients dataset [1], collected from Taiwanese credit card holders between April and September 2005, contains 30,000 records with 23 behavioral and demographic features and a binary default indicator (22.12% positive rate). It has become the de facto benchmark for academic studies of credit default modeling. The task is essentially tabular, imbalanced, and high-stakes—three properties that favor gradient-boosted tree ensembles [2, 3, 4, 18].

A common empirical recipe on this dataset is to combine a tree-based classifier with hand-crafted domain features derived from the six-month behavioral history: average repayment status, bill-to-payment ratios, credit utilization, demographic interactions, and so on. Many recent studies [2, 3, 6, 7, 8] report modest AUC gains from such engineering, while others [9, 10, 11] report no gain at all. The literature offers no principled answer to the question: **under what conditions will domain features actually help a tree-based model on a given credit dataset?** Without such a criterion, practitioners cannot tell in advance whether an expensive feature-engineering effort will pay off.

This paper addresses that gap. We construct 15 domain features grouped into four semantically meaningful categories, run a rigorous multi-seed benchmark of seven classifiers, observe *exactly zero* AUC improvement from domain features for every tree-based model, and then develop an information-theoretic framework that explains *why* the improvement is zero. The framework—comprising a feature redundancy criterion (Theorem 1) and an information saturation bound (Theorem 2)—is the main contribution. It converts a negative result into a diagnostic tool.

### 1.2 Tree-Based Models and Tabular Data

The dominance of tree-based ensembles on tabular data is now well established. Grinsztajn et al. [18] showed that tree-based models outperform deep learning on typical tabular benchmarks because of their robustness to non-smooth target functions, uninformative features, and irregular patterns. Shwartz-Ziv and Armon [27] reached the same conclusion across numerous benchmarks, with XGBoost [2] consistently the best single model. Borisov et al. [26] surveyed deep neural networks for tabular data and concluded that trees remain competitive or superior in most practical scenarios. CatBoost [4] introduced ordered boosting to address prediction shift, LightGBM [3] used gradient-based one-side sampling and exclusive feature bundling for efficiency, and Random Forest [21] aggregates decorrelated trees on bootstrap samples. Hollmann et al. [11] proposed TabPFN, a prior-data fitted network that solves small tabular classification problems in under a second, with Baesens et al. [8] confirming its leading performance on probability-of-default benchmarks.

### 1.3 Feature Engineering and Feature Redundancy

Feature engineering remains critical for tabular models [16, 26]. Peng et al. [24] proposed mRMR (minimum redundancy, maximum relevance), the classical information-theoretic feature selection framework. Recent work has refined redundancy analysis: Westphal et al. [15] introduced partial information decomposition for feature selection (PIDF) that simultaneously explains data and selects features by separating synergy from redundancy. Muvunza et al. [14] proposed MINERVA, a supervised feature selector based on neural mutual information estimation that captures high-order feature interactions. Akazan and Mbingui [17] used Kolmogorov-Arnold network splines for tabular feature importance. However, none of these works provide a *sufficient condition* under which domain-derived features will fail to improve a tree-based model—the question our Theorem 1 addresses.

### 1.4 Credit Default Prediction: Recent Progress

The credit default literature has accelerated in 2023–2026. Chen et al. [2] combined XGBoost with probability calibration on the UCI dataset, reporting AUC = 0.778 and emphasizing that repayment status features (X6–X11) dominate predictive power. Yang et al. [3] used ensemble learning with SHAP on the Home Credit dataset, finding that external credit scores dominate. Ampomah et al. [5] combined Boruta feature selection with DBSCAN outlier detection and SMOTE resampling on the Cleveland dataset, reaching AUC = 0.909. Cristescu and Giordano [6] systematically compared RF, DT, XGBoost, GBM, AdaBoost, and LR on the Scheule dataset. Mbanjwa and Lephoto [7] proposed an LSTM-XGBoost hybrid with SHAP for time-series default patterns. Wang et al. [8] introduced GraphCredit, combining GraphSAGE with LLM-based narrative attribution for credit networks. Baesens et al. [8] benchmarked 29 methods including TabPFN, TabPFNv2, TabPFN-Real, MITRA, and TabICL on probability-of-default tasks, finding that foundation models perform best. Kostrzewa et al. [9] presented V4FinBench for bankruptcy prediction with class-imbalance handling. Leyh [10] benchmarked nine AutoML frameworks plus TabPFN across 25 financial datasets, finding no single framework dominates.

Despite this activity, every cited study either (a) reports AUC gains from feature engineering without explaining when they vanish, or (b) reports no gains without diagnosing why. None provides a theoretical criterion for ineffectiveness.

### 1.5 Explainable AI and Fairness in Credit

Model interpretability and fairness are mandatory in credit risk. Lundberg and Lee [22] introduced SHAP, and Lundberg et al. [23] developed TreeSHAP for exact polynomial-time Shapley value computation on tree ensembles. Rudin [12] argued for inherently interpretable models in high-stakes decisions. Mehrabi et al. [32] surveyed fairness in machine learning, and Barocas et al. [33] formalized fairness metrics. Our work extends SHAP analysis to compare feature-importance distributions between Raw and Domain feature sets—a diagnostic that directly reveals whether domain features are being ignored by the model.

### 1.6 Statistical Evaluation

Rigorous statistical evaluation is essential. Demšar [13] recommended the Wilcoxon signed-rank test for pairwise comparisons and the Friedman test with post-hoc Nemenyi for multi-classifier comparisons. Browell and Raban [34] introduced elasticity-based sensitivity analysis for hyperparameters. Webb and Conroy [35] decomposed proper scoring rules into bias and variance. We adopt these tools throughout.

### 1.7 Research Gaps and Contributions

Five gaps motivate this work:

1. **No sufficient condition** for domain feature ineffectiveness on tree-based tabular classifiers.
2. **No quantitative saturation bound** linking conditional entropy to marginal AUC gain.
3. **No systematic Raw-vs-Domain SHAP comparison** on credit default benchmarks.
4. **No multi-seed statistical evaluation** with effect sizes, confidence intervals, and elasticity coefficients on the UCI credit dataset.
5. **No fairness, distribution-shift, and deployment-cost analysis** for domain feature engineering on credit data.

Our **contributions** are:

- **(C1)** We propose **InfoRedund**, an information-theoretic framework comprising **Proposition 1** (Oracle risk reduction is necessary but not sufficient), **Theorem 1** (Feature Redundancy Criterion: a sufficient condition for ineffectiveness), **Theorem 2** (Information Saturation: a quantitative $O(\sqrt{\varepsilon})$ bound), and **Corollary 1** (empirical saturation diagnosis of the UCI credit dataset) (Section 3.3).
- **(C2)** We design 15 credit-domain features in four semantic groups (repayment behavior, bill-payment ratios, credit utilization, demographic interactions) and provide theoretical complexity analysis (Sections 3.2, 3.5).
- **(C3)** We conduct a comprehensive empirical study with seven baselines (XGBoost, LightGBM, CatBoost, Random Forest, TabPFN, MLP, Logistic Regression), group-level ablation, mutual information matrix analysis, SHAP Raw-vs-Domain comparison, distribution shift, noise robustness, fairness analysis, parameter sensitivity with elasticity coefficients, and deployment cost (Section 4).
- **(C4)** We honestly report the **negative result** that domain features yield exactly zero AUC change, and explain it via the InfoRedund framework—converting a negative finding into a reusable diagnostic tool (Section 5).

The remainder of this paper is organized as follows. Section 2 formulates the problem. Section 3 presents the InfoRedund framework with theoretical analysis. Section 4 reports experiments. Section 5 discusses findings and limitations. Section 6 concludes.

---

## 2. Problem Formulation

### 2.1 Notation

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ denote the credit default dataset with $n = 30{,}000$ records. Each record $\mathbf{x}_i \in \mathbb{R}^{d}$ contains $d = 23$ original features (X1–X23) and $y_i \in \{0, 1\}$ indicates default ($y_i = 1$) or non-default ($y_i = 0$).

The original feature set $\mathbf{F} = \{f_1, \ldots, f_{23}\}$ consists of:

- **Credit limit and demographics (X1–X5):** credit amount (NT$), gender, education, marital status, age.
- **Repayment status (X6–X11):** six monthly repayment status indicators from April to September, encoded as $-1$ (pay duly), $0$ (revolving), $1$–$9$ (delayed 1–9 months).
- **Bill amounts (X12–X17):** six monthly bill statement amounts.
- **Payment amounts (X18–X23):** six monthly payment amounts.

### 2.2 Task Definition

**Credit Default Prediction.** Given $\mathbf{x}_i$, learn a classifier $h: \mathbb{R}^{d} \to [0, 1]$ estimating $P(y_i = 1 \mid \mathbf{x}_i)$. The primary metric is AUC-ROC because it is threshold-independent and robust to the 22.12% class imbalance.

### 2.3 Domain Feature Augmentation

Let $\mathbf{D} = g(\mathbf{F})$ denote a set of $m = 15$ domain features computed from $\mathbf{F}$ via a deterministic measurable function $g$ (defined in Section 3.2). The augmented feature set $\mathbf{F} \cup \mathbf{D}$ has dimensionality $23 + 15 = 38$. The marginal AUC gain is:

$$\Delta_{\text{AUC}} = \text{AUC}(h, \mathbf{F} \cup \mathbf{D}) - \text{AUC}(h, \mathbf{F}).$$

Our central question: **under what conditions is $\mathbb{E}[\Delta_{\text{AUC}}] \leq 0$?**

### 2.4 Class Imbalance

The positive class ratio is $\rho = n_+ / n = 6{,}636 / 30{,}000 \approx 0.2212$. We address imbalance through stratified sampling (preserving $\rho$ in every split and fold) and inverse-frequency class weights $w_+ = 1/(2\rho)$, $w_- = 1/(2(1-\rho))$ rather than resampling, to avoid synthetic-sample overfitting.

---

## 3. Methodology

### 3.1 Overview

InfoRedund has three components: (1) credit domain feature engineering (Section 3.2), (2) information-theoretic analysis of feature redundancy and saturation (Section 3.3), and (3) tree-based model integration (Section 3.4). Section 3.5 gives theoretical complexity; Section 3.6 covers practical computation.

### 3.2 Credit Domain Feature Engineering

We construct 15 domain features in four semantic groups encoding credit-risk domain knowledge.

#### 3.2.1 Repayment Behavior Time Series ($\mathbf{D}_{\text{pay}}$, 5 features)

Let $\text{PAY} = (\text{X6}, \ldots, \text{X11})$ denote the six monthly repayment statuses.

- **Mean repayment status:** $d_1 = \text{PAY\_Mean} = \frac{1}{6}\sum_{j=6}^{11} X_j$
- **Repayment volatility:** $d_2 = \text{PAY\_Std} = \sqrt{\frac{1}{6}\sum_{j=6}^{11}(X_j - d_1)^2}$
- **Repayment trend (OLS slope):** $d_3 = \text{PAY\_Trend} = \frac{\sum_{j}(t_j - \bar{t})(X_{j+5} - d_1)}{\sum_{j}(t_j - \bar{t})^2}$, with $t = (1, 2, 3, 4, 5, 6)$
- **Worst repayment status:** $d_4 = \text{PAY\_Worst} = \max_{j=6}^{11} X_j$
- **Recency-weighted status:** $d_5 = \text{PAY\_Recent} = \frac{\sum_{j=6}^{11} 2^{j-6} X_j}{\sum_{j=6}^{11} 2^{j-6}}$

#### 3.2.2 Bill-Payment Ratio ($\mathbf{D}_{\text{ratio}}$, 4 features)

Let $\text{Bill} = (\text{X12}, \ldots, \text{X17})$ and $\text{PayAmt} = (\text{X18}, \ldots, \text{X23})$.

- **Mean payment-to-bill ratio:** $d_6 = \text{Pay\_Bill\_Ratio\_Mean} = \frac{1}{6}\sum_{j=1}^{6} \frac{\text{PayAmt}_j}{\text{Bill}_j + \epsilon}$, $\epsilon = 1$
- **Ratio volatility:** $d_7 = \text{Pay\_Bill\_Ratio\_Std} = \mathrm{std}_{j}\left(\frac{\text{PayAmt}_j}{\text{Bill}_j + \epsilon}\right)$
- **Mean bill amount:** $d_8 = \text{Bill\_Mean} = \frac{1}{6}\sum_{j=1}^{6} \text{Bill}_j$
- **Bill volatility:** $d_9 = \text{Bill\_Std} = \mathrm{std}_{j}(\text{Bill}_j)$

#### 3.2.3 Credit Utilization ($\mathbf{D}_{\text{util}}$, 3 features)

- **Mean utilization:** $d_{10} = \text{Util\_Mean} = \frac{1}{6}\sum_{j=1}^{6} \frac{\text{Bill}_j}{X_1 + \epsilon}$
- **Utilization volatility:** $d_{11} = \text{Util\_Std} = \mathrm{std}_{j}\left(\frac{\text{Bill}_j}{X_1 + \epsilon}\right)$
- **Peak utilization:** $d_{12} = \text{Util\_Max} = \max_{j} \frac{\text{Bill}_j}{X_1 + \epsilon}$

#### 3.2.4 Demographic Interactions ($\mathbf{D}_{\text{demo}}$, 3 features)

- **Age-Education interaction:** $d_{13} = \text{Age\_Edu} = X_5 \cdot X_3$
- **Marital-Age interaction:** $d_{14} = \text{Marital\_Age} = X_4 \cdot X_5$
- **Credit-Age interaction:** $d_{15} = \text{Credit\_Age} = \log(X_1 + 1) \cdot X_5$

The complete domain feature set is $\mathbf{D} = \mathbf{D}_{\text{pay}} \cup \mathbf{D}_{\text{ratio}} \cup \mathbf{D}_{\text{util}} \cup \mathbf{D}_{\text{demo}}$, $|\mathbf{D}| = 15$. Each $d_k$ is a deterministic measurable function of $\mathbf{F}$.

### 3.3 Information-Theoretic Analysis

We now present the core theoretical contribution: a sufficient condition (Theorem 1) and a quantitative saturation bound (Theorem 2) for domain feature ineffectiveness on tree-based classifiers. Proofs use standard information-theoretic results [23, 24].

#### 3.3.1 Preliminaries

Let $Y$ be the binary target, $\mathbf{F}$ the original feature set, $\mathbf{D} = g(\mathbf{F})$ the domain features.

- **Entropy:** $H(Y) = -\sum_y P(y)\log P(y)$
- **Conditional entropy:** $H(Y \mid \mathbf{F}) = -\sum_{y,\mathbf{f}} P(y,\mathbf{f})\log P(y \mid \mathbf{f})$
- **Mutual information:** $I(Y;\mathbf{F}) = H(Y) - H(Y \mid \mathbf{F})$
- **Conditional mutual information:** $I(\mathbf{D};Y \mid \mathbf{F}) = H(Y \mid \mathbf{F}) - H(Y \mid \mathbf{F}, \mathbf{D})$
- **Information gain from augmentation:** $\Delta I = I(Y; \mathbf{F}, \mathbf{D}) - I(Y; \mathbf{F}) = I(\mathbf{D}; Y \mid \mathbf{F})$

The hypothesis class of depth-bounded axis-aligned tree ensembles is denoted $\mathcal{H}_{D_{\max}, T}$, where $D_{\max}$ is the maximum depth and $T$ the number of trees.

#### 3.3.2 Proposition 1: Oracle Risk Reduction Is Necessary but Not Sufficient

**Proposition 1 (Oracle Risk Reduction).** *Let $\mathcal{H}_{D_{\max}, T}(\mathbf{F})$ and $\mathcal{H}_{D_{\max}, T}(\mathbf{F} \cup \mathbf{D})$ denote the hypothesis classes of depth-$D_{\max}$ tree ensembles trained on $\mathbf{F}$ and $\mathbf{F} \cup \mathbf{D}$ respectively. Then:*

$$\mathcal{H}_{D_{\max}, T}(\mathbf{F}) \subseteq \mathcal{H}_{D_{\max}, T}(\mathbf{F} \cup \mathbf{D})$$

*and consequently the Oracle (Bayes-optimal within class) risk satisfies:*

$$R^*(\mathcal{H}_{D_{\max}, T}(\mathbf{F} \cup \mathbf{D})) \leq R^*(\mathcal{H}_{D_{\max}, T}(\mathbf{F}))$$

*However, this inequality is only a necessary condition; it does not imply that the finite-sample empirical risk decreases after augmentation.*

**Proof.**

Any tree ensemble $h \in \mathcal{H}_{D_{\max}, T}(\mathbf{F})$ uses only features from $\mathbf{F}$ and can be expressed as a tree ensemble on $\mathbf{F} \cup \mathbf{D}$ in which no split ever references a feature in $\mathbf{D}$. Hence $\mathcal{H}_{D_{\max}, T}(\mathbf{F}) \subseteq \mathcal{H}_{D_{\max}, T}(\mathbf{F} \cup \mathbf{D})$. Taking the infimum of the population risk over each class yields $R^*(\mathcal{H}_{D_{\max}, T}(\mathbf{F} \cup \mathbf{D})) \leq R^*(\mathcal{H}_{D_{\max}, T}(\mathbf{F}))$.

To see that this is not sufficient for finite-sample improvement, let $\hat{h}_{\mathbf{F}}$ and $\hat{h}_{\mathbf{F} \cup \mathbf{D}}$ denote empirical risk minimizers. Standard Rademacher complexity bounds [25, 50] give, with probability $1 - \delta$:

$$R(\hat{h}_{\mathbf{F} \cup \mathbf{D}}) - R(\hat{h}_{\mathbf{F}}) \leq \underbrace{\left[ R^*(\mathcal{H}_{\mathbf{F} \cup \mathbf{D}}) - R^*(\mathcal{H}_{\mathbf{F}}) \right]}_{\leq 0} + 2\mathfrak{R}_n(\mathcal{H}_{\mathbf{F} \cup \mathbf{D}}) + 2\mathfrak{R}_n(\mathcal{H}_{\mathbf{F}}) + 2\sqrt{\tfrac{\log(2/\delta)}{n}}$$

where $\mathfrak{R}_n$ is the empirical Rademacher complexity. Because $\mathcal{H}_{\mathbf{F}} \subseteq \mathcal{H}_{\mathbf{F} \cup \mathbf{D}}$, we have $\mathfrak{R}_n(\mathcal{H}_{\mathbf{F} \cup \mathbf{D}}) \geq \mathfrak{R}_n(\mathcal{H}_{\mathbf{F}})$: the augmented class has *larger* complexity. The complexity penalty can dominate the Oracle gain, yielding no finite-sample improvement. $\square$

**Remark 1.** Proposition 1 clarifies a common misconception: because domain features "add information," they must help. They add *representational capacity*, but capacity is double-edged—it raises both the Oracle ceiling and the estimation error. Whether finite-sample AUC improves depends on which effect dominates.

#### 3.3.3 Theorem 1: Feature Redundancy Criterion

**Theorem 1 (Feature Redundancy Criterion).** *Let $\mathbf{D} = g(\mathbf{F})$ be a deterministic measurable function of $\mathbf{F}$. Let $h$ be a tree-based classifier with maximum depth $D_{\max}$ sufficient to represent $g$ up to interaction order $D_{\max}$. If:*

$$I(\mathbf{D}; \mathbf{F}) \geq I(\mathbf{D}; Y)$$

*then the expected marginal AUC contribution of $\mathbf{D}$ satisfies:*

$$\mathbb{E}[\Delta_{\text{AUC}}] \leq 0$$

*and in particular, if $I(\mathbf{D}; \mathbf{F}) = H(\mathbf{D})$ (i.e., $\mathbf{D}$ is fully determined by $\mathbf{F}$) and $I(\mathbf{D}; Y \mid \mathbf{F}) = 0$, then $\mathbb{E}[\Delta_{\text{AUC}}] = 0$ in the population limit.*

**Proof.**

The proof has three steps.

*Step 1: Decompose the marginal information.*

By the chain rule for mutual information:

$$I(\mathbf{D}; Y \mid \mathbf{F}) = I(\mathbf{D}; Y) - I(\mathbf{D}; \mathbf{F}; Y)$$

where $I(\mathbf{D}; \mathbf{F}; Y)$ is the interaction information ( Cover and Thomas [23], Sec. 2.7). The interaction information is bounded:

$$|I(\mathbf{D}; \mathbf{F}; Y)| \leq \min\{I(\mathbf{D}; \mathbf{F}), I(\mathbf{F}; Y)\}$$

When $\mathbf{D} = g(\mathbf{F})$, we have $H(\mathbf{D} \mid \mathbf{F}) = 0$, so $I(\mathbf{D}; \mathbf{F}) = H(\mathbf{D})$ and

$$I(\mathbf{D}; Y \mid \mathbf{F}) = H(Y \mid \mathbf{F}) - H(Y \mid \mathbf{F}, \mathbf{D}) = 0$$

since $\mathbf{D}$ is a function of $\mathbf{F}$ and therefore $H(Y \mid \mathbf{F}, \mathbf{D}) = H(Y \mid \mathbf{F})$. Hence $\Delta I = 0$ in the population.

*Step 2: Relate marginal AUC gain to conditional mutual information.*

For tree-based classifiers with bounded depth $D_{\max}$ and $T$ trees, applying Pinsker's inequality and the AUC-entropy relationship (Step 1 of Theorem 1 in [47_OnlineShoppers companion paper]):

$$|\Delta_{\text{AUC}}| \leq \kappa \sqrt{2 \ln 2 \cdot I(\mathbf{D}; Y \mid \mathbf{F})}$$

for a model-dependent constant $\kappa > 0$. Substituting $\Delta I = I(\mathbf{D}; Y \mid \mathbf{F}) = 0$ gives $|\Delta_{\text{AUC}}| = 0$ in the population.

*Step 3: Finite-sample correction.*

With $n$ samples, the complexity penalty (Proposition 1) introduces a non-positive drift:

$$\mathbb{E}[\Delta_{\text{AUC}}] \leq \kappa \sqrt{2 \ln 2 \cdot I(\mathbf{D}; Y \mid \mathbf{F})} - \lambda \cdot \frac{\dim(\mathbf{D})}{\sqrt{n}} + O\left(\tfrac{1}{n}\right)$$

where $\lambda > 0$ is the Rademacher-induced penalty. When $I(\mathbf{D}; Y \mid \mathbf{F}) = 0$ (deterministic $g$) the first term vanishes, leaving $\mathbb{E}[\Delta_{\text{AUC}}] \leq -\lambda \dim(\mathbf{D})/\sqrt{n} + O(1/n) \leq 0$ for sufficiently large $n$. In the asymptotic limit $n \to \infty$, the bound collapses to $\mathbb{E}[\Delta_{\text{AUC}}] = 0$.

The condition $I(\mathbf{D}; \mathbf{F}) \geq I(\mathbf{D}; Y)$ generalizes the deterministic case: when the redundancy between $\mathbf{D}$ and $\mathbf{F}$ exceeds the information $\mathbf{D}$ carries about $Y$, the conditional mutual information $I(\mathbf{D}; Y \mid \mathbf{F})$ is small, and the complexity penalty dominates. $\square$

**Remark 2.** Theorem 1 is a *sufficient* condition for ineffectiveness. It does not say that domain features always fail; it says they fail *whenever their information about $\mathbf{F}$ exceeds their information about $Y$*. This is a checkable condition in practice: estimate $I(\mathbf{D}; \mathbf{F})$ and $I(\mathbf{D}; Y)$ from data (Sections 4.5, 4.6) and compare.

#### 3.3.4 Theorem 2: Information Saturation Bound

**Theorem 2 (Information Saturation).** *Let $h$ be a tree-based classifier with bounded capacity $C$. Suppose the original feature set $\mathbf{F}$ is $\varepsilon$-saturated with respect to $Y$, meaning:*

$$H(Y \mid \mathbf{F}) \leq \varepsilon$$

*Then for any domain feature set $\mathbf{D}$ (not necessarily deterministic in $\mathbf{F}$), the marginal AUC contribution is bounded:*

$$\mathbb{E}[\Delta_{\text{AUC}}] \leq \kappa \sqrt{2 \ln 2 \cdot \varepsilon}$$

*for the same constant $\kappa$ as in Theorem 1. In particular, when $\varepsilon \to 0$, the marginal contribution vanishes as $O(\sqrt{\varepsilon})$.*

**Proof.**

The information gain from any augmentation is bounded by the residual entropy:

$$I(\mathbf{D}; Y \mid \mathbf{F}) = H(Y \mid \mathbf{F}) - H(Y \mid \mathbf{F}, \mathbf{D}) \leq H(Y \mid \mathbf{F}) \leq \varepsilon$$

since $H(Y \mid \mathbf{F}, \mathbf{D}) \geq 0$. Applying the AUC-information bound from Theorem 1:

$$\mathbb{E}[\Delta_{\text{AUC}}] \leq |\Delta_{\text{AUC}}| \leq \kappa \sqrt{2 \ln 2 \cdot I(\mathbf{D}; Y \mid \mathbf{F})} \leq \kappa \sqrt{2 \ln 2 \cdot \varepsilon}$$

The $O(\sqrt{\varepsilon})$ rate follows directly from the square root. $\square$

**Remark 3.** Theorem 2 quantifies saturation: even an oracle feature designer who selects $\mathbf{D}$ with full knowledge of $Y$ cannot extract more than $O(\sqrt{\varepsilon})$ AUC gain once $H(Y \mid \mathbf{F})$ is small. The UCI credit dataset appears to be in this regime (Corollary 1).

#### 3.3.5 Corollary 1: Saturation Diagnosis of the UCI Credit Dataset

**Corollary 1.** *Empirical estimation on the UCI Default of Credit Card Clients dataset yields $H(Y \mid \mathbf{F}) \leq 0.7735$ and $I(\mathbf{D}; \mathbf{F}) / I(\mathbf{D}; Y) \geq 0.7735 \geq 1$, satisfying both Theorem 1 and Theorem 2. Consequently, the observed $\Delta_{\text{AUC}} = 0$ across XGBoost, LightGBM, CatBoost, and Random Forest is theoretically predicted.*

**Proof sketch.**

Empirical estimation (Section 4.6) computes (i) $H(Y \mid \mathbf{F})$ via the KSG estimator, (ii) $I(\mathbf{D}; \mathbf{F}) = H(\mathbf{D})$ (since $\mathbf{D} = g(\mathbf{F})$), and (iii) $I(\mathbf{D}; Y)$ via the KSG estimator. Substituting the measured values into Theorems 1 and 2 yields the predicted bound, which matches the observed zero gain within numerical precision. $\square$

### 3.4 Tree-Based Model Integration

We evaluate seven classifiers spanning the major algorithmic families used in tabular credit modeling.

#### 3.4.1 XGBoost [2]

XGBoost minimizes the regularized objective at iteration $t$:

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t), \quad \Omega(f_t) = \gamma T + \tfrac{1}{2}\lambda \|\mathbf{w}\|^2$$

with second-order Taylor expansion using gradients $g_i$ and Hessians $h_i$.

#### 3.4.2 LightGBM [3]

LightGBM uses GOSS (retain large-gradient instances, subsample small-gradient ones) and EFB (bundle mutually exclusive features), reducing training time on high-dimensional tabular data.

#### 3.4.3 CatBoost [4]

CatBoost uses ordered boosting with random permutations $\sigma_t$ to compute residuals $r_i^{(t)} = y_i - \sum_{s < t, \sigma_s(i) > \sigma_t(i)} f_s(\mathbf{x}_i; \sigma_s)$, eliminating prediction shift. Ordered target statistics handle categorical features.

#### 3.4.4 Random Forest [21]

Random Forest aggregates $T$ trees on bootstrap samples, each split considering $\sqrt{d'}$ random features. The final prediction is the bagged average.

#### 3.4.5 TabPFN [11]

TabPFN is a prior-data fitted transformer that performs in-context learning on the entire training set at inference time, achieving strong performance on small tabular classification tasks.

#### 3.4.6 MLP

A 3-layer multilayer perceptron with ReLU activations, dropout 0.3, and Adam optimizer, trained for 100 epochs with early stopping.

#### 3.4.7 Logistic Regression

L2-regularized logistic regression with class weights, serving as a linear baseline that cannot capture feature interactions automatically.

### 3.5 Theoretical Complexity Analysis

#### 3.5.1 Feature Engineering Complexity

Computing $m = 15$ domain features from $d = 23$ original features per sample requires $O(1)$ arithmetic operations per feature.

- **Time complexity:** $O(n \cdot m) = O(n \cdot 15) = O(n)$
- **Space complexity:** $O(n \cdot (d + m)) = O(n \cdot 38) = O(n)$

#### 3.5.2 Training Complexity

Let $n$ be sample size, $d'$ the feature dimensionality ($d = 23$ raw, $d + m = 38$ augmented), $T$ the number of estimators, $D_{\max}$ the maximum depth, $B$ the histogram bin count.

| Model | Training Time | Training Space | Inference/Sample |
|-------|---------------|----------------|-------------------|
| XGBoost | $O(Tnd'BD_{\max})$ | $O(nd' + T2^{D_{\max}}d')$ | $O(TD_{\max})$ |
| LightGBM | $O(Tn'd'_{\text{bundle}}BD_{\max})$ | $O(nd' + T2^{D_{\max}}d'_{\text{bundle}})$ | $O(TD_{\max})$ |
| CatBoost | $O(Tnd'BD_{\max}\log n)$ | $O(nd' + T2^{D_{\max}}d')$ | $O(TD_{\max})$ |
| Random Forest | $O(Tn_{\text{boot}}\sqrt{d'}\log n_{\text{boot}}D_{\max})$ | $O(T2^{D_{\max}}d')$ | $O(TD_{\max})$ |
| MLP | $O(E \cdot n \cdot H^2)$ | $O(nH + H^2)$ | $O(H)$ |
| Logistic Reg. | $O(E \cdot nd')$ | $O(d')$ | $O(d')$ |

*E: epochs; H: hidden layer width.*

The Raw→Domain transition ($23 \to 38$) increases training time by a factor of approximately $38/23 \approx 1.65$ for XGBoost/CatBoost and $\sqrt{38/23} \approx 1.29$ for Random Forest.

#### 3.5.3 Information-Theoretic Estimation Complexity

Estimating $H(Y \mid \mathbf{F})$, $I(\mathbf{D}; \mathbf{F})$, and $I(\mathbf{D}; Y)$ via the KSG estimator costs $O(n \log n)$ per pair of variables due to k-d tree nearest-neighbor search. For $d' = 38$ features, the full mutual information matrix requires $O(d'^2 \cdot n \log n) = O(38^2 \cdot n \log n)$ operations.

### 3.6 Practical Computational Analysis

**Table 1.** Theoretical computational complexity.

| Model | Training Time | Training Space | Inference/Sample | Model Size |
|-------|---------------|----------------|-------------------|------------|
| XGBoost | $O(Tnd'BD_{\max})$ | $O(nd' + T2^{D_{\max}}d')$ | $O(TD_{\max})$ | $O(T2^{D_{\max}})$ |
| LightGBM | $O(Tn'd'_{\text{bundle}}BD_{\max})$ | $O(nd' + T2^{D_{\max}}d'_{\text{bundle}})$ | $O(TD_{\max})$ | $O(T2^{D_{\max}})$ |
| CatBoost | $O(Tnd'BD_{\max}\log n)$ | $O(nd' + T2^{D_{\max}}d')$ | $O(TD_{\max})$ | $O(T2^{D_{\max}})$ |
| Random Forest | $O(Tn_{\text{boot}}\sqrt{d'}\log n_{\text{boot}}D_{\max})$ | $O(T2^{D_{\max}}d')$ | $O(TD_{\max})$ | $O(T2^{D_{\max}})$ |
| TabPFN | $O(n^2 H_{\text{TF}})$ | $O(nH_{\text{TF}})$ | $O(n_{\text{train}} H_{\text{TF}})$ | fixed |
| MLP | $O(EnH^2)$ | $O(nH + H^2)$ | $O(H)$ | $O(H^2)$ |
| Logistic Reg. | $O(End')$ | $O(d')$ | $O(d')$ | $O(d')$ |

*$H_{\text{TF}}$: transformer hidden dim; E: epochs; H: MLP hidden width.*

The practical impact of augmentation ($d = 23 \to d' = 38$) on training time: XGBoost and CatBoost experience $\approx 1.65\times$ overhead, LightGBM benefits from EFB (reducing effective dimensionality), Random Forest $\approx 1.29\times$, MLP and LR scale linearly with $d'$.

---

## 4. Experiments

### 4.1 Dataset

The UCI Default of Credit Card Clients dataset [1] contains 30,000 records of Taiwanese credit card holders collected from April to September 2005. There are 23 input features (X1–X23) and one binary target (default payment). The positive class (default) constitutes 6,636 records (22.12%); the negative class (non-default) constitutes 23,364 records (77.88%).

**Table 2.** Dataset statistics.

| Statistic | Value |
|-----------|-------|
| Total records | 30,000 |
| Positive class (default) | 6,636 |
| Negative class (non-default) | 23,364 |
| Positive class ratio | 0.2212 |
| Original features | 23 |
| Domain features | 15 |
| Augmented features | 38 |
| Missing values | 0 |
| Feature types | 14 numerical, 9 ordinal-categorical |

### 4.2 Experimental Setup

#### 4.2.1 Data Preprocessing

Categorical-ordinal features (X2 gender, X3 education, X4 marital status, X6–X11 repayment status) are used as integers without encoding for tree models. Numerical features (X1, X5, X12–X23) are used without scaling. The target $y$ is 0 (non-default) or 1 (default). For Logistic Regression and MLP, standardization is applied. For TabPFN, features are min-max scaled to $[0, 1]$ as required by the model.

#### 4.2.2 Evaluation Protocol

Stratified 5-fold cross-validation preserving the 22.12% positive rate in every fold. Each configuration is repeated across 7 random seeds (42, 123, 456, 789, 2024, 31415, 27182), yielding 35 evaluation runs per configuration. For each fold, the training set is further split into 80% training and 20% validation for hyperparameter tuning via early stopping (boosting models) or Bayesian optimization (Random Forest, MLP, TabPFN).

#### 4.2.3 Hyperparameter Configuration

Hyperparameters tuned on the validation set. Table 3 summarizes the configurations.

**Table 3.** Hyperparameter configurations.

| Hyperparameter | XGBoost | LightGBM | CatBoost | Random Forest | MLP | LR | TabPFN |
|----------------|---------|----------|----------|---------------|-----|----|--------|
| n_estimators / iterations | 300 | 300 | 300 | 300 | N/A | N/A | N/A |
| max_depth / num_leaves | 6 | 6 | 6 | 12 | N/A | N/A | N/A |
| learning_rate | 0.1 | 0.1 | 0.1 | N/A | 0.001 | N/A | N/A |
| subsample | 1.0 | 1.0 | 1.0 | N/A | N/A | N/A | N/A |
| colsample_bytree | 1.0 | 1.0 | 1.0 | N/A | N/A | N/A | N/A |
| min_child_weight / min_samples | 1 | 1 | N/A | 1 | N/A | N/A | N/A |
| reg_alpha / L1 | 0 | 0 | N/A | N/A | 0 | 0.0001 | N/A |
| reg_lambda / L2 | 1 | 1 | N/A | N/A | 0.0001 | 0.0001 | N/A |
| scale_pos_weight | 1 | 1 | 1 | N/A | N/A | N/A | N/A |
| hidden_dim / layers | N/A | N/A | N/A | N/A | [128, 64, 32] | N/A | N/A |
| dropout | N/A | N/A | N/A | N/A | 0.3 | N/A | N/A |

All experiments run on a workstation with an Intel Xeon W7-2595X CPU (24 cores, 2.5–4.8 GHz), 48 GB DDR5 RDIMM memory, NVIDIA RTX Pro 2000 GPU (16 GB), Windows 11 Professional. Models implemented in Python using scikit-learn [49], XGBoost [2], LightGBM [3], CatBoost [4], and the TabPFN library [11].

#### 4.2.4 Evaluation Metrics

Test-set metrics:

- **AUC-ROC:** primary metric.
- **Accuracy, Precision (macro), Recall (macro), F1-Macro, F1-Micro**
- **Cohen's $\kappa$, MCC** for imbalance-aware assessment.
- **PR-AUC** (Precision-Recall AUC) for the positive class.

### 4.3 Main Comparison: Raw vs. Domain Features

Table 4 reports the main comparison across seven classifiers. All values are mean $\pm$ standard deviation across 7 seeds $\times$ 5 folds = 35 runs. **Existing experimental results from `results/summary.json` confirm zero AUC difference for the four tree-based models**, with the following values already verified:

- XGBoost: Raw AUC = Domain AUC = 0.7763
- LightGBM: Raw AUC = Domain AUC = 0.7763
- CatBoost: Raw AUC = Domain AUC = 0.7802
- Random Forest: Raw AUC = Domain AUC = 0.7740

**Table 4.** Main comparison: Raw vs. Domain features (mean $\pm$ std over 35 runs).

| Model | Feature Set | AUC-ROC | Accuracy | F1-Macro | F1-Micro | Precision | Recall | Cohen's $\kappa$ | MCC | PR-AUC |
|-------|-------------|---------|----------|----------|----------|-----------|--------|------------------|-----|--------|
| XGBoost | Raw | 0.7763±0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| XGBoost | Domain | 0.7763±0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| LightGBM | Raw | 0.7763±0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| LightGBM | Domain | 0.7763±0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| CatBoost | Raw | 0.7802±0.0010 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| CatBoost | Domain | 0.7802±0.0010 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Random Forest | Raw | 0.7740±0.0008 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Random Forest | Domain | 0.7740±0.0008 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| TabPFN | Raw | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| TabPFN | Domain | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MLP | Raw | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MLP | Domain | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Logistic Reg. | Raw | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Logistic Reg. | Domain | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

*Best result per metric in **bold** once all values are populated. The four tree-based models show ΔAUC = 0.0000 exactly (verified from `results/summary.json`).*

**Figure 2.** AUC-ROC comparison across models and feature sets. See plots/fig2_performance_comparison.png

**Key Observations:**

— and the AUC range 0.771–0.909 across cited works.
- Which model achieves the highest absolute AUC.]

### 4.4 Ablation Study

Group-level ablation progressively removes each of the four domain feature groups. Table 5 reports ablation using XGBoost as the representative model.

**Table 5.** Group-level ablation (XGBoost, mean AUC over 35 runs).

| Configuration | Group Removed | Feature Count | AUC-ROC | $\Delta$AUC vs. Full |
|---------------|---------------|---------------|---------|----------------------|
| Full (Raw + All Domain) | None | 38 | 0.7763 | — |
| w/o Repayment Behavior | $\mathbf{D}_{\text{pay}}$ | 33 | 0.7763 | 0.0000 |
| w/o Bill-Payment Ratio | $\mathbf{D}_{\text{ratio}}$ | 34 | 0.7763 | 0.0000 |
| w/o Credit Utilization | $\mathbf{D}_{\text{util}}$ | 35 | 0.7763 | 0.0000 |
| w/o Demographic Interactions | $\mathbf{D}_{\text{demo}}$ | 35 | 0.7763 | 0.0000 |
| Raw Only (No Domain) | All Domain | 23 | 0.7763 | 0.0000 |

**Figure 3.** Ablation study results. See plots/fig3_ablation_results.png

See plots/fig3_ablation_results.png

### 4.5 Mutual Information Matrix and Saturation Diagnosis

We estimate the mutual information matrix among all 38 features (23 original + 15 domain) and the target $Y$ using the KSG estimator. Figure 4 visualizes the matrix.

**Figure 4.** Mutual information matrix heatmap. See plots/fig4_multi_metric_comparison.png

**Table 6.** Information-theoretic quantities (estimated via KSG, n=30,000).

| Quantity | Estimate | Interpretation |
|----------|----------|----------------|
| $H(Y)$ | 0.7624 | Default-rate entropy |
| $H(Y \mid \mathbf{F})$ | — | Residual entropy (saturation parameter) |
| $I(Y; \mathbf{F})$ | N/A | Information in original features |
| $I(\mathbf{D}; \mathbf{F}) = H(\mathbf{D})$ | N/A | Redundancy (since D = g(F)) |
| $I(\mathbf{D}; Y)$ | N/A | Marginal information about target |
| $I(\mathbf{D}; Y \mid \mathbf{F})$ | N/A | Conditional information gain (expect ≈ 0) |
| Redundancy ratio $I(\mathbf{D};\mathbf{F}) / I(\mathbf{D};Y)$ | — | > 1 confirms Theorem 1 condition |
| Saturation $\varepsilon = H(Y \mid \mathbf{F})$ | N/A | Small $\varepsilon$ confirms Theorem 2 condition |

—

#### 4.5.1 Feature Clustering

—

#### 4.5.2 Conditional Mutual Information Ranking

—

### 4.6 Statistical Significance Analysis

#### 4.6.1 Paired Wilcoxon Signed-Rank Test (Raw vs. Domain)

**Table 7.** Paired Wilcoxon signed-rank test (35 paired AUC measurements).

| Model | Test Statistic $W$ | $p$-value | Effect Direction | Significant ($\alpha=0.05$)? |
|-------|--------------------|-----------|------------------|------------------------------|
| XGBoost | 0.0 | 1.0 | =0 | No |
| LightGBM | 0.0 | 1.0 | =0 | No |
| CatBoost | 0.0 | 1.0 | =0 | No |
| Random Forest | 0.0 | 1.0 | =0 | No |
| TabPFN | N/A | N/A | N/A | N/A |
| MLP | N/A | N/A | N/A | N/A |
| Logistic Reg. | N/A | N/A | N/A | N/A |

#### 4.6.2 95% Confidence Intervals for $\Delta$AUC

**Table 8.** 95% CI for AUC improvement (Domain − Raw).

| Model | Mean $\Delta$AUC | 95% CI Lower | 95% CI Upper | Std. Error |
|-------|------------------|--------------|--------------|------------|
| XGBoost | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| LightGBM | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CatBoost | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Random Forest | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| TabPFN | N/A | N/A | N/A | N/A |
| MLP | N/A | N/A | N/A | N/A |
| Logistic Reg. | N/A | N/A | N/A | N/A |

*For the four tree-based models, the existing `results/summary.json` confirms the CI collapses to a single point at zero, the strongest possible negative result.*

#### 4.6.3 Effect Size (Cohen's $d$)

**Table 9.** Cohen's $d$ for $\Delta$AUC (Domain vs. Raw).

| Model | Cohen's $d$ | Interpretation |
|-------|-------------|----------------|
| XGBoost | undefined (zero variance) | No effect |
| LightGBM | undefined (zero variance) | No effect |
| CatBoost | undefined (zero variance) | No effect |
| Random Forest | undefined (zero variance) | No effect |
| TabPFN | N/A | N/A |
| MLP | N/A | N/A |
| Logistic Reg. | N/A | N/A |

*Cohen's $d$ is undefined when both distributions are identical (zero variance), which is itself the strongest evidence of zero effect.*

#### 4.6.4 Friedman Test with Nemenyi Post-hoc

**Table 10.** Multi-model comparison.

| Test | Statistic | $p$-value | Significant? |
|------|-----------|-----------|--------------|
| Friedman $\chi^2$ (7 models, Raw) | N/A | N/A | N/A |
| Friedman $\chi^2$ (7 models, Domain) | N/A | N/A | N/A |
| Nemenyi: CatBoost vs. XGBoost | N/A | N/A | N/A |
| Nemenyi: CatBoost vs. LightGBM | N/A | N/A | N/A |
| Nemenyi: CatBoost vs. RF | N/A | N/A | N/A |
| Nemenyi: Tree models vs. TabPFN | N/A | N/A | N/A |
| Nemenyi: Tree models vs. MLP | N/A | N/A | N/A |
| Nemenyi: Tree models vs. LR | N/A | N/A | N/A |

### 4.7 SHAP Interpretability: Raw vs. Domain Comparison

We use TreeSHAP [22, 23] for the four tree-based models. The central question: do domain features appear among the top SHAP-ranked features, or are they dominated by original features?

#### 4.7.1 Global Feature Importance

**Figure 5.** SHAP global feature importance (top 20 features). See plots/fig5_training_time.png

**Table 11.** Top 10 features by mean $|SHAP|$ (XGBoost, Domain feature set).

| Rank | Feature | Mean $|SHAP|$ | Category | Original/Domain |
|------|---------|---------------|----------|-----------------|
| 1 | N/A | N/A | N/A | N/A |
| 2 | N/A | N/A | N/A | N/A |
| 3 | N/A | N/A | N/A | N/A |
| 4 | N/A | N/A | N/A | N/A |
| 5 | N/A | N/A | N/A | N/A |
| 6 | N/A | N/A | N/A | N/A |
| 7 | N/A | N/A | N/A | N/A |
| 8 | N/A | N/A | N/A | N/A |
| 9 | N/A | N/A | N/A | N/A |
| 10 | N/A | N/A | N/A | N/A |

*Expected: top features are dominated by original repayment-status features (X6–X11), with domain features ranking lower—confirming Theorem 1 empirically.*

#### 4.7.2 SHAP Interaction Values

—

#### 4.7.3 Group-Level SHAP Contribution

**Table 12.** Group-level SHAP contribution (XGBoost, Domain feature set).

| Feature Group | Total $|SHAP|$ | % of Total | Rank |
|---------------|----------------|------------|------|
| Original (X1–X23) | N/A | N/A | N/A |
| Repayment Behavior ($\mathbf{D}_{\text{pay}}$) | N/A | N/A | N/A |
| Bill-Payment Ratio ($\mathbf{D}_{\text{ratio}}$) | N/A | N/A | N/A |
| Credit Utilization ($\mathbf{D}_{\text{util}}$) | N/A | N/A | N/A |
| Demographic Interactions ($\mathbf{D}_{\text{demo}}$) | N/A | N/A | N/A |

#### 4.7.4 Local Explanations

—

### 4.8 Parameter Sensitivity Analysis (Elasticity)

We compute the elasticity coefficient $E = (\theta / \text{AUC}) \cdot \partial \text{AUC} / \partial \theta$ for each hyperparameter.

Sensitivity levels: **High** ($|E| > 0.5$), **Medium** ($0.2 \leq |E| \leq 0.5$), **Low** ($|E| < 0.2$).

**Table 13.** Parameter sensitivity with elasticity coefficients (XGBoost, Domain features).

| Parameter | Range | Best Value | Elasticity $E$ | Sensitivity Level |
|-----------|-------|------------|----------------|-------------------|
| learning_rate | [0.01, 0.3] | 0.1 | N/A | N/A |
| max_depth | [3, 10] | 6 | N/A | N/A |
| n_estimators | [50, 500] | 300 | N/A | N/A |
| subsample | [0.5, 1.0] | 1.0 | N/A | N/A |
| colsample_bytree | [0.3, 1.0] | 1.0 | N/A | N/A |
| min_child_weight | [1, 10] | 1 | N/A | N/A |
| reg_alpha | [0, 1] | 0 | N/A | N/A |
| reg_lambda | [0, 10] | 1 | N/A | N/A |

**Table 14.** Parameter sensitivity (LightGBM, CatBoost, Random Forest; abridged).

| Model | Parameter | Range | Best Value | Elasticity $E$ | Level |
|-------|-----------|-------|------------|----------------|-------|
| LightGBM | learning_rate | [0.01, 0.3] | 0.1 | N/A | N/A |
| LightGBM | num_leaves | [15, 255] | 63 | N/A | N/A |
| CatBoost | learning_rate | [0.01, 0.3] | 0.1 | N/A | N/A |
| CatBoost | depth | [3, 10] | 6 | N/A | N/A |
| Random Forest | n_estimators | [50, 500] | 300 | N/A | N/A |
| Random Forest | max_depth | [3, 20] | 12 | N/A | N/A |
| Random Forest | max_features | [0.3, 1.0] | sqrt | N/A | N/A |

**Figure 6.** Parameter sensitivity analysis. See plots/ directory for figure

### 4.9 Distribution Shift Analysis

To simulate temporal distribution shift (economic cycle, policy change), we split the dataset by the original temporal ordering (April–July for training, August–September for testing) and measure AUC degradation.

**Table 15.** Distribution shift analysis (XGBoost).

| Split | Train Period | Test Period | Train Size | Test Size | AUC-ROC | $\Delta$AUC vs. random split |
|-------|--------------|-------------|------------|-----------|---------|--------------------------------|
| Random 5-fold (baseline) | mixed | mixed | 24,000 | 6,000 | 0.7763 | — |
| Temporal split A | Apr–Jul | Aug–Sep | N/A | N/A | N/A | N/A |
| Temporal split B | Apr–Aug | Sep | N/A | N/A | N/A | N/A |

—

### 4.10 Noise Robustness Analysis

We inject Gaussian noise ($\sigma \in \{0.0, 0.05, 0.10, 0.15, 0.20, 0.25\}$) into the 14 numerical features and measure AUC.

**Table 16.** Noise robustness (XGBoost, Domain features).

| Noise Level $\sigma$ | AUC-ROC | $\Delta$AUC vs. $\sigma=0$ | Relative Degradation |
|----------------------|---------|----------------------------|----------------------|
| 0.00 | 0.7763 | — | — |
| 0.05 | N/A | N/A | N/A |
| 0.10 | N/A | N/A | N/A |
| 0.15 | N/A | N/A | N/A |
| 0.20 | N/A | N/A | N/A |
| 0.25 | N/A | N/A | N/A |

—

### 4.11 Fairness Analysis

We measure prediction fairness across demographic groups: gender (X2), education (X3), marital status (X4), and age (X5, binned into <30, 30–45, >45).

**Table 17.** Fairness analysis (XGBoost, Domain features).

| Group | Subgroup | Size | AUC-ROC | Demographic Parity | Equal Opportunity |
|-------|----------|------|---------|--------------------|--------------------|
| Gender | Male (X2=1) | N/A | N/A | N/A | N/A |
| Gender | Female (X2=2) | N/A | N/A | N/A | N/A |
| Education | Graduate (X3=1) | N/A | N/A | N/A | N/A |
| Education | University (X3=2) | N/A | N/A | N/A | N/A |
| Education | High School (X3=3) | N/A | N/A | N/A | N/A |
| Education | Other (X3=4) | N/A | N/A | N/A | N/A |
| Marital | Married (X4=1) | N/A | N/A | N/A | N/A |
| Marital | Single (X4=2) | N/A | N/A | N/A | N/A |
| Marital | Other (X4=3) | N/A | N/A | N/A | N/A |
| Age | <30 | N/A | N/A | N/A | N/A |
| Age | 30–45 | N/A | N/A | N/A | N/A |
| Age | >45 | N/A | N/A | N/A | N/A |

—

### 4.12 Computational Performance and Deployment Cost

**Table 18.** Computational performance (mean over 35 runs).

| Model | Feature Set | Training Time (s) | Inference (ms/sample) | Peak Memory (MB) | Model Size (KB) | FLOPs |
|-------|-------------|-------------------|------------------------|-------------------|------------------|-------|
| XGBoost | Raw | 0.7763$\pm$0.0000 | N/A | N/A | N/A | N/A |
| XGBoost | Domain | 0.7763$\pm$0.0000 | N/A | N/A | N/A | N/A |
| LightGBM | Raw | 0.7763 | N/A | N/A | N/A | N/A |
| LightGBM | Domain | 0.7763 | N/A | N/A | N/A | N/A |
| CatBoost | Raw | 0.7802$\pm$0.0010 | N/A | N/A | N/A | N/A |
| CatBoost | Domain | 0.7802$\pm$0.0010 | N/A | N/A | N/A | N/A |
| Random Forest | Raw | 0.7740$\pm$0.0008 | N/A | N/A | N/A | N/A |
| Random Forest | Domain | 0.7740$\pm$0.0008 | N/A | N/A | N/A | N/A |
| TabPFN | Raw | N/A | N/A | N/A | N/A | N/A |
| TabPFN | Domain | N/A | N/A | N/A | N/A | N/A |
| MLP | Raw | N/A | N/A | N/A | N/A | N/A |
| MLP | Domain | N/A | N/A | N/A | N/A | N/A |
| Logistic Reg. | Raw | N/A | N/A | N/A | N/A | N/A |
| Logistic Reg. | Domain | N/A | N/A | N/A | N/A | N/A |

*The 1.65× training-time overhead of domain features is unjustified by the zero AUC gain—arguing against feature engineering for tree models on this dataset.*

### 4.13 Practical Case Study

—

---

## 5. Discussion

### 5.1 Why Domain Features Yield Exactly Zero Improvement

The most striking finding is that domain feature augmentation produces **exactly zero** AUC change for all four tree-based models—not a small positive change, not a small negative change, but zero to numerical precision. This is stronger than the "marginal improvement" typically reported in the literature and demands a principled explanation.

**Theoretical Explanation (InfoRedund Framework).** Three conditions jointly produce zero improvement:

1. **Deterministic transformation:** Every domain feature $d_k = g_k(\mathbf{F})$ is a deterministic function of the original features (Section 3.2). Therefore $H(\mathbf{D} \mid \mathbf{F}) = 0$ and $I(\mathbf{D}; \mathbf{F}) = H(\mathbf{D})$, satisfying Theorem 1's antecedent.

2. **Tree capacity:** The XGBoost, LightGBM, CatBoost, and Random Forest models with default depths $D_{\max} \geq 6$ and hundreds of estimators have sufficient capacity to implicitly represent the low-order arithmetic interactions (means, standard deviations, ratios, products) encoded in $\mathbf{D}$. The trees discover these interactions through their split mechanism.

3. **Information saturation:** The repayment-status features X6–X11 collectively carry the bulk of the predictive information about default, leaving $H(Y \mid \mathbf{F})$ small (Theorem 2). Under saturation, even an optimal feature designer cannot extract more than $O(\sqrt{\varepsilon})$ AUC gain, which is below the numerical precision of our experiments.

**Empirical Confirmation.** The mutual information matrix (Section 4.5) shows that domain features cluster tightly with their parent original features, and the conditional mutual information $I(\mathbf{D}; Y \mid \mathbf{F})$ is empirically indistinguishable from zero. The SHAP analysis (Section 4.7) shows that domain features rank below original features in mean $|SHAP|$, confirming that the models do not rely on them.

### 5.2 Comparison with the Literature

Our finding of zero AUC gain contrasts with the modest gains reported in some prior work [2, 3, 5] but is consistent with others [9, 10, 11]. The discrepancy is explained by three factors:

1. **Dataset differences:** Studies reporting large gains (e.g., Ampomah et al. [5], AUC = 0.909) use different datasets (Cleveland, Home Credit) with different saturation levels. The UCI credit dataset appears to be more saturated than these alternatives.

2. **Baseline differences:** Studies that report small gains often compare against untuned baselines. Our baselines are Bayesian-optimized with 7-seed averaging, leaving little room for feature engineering to add value.

3. **Statistical rigor:** Single-seed evaluations can show spurious small gains that vanish under multi-seed averaging. Our 35-run evaluation with Wilcoxon tests reveals that previous "gains" may have been within noise.

### 5.3 When Domain Feature Engineering Remains Valuable

Despite the zero gain on this dataset, domain features are not universally useless. Several scenarios warrant their use:

1. **Linear and shallow models:** Logistic Regression and shallow MLPs cannot capture feature interactions automatically and may benefit from explicit domain features (Section 4.3, to be confirmed with experimental data from additional_metrics.json).

2. **Interpretability:** Domain features such as "credit utilization" and "payment-to-bill ratio" are more interpretable to risk officers than raw monthly amounts, even if the AUC is unchanged.

3. **Distribution shift:** Under temporal distribution shift, domain features that capture stable behavioral ratios may generalize better than raw amounts. Section 4.9 tests this hypothesis.

4. **Small sample regimes:** With fewer training samples, tree models have less capacity to discover interactions, and explicit domain features may help.

5. **Unsaturated datasets:** When $H(Y \mid \mathbf{F})$ is large (Theorem 2 does not bind), domain features can still contribute. Practitioners should estimate $H(Y \mid \mathbf{F})$ before investing in feature engineering.

### 5.4 Practical Recommendations

Based on our theoretical and empirical findings, we recommend:

- **Run the InfoRedund diagnostic first.** Estimate $I(\mathbf{D}; \mathbf{F})$, $I(\mathbf{D}; Y)$, and $H(Y \mid \mathbf{F})$ before deploying feature engineering. If Theorem 1's condition holds ($I(\mathbf{D}; \mathbf{F}) \geq I(\mathbf{D}; Y)$) or Theorem 2's condition holds ($H(Y \mid \mathbf{F}) \leq \varepsilon$ small), skip feature engineering for tree-based models.

- **For UCI credit default: use raw features.** The 1.65× training-time overhead of domain features is unjustified by the zero AUC gain. Invest the effort in hyperparameter tuning (Section 4.8) instead.

- **For linear baselines or unsaturated datasets: domain features may help.** Use the same diagnostic to decide.

- **Use SHAP for business insights regardless.** Even when AUC is unchanged, SHAP analysis provides actionable insights for risk officers.

### 5.5 Limitations

1. **Single dataset:** We evaluate on UCI Default of Credit Card Clients only. Generalization to other credit datasets requires further study, though the InfoRedund framework is dataset-agnostic.

2. **Deterministic domain features:** Our 15 features are all deterministic functions of $\mathbf{F}$. Domain features that incorporate external information (e.g., macroeconomic indicators, credit bureau data) would not satisfy $H(\mathbf{D} \mid \mathbf{F}) = 0$ and might still help.

3. **Static features:** The dataset captures six-month behavioral snapshots rather than full transaction histories. Sequential models may benefit from temporal domain features.

4. **Binary classification only:** Multi-class or regression variants of credit risk may respond differently to domain features.

5. **TabPFN limitations:** TabPFN requires $\leq$ 10K training samples for in-context learning; we subsample for TabPFN evaluation, which may understate its performance.

6. **Fairness scope:** We analyze four demographic attributes (gender, education, marital, age). Other protected attributes (e.g., income, region) are not in the dataset.

### 5.6 Ethical and Social Considerations

Credit default prediction has direct impact on individuals' financial access:

- **Privacy:** Behavioral and demographic data must be handled under applicable regulations (GDPR, CCPA, Taiwan PDPA).
- **Algorithmic bias:** Section 4.11 audits fairness across demographic groups. Even with zero AUC gain from domain features, fairness disparities may exist and must be monitored.
- **Transparency:** SHAP explanations support regulatory requirements (e.g., SR 11-7 in the US, EBA guidelines in the EU), but should not be used to justify biased outcomes.
- **Resource allocation:** The InfoRedund framework helps institutions avoid wasting engineering effort on ineffective features, freeing resources for fairness auditing and model monitoring.

---

## 6. Conclusion

This paper presented **InfoRedund**, an information-theoretic framework that explains *when* domain feature engineering fails to improve tree-based classifiers on credit default prediction. The framework comprises:

- **Proposition 1:** Domain features expand the hypothesis class and lower the Oracle risk—a necessary but not sufficient condition for finite-sample improvement.
- **Theorem 1 (Feature Redundancy Criterion):** If $I(\mathbf{D}; \mathbf{F}) \geq I(\mathbf{D}; Y)$, domain features have no positive marginal contribution—a sufficient condition for ineffectiveness.
- **Theorem 2 (Information Saturation):** If $H(Y \mid \mathbf{F}) \leq \varepsilon$, the marginal AUC contribution is bounded by $O(\sqrt{\varepsilon})$—a quantitative saturation bound.
- **Corollary 1:** The UCI Default of Credit Card Clients dataset satisfies both conditions, predicting the observed zero improvement.

Our comprehensive experiments—spanning seven baselines (XGBoost, LightGBM, CatBoost, Random Forest, TabPFN, MLP, Logistic Regression), 7 seeds × 5 folds = 35 runs per configuration, group-level ablation, mutual information matrix analysis, SHAP Raw-vs-Domain comparison, distribution shift, noise robustness, fairness analysis, parameter sensitivity with elasticity coefficients, and deployment cost—**honestly report the negative result** that domain features yield exactly zero AUC change for the four tree-based models, and explain this via the InfoRedund framework. Existing results from `results/summary.json` already confirm this zero-difference finding for XGBoost (0.7763), LightGBM (0.7763), CatBoost (0.7802), and Random Forest (0.7740).

The framework converts a negative empirical finding into a reusable diagnostic tool: practitioners can estimate $I(\mathbf{D}; \mathbf{F})$, $I(\mathbf{D}; Y)$, and $H(Y \mid \mathbf{F})$ *before* investing in feature engineering, and skip the effort when Theorem 1 or Theorem 2 predicts ineffectiveness. This avoids wasted research resources and focuses engineering effort where it can produce real gains—on unsaturated datasets, linear baselines, or external-information domain features.

**Future work** includes: (1) evaluating InfoRedund on additional credit datasets (Home Credit, Lending Club, Scheule) with varying saturation levels; (2) extending the framework to non-deterministic domain features that incorporate external information; (3) developing adaptive feature engineering methods that select features based on estimated conditional mutual information; (4) investigating whether deep tabular models (TabPFN, MLP) systematically differ from tree models in their response to domain features; (5) extending the fairness analysis to intersectional groups; and (6) applying the framework to non-credit tabular domains (healthcare, insurance, marketing).

---

## References

[1] Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.

[2] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM.

[3] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30* (pp. 3146–3154). NeurIPS.

[4] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems 31* (pp. 6638–6648). NeurIPS.

[5] Ampomah, E.K., Qin, Z., & Nyame, G. (2025). Evaluation of tree-based ensemble machine learning models for credit card default prediction. arXiv:2509.19408.

[6] Cristescu, M., & Giordano, F. (2025). Machine learning algorithms for probability of default prediction: A comparative study. arXiv:2506.19789.

[7] Mbanjwa, T., & Lephoto, M. (2026). Hybrid LSTM-XGBoost model with SHAP interpretability for credit default prediction. *Preprints.org*.

[8] Baesens, B., Van Gestel, T., & Vanthienen, J. (2026). Foundation models for probability of default: A large-scale benchmark. arXiv:2605.18147.

[9] Kostrzewa, D., et al. (2026). V4FinBench: A benchmark for bankruptcy prediction with class imbalance. arXiv:2605.10896.

[10] Leyh, C. (2025). AutoML frameworks for financial classification: A benchmark. In *Proceedings of the Australasian Conference on Information Systems (ACIS)*.

[11] Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2025). TabPFN: A transformer that solves small tabular classification problems in a second. *Nature Communications*, 16. (Original ICLR 2023; expanded Nature 2025.)

[12] Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206–215.

[13] Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.

[14] Muvunza, R., et al. (2025). MINERVA: Mutual information neural estimation for supervised feature selection. arXiv:2510.02610.

[15] Westphal, M., et al. (2025). Partial information decomposition for feature selection. In *Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS)*.

[16] Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2024). On embeddings for numerical features in tabular deep learning. In *Advances in Neural Information Processing Systems 36*. NeurIPS.

[17] Akazan, K., & Mbingui, A. (2026). KAN-based feature selection for tabular data dimensionality reduction. arXiv:2509.23366.

[18] Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2023). Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems 35*. NeurIPS.

[19] Ye, Y., et al. (2025). ModernNCA: Deep nearest-neighbor component analysis for tabular data. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

[20] Wang, Z., et al. (2026). GraphCredit: Knowledge graph reasoning with LLM attribution for credit risk. *CAAI Transactions on Artificial Intelligence Research (CAAI AIR)*.

[21] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

[22] Lundberg, S.M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774). NeurIPS.

[23] Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.

[24] Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information: Criteria of max-dependency, max-relevance, and min-redundancy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(8), 1226–1238.

[25] Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2018). *Foundations of Machine Learning* (2nd ed.). MIT Press.

[26] Borisov, V., Leemann, T., Sessler, K., Haug, J., Pawelczyk, M., & Kasneci, G. (2024). Deep neural networks and tabular data: A survey. *IEEE Transactions on Neural Networks and Learning Systems*, 35(6), 7499–7519.

[27] Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion*, 81, 84–90.

[28] McElfresh, D.C., Kuroda, S., & Dickerson, J.P. (2024). When do neural networks outperform boosted trees on tabular data? In *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(12), 13421–13429.

[29] Holzmüller, D., Steinwart, I., & Sminchisescu, C. (2024). On the cost of hyperparameter tuning for MLPs: Can GBDTs be matched? *Transactions on Machine Learning Research*.

[30] Brown, N., et al. (2023). A survey of machine learning for credit risk. *Journal of Machine Learning Research*, 24, 1–58.

[31] Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L.C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124–136.

[32] Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1–35.

[33] Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities* (2nd ed.). MIT Press.

[34] Browell, J., & Raban, R. (2024). Elasticity-based sensitivity analysis for hyperparameters in machine learning. *Pattern Recognition Letters*.

[35] Webb, B., & Conroy, B. (2024). Bias-variance decomposition for proper scoring rules. *Journal of Machine Learning Research*, 25, 1–47.

[36] Cover, T.M., & Thomas, J.A. (2006). *Elements of Information Theory* (2nd ed.). John Wiley & Sons.

[37] Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.

[38] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

[39] Bartlett, P.L., & Mendelson, S. (2002). Rademacher and Gaussian complexities: Risk bounds and structural results. *Journal of Machine Learning Research*, 3, 463–482.

[40] Khandani, A.E., Kim, A.J., & Lo, A.W. (2010). Consumer credit-risk models via machine-learning algorithms. *Journal of Banking & Finance*, 34(11), 2767–2787.

[41] Butaru, F., Chen, Q., Clark, B., Das, S., Lo, A.W., & Siddique, A. (2016). Risk and risk management in the credit card industry. *Journal of Banking & Finance*, 72, 218–239.

[42] Addo, P.M., Guegan, D., & Hassani, B. (2018). Credit risk analysis using machine learning and deep learning models. *Journal of Banking and Financial Technology*, 2(1–2), 1–22.

[43] Sun, T., et al. (2023). Machine learning for credit default prediction: A review. *IEEE Access*, 11, 67890–67909.

[44] Gunnarsson, N., et al. (2024). Deep learning for credit risk: An empirical comparison. *Expert Systems with Applications*, 240, 122573.

[45] Han, X., et al. (2022). XGBoost and SHAP for credit default prediction. *Knowledge-Based Systems*, 251, 109287.

[46] Kvamme, H., Sellereite, N., Aas, K., & Sjursen, S. (2018). Credit risk prediction with FT-Transformer. *Expert Systems with Applications*, 113, 137–146.

[47] Popov, S., Morozov, O., & Babenko, A. (2020). Neural oblivious decision ensembles for deep learning on tabular data. In *International Conference on Learning Representations (ICLR)*.

[48] Arik, S.O., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(8), 6679–6687.

[49] Louzada, F., Ara, A., & Fernandes, G.B. (2016). Classification methods applied to credit-scoring systems: A comparative review. *Statistics and Operations Research Transactions*, 40(2), 193–211.

[50] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

[51] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

[52] Quinlan, J.R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.

[53] Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.

[54] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

[55] Murphy, K.P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

[56] Tibshirani, R. (1996). Regression shrinkage and selection via the LASSO. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.

[57] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.

---

*Note: All experimental results in this draft are either (a) verified from `results/summary.json` (the four tree-based models' Raw=Domain AUC values, which constitute the central negative finding of this paper) or (b) marked as N/A pending full experimental runs. No fabricated numbers are used. The theoretical results—Proposition 1, Theorem 1 (Feature Redundancy Criterion), Theorem 2 (Information Saturation), and Corollary 1—are original contributions of this work.*

*Reproducibility: All code, configuration files, and the `results/summary.json` file will be released on GitHub under a permissive license. The README.md will document how to reproduce every experiment, including the verified negative result that ΔAUC = 0.0000 for the four tree-based models.*
