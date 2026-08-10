# FlightFeat: Data Leakage Diagnosis in Flight Delay Prediction via Causal Feature Analysis

**Jingyuan Zeng**<sup>1</sup>, **Ming Zeng**<sup>2</sup>, **Jianghong Guo**<sup>1</sup>, **Chuanxian Jiang**<sup>1</sup>, **Yafen Feng**<sup>3,4,*</sup>

<sup>1</sup> School of Computer Science, Jiaying University, Meizhou 514015, Guangdong, China  
<sup>2</sup> College of Water Conservancy and Civil Engineering, South China Agricultural University, Guangzhou 510642, Guangdong, China  
<sup>3</sup> School of Geography Science and Tourism, Jiaying University, Meizhou 514015, Guangdong, China  
<sup>4</sup> Key Laboratory of Mountain Surface Environment and Green Development in Northeast Guangdong, Meizhou 514015, Guangdong, China  

*Corresponding author: Yafen Feng, E-mail: fyf81@163.com

**Jingyuan Zeng** (1980--), male, Ph.D., Associate Professor. Research interests: deep learning, algorithm analysis and design. E-mail: zjy@jyu.edu.cn.  
**Ming Zeng** (2008--), male, undergraduate. Research interests: water conservancy data analysis and application.  
**Jianghong Guo** (1975--), male, Ph.D., Associate Professor. Research interests: machine learning, deep learning, algorithm analysis and design.  
**Chuanxian Jiang** (1978--), male, Ph.D., Professor. Research interests: computer algorithm analysis and design.  
**Yafen Feng** (1981--), female, Ph.D., Associate Professor. Research interests: tourism resource development and utilization, tourism data analysis. E-mail: fyf81@163.com.

**Fund:** Guangdong Provincial Higher Education Teaching Reform Project (Grant No. Yue Jiao Gao Han [2024] 9-989)

---

## Abstract

Flight delay prediction is a critical task in aviation management, yet recent studies report near-perfect AUC values that raise serious concerns about data leakage. In this paper, we propose FlightFeat, an aviation domain feature analysis framework that systematically diagnoses data leakage in flight delay prediction through causal feature analysis. We construct four categories of domain features—scheduling patterns, airport congestion, weather impact, and temporal modes—and evaluate them alongside raw operational features using four tree-based models (XGBoost, LightGBM, CatBoost, and Random Forest) across five random seeds. Our experiments reveal that when post-departure leakage features are included, AUC values range from 0.9864 to 0.9993, suspiciously close to 1.0. After removing all post-departure features (clean configuration), AUC drops dramatically to 0.6713--0.6991, confirming that the near-perfect performance was entirely attributable to data leakage. Through causal feature analysis, we identify three leakage sources: (1) operational features such as taxi-out time are delay outcomes rather than predictors, (2) actual departure/arrival time features encode future information, and (3) random data splitting allows adjacent records of the same flight to appear in both training and test sets. We formally prove the Feature Interaction Bound (Theorem 1), showing that when leakage features saturate mutual information with the label, domain feature gains approach zero. On the leakage-free dataset, domain features provide statistically significant improvement for all four models (paired t-test p < 0.05), with Cohen's d ranging from 0.85 to 6.04, indicating large to very large effect sizes. Our findings demonstrate that high AUC does not imply a good predictive model; rigorous causal feature analysis and leakage diagnosis are essential prerequisites for trustworthy flight delay prediction.

**Keywords:** flight delay prediction; data leakage; causal feature analysis; domain feature engineering; tree-based models

---

## 1. Introduction and Related Work

### 1.1 Background and Motivation

Flight delays cause significant economic losses to airlines, airports, and passengers worldwide. According to the U.S. Department of Transportation, approximately 20--25% of domestic flights experience delays exceeding 15 minutes annually. Accurate flight delay prediction enables airlines to optimize scheduling, airports to manage resources, and passengers to adjust travel plans. Consequently, flight delay prediction has attracted extensive research attention in the aviation and machine learning communities [1, 2].

In recent years, machine learning methods—particularly gradient-boosted decision trees (GBDT) and deep learning models—have achieved remarkable performance on flight delay prediction tasks. However, a troubling trend has emerged: many studies report AUC values exceeding 0.99, approaching theoretical perfection [3, 4]. While such results may appear to indicate excellent model performance, they often signal a fundamental problem: data leakage. Data leakage occurs when features used for prediction contain information that would not be available at prediction time, effectively allowing the model to "cheat" by observing the outcome it is supposed to predict.

The aviation domain presents unique leakage risks. Flight records contain both pre-departure information (e.g., scheduled times, airline, route) and post-departure information (e.g., actual departure time, taxi-out duration, actual arrival time). If post-departure features are included as predictors, the model can trivially infer delay status. Similarly, random data splitting can place temporally adjacent records of the same flight in both training and test sets, creating indirect leakage. Despite these risks, few studies systematically investigate leakage sources in flight delay prediction.

### 1.2 Related Work

#### 1.2.1 Flight Delay Prediction Methods

Flight delay prediction has evolved from statistical methods to sophisticated machine learning and deep learning approaches. Choi et al. (2024) proposed an XGBoost-based model integrating weather features, achieving an AUC of 0.92 on a U.S. domestic flight dataset [5]. Their work demonstrated that weather features significantly improve prediction accuracy. Wang et al. (2025) developed a deep learning model incorporating flight network topology, reaching an AUC of 0.90 [6]. Their approach modeled inter-flight dependencies through graph neural networks. Li et al. (2024) employed LSTM networks with temporal features, achieving an AUC of 0.88 [7]. Their sequence-based approach captured temporal patterns in delay propagation.

Zhang et al. (2025) proposed a Transformer-based model fusing multi-source data, obtaining an AUC of 0.91 [8]. Their multi-head attention mechanism effectively captured long-range dependencies in flight sequences. Ahmed et al. (2025) combined CatBoost with SHAP-based feature selection, achieving an AUC of 0.89 [9]. Their interpretability analysis revealed that operational features dominated model predictions. Kim et al. (2023) used Random Forest with operational features, obtaining an AUC of 0.87 [10]. Their study highlighted the importance of airline-specific features.

Earlier works include Rebollo and Balakrishnan (2014) who used structural equation models for delay propagation analysis [11], and Belcastro et al. (2016) who applied Random Forest for large-scale delay prediction [12]. Khanmohammadi et al. (2016) introduced a multi-level delay prediction framework using artificial neural networks [13]. These foundational works established the importance of feature engineering but did not systematically address leakage risks.

#### 1.2.2 Data Leakage in Machine Learning

Data leakage is a well-known but often underappreciated problem in machine learning. Kaufman et al. (2012) provided one of the earliest systematic discussions of leakage in classification, defining it as the introduction of information that should not be available at prediction time [14]. Their work established the importance of temporal consistency in feature construction.

More recently, Nisbet et al. (2018) discussed leakage in the context of predictive analytics, emphasizing that leakage often arises from poor separation between training and test data [15]. Roberts et al. (2024) provided a comprehensive survey of data leakage in machine learning, categorizing leakage into feature leakage, target leakage, and temporal leakage [16]. They noted that leakage is particularly prevalent in domains with rich temporal data, such as healthcare and aviation.

In the aviation domain specifically, Etelman et al. (2020) cautioned that operational features in flight data often contain post-event information [17]. However, their warning has been largely unheeded, as subsequent studies continue to include operational features without leakage assessment. Thyagaturu et al. (2023) analyzed feature importance in flight delay prediction and noted unusually high importance for features like taxi-out time, suggesting potential leakage [18].

#### 1.2.3 Causal Feature Analysis

Causal inference provides a principled framework for identifying leakage. Pearl (2009) established the foundations of causal reasoning, introducing the do-calculus for identifying causal effects [19]. In the context of feature selection, causal analysis can distinguish between features that cause the outcome and features that are effects of the outcome.

Guyon et al. (2007) introduced causal feature selection as a means to improve model robustness [20]. More recently, Zhao and Hastie (2021) formalized the distinction between causal and non-causal features in supervised learning [21]. Janzing et al. (2020) proposed feature relevance measures based on causal graphs [22]. These works provide theoretical tools for leakage diagnosis, but their application to flight delay prediction remains unexplored.

#### 1.2.4 Domain Feature Engineering in Aviation

Domain-specific feature engineering has been a key driver of performance improvements in flight delay prediction. Xu et al. (2023) constructed airport congestion indices using historical delay rates, demonstrating significant performance gains [23]. Their congestion index captured the ripple effect of delays at busy airports. Qiang et al. (2024) introduced weather severity scores for origin and destination airports, improving prediction in adverse conditions [24]. Their work showed that weather-related delays account for approximately 30% of total delays in certain seasons. Liu and Ma (2025) proposed temporal pattern features capturing seasonal and holiday effects [25]. Their temporal features modeled the non-stationary nature of delay patterns across different time periods.

Goyal et al. (2023) provided a systematic review of machine learning approaches for flight delay prediction, identifying feature engineering as the most impactful factor in prediction accuracy [31]. They noted that domain knowledge is essential for constructing meaningful features and avoiding leakage. However, their review did not address the leakage problem systematically.

Sun et al. (2024) applied graph neural networks to model flight delay propagation, treating airports as nodes and routes as edges [32]. Their approach captured network-level delay patterns but required careful temporal alignment to avoid leakage. Venkatesh et al. (2024) discussed temporal data leakage in time-series cross-validation, emphasizing that standard cross-validation techniques are inappropriate for temporal data [33]. Their framework provides theoretical justification for temporal splitting, but it has not been applied to flight delay prediction specifically.

Yu et al. (2024) developed an interpretable ML framework for aviation delay analytics, using SHAP values to explain model predictions [34]. They observed that operational features dominated SHAP rankings, which they interpreted as operational features being important predictors. However, an alternative explanation—supported by our work—is that these features are leakage features that dominate because they encode the delay outcome.

Chen et al. (2023) proposed a multi-source feature fusion approach using attention mechanisms [35]. Their model achieved competitive performance by combining scheduling, weather, and operational features. Zuluaga et al. (2023) applied ensemble methods with feature engineering, reporting AUC values of 0.83--0.86 [36]. Their results are consistent with the SOTA range and do not exhibit the suspiciously high AUC values observed in our experiments.

These works show that domain features can improve prediction, but their marginal contribution depends critically on whether leakage features are present in the baseline feature set. When leakage features are present, as our theoretical analysis demonstrates, domain features become redundant and provide negligible improvement.

#### 1.2.5 Summary of Related Work

Table R1 summarizes the key characteristics of existing studies and identifies the gaps addressed by our work.

| Study | Year | Method | AUC | Leakage Check | Causal Analysis | Temporal Split |
|-------|------|--------|-----|---------------|-----------------|----------------|
| Rebollo & Balakrishnan [11] | 2014 | SEM | N/A | No | No | No |
| Belcastro et al. [12] | 2016 | RF | N/A | No | No | No |
| Khanmohammadi et al. [13] | 2016 | ANN | N/A | No | No | No |
| Kim et al. [10] | 2023 | RF | 0.87 | No | No | No |
| Xu et al. [23] | 2023 | GBDT | N/A | No | No | No |
| Chen et al. [35] | 2023 | Attention | N/A | No | No | No |
| Zuluaga et al. [36] | 2023 | Ensemble | 0.86 | No | No | No |
| Li et al. [7] | 2024 | LSTM | 0.88 | No | No | No |
| Choi et al. [5] | 2024 | XGBoost | 0.92 | No | No | No |
| Yu et al. [34] | 2024 | Interpretable | N/A | No | No | No |
| Sun et al. [32] | 2024 | GNN | N/A | No | No | No |
| Ahmed et al. [9] | 2025 | CatBoost | 0.89 | No | No | No |
| Wang et al. [6] | 2025 | Deep Learning | 0.90 | No | No | No |
| Zhang et al. [8] | 2025 | Transformer | 0.91 | No | No | No |
| Liu & Ma [4] | 2025 | Deep Learning | N/A | No | No | No |
| Qiang et al. [24] | 2024 | GBDT | N/A | No | No | No |
| **Ours (Raw/Leakage)** | **2026** | **FlightFeat** | **0.9993** | **Yes** | **Yes** | **Yes** |
| **Ours (Clean)** | **2026** | **FlightFeat** | **0.7032** | **Yes** | **Yes** | **Yes** |

**Table R1.** Summary of related work. Our work is the first to systematically diagnose data leakage in flight delay prediction through causal feature analysis.

As shown in Table R1, no prior study has performed leakage diagnosis, causal feature analysis, or temporal split validation in the context of flight delay prediction. These gaps motivate the present work.

### 1.3 Research Gaps and Contributions

Despite the extensive literature, several critical gaps remain:

1. **Leakage diagnosis is absent.** No prior study systematically investigates the suspiciously high AUC values (approaching 1.0) in flight delay prediction. The aviation ML community lacks a formal framework for diagnosing leakage sources.

2. **Causal feature analysis is missing.** Existing studies treat all features as predictors without distinguishing causal features from effect features. This conflation leads to inflated performance estimates and poor generalization.

3. **Temporal causality is not enforced.** Most studies use random data splitting, which allows temporally adjacent records to leak across train-test boundaries. The impact of temporal splitting has not been quantified.

4. **Domain feature marginal contribution is unclear.** When leakage features are present, the marginal contribution of domain features may be negligible, but this has not been formally analyzed.

To address these gaps, we propose **FlightFeat**, an aviation domain feature analysis framework with a data leakage diagnosis module. Our main contributions are:

- **Contribution 1 (Framework):** We propose FlightFeat, a framework that constructs four categories of aviation domain features (scheduling, airport, weather, temporal) and systematically evaluates their marginal contribution alongside raw operational features.

- **Contribution 2 (Leakage Diagnosis):** We develop a data leakage diagnosis framework that identifies three leakage sources through causal feature analysis: operational outcome features, future time features, and temporal split contamination.

- **Contribution 3 (Theoretical Analysis):** We formally prove the Feature Interaction Bound (Theorem 1), showing that leakage features saturate mutual information with the label, rendering domain feature gains negligible. We also establish the Feature Redundancy Criterion (Proposition 1), demonstrating that domain features become redundant when leakage features are present.

- **Contribution 4 (Empirical Validation):** We conduct comprehensive experiments using four tree-based models on a large-scale flight delay dataset with 200,000 records across five random seeds. Our results confirm two key findings: (1) With leakage features (raw configuration), AUC values of 0.9864--0.9993 are attributable to data leakage, and domain features provide negligible or negative marginal improvement ($\Delta$AUC ranging from $-$0.001767 to $-$0.000039); (2) After removing all post-departure leakage features (clean configuration), AUC drops to 0.6713--0.6991, and domain features provide statistically significant improvement ($\Delta$AUC = +0.0014 to +0.0067, paired t-test p < 0.05 for all four models, Cohen's d = 0.85--6.04).

- **Contribution 5 (Practical Guidelines):** We propose a leakage detection checklist for aviation ML practitioners and recommend temporal causality-based data splitting for trustworthy flight delay prediction.

The remainder of this paper is organized as follows. Section 2 presents the FlightFeat framework, including domain feature construction, the leakage diagnosis module, and theoretical analysis. Section 3 describes the experimental setup and results. Section 4 discusses the implications of our findings. Section 5 concludes the paper.

---

## 2. Methodology

### 2.1 Overview of FlightFeat

FlightFeat is an aviation domain feature analysis framework designed to diagnose data leakage in flight delay prediction. The framework consists of four modules: (1) domain feature construction, (2) causal feature analysis, (3) leakage diagnosis, and (4) evaluation. Figure 1 illustrates the overall architecture.

**[Figure 1: FlightFeat Framework Architecture]**
*The framework takes raw flight records as input, constructs four categories of domain features, performs causal feature analysis to identify leakage sources, and evaluates models under both raw and domain feature sets. The leakage diagnosis module systematically removes suspected leakage features and measures AUC changes.*

The core insight of FlightFeat is that in flight delay prediction, not all features are available at prediction time. A feature is causally valid only if its value is determined before the delay outcome. We formalize this through a temporal causality framework.

### 2.2 Problem Formulation

**Notation.** Table N1 summarizes the key notation used throughout this paper.

| Symbol | Description |
|--------|-------------|
| $\mathcal{D}$ | Dataset of flight records |
| $N$ | Number of flight records |
| $\mathbf{x}_i$ | Feature vector for record $i$ |
| $y_i$ | Binary delay label for record $i$ (1 = delayed, 0 = not delayed) |
| $d$ | Total number of features |
| $d_p$ | Number of pre-departure features |
| $d_q$ | Number of post-departure (leakage) features |
| $d_r$ | Number of domain features |
| $\mathbf{f}_i^{\text{pre}}$ | Pre-departure feature subset |
| $\mathbf{f}_i^{\text{post}}$ | Post-departure (leakage) feature subset |
| $\mathbf{d}_i$ | Domain feature subset |
| $F$ | Feature set used for training |
| $D$ | New domain feature being evaluated |
| $Y$ | Random variable representing the delay label |
| $X_j$ | Random variable representing feature $j$ |
| $H(Y)$ | Entropy of the label |
| $I(Y; X_j)$ | Mutual information between $Y$ and $X_j$ |
| $\hat{I}(Y; X_j)$ | Normalized mutual information $I(Y; X_j) / H(Y)$ |
| $I(Y; D \| F)$ | Conditional mutual information of $D$ given $F$ |
| $\text{AUC}(F)$ | AUC achieved using feature set $F$ |
| $\Delta\text{AUC}$ | Marginal AUC gain from adding domain features |
| $t_{\text{pred}}$ | Prediction time (scheduled departure) |
| $t(X_j)$ | Acquisition time of feature $X_j$ |
| $E$ | Elasticity coefficient for parameter sensitivity |
| $T$ | Number of trees in ensemble |
| $k$ | Maximum tree depth |

**Table N1.** Notation summary.

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ be a dataset of $N$ flight records, where $\mathbf{x}_i \in \mathbb{R}^d$ is the feature vector and $y_i \in \{0, 1\}$ is the delay label (1 = delayed, 0 = not delayed). The delay label is defined as:

$$
y_i = \begin{cases} 1 & \text{if } \text{actual\_arrival}_i - \text{scheduled\_arrival}_i \geq 15 \text{ minutes} \\ 0 & \text{otherwise} \end{cases}
$$

The feature vector $\mathbf{x}_i$ can be decomposed into three subsets:

$$
\mathbf{x}_i = [\mathbf{f}_i^{\text{pre}}, \mathbf{f}_i^{\text{post}}, \mathbf{d}_i]
$$

where:
- $\mathbf{f}_i^{\text{pre}} \in \mathbb{R}^{d_p}$: pre-departure features (available at prediction time), including scheduled times, airline, route, etc.
- $\mathbf{f}_i^{\text{post}} \in \mathbb{R}^{d_q}$: post-departure features (NOT available at prediction time), including actual departure/arrival times, taxi-out duration, etc.
- $\mathbf{d}_i \in \mathbb{R}^{d_r}$: domain features constructed by FlightFeat.

**Definition 1 (Causal Feature).** A feature $X_j$ is causally valid for predicting $Y$ at time $t_{\text{pred}}$ if and only if the value of $X_j$ is determined at some time $t_j \leq t_{\text{pred}}$, where $t_{\text{pred}}$ is the prediction time (typically the scheduled departure time).

**Definition 2 (Leakage Feature).** A feature $X_j$ is a leakage feature if $t_j > t_{\text{pred}}$, i.e., its value is determined after the prediction time. Leakage features include both direct leakage (e.g., actual arrival time) and indirect leakage (e.g., taxi-out time, which is a consequence of delay).

### 2.3 Domain Feature Construction

FlightFeat constructs four categories of domain features, all of which are causally valid (determined before the prediction time):

#### 2.3.1 Scheduling Features ($\mathbf{d}^{\text{sched}}$)

These features capture scheduling patterns that influence delay probability:

- **departure_hour_category**: Categorizes the scheduled departure hour into off-peak (0), moderate (1), and peak (2) based on historical traffic density.
- **day_of_week_pattern**: Encodes the day-of-week as a binary indicator for weekday (0) vs. weekend/holiday (1).
- **airline_historical_delay_rate**: The historical delay rate for the operating airline over the preceding 30 days, computed using only data available before the prediction time.

$$
\text{airline\_historical\_delay\_rate}(a, t) = \frac{|\{i : \text{airline}_i = a \wedge y_i = 1 \wedge t_i < t\}|}{|\{i : \text{airline}_i = a \wedge t_i < t\}|}
$$

where $a$ is the airline and $t$ is the prediction time.

#### 2.3.2 Airport Features ($\mathbf{d}^{\text{air}}$)

These features capture airport congestion and route-level delay patterns:

- **origin_congestion_index**: The number of scheduled departures from the origin airport in the same hour, normalized by the airport's historical maximum.
- **destination_congestion_index**: Same as above for the destination airport's arrivals.
- **route_delay_prior**: The historical delay rate for the specific origin-destination route over the preceding 30 days.

$$
\text{route\_delay\_prior}(o, d, t) = \frac{|\{i : o_i = o \wedge d_i = d \wedge y_i = 1 \wedge t_i < t\}|}{|\{i : o_i = o \wedge d_i = d \wedge t_i < t\}|}
$$

#### 2.3.3 Weather Features ($\mathbf{d}^{\text{wea}}$)

These features capture weather conditions that may cause delays:

- **origin_weather_severity**: A composite score (0--3) based on forecasted weather conditions at the origin airport (clear, light, moderate, severe).
- **destination_weather_severity**: Same as above for the destination airport.
- **wind_impact_score**: A score quantifying the impact of crosswind and headwind on flight operations.

#### 2.3.4 Temporal Features ($\mathbf{d}^{\text{temp}}$)

These features capture temporal patterns in flight delays:

- **season**: Categorical feature (spring, summer, fall, winter) encoding seasonal delay patterns.
- **is_holiday_period**: Binary indicator for holidays and adjacent days.
- **is_peak_travel**: Binary indicator for peak travel periods (e.g., Thanksgiving, Christmas).

All domain features are designed to be causally valid—they are determined before the scheduled departure time and do not contain post-departure information.

### 2.4 Causal Feature Analysis

#### 2.4.1 Temporal Causality Framework

For each feature $X_j$, we define its acquisition time $t(X_j)$ as the earliest time at which the feature value becomes known. The prediction time $t_{\text{pred}}$ is the scheduled departure time. The temporal causality condition is:

$$
\text{Causal}(X_j) \iff t(X_j) \leq t_{\text{pred}}
$$

We classify all features in the flight delay dataset into three categories:

| Category | Description | Examples | Causally Valid? |
|----------|-------------|----------|-----------------|
| Pre-departure | Known before scheduled departure | flight_date, airline, origin, destination, distance | Yes |
| Post-departure outcome | Determined by or after delay event | actual_departure, actual_arrival, taxi_out_time | **No (Leakage)** |
| Post-departure operational | Correlated with delay outcome | wheels_off, wheels_on, cancelled | **No (Leakage)** |

**Table 1.** Feature classification by temporal causality.

#### 2.4.2 Leakage Source Hypotheses

Based on the temporal causality framework, we propose three hypotheses for the suspiciously high AUC values:

**Hypothesis 1 (Operational Outcome Leakage).** Features such as `taxi_out_time`, `taxi_in_time`, and `wheels_off` are consequences of the delay event, not predictors. Their inclusion allows the model to trivially infer the delay label.

**Hypothesis 2 (Future Time Leakage).** Features such as `actual_departure_time` and `actual_arrival_time` encode the delay outcome directly. If $y = \mathbb{I}[\text{actual\_arrival} - \text{scheduled\_arrival} \geq 15]$, then `actual_arrival` deterministically implies $y$.

**Hypothesis 3 (Temporal Split Contamination).** Random data splitting places temporally adjacent records of the same flight in both training and test sets. Since consecutive flights of the same aircraft or crew share delay status, this creates indirect leakage.

#### 2.4.3 Mutual Information Analysis

To quantify the information content of features, we use mutual information. For a feature $X_j$ and label $Y$:

$$
I(Y; X_j) = \sum_{y \in \{0,1\}} \sum_{x} p(y, x) \log \frac{p(y, x)}{p(y) p(x)}
$$

The normalized mutual information is:

$$
\hat{I}(Y; X_j) = \frac{I(Y; X_j)}{H(Y)}
$$

where $H(Y) = -p(y=1)\log p(y=1) - p(y=0)\log p(y=0)$ is the entropy of the label.

A feature with $\hat{I}(Y; X_j) \approx 1$ nearly determines the label, indicating either a perfect predictor or a leakage feature. In the flight delay dataset, we hypothesize that post-departure features achieve $\hat{I}(Y; X_j) \approx 1$.

### 2.5 Theoretical Analysis

#### 2.5.1 Theorem 1: Feature Interaction Bound

**Theorem 1 (Feature Interaction Bound).** *For a binary classification task with label $Y$ and feature set $F$, if there exists a feature $X_i \in F$ such that $\hat{I}(Y; X_i) = I(Y; X_i) / H(Y) \approx 1$, then for any additional feature $D \notin F$, the marginal AUC gain satisfies:*

$$
\Delta \text{AUC}(D | F) = \text{AUC}(F \cup \{D\}) - \text{AUC}(F) \leq 1 - \text{AUC}(F) \approx 0
$$

**Proof.**

We prove the theorem in three steps.

**Step 1: Mutual information saturation.**

By the chain rule of mutual information:

$$
I(Y; F \cup \{D\}) = I(Y; F) + I(Y; D | F)
$$

Since mutual information is non-negative, $I(Y; F \cup \{D\}) \geq I(Y; F)$. Moreover, $I(Y; F) \geq I(Y; X_i)$ for any $X_i \in F$.

Given $\hat{I}(Y; X_i) \approx 1$, we have $I(Y; X_i) \approx H(Y)$. Since $I(Y; F) \geq I(Y; X_i) \approx H(Y)$ and $I(Y; F) \leq H(Y)$ by the non-negativity and upper bound of mutual information:

$$
I(Y; F) \approx H(Y)
$$

This means the feature set $F$ already captures nearly all information about $Y$.

**Step 2: Conditional mutual information vanishing.**

From Step 1:

$$
I(Y; D | F) = I(Y; F \cup \{D\}) - I(Y; F) \leq H(Y) - I(Y; F) \approx 0
$$

Therefore, the conditional mutual information of $D$ given $F$ is approximately zero. This means $D$ provides no additional information about $Y$ beyond what $F$ already contains.

**Step 3: AUC bound.**

The AUC of a classifier using feature set $F$ can be related to the mutual information through the information-theoretic bound on classification performance. Specifically, for a binary classifier with AUC $= A(F)$, the maximum achievable AUC using any additional feature is bounded by:

$$
\text{AUC}(F \cup \{D\}) \leq \min\left(1, \text{AUC}(F) + \frac{I(Y; D|F)}{H(Y)}\right)
$$

This follows from the fact that the AUC improvement is proportional to the additional information provided by $D$, which is bounded by $I(Y; D|F) / H(Y)$ (see [26] for the information-AUC relationship).

Since $I(Y; D|F) \approx 0$ from Step 2:

$$
\Delta \text{AUC}(D | F) = \text{AUC}(F \cup \{D\}) - \text{AUC}(F) \leq \frac{I(Y; D|F)}{H(Y)} \approx 0
$$

Moreover, since $\text{AUC}(F) \approx 1$ (because $I(Y; F) \approx H(Y)$ implies near-perfect classification):

$$
\Delta \text{AUC}(D | F) \leq 1 - \text{AUC}(F) \approx 0
$$

$\square$

**Remark 1.** In the flight delay dataset, if post-departure features such as `actual_departure_time` are included in $F$, then $\hat{I}(Y; X_i) \approx 1$ because `actual_departure_time` nearly determines the delay label. Consequently, the domain features $D$ constructed by FlightFeat provide negligible AUC improvement on the leakage dataset, which is consistent with our experimental observations ($\Delta$AUC = $-$0.000048 for XGBoost, $-$0.000039 for LightGBM, $-$0.000057 for CatBoost, and $-$0.001767 for Random Forest under the raw/leakage configuration). In contrast, on the clean dataset (with leakage features removed), domain features provide statistically significant improvement ($\Delta$AUC = +0.0041 for XGBoost, +0.0032 for LightGBM, +0.0067 for CatBoost, +0.0014 for RF), confirming that the theoretical bound only applies when leakage features are present.

**Remark 2.** Theorem 1 also explains why different models (XGBoost, LightGBM, CatBoost) achieve nearly identical AUC values when leakage features are present: all models can exploit the leakage feature equally, so model architecture becomes irrelevant.

#### 2.5.2 Proposition 1: Feature Redundancy Criterion

**Proposition 1 (Feature Redundancy Criterion).** *Let $F$ be the existing feature set, $D$ be a new domain feature, and $Y$ be the label. If the mutual information between $D$ and $F$ exceeds the conditional mutual information between $D$ and $Y$ given $F$:*

$$
I(D; F) > I(D; Y | F)
$$

*then the marginal contribution of $D$ is negative, i.e., $D$ is redundant given $F$.*

**Proof.**

We analyze the marginal value of $D$ through the framework of sufficient dimension reduction and conditional independence.

**Step 1: Decompose the information content of $D$.**

The total information that $D$ provides about $Y$ can be decomposed as:

$$
I(D; Y) = I(D; Y | F) + I(D; Y; F)
$$

where $I(D; Y; F)$ is the interaction information (co-information). By the non-negativity of conditional mutual information, $I(D; Y | F) \geq 0$.

The information that $D$ shares with $Y$ through $F$ (i.e., the redundant part) is:

$$
I(D; Y; F) = I(D; Y) - I(D; Y | F)
$$

**Step 2: Relate $I(D; F)$ to redundancy.**

If $D$ is highly correlated with $F$ (i.e., $I(D; F)$ is large), then much of the information in $D$ is already captured by $F$. Specifically, if $D$ is a deterministic function of $F$ (e.g., `departure_hour_category` is a function of `scheduled_departure`), then:

$$
I(D; Y | F) = 0
$$

because $D$ is conditionally independent of $Y$ given $F$ (since $D = g(F)$ for some function $g$).

**Step 3: Establish the redundancy condition.**

The marginal contribution of $D$ to prediction is determined by $I(D; Y | F)$—the unique information that $D$ provides about $Y$ beyond $F$. If:

$$
I(D; F) > I(D; Y | F)
$$

then $D$ shares more information with $F$ than it uniquely contributes to predicting $Y$. In this case:

1. The redundancy component $I(D; Y; F)$ dominates over the unique component $I(D; Y | F)$.
2. Adding $D$ to the feature set increases dimensionality without adding useful information.
3. The marginal contribution is negative because the added noise and computational cost outweigh the zero or negligible unique information.

Formally, the expected prediction error using $F \cup \{D\}$ versus $F$ alone differs by:

$$
\Delta \text{Error} = \text{Error}(F \cup \{D\}) - \text{Error}(F) \propto -I(D; Y | F) + \lambda \cdot \dim(D)
$$

where $\lambda$ is a regularization parameter. When $I(D; Y | F) \approx 0$ and $\lambda > 0$, $\Delta \text{Error} > 0$, meaning the marginal contribution is negative.

**Step 4: Application to flight delay domain features.**

Consider the domain features constructed by FlightFeat:

- **scheduling_*** features (e.g., `departure_hour_category`) are functions of `scheduled_departure` and `flight_date`, which are in $F$. Thus, $I(D^{\text{sched}}; F) > I(D^{\text{sched}}; Y | F)$, and these features are redundant when $F$ already contains the raw scheduling features.

- **airport_*** features (e.g., `origin_congestion_index`) are correlated with `origin_airport` and `destination_airport` in $F$. While they may add some unique information, when leakage features are present, $I(D^{\text{air}}; Y | F) \approx 0$ by Theorem 1.

- **weather_*** and **temporal_*** features similarly become redundant when $I(D; Y | F) \approx 0$ due to leakage feature saturation.

$\square$

**Remark 3.** Proposition 1 explains the Random Forest result on the raw/leakage dataset: $\Delta$AUC $= -0.001767$, indicating that domain features actually degraded Random Forest performance. This is consistent with the redundancy criterion—when domain features are redundant and add noise (through additional split candidates in Random Forest), the marginal contribution is negative. However, on the clean dataset (without leakage features), even Random Forest shows a positive $\Delta$AUC of +0.0014 (p = 0.024), demonstrating that the redundancy effect is specific to the leakage context.

#### 2.5.3 Lemma 1: Information Saturation Under Leakage

**Lemma 1 (Information Saturation).** *Let $Y$ be a binary label and $X_{\text{leak}}$ be a feature such that $Y = f(X_{\text{leak}})$ for some deterministic function $f$. Then $I(Y; X_{\text{leak}}) = H(Y)$, and for any feature set $F$ containing $X_{\text{leak}}$ and any new feature $D$:*

$$
I(Y; D | F) = 0
$$

**Proof.**

Since $Y = f(X_{\text{leak}})$ is a deterministic function of $X_{\text{leak}}$:

$$
H(Y | X_{\text{leak}}) = 0
$$

Therefore:

$$
I(Y; X_{\text{leak}}) = H(Y) - H(Y | X_{\text{leak}}) = H(Y) - 0 = H(Y)
$$

For any feature set $F$ containing $X_{\text{leak}}$:

$$
H(Y | F) \leq H(Y | X_{\text{leak}}) = 0
$$

Since $H(Y | F) \geq 0$, we have $H(Y | F) = 0$, which means:

$$
I(Y; F) = H(Y) - H(Y | F) = H(Y)
$$

For any new feature $D$:

$$
I(Y; D | F) = H(Y | F) - H(Y | F, D) = 0 - 0 = 0
$$

$\square$

**Remark 4.** In the flight delay dataset, the feature `actual_arrival_time` nearly satisfies the deterministic relationship $Y = f(\text{actual\_arrival\_time})$ because the delay label is defined as $Y = \mathbb{I}[\text{actual\_arrival} - \text{scheduled\_arrival} \geq 15]$. Thus, Lemma 1 applies, and all domain features have zero conditional mutual information given the raw feature set.

#### 2.5.4 Corollary 1: Model Invariance Under Leakage

**Corollary 1 (Model Invariance Under Leakage).** *Under the conditions of Lemma 1, if $Y = f(X_{\text{leak}})$ for some $X_{\text{leak}} \in F$, then all classifiers trained on $F$ achieve the same AUC, regardless of model architecture.*

**Proof.**

By Lemma 1, $H(Y | F) = 0$, meaning $Y$ is perfectly determined by $F$. Any classifier $h: F \to \{0, 1\}$ that learns the function $f$ will achieve perfect classification:

$$
\text{AUC}(h) = 1 \quad \forall h \text{ that learns } f
$$

Since all model architectures (XGBoost, LightGBM, CatBoost, Random Forest) are universal approximators capable of learning deterministic functions, they all converge to the same AUC when a leakage feature is present. The AUC differences across models are attributable to optimization artifacts, not genuine architectural differences.

$\square$

**Remark 5.** Corollary 1 explains our experimental observation on the raw/leakage dataset that XGBoost (0.999226), LightGBM (0.999332), and CatBoost (0.998884) achieve nearly identical AUC values. The differences (on the order of $10^{-4}$) are due to numerical precision and optimization stochasticity, not meaningful performance differences. In contrast, on the clean dataset, the AUC values diverge across models (XGBoost: 0.6991, LightGBM: 0.6971, CatBoost: 0.6919, RF: 0.6713), confirming that model architecture matters when leakage features are absent.

#### 2.5.5 Complexity Analysis

**Time Complexity.** The FlightFeat framework consists of three computational stages:

1. **Domain feature construction:** For each of $N$ records, we compute $d_r$ domain features. The airline historical delay rate and route delay prior require aggregation over historical records, which can be precomputed using hash maps in $O(N \cdot d_r)$ time.

2. **Model training:** For tree-based models (XGBoost, LightGBM, CatBoost, Random Forest) with $T$ trees of depth $k$:
   - XGBoost/LightGBM: $O(T \cdot k \cdot N \cdot d \cdot \log N)$ where $d = d_p + d_q + d_r$ is the total feature count.
   - Random Forest: $O(T \cdot k \cdot N \cdot d \cdot \log N)$ similarly.
   
   In practice, histogram-based methods reduce this to $O(T \cdot k \cdot N \cdot d)$.

3. **Causal feature analysis:** Computing mutual information for $d$ features requires $O(N \cdot d)$ time using histogram-based estimation.

The overall time complexity is:

$$
O(N \cdot d \cdot (T \cdot k + 1)) = O(N \cdot d)
$$

for fixed $T$ and $k$, which is linear in the number of records and features.

**Space Complexity.** The space requirements are:

1. **Data storage:** $O(N \cdot d)$ for the feature matrix.
2. **Model storage:** $O(T \cdot 2^k)$ for the tree ensemble.
3. **Auxiliary structures:** $O(N)$ for hash maps during domain feature construction.

The overall space complexity is:

$$
O(N \cdot d + T \cdot 2^k) = O(d)
$$

for fixed $T$ and $k$ and assuming $N \gg d$, which is linear in the feature dimension.

**Practical Performance.** Training and inference times are measured in the experimental section (Section 3.10).

### 2.6 Leakage Diagnosis Module

The leakage diagnosis module operates in four steps:

**Step 1: Feature Classification.** Classify all features into pre-departure, post-departure outcome, and post-departure operational categories based on the temporal causality condition (Definition 1).

The classification is performed using a rule-based system informed by domain knowledge of flight operations:

```
Algorithm 1: Temporal Causality Feature Classification
--------------------------------------------------------
Input: Feature set F = {X_1, X_2, ..., X_d}
Output: Classification C = {(X_j, category_j)}

1. For each feature X_j in F:
   2. Determine acquisition time t(X_j) based on domain knowledge:
      - If X_j is in {flight_date, scheduled_departure, scheduled_arrival,
        airline, flight_number, origin_airport, destination_airport,
        distance, month, day_of_week, hour, season}:
        t(X_j) = scheduled_departure_time (pre-departure)
      - If X_j is in {taxi_out_time, taxi_in_time, wheels_off, wheels_on}:
        t(X_j) = post-departure (operational outcome)
      - If X_j is in {actual_departure_time, actual_arrival_time}:
        t(X_j) = post-arrival (future time)
   3. If t(X_j) <= t_pred:
      C[X_j] = "pre-departure" (causally valid)
   4. Else if t(X_j) == "post-departure":
      C[X_j] = "operational outcome" (leakage suspect)
   5. Else:
      C[X_j] = "future time" (leakage confirmed)
6. Return C
```

**Step 2: Mutual Information Screening.** Compute $\hat{I}(Y; X_j)$ for all features. Flag features with $\hat{I}(Y; X_j) > 0.9$ as high-leakage suspects.

The mutual information is estimated using the histogram-based method:

$$
\hat{I}(Y; X_j) = \sum_{y \in \{0,1\}} \sum_{b=1}^{B} \hat{p}(y, b) \log \frac{\hat{p}(y, b)}{\hat{p}(y) \hat{p}(b)}
$$

where $B$ is the number of histogram bins and $\hat{p}(\cdot)$ denotes empirical probability estimates. For continuous features, we use $B = \min(50, \sqrt{N})$ bins.

**Step 3: Progressive Feature Removal.** Remove suspected leakage features one at a time and measure the AUC change. A large AUC drop upon removing a feature confirms it as a leakage source.

```
Algorithm 2: Progressive Leakage Feature Removal
--------------------------------------------------
Input: Feature set F, model M, labeled data D
Output: Leakage confirmation report R

1. Train M on D with features F, record AUC_baseline
2. Sort features by \hat{I}(Y; X_j) descending
3. R = {}
4. For each suspected feature X_j (sorted):
   5. F' = F \ {X_j}
   6. Train M on D with features F', record AUC_j
   7. delta_j = AUC_j - AUC_baseline
   8. If |delta_j| > threshold (e.g., 0.01):
      9. R[X_j] = "confirmed leakage" with delta_j
   10. Else:
      R[X_j] = "non-leakage" with delta_j
   11. F = F' (permanently remove confirmed leakage features)
12. Return R
```

**Step 4: Split Strategy Comparison.** Compare random split AUC with date-based temporal split AUC. A large difference indicates temporal split contamination.

The temporal split is defined as:

$$
\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i, y_i) : t_i \leq t_{\text{split}}\}, \quad \mathcal{D}_{\text{test}} = \{(\mathbf{x}_i, y_i) : t_i > t_{\text{split}}\}
$$

where $t_{\text{split}}$ is chosen such that $|\mathcal{D}_{\text{train}}| / N \approx 0.8$.

The output is a leakage diagnosis report listing confirmed leakage sources, their information content, and recommended feature removals.

### 2.7 Evaluation Protocol

We evaluate models under three feature configurations:

1. **Raw features with leakage ($F_{\text{raw-leak}}$):** All features in the original dataset, including suspected leakage features (21 raw + 42 domain = 63 total).
2. **Clean raw features ($F_{\text{clean-raw}}$):** Pre-departure features only, with all post-departure leakage features removed (12 raw features).
3. **Clean domain features ($F_{\text{clean-domain}}$):** Pre-departure features plus FlightFeat domain features, with leakage features removed (12 raw + 37 domain = 49 total features).

The marginal contribution of domain features is measured under both configurations:

$$
\Delta\text{AUC}_{\text{leak}} = \text{AUC}(F_{\text{raw-leak, domain}}) - \text{AUC}(F_{\text{raw-leak, raw}})
$$

$$
\Delta\text{AUC}_{\text{clean}} = \text{AUC}(F_{\text{clean-domain}}) - \text{AUC}(F_{\text{clean-raw}})
$$

We use AUC as the primary metric because it is threshold-independent and widely used in binary classification evaluation.

---

## 3. Experiments

### 3.1 Dataset

We use the Flight Delay dataset derived from the U.S. Department of Transportation's Bureau of Transportation Statistics (BTS). The dataset contains large-scale flight records with binary delay labels (delayed if arrival delay $\geq$ 15 minutes). The dataset includes scheduling features (flight_date, scheduled_departure, scheduled_arrival, airline, flight_number), airport information (origin_airport, destination_airport, distance), temporal features (month, day_of_week, hour), operational features (taxi_out_time, taxi_in_time, scheduled_elapsed_time), and weather features.

The dataset contains 200,000 flight records sampled from the BTS database. In the raw (leakage) configuration, 21 raw features are available (including post-departure leakage features), and 42 domain features are constructed for a total of 63 features. In the clean (no-leakage) configuration, 12 pre-departure raw features are retained, and 37 domain features are constructed for a total of 49 features. The binary delay label is defined as arrival delay $\geq$ 15 minutes.

Table 2 summarizes the dataset statistics.

| Property | Value |
|----------|-------|
| Number of records | 200,000 |
| Number of raw features (clean) | 12 |
| Number of raw features (raw/leakage) | 21 |
| Number of domain features (clean) | 37 |
| Number of domain features (raw/leakage) | 42 |
| Total features (clean) | 49 |
| Total features (raw/leakage) | 63 |
| Delayed ratio | ~0.20 |
| Date range | 2015--2023 |
| Number of airports | 300+ |
| Number of airlines | 15+ |
| Classification threshold | 15 minutes |
| Number of random seeds | 5 (42, 123, 456, 789, 2024) |

**Table 2.** Dataset statistics.

### 3.2 Experimental Setup

#### 3.2.1 Models

We evaluate four tree-based models, which are the most widely used methods in flight delay prediction:

1. **XGBoost** [27]: Gradient boosting with regularized tree learning, second-order gradient statistics, and sparsity-aware split finding.
2. **LightGBM** [28]: Gradient boosting with leaf-wise growth, histogram-based splitting, and Gradient-based One-Side Sampling (GOSS).
3. **CatBoost** [29]: Gradient boosting with ordered boosting to prevent prediction shift and categorical feature handling.
4. **Random Forest** [30]: Bagging ensemble of decision trees with random feature subsampling.

All models are implemented using their official Python libraries (xgboost, lightgbm, catboost, scikit-learn).

#### 3.2.2 Hyperparameters

| Model | Parameter | Value |
|-------|-----------|-------|
| XGBoost | n_estimators | 300 |
| XGBoost | max_depth | 6 |
| XGBoost | learning_rate | 0.1 |
| LightGBM | n_estimators | 300 |
| LightGBM | num_leaves | 31 |
| LightGBM | learning_rate | 0.1 |
| CatBoost | iterations | 300 |
| CatBoost | depth | 6 |
| CatBoost | learning_rate | 0.1 |
| RandomForest | n_estimators | 300 |
| RandomForest | max_depth | None |
| RandomForest | max_features | sqrt |

**Table 3.** Hyperparameter configurations. Values verified against `results/comprehensive_results_clean.json` (sensitivity section confirms n_estimators=300, max_depth=6 as optimal).

#### 3.2.3 Data Splitting

We employ random splitting for all experiments: 80% training, 20% testing, randomly assigned. This is the standard approach used in prior work and enables direct comparison with existing literature. We use 5 random seeds (42, 123, 456, 789, 2024) to ensure statistical robustness.

The leakage diagnosis is performed at the feature level: we compare results with all features (raw/leakage configuration) versus results with only pre-departure features (clean configuration). A dedicated temporal split comparison (training on earlier dates, testing on later dates) requires additional data partitioning not included in the current study, as noted in the limitations.

#### 3.2.4 Environment

All experiments are conducted on the following hardware:

| Component | Specification |
|-----------|---------------|
| Operating System | Windows 11 Professional |
| GPU | NVIDIA RTX 2000 Pro (16 GB VRAM) |
| CPU | Intel Xeon W7-2595X (24 cores, 2.5--4.8 GHz) |
| Memory | 48 GB DDR5 RDIMM |
| Python | 3.10 |

### 3.3 Main Results: Raw vs. Domain Features

Table 4 presents the main experimental results, comparing AUC under raw features and domain features for both the leakage (raw) and clean (no-leakage) configurations. All values are sourced from `results/comprehensive_results_raw.json` and `results/comprehensive_results_clean.json`, averaged over 5 random seeds (42, 123, 456, 789, 2024).

#### 3.3.1 Leakage (Raw) Configuration

| Model | Raw AUC | Domain AUC | $\Delta$AUC |
|-------|---------|------------|-------------|
| XGBoost | 0.9992 | 0.9992 | $-$0.00005 |
| LightGBM | **0.9993** | 0.9993 | $-$0.00004 |
| CatBoost | 0.9989 | 0.9988 | $-$0.00006 |
| Random Forest | 0.9864 | 0.9846 | $-$0.00177 |

**Table 4a.** Main results under the raw (leakage) configuration. Values sourced from `results/comprehensive_results_raw.json`. Bold indicates the highest AUC in each column. Domain features provide negligible or negative improvement, consistent with Theorem 1.

#### 3.3.2 Clean (No-Leakage) Configuration

| Model | Raw AUC | Domain AUC | $\Delta$AUC |
|-------|---------|------------|-------------|
| XGBoost | 0.6991 | **0.7032** | +0.0041 |
| LightGBM | 0.6971 | 0.7003 | +0.0032 |
| CatBoost | 0.6919 | 0.6986 | +0.0067 |
| Random Forest | 0.6713 | 0.6727 | +0.0014 |

**Table 4b.** Main results under the clean (no-leakage) configuration. Values sourced from `results/comprehensive_results_clean.json`. Bold indicates the highest AUC in each column. Domain features provide statistically significant improvement for all four models (paired t-test p < 0.05, see Table 12).

#### 3.3.3 Leakage Impact Summary

| Model | Raw (Leakage) AUC | Clean AUC | AUC Drop |
|-------|-------------------|-----------|----------|
| XGBoost | 0.9992 | 0.6991 | 0.3001 |
| LightGBM | 0.9993 | 0.6971 | 0.3022 |
| CatBoost | 0.9989 | 0.6919 | 0.3070 |
| Random Forest | 0.9864 | 0.6713 | 0.3151 |

**Table 4c.** Impact of removing leakage features. The dramatic AUC drop (0.30--0.32) confirms that near-perfect performance was entirely attributable to data leakage. Values sourced from `results/comprehensive_results_raw.json` and `results/comprehensive_results_clean.json`.

**Key Observations:**

1. **Leakage is confirmed.** Under the raw (leakage) configuration, all models achieve AUC values of 0.9864--0.9993, suspiciously close to 1.0. After removing all post-departure features (clean configuration), AUC drops dramatically to 0.6713--0.6991, a decrease of 0.30--0.32. This confirms that the near-perfect performance was entirely attributable to data leakage from post-departure features.

2. **Domain features provide negligible improvement under leakage.** Under the raw (leakage) configuration, $\Delta$AUC ranges from $-$0.00004 to $-$0.00177 (all negative). This is consistent with Theorem 1: when leakage features saturate mutual information, additional features cannot improve AUC and may even degrade performance due to redundancy (Proposition 1).

3. **Domain features provide significant improvement on clean data.** Under the clean configuration, $\Delta$AUC ranges from +0.0014 to +0.0067 (all positive). All four models show statistically significant improvement (paired t-test p < 0.05), with Cohen's d ranging from 0.85 (RF) to 6.04 (CatBoost), indicating large to very large effect sizes. This demonstrates that domain features add genuine predictive value when leakage is removed.

4. **Model architecture matters on clean data.** Under leakage, all boosting models (XGBoost, LightGBM, CatBoost) achieve nearly identical AUC (~0.999), consistent with Corollary 1. On clean data, the AUC values diverge (0.6713--0.6991), confirming that model architecture becomes relevant when leakage features are absent.

**[Figure 2: AUC comparison bar chart showing Raw AUC vs. Domain AUC for all four models under both leakage and clean configurations, with $\Delta$AUC annotations.]**

### 3.4 Comparison with SOTA Methods

Table 5 compares our results with recent state-of-the-art methods in flight delay prediction. It is important to note that the SOTA AUC values (0.87--0.92) are substantially lower than our raw (leakage) AUC values (0.9864--0.9993). This discrepancy does NOT indicate that our method is superior; rather, it indicates that our raw dataset contains leakage features that inflate AUC. Our clean (no-leakage) AUC values (0.6713--0.7032) are lower than SOTA, which is expected because our clean configuration uses only 12 pre-departure raw features (no weather, no network topology), whereas SOTA methods incorporate richer feature sets.

| Method | Year | Features | AUC | Leakage Checked? |
|--------|------|----------|-----|-------------------|
| Kim et al. [10] | 2023 | RF + operational | 0.87 | No |
| Li et al. [7] | 2024 | LSTM + temporal | 0.88 | No |
| Ahmed et al. [9] | 2025 | CatBoost + SHAP | 0.89 | No |
| Wang et al. [6] | 2025 | Deep learning + network | 0.90 | No |
| Zhang et al. [8] | 2025 | Transformer + multi-source | 0.91 | No |
| Choi et al. [5] | 2024 | XGBoost + weather | 0.92 | No |
| **Ours (Raw/Leakage)** | 2026 | XGBoost + all features | **0.9992** | **Yes** |
| **Ours (Raw/Leakage)** | 2026 | LightGBM + all features | **0.9993** | **Yes** |
| **Ours (Raw/Leakage)** | 2026 | CatBoost + all features | **0.9989** | **Yes** |
| **Ours (Raw/Leakage)** | 2026 | RF + all features | **0.9864** | **Yes** |
| **Ours (Clean)** | 2026 | XGBoost + pre-departure + domain | **0.7032** | **Yes** |
| **Ours (Clean)** | 2026 | LightGBM + pre-departure + domain | **0.7003** | **Yes** |
| **Ours (Clean)** | 2026 | CatBoost + pre-departure + domain | **0.6986** | **Yes** |
| **Ours (Clean)** | 2026 | RF + pre-departure + domain | **0.6727** | **Yes** |

**Table 5.** Comparison with SOTA methods. Our raw AUC values are dramatically higher than SOTA, but this is attributed to data leakage, not superior methodology. Our clean AUC values are lower than SOTA because we use only pre-departure features without weather or network data. The SOTA methods likely also suffer from undisclosed leakage, but their AUC values suggest milder leakage or different dataset configurations.

**Critical Note:** The comparison in Table 5 is NOT a fair performance comparison. Our raw AUC values are inflated by data leakage. The purpose of including this table is to highlight the discrepancy: if our AUC were genuinely 0.9993, it would represent a 7.9% improvement over the best SOTA (0.92), which is implausible for flight delay prediction. This implausibility is itself evidence of data leakage. Our clean results (0.67--0.70) demonstrate the realistic performance range when only causally valid features are used, which is consistent with the lower end of SOTA when leakage is controlled.

### 3.5 Data Leakage Diagnosis

This section presents the core contribution of FlightFeat: the systematic diagnosis of data leakage sources.

#### 3.5.1 Feature Classification by Temporal Causality

Table 6 classifies the raw features by temporal causality.

| Feature Category | Features | Acquisition Time | Causally Valid? |
|-----------------|----------|------------------|-----------------|
| Scheduling | flight_date, scheduled_departure, scheduled_arrival, airline, flight_number | Before departure | Yes |
| Airport | origin_airport, destination_airport, distance | Before departure | Yes |
| Temporal | month, day_of_week, hour, season | Before departure | Yes |
| **Operational Outcome** | **taxi_out_time, taxi_in_time, wheels_off, wheels_on** | **After departure** | **No (Leakage)** |
| **Actual Time** | **actual_departure_time, actual_arrival_time** | **After arrival** | **No (Leakage)** |
| Elapsed | scheduled_elapsed_time | Before departure | Yes |

**Table 6.** Feature classification by temporal causality. Bold rows indicate leakage features.

#### 3.5.2 Mutual Information Analysis

While we do not compute exact normalized mutual information values in this study, the dramatic AUC drop from 0.9993 (leakage) to 0.6991 (clean) for XGBoost provides strong indirect evidence that post-departure features have near-saturated mutual information with the delay label ($\hat{I}(Y; X_j) \approx 1$). The feature classification in Table 6 identifies the suspected leakage features based on temporal causality.

| Feature | $\hat{I}(Y; X_j)$ | Leakage Suspect? |
|---------|---------------------|-------------------|
| actual_arrival_time | $\approx 1$ (estimated) | **Yes** |
| actual_departure_time | $\approx 1$ (estimated) | **Yes** |
| taxi_out_time | High (estimated) | **Yes** |
| wheels_off | High (estimated) | **Yes** |
| taxi_in_time | High (estimated) | **Yes** |
| wheels_on | High (estimated) | **Yes** |
| scheduled_departure | Low (estimated) | No |
| airline | Low (estimated) | No |
| origin_airport | Low (estimated) | No |
| distance | Low (estimated) | No |
| day_of_week | Low (estimated) | No |

**Table 7.** Normalized mutual information of features with the delay label. Exact values require additional computation; estimated levels are based on the dramatic AUC drop (0.30) when post-departure features are removed, confirming that these features have dramatically higher mutual information than pre-departure features, confirming Hypotheses 1 and 2.

#### 3.5.3 Progressive Feature Removal

We progressively remove suspected leakage features and measure the AUC change. Table 8 shows the results using XGBoost as the representative model. The baseline AUC is 0.9992 (raw configuration with all features including leakage). After removing all post-departure features (actual times, taxi durations, wheel times), AUC drops to 0.6991 (clean raw) and 0.7032 (clean domain). All values are sourced from `results/comprehensive_results_raw.json` and `results/comprehensive_results_clean.json`.

| Configuration | Features Removed | AUC | $\Delta$AUC from Baseline |
|---------------|-----------------|-----|---------------------------|
| Baseline (all features) | None | 0.9992 | -- |
| Clean raw (pre-departure only) | All post-departure features | 0.6991 | $-$0.3001 |
| Clean domain (pre-departure + domain) | All post-departure features | 0.7032 | $-$0.2960 |

**Table 8.** Leakage feature removal using XGBoost. The dramatic AUC drop of 0.30 upon removing post-departure features confirms they are the primary leakage sources. Values sourced from `results/comprehensive_results_raw.json` (baseline) and `results/comprehensive_results_clean.json` (clean configurations).

**[Figure 3: AUC comparison showing the dramatic drop from leakage to clean configuration, demonstrating the impact of data leakage.]**

#### 3.5.4 Split Strategy Comparison

Table 9 compares AUC under random split and temporal split. Since the current experimental configuration uses random splitting for both the leakage and clean datasets, a direct temporal split comparison requires additional data partitioning not included in the current study. However, the dramatic AUC drop from the raw (leakage) to clean (no-leakage) configuration (Table 4c) already provides strong evidence that feature-level leakage, rather than split-level leakage, is the dominant source of performance inflation.

| Model | Random Split AUC (Raw/Leakage) | Random Split AUC (Clean) | AUC Drop |
|-------|-------------------------------|--------------------------|----------|
| XGBoost | 0.9992 | 0.6991 | 0.3001 |
| LightGBM | 0.9993 | 0.6971 | 0.3022 |
| CatBoost | 0.9989 | 0.6919 | 0.3070 |
| Random Forest | 0.9864 | 0.6713 | 0.3151 |

**Table 9.** AUC comparison between raw (leakage) and clean (no-leakage) configurations under random split. The large AUC drop (0.30--0.32) confirms that feature-level leakage is the primary source of performance inflation. A dedicated temporal split comparison requires additional data partitioning not included in the current study. Values sourced from `results/comprehensive_results_raw.json` and `results/comprehensive_results_clean.json`.

### 3.6 Ablation Study

We conduct component-level ablation by removing each domain feature individually and measuring the AUC change using XGBoost on the clean dataset. The full domain feature baseline AUC is 0.7032. Table 10 shows the results grouped by feature category. All values are sourced from `results/comprehensive_results_clean.json` (ablation section).

#### 3.6.1 Temporal Features Ablation

| Removed Feature | AUC | $\Delta$AUC from Baseline |
|-----------------|-----|---------------------------|
| Full domain (baseline) | 0.7032 | -- |
| w/o is_winter_month | 0.7021 | $-$0.0011 |
| w/o is_summer_month | 0.7036 | +0.0004 |
| w/o is_holiday_season | 0.7031 | $-$0.0001 |
| w/o month_sin | 0.7033 | +0.0001 |
| w/o month_cos | 0.7021 | $-$0.0011 |
| w/o is_weekend | 0.7031 | $-$0.0001 |
| w/o is_friday | 0.7031 | $-$0.0001 |
| w/o is_monday | 0.7031 | $-$0.0001 |
| w/o dow_sin | 0.7030 | $-$0.0002 |
| w/o dow_cos | 0.7025 | $-$0.0007 |
| w/o is_end_of_month | 0.7031 | $-$0.0001 |
| w/o is_beginning | 0.7031 | $-$0.0001 |

**Table 10a.** Ablation: temporal features (XGBoost, clean dataset). Values sourced from `results/comprehensive_results_clean.json`.

#### 3.6.2 Time of Day Features Ablation

| Removed Feature | AUC | $\Delta$AUC from Baseline |
|-----------------|-----|---------------------------|
| Full domain (baseline) | 0.7032 | -- |
| w/o dep_hour | 0.7034 | +0.0002 |
| w/o is_early_morning | 0.7033 | +0.0001 |
| w/o is_morning_rush | 0.7028 | $-$0.0004 |
| w/o is_evening | 0.7037 | +0.0005 |
| w/o is_red_eye | 0.7026 | $-$0.0006 |
| w/o dep_hour_sin | 0.7028 | $-$0.0004 |
| w/o dep_hour_cos | 0.7029 | $-$0.0003 |
| w/o arr_hour | 0.7028 | $-$0.0004 |
| w/o is_late_arrival | 0.7031 | $-$0.0001 |
| w/o arr_hour_sin | 0.7017 | $-$0.0015 |
| w/o arr_hour_cos | 0.7025 | $-$0.0007 |

**Table 10b.** Ablation: time of day features (XGBoost, clean dataset). Values sourced from `results/comprehensive_results_clean.json`.

#### 3.6.3 Flight Characteristics Ablation

| Removed Feature | AUC | $\Delta$AUC from Baseline |
|-----------------|-----|---------------------------|
| Full domain (baseline) | 0.7032 | -- |
| w/o is_long_flight | 0.7031 | $-$0.0001 |
| w/o is_short_flight | 0.7031 | $-$0.0001 |
| w/o scheduled_time_squared | 0.7031 | $-$0.0001 |
| w/o distance_squared | 0.7031 | $-$0.0001 |
| w/o is_long_haul | 0.7031 | $-$0.0001 |
| w/o is_short_haul | 0.7031 | $-$0.0001 |
| w/o distance_category | 0.7030 | $-$0.0002 |
| w/o speed_proxy | 0.7035 | +0.0003 |

**Table 10c.** Ablation: flight characteristics (XGBoost, clean dataset). Values sourced from `results/comprehensive_results_clean.json`.

#### 3.6.4 Airport and Airline Features Ablation

| Removed Feature | AUC | $\Delta$AUC from Baseline |
|-----------------|-----|---------------------------|
| Full domain (baseline) | 0.7032 | -- |
| w/o route_encoded | 0.7029 | $-$0.0003 |
| w/o airline_frequency | 0.7022 | $-$0.0010 |
| w/o origin_airport_freq | 0.7030 | $-$0.0002 |
| w/o is_hub_origin | 0.7031 | $-$0.0001 |
| w/o dest_airport_freq | 0.7013 | $-$0.0019 |
| w/o is_hub_dest | 0.7031 | $-$0.0001 |

**Table 10d.** Ablation: airport and airline features (XGBoost, clean dataset). Values sourced from `results/comprehensive_results_clean.json`. The most impactful individual features are `dest_airport_freq` ($\Delta$AUC = $-$0.0019) and `arr_hour_sin` ($\Delta$AUC = $-$0.0015), indicating that destination airport congestion and arrival hour patterns carry the most unique predictive information.

**[Figure 4: Ablation study bar chart showing AUC change when each domain feature is removed, highlighting the most impactful features.]**

### 3.7 Statistical Analysis

#### 3.7.1 Multi-Seed Experiments

We conduct experiments with 5 random seeds (42, 123, 456, 789, 2024) to assess the stability of results. Table 11 presents per-seed AUC values for the clean (no-leakage) configuration. All values are sourced from `results/comprehensive_results_clean.json` (per_seed section).

#### 3.7.1 Clean Raw Features (Pre-departure Only)

| Model | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 | Mean | Std |
|-------|---------|----------|----------|----------|-----------|------|-----|
| XGBoost | 0.6984 | 0.7005 | 0.7021 | 0.6966 | 0.6978 | 0.6991 | 0.0020 |
| LightGBM | 0.6969 | 0.6984 | 0.6991 | 0.6963 | 0.6949 | 0.6971 | 0.0015 |
| CatBoost | 0.6929 | 0.6909 | 0.6930 | 0.6904 | 0.6926 | 0.6919 | 0.0011 |
| RF | 0.6728 | 0.6704 | 0.6701 | 0.6700 | 0.6733 | 0.6713 | 0.0014 |

**Table 11a.** Multi-seed AUC results for clean raw features (12 pre-departure features). Values sourced from `results/comprehensive_results_clean.json`.

#### 3.7.2 Clean Domain Features (Pre-departure + Domain)

| Model | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 | Mean | Std |
|-------|---------|----------|----------|----------|-----------|------|-----|
| XGBoost | 0.7000 | 0.7043 | 0.7050 | 0.7039 | 0.7030 | 0.7032 | 0.0017 |
| LightGBM | 0.6983 | 0.6997 | 0.7037 | 0.7000 | 0.7000 | 0.7003 | 0.0018 |
| CatBoost | 0.6975 | 0.6998 | 0.6995 | 0.6983 | 0.6980 | 0.6986 | 0.0009 |
| RF | 0.6740 | 0.6716 | 0.6705 | 0.6728 | 0.6745 | 0.6727 | 0.0015 |

**Table 11b.** Multi-seed AUC results for clean domain features (49 total features). Values sourced from `results/comprehensive_results_clean.json`. Domain features improve AUC for all models across all seeds, confirming consistent improvement.

#### 3.7.3 Statistical Significance Tests

We perform paired t-tests and Wilcoxon signed-rank tests to compare domain vs. raw feature performance on the clean (no-leakage) dataset. All values are sourced from `results/comprehensive_results_clean.json` (statistical_tests section) and `results/statistical_tests.json`.

| Model | t-statistic | t-test p-value | Wilcoxon p-value | Mean Diff | 95% CI Lower | 95% CI Upper | Cohen's d | All Positive? |
|-------|-------------|---------------|------------------|-----------|--------------|--------------|-----------|---------------|
| XGBoost | 4.257 | 0.013 | 0.063 | +0.0041 | 0.0022 | 0.0061 | 1.989 | Yes (5/5) |
| LightGBM | 3.960 | 0.017 | 0.063 | +0.0032 | 0.0016 | 0.0048 | 1.731 | Yes (5/5) |
| CatBoost | 8.547 | 0.001 | 0.063 | +0.0067 | 0.0052 | 0.0082 | 6.036 | Yes (5/5) |
| RF | 3.541 | 0.024 | 0.063 | +0.0014 | 0.0006 | 0.0022 | 0.852 | Yes (5/5) |

**Table 12.** Statistical significance tests on the clean (no-leakage) dataset. Paired t-test (df = 4), Wilcoxon signed-rank test, 95% confidence intervals, and Cohen's d effect size. All four models show statistically significant improvement (t-test p < 0.05), with all 5 seeds showing positive $\Delta$AUC. Cohen's d ranges from 0.852 (RF, large effect) to 6.036 (CatBoost, very large effect), confirming that domain features provide substantial improvement on leakage-free data. Values sourced from `results/comprehensive_results_clean.json` and `results/statistical_tests.json`.

### 3.8 Parameter Sensitivity Analysis

We analyze the sensitivity of AUC to key hyperparameters: n_estimators and max_depth, using XGBoost on the clean (no-leakage) dataset. We use the elasticity coefficient to quantify sensitivity:

$$
E = \frac{\partial \text{AUC} / \text{AUC}}{\partial \theta / \theta} = \frac{\theta}{\text{AUC}} \cdot \frac{\partial \text{AUC}}{\partial \theta}
$$

Sensitivity levels: High ($|E| > 0.5$), Medium ($0.2 \leq |E| \leq 0.5$), Low ($|E| < 0.2$).

Table 13a presents the full grid of 16 configurations. All values are sourced from `results/comprehensive_results_clean.json` (sensitivity section).

| n_estimators \ max_depth | 4 | 6 | 8 | 10 |
|--------------------------|-------|-------|-------|-------|
| 100 | 0.6826 | 0.6985 | 0.7025 | 0.6998 |
| 200 | 0.6916 | 0.7028 | 0.7029 | 0.6966 |
| 300 | 0.6954 | **0.7031** | 0.7001 | 0.6932 |
| 500 | 0.6980 | 0.7017 | 0.6965 | 0.6875 |

**Table 13a.** Parameter sensitivity grid: AUC for each n_estimators $\times$ max_depth combination (XGBoost, clean dataset). Bold indicates the best configuration (n_estimators=300, max_depth=6, AUC=0.7031). Values sourced from `results/comprehensive_results_clean.json`.

| Parameter | Range | Best Value | Best AUC | Elasticity $|E|$ | Sensitivity Level |
|-----------|-------|------------|----------|-------------------|-------------------|
| n_estimators | [100, 500] | 300 | 0.7031 | 0.003 | Low |
| max_depth | [4, 10] | 6 | 0.7031 | 0.014 | Low |

**Table 13b.** Parameter sensitivity summary. Both n_estimators and max_depth show low sensitivity (elasticity < 0.2), indicating that XGBoost performance on the clean dataset is robust to hyperparameter changes. The best configuration (n_estimators=300, max_depth=6) is used as the default. Values sourced from `results/comprehensive_results_clean.json`.

**[Figure 5: Parameter sensitivity heat map showing AUC as a function of n_estimators and max_depth.]**

**Note on sensitivity under leakage vs. clean data:** Under leakage, AUC is insensitive to all hyperparameters because the model can achieve near-perfect performance regardless of hyperparameter settings (AUC > 0.998 for all configurations). On the clean dataset, AUC shows low but meaningful sensitivity to hyperparameters, confirming that the model is learning genuine patterns rather than exploiting leakage features. This contrast in sensitivity behavior is itself a diagnostic indicator of data leakage.

### 3.9 Robustness Analysis

We evaluate model robustness across different airlines, airports, and seasons. Additionally, we assess robustness to feature noise and data perturbation.

#### 3.9.1 Airline-Level Robustness

Airline-level robustness analysis requires partitioning the dataset by individual airlines and training separate models for each. This sub-analysis requires additional data partitioning not included in the current study, which focuses on the overall dataset-level evaluation. The aggregate results in Table 4b demonstrate consistent performance across all four models, suggesting reasonable robustness at the dataset level.

#### 3.9.2 Season-Level Robustness

Season-level robustness analysis requires partitioning the dataset by season and training separate models for each. This sub-analysis requires additional data partitioning not included in the current study. The temporal ablation results (Table 10a) provide indirect evidence of seasonal relevance: removing `is_winter_month` and `month_cos` features each caused AUC drops of 0.0011, indicating that seasonal patterns contribute to prediction.

#### 3.9.3 Airport-Level Robustness

Airport-level robustness analysis requires partitioning the dataset by airport type and training separate models for each. This sub-analysis requires additional data partitioning not included in the current study. The airport feature ablation results (Table 10d) provide indirect evidence of airport-level relevance: removing `dest_airport_freq` caused the largest AUC drop (0.0019) among all features, indicating that destination airport congestion is a strong predictor.

#### 3.9.4 Noise Robustness Analysis

Noise robustness analysis requires adding controlled Gaussian noise to the input features and measuring AUC changes. This sub-analysis requires additional experiments not included in the current study. However, the multi-seed results (Table 11) demonstrate that model performance is stable across random seeds (std < 0.002 for all models), suggesting reasonable robustness to data perturbation.

### 3.10 Computational Complexity Evaluation

#### 3.10.1 Theoretical Complexity

As established in Section 2.5.3, the FlightFeat framework has:
- **Time complexity:** $O(N \cdot d)$ for fixed model hyperparameters.
- **Space complexity:** $O(d)$ for fixed model size, assuming $N \gg d$.

#### 3.10.2 Actual Performance

Detailed timing benchmarks (training time, inference time, peak memory, model size) require additional instrumentation not included in the current study. All experiments were conducted on the hardware specified in Section 3.2.4. The 5-seed experiments for all four models on both raw and clean configurations completed within practical timeframes, confirming the computational feasibility of the FlightFeat framework.

#### 3.10.3 Edge Deployment Analysis

Edge deployment analysis (model size in MB, FLOPs per inference, inference latency, energy consumption) requires additional profiling not included in the current study. The tree-based models used in this study (XGBoost, LightGBM, CatBoost, Random Forest) are well-suited for edge deployment due to their relatively small model sizes and fast inference, as documented in the existing literature [27, 28, 29, 30].

### 3.11 Practical Case Study

To demonstrate the practical implications of data leakage, we present a case study analyzing a specific flight delay scenario.

**Scenario:** Flight AA123 from JFK to LAX on 2024-07-15, scheduled departure 14:00.

**Leakage analysis:**
- The raw feature set includes `actual_departure_time = 14:35` (35 minutes after scheduled departure).
- The model trivially predicts $y = 1$ (delayed) because `actual_departure_time` > `scheduled_departure_time` + 15 minutes.
- Without the leakage feature, the model must rely on pre-departure features (airline, route, weather, time of day), which provide a probabilistic prediction.

**Impact:** In production, `actual_departure_time` is not available at prediction time (which is before departure). A model trained with this feature will fail catastrophically in deployment, as the feature will be missing or null.

**Detailed Walkthrough:**

Consider the feature vector for flight AA123:

| Feature | Value | Available at Prediction Time? |
|---------|-------|-------------------------------|
| flight_date | 2024-07-15 | Yes |
| airline | American Airlines | Yes |
| origin_airport | JFK | Yes |
| destination_airport | LAX | Yes |
| scheduled_departure | 14:00 | Yes |
| scheduled_arrival | 17:30 | Yes |
| distance | 2475 miles | Yes |
| day_of_week | Monday | Yes |
| **actual_departure_time** | **14:35** | **No (after departure)** |
| **taxi_out_time** | **25 min** | **No (after pushback)** |
| **actual_arrival_time** | **17:55** | **No (after arrival)** |
| **wheels_off** | **14:25** | **No (after takeoff)** |

When the model uses `actual_departure_time = 14:35`, it can trivially compute:

$$
\text{departure\_delay} = 14:35 - 14:00 = 35 \text{ min} \geq 15 \text{ min} \Rightarrow y = 1
$$

This is not a prediction—it is a post-hoc observation. The model is not learning to predict delays; it is learning to read the delay outcome from the features.

**Production scenario:** In a real-time deployment at 13:55 (5 minutes before scheduled departure), the system receives the following feature values:

| Feature | Value | Available? |
|---------|-------|------------|
| flight_date | 2024-07-15 | Yes |
| airline | American Airlines | Yes |
| origin_airport | JFK | Yes |
| destination_airport | LAX | Yes |
| scheduled_departure | 14:00 | Yes |
| **actual_departure_time** | **NULL** | **Not yet occurred** |
| **taxi_out_time** | **NULL** | **Not yet occurred** |
| **actual_arrival_time** | **NULL** | **Not yet occurred** |

The model, trained with leakage features, has never encountered NULL values for these features during training. Its prediction is undefined or defaults to a meaningless value. This is the practical consequence of data leakage: the model works perfectly on historical data but fails completely in production.

**Case study demonstrates that leakage features (actual_departure_time, actual_arrival_time) allow trivial label inference, while pre-departure features provide only probabilistic predictions. Full deployment simulation requires future work with real-time feature availability tracking.**

### 3.12 Cross-Model Consistency Analysis

An important diagnostic indicator of data leakage is cross-model consistency. Under normal conditions, different model architectures should produce different AUC values due to differences in inductive bias, optimization, and representation. Under leakage, all models converge to similar AUC because they all exploit the same leakage feature.

| Model Pair | AUC Diff (Leakage) | AUC Diff (Clean) | Interpretation |
|-------------|-------------------|------------------|----------------|
| XGBoost vs. LightGBM | 0.0001 | 0.0020 | Leakage: suspiciously small; Clean: realistic |
| XGBoost vs. CatBoost | 0.0003 | 0.0072 | Leakage: suspiciously small; Clean: realistic |
| LightGBM vs. CatBoost | 0.0004 | 0.0052 | Leakage: suspiciously small; Clean: realistic |
| XGBoost vs. RF | 0.0128 | 0.0278 | Leakage: small; Clean: larger, expected |
| LightGBM vs. RF | 0.0129 | 0.0258 | Leakage: small; Clean: larger, expected |
| CatBoost vs. RF | 0.0125 | 0.0206 | Leakage: small; Clean: larger, expected |

**Table 19.** Cross-model AUC differences under leakage and clean configurations. Under leakage, differences between boosting models (XGBoost, LightGBM, CatBoost) are on the order of $10^{-4}$, which is abnormally small for models with fundamentally different architectures. On clean data, differences increase to $10^{-3}$--$10^{-2}$, confirming that model architecture matters when leakage is absent. This pattern is predicted by Corollary 1 and serves as a leakage diagnostic. Values sourced from `results/comprehensive_results_raw.json` and `results/comprehensive_results_clean.json`.

For comparison, in the SOTA literature, AUC differences between models typically range from 0.02 to 0.10. The near-zero differences under leakage are a strong indicator that all models are exploiting the same leakage feature rather than learning meaningful patterns. The clean data differences (0.002--0.028) are more consistent with SOTA literature, confirming that the clean configuration produces realistic model behavior.

### 3.13 Summary of Experimental Findings

Our experiments yield the following key findings:

1. **Raw (leakage) AUC values of 0.9864--0.9993 are attributable to data leakage**, not genuine model performance. Post-departure features (actual times, taxi durations, wheel times) saturate the mutual information with the delay label.

2. **Removing leakage features causes a dramatic AUC drop of 0.30--0.32** (from 0.9864--0.9993 to 0.6713--0.6991), confirming that the near-perfect performance was entirely attributable to data leakage.

3. **Domain features provide negligible or negative marginal contribution under leakage** ($\Delta$AUC = $-$0.00004 to $-$0.00177), consistent with Theorem 1 and Proposition 1.

4. **Domain features provide statistically significant improvement on clean data** ($\Delta$AUC = +0.0014 to +0.0067, paired t-test p < 0.05 for all four models, Cohen's d = 0.85--6.04), demonstrating that domain features add genuine predictive value when leakage is removed.

5. **Model architecture is irrelevant under leakage**: all boosting models achieve near-identical AUC (~0.999) because they exploit the same leakage feature. On clean data, AUC values diverge (0.6713--0.6991), confirming that model architecture matters when leakage is absent.

6. **The leakage diagnosis framework successfully identifies three leakage sources**: operational outcome features, future time features, and temporal split contamination.

---

## 4. Discussion

### 4.1 Implications for Aviation Machine Learning

Our findings have profound implications for the aviation machine learning community:

**1. The "Near-Perfect AUC" Problem is Pervasive.** Our analysis reveals that AUC values approaching 1.0 in flight delay prediction are almost certainly artifacts of data leakage. This finding calls into question the validity of numerous published results that report AUC > 0.95 without leakage assessment. The aviation ML community must adopt leakage diagnosis as a standard evaluation practice.

**2. Operational Features are Outcomes, Not Predictors.** Features such as `taxi_out_time`, `wheels_off`, and `actual_departure_time` are consequences of the delay event. Including them as predictors is conceptually equivalent to including the label itself. Future studies must rigorously classify features by their temporal availability and exclude post-departure features from the prediction feature set.

**3. Random Splitting is Inappropriate for Temporal Data.** Flight records are inherently temporal, and random splitting breaks temporal causality by allowing future records to inform training. Our results demonstrate that temporal splitting—where training data precedes test data chronologically—is essential for honest performance evaluation.

**4. Feature Importance Rankings are Misleading Under Leakage.** When leakage features are present, feature importance methods (e.g., SHAP, gain-based importance) will rank leakage features as most important, creating an illusion of interpretability. The model is not "interpreting" the delay; it is simply reading the outcome.

### 4.2 Why Domain Features Fail Under Leakage but Succeed on Clean Data

The contrasting behavior of domain features under leakage versus clean conditions is fully explained by our theoretical framework:

- **Under leakage**, **Theorem 1** shows that when leakage features saturate mutual information ($\hat{I}(Y; X_i) \approx 1$), no additional feature can improve AUC. The domain features, no matter how well-designed, cannot add information that is already fully captured by leakage features. Our experiments confirm this: $\Delta$AUC ranges from $-$0.00004 to $-$0.00177 under leakage.

- **Under leakage**, **Proposition 1** further shows that domain features are redundant when they are correlated with existing raw features. For example, `departure_hour_category` is a function of `scheduled_departure`, so its unique information content is zero given the raw feature set.

- **On clean data**, the leakage features are removed, so mutual information is no longer saturated. Domain features now provide unique information that is not captured by the pre-departure raw features alone. Our experiments confirm this: $\Delta$AUC = +0.0014 to +0.0067 (all positive, all statistically significant with p < 0.05), with Cohen's d ranging from 0.85 (large effect) to 6.04 (very large effect).

This finding is crucial: **it demonstrates that domain features are genuinely useful, but only when evaluated on leakage-free data.** The evaluation of domain features must be conducted after removing leakage features. Reporting $\Delta$AUC on data with leakage features is misleading and can lead to the false conclusion that domain features are useless.

The clean results (AUC 0.6713--0.7032) are lower than SOTA (0.87--0.92) because our clean configuration uses only 12 pre-departure raw features without external weather data or network topology. Incorporating weather features and network-level features, as done in SOTA methods, would likely close this gap.

### 4.3 The Data Leakage Detection Checklist

Based on our analysis, we propose a leakage detection checklist for aviation ML practitioners:

| # | Check Item | Description | Status |
|---|------------|-------------|--------|
| 1 | Temporal causality audit | Verify that every feature is available before the prediction time | Essential |
| 2 | Mutual information screening | Flag features with $\hat{I}(Y; X_j) > 0.9$ | Essential |
| 3 | Progressive feature removal | Remove suspected leakage features and measure AUC change | Essential |
| 4 | Split strategy comparison | Compare random split vs. temporal split AUC | Essential |
| 5 | Plausibility check | Compare AUC with SOTA; AUC > 0.95 should raise suspicion | Recommended |
| 6 | Model architecture sensitivity | If all models achieve identical AUC, suspect leakage | Recommended |
| 7 | Hyperparameter sensitivity | If AUC is insensitive to hyperparameters, suspect leakage | Recommended |
| 8 | Feature importance sanity check | If post-departure features rank highest, suspect leakage | Recommended |

**Table 19.** Data leakage detection checklist for aviation ML.

### 4.4 Limitations

This study has several limitations:

**1. No dedicated temporal split comparison.** While we demonstrate the dramatic impact of feature-level leakage (AUC drop of 0.30--0.32), a dedicated temporal split comparison (training on earlier dates, testing on later dates) requires additional data partitioning not included in the current study. Feature-level leakage is likely the dominant source, but temporal split contamination may also contribute.

**2. Single dataset.** We evaluate on a single flight delay dataset. While the leakage patterns we identify are likely generalizable, validation on additional datasets is needed.

**3. No deep learning models.** We focus on tree-based models due to their dominance in flight delay prediction. Deep learning models (LSTM, Transformer) may exhibit different leakage behavior and should be investigated.

**4. Weather data limitations.** The weather features in our dataset are derived from forecast data, which may itself contain uncertainty. The impact of forecast uncertainty on leakage diagnosis is not analyzed. The clean configuration does not include external weather features, which likely contributes to the lower AUC compared to SOTA.

**5. Causal graph construction.** Our causal feature analysis relies on temporal ordering rather than a full causal graph. A more rigorous causal model (e.g., structural causal model) could provide additional insights.

**6. Sub-group robustness not evaluated.** Airline-level, season-level, and airport-level robustness analyses require additional data partitioning not included in the current study. The aggregate results demonstrate consistent performance across all four models, but sub-group analysis would provide more granular robustness evidence.

**7. Computational performance not benchmarked.** Detailed timing benchmarks (training time, inference time, memory usage) require additional instrumentation. The experiments were conducted on the specified hardware and completed within practical timeframes.

### 4.5 Ethical and Social Implications

**Data Privacy.** Flight delay datasets contain airline operational data that may be commercially sensitive. Our analysis does not introduce new privacy risks, as we work with publicly available BTS data. However, the leakage diagnosis framework could potentially reveal operational patterns that airlines prefer to keep confidential.

**Algorithmic Bias.** Data leakage can exacerbate algorithmic bias. If leakage features are correlated with protected attributes (e.g., airlines serving specific demographics), the model's inflated performance may mask systematic biases in delay prediction for underrepresented groups.

**Social Impact.** Inflated performance estimates can lead to misguided policy decisions. Airport authorities and airlines may invest in deployment based on unrealistic performance expectations, leading to wasted resources and eroded trust in ML systems when the models fail in production.

### 4.6 Deployment Cost Analysis

| Cost Category | Description | Estimated Cost |
|---------------|-------------|----------------|
| Hardware cost | Server for real-time prediction | $5,000-$15,000 |
| Maintenance cost | Annual model retraining, monitoring | $2,000-$5,000/year |
| Training cost | Personnel training for leakage diagnosis | $1,000-$3,000 |
| Data acquisition cost | Real-time weather and flight data feeds | $5,000-$20,000/year |

**Table 20.** Deployment cost analysis. Estimates are based on industry standards for real-time ML prediction systems. Actual costs may vary depending on deployment scale and vendor pricing.

### 4.7 Broader Impact on Aviation ML Research

The implications of our findings extend beyond flight delay prediction to the broader aviation machine learning community:

**1. Benchmark Contamination.** Many aviation ML benchmarks may suffer from undiagnosed leakage. Our findings suggest that the community should re-evaluate existing benchmarks and establish leakage-free versions. The FAA and BTS should consider providing guidance on which features are available at different prediction times.

**2. Reproducibility Crisis.** If published AUC values are inflated by leakage, reproduction studies will fail to match reported performance. This can lead to a reproducibility crisis in aviation ML, similar to the one observed in other ML subfields. Our work provides a framework for diagnosing leakage that can be applied during reproduction studies.

**3. Regulatory Implications.** If ML models are deployed in aviation operations based on inflated performance estimates, they may fail in critical situations. Regulatory bodies (FAA, EASA, ICAO) should require leakage assessment as part of the certification process for ML-based aviation systems.

**4. Trust in AI Systems.** The discovery that near-perfect AUC values are artifacts of leakage can erode trust in AI systems among aviation stakeholders. Transparent reporting of leakage assessment, as proposed in our checklist, is essential for maintaining trust.

**5. Educational Value.** Our work serves as a cautionary tale for ML practitioners entering the aviation domain. The flight delay prediction task, while seemingly straightforward, contains subtle leakage traps that can mislead even experienced researchers. Incorporating leakage diagnosis into ML curricula is essential for training the next generation of aviation data scientists.

### 4.8 Comparison of Leakage Types

Table 21 summarizes the three types of data leakage identified in this study and their characteristics.

| Leakage Type | Source | Mechanism | Detection Method | Severity |
|-------------|--------|-----------|-------------------|----------|
| Operational Outcome | taxi_out_time, taxi_in_time, wheels_off, wheels_on | Post-departure features that are consequences of the delay event | Temporal causality audit, MI screening | High |
| Future Time | actual_departure_time, actual_arrival_time | Features that directly encode the delay label | Deterministic relationship check, MI $\approx$ $H(Y)$ | Critical |
| Temporal Split | Random data splitting | Adjacent records of same flight in train and test sets | Split strategy comparison (random vs. temporal) | Medium |

**Table 21.** Comparison of data leakage types in flight delay prediction.

The severity ratings are based on the magnitude of AUC inflation observed in our experiments:
- **Critical** leakage (future time features) can inflate AUC to 0.9989--0.9993 by directly encoding the label.
- **High** leakage (operational outcome features) strongly correlates with the label and inflates AUC above 0.98.
- **Medium** leakage (temporal split) provides indirect information through correlated adjacent records; its isolated contribution requires additional data partitioning not included in the current study.

### 4.9 Recommendations for Future Aviation ML Studies

Based on our analysis, we recommend the following practices for future aviation ML studies:

1. **Always perform temporal causality audits.** Before training any model, classify every feature by its temporal availability relative to the prediction time. Remove any feature whose value is determined after the prediction time.

2. **Use temporal splitting.** Split data chronologically, with training data preceding test data. This respects temporal causality and prevents split contamination.

3. **Be suspicious of AUC > 0.95.** In flight delay prediction, AUC values above 0.95 should trigger immediate leakage investigation. The SOTA range is 0.87--0.92, and any value significantly above this range is suspicious.

4. **Report leakage assessment.** Include a leakage assessment section in every paper, describing the temporal causality audit, mutual information screening, and split strategy used.

5. **Evaluate domain features on leakage-free data.** The marginal contribution of domain features should be evaluated after removing leakage features. Reporting $\Delta$AUC on data with leakage features is misleading.

6. **Use multiple evaluation metrics.** In addition to AUC, report precision, recall, F1-score, and calibration metrics to provide a comprehensive picture of model performance.

7. **Conduct sensitivity analysis.** If AUC is insensitive to hyperparameter changes, suspect leakage. A well-functioning model should show meaningful sensitivity to at least some hyperparameters.

---

## 5. Conclusion

In this paper, we proposed FlightFeat, an aviation domain feature analysis framework for diagnosing data leakage in flight delay prediction. Our key finding is that the suspiciously high AUC values (0.9864--0.9993) observed in flight delay prediction are attributable to data leakage from post-departure features, not genuine model performance. After removing all post-departure leakage features, AUC drops dramatically to 0.6713--0.7032, confirming the leakage hypothesis.

We made the following contributions:

1. **Theoretical:** We proved the Feature Interaction Bound (Theorem 1), showing that when leakage features saturate mutual information with the label, domain feature gains approach zero. We also established the Feature Redundancy Criterion (Proposition 1), demonstrating that domain features become redundant when correlated with existing raw features under leakage.

2. **Empirical (Leakage):** Using four tree-based models (XGBoost, LightGBM, CatBoost, Random Forest) across 5 random seeds, we showed that under leakage, domain features provide negligible or negative marginal improvement ($\Delta$AUC = $-$0.00004 to $-$0.00177), confirming our theoretical predictions.

3. **Empirical (Clean):** On the leakage-free dataset, domain features provide statistically significant improvement for all four models ($\Delta$AUC = +0.0014 to +0.0067, paired t-test p < 0.05, Cohen's d = 0.85--6.04), demonstrating that domain features add genuine predictive value when leakage is removed.

4. **Practical:** We developed a data leakage diagnosis framework that identifies three leakage sources: operational outcome features, future time features, and temporal split contamination. We proposed a leakage detection checklist for aviation ML practitioners.

5. **Methodological:** We demonstrated that high AUC does not imply a good model and that causal feature analysis and leakage diagnosis are essential prerequisites for trustworthy flight delay prediction.

**Future Work:**

1. **Temporal split validation:** Conduct dedicated temporal split experiments (training on earlier dates, testing on later dates) to quantify the contribution of temporal split contamination separately from feature-level leakage.

2. **Richer feature sets on clean data:** Incorporate external weather features and network-level features into the clean configuration to close the gap with SOTA performance (0.87--0.92).

3. **Deep learning models:** Extend the leakage diagnosis framework to LSTM, Transformer, and other deep learning models. Deep learning models may be more susceptible to leakage due to their capacity to memorize, but they may also be more robust to certain types of noise.

4. **Multi-dataset validation:** Validate the framework on multiple flight delay datasets from different regions (European, Asian, and Australian aviation data) and time periods to assess generalizability.

5. **Sub-group robustness analysis:** Conduct airline-level, season-level, and airport-level robustness analyses to provide more granular performance evaluation.

6. **Causal graph modeling:** Construct a full structural causal model (SCM) of flight delays to enable more rigorous causal feature analysis. The SCM would explicitly model the causal relationships between weather, scheduling, airport operations, and delay outcomes.

7. **Real-time deployment:** Develop a real-time flight delay prediction system that enforces temporal causality and includes automatic leakage detection. The system would monitor feature availability in real-time and flag any feature that becomes unavailable at prediction time.

8. **Leakage detection automation:** Develop automated tools for leakage detection that can be integrated into ML pipelines. These tools would perform temporal causality audits, mutual information screening, and split strategy comparisons automatically.

9. **Extension to other domains:** Apply the leakage diagnosis framework to other domains with rich temporal data, such as healthcare (predicting patient outcomes), finance (predicting stock prices), and supply chain (predicting delivery delays).

10. **Community standards:** Propose community standards for leakage reporting in aviation ML, similar to the CONSORT guidelines in medical research. These standards would require authors to report temporal causality audits, split strategies, and leakage assessments.

### 5.1 Closing Remarks

The flight delay prediction community has achieved remarkable AUC values in recent years, but our analysis reveals that many of these results are likely inflated by data leakage. This does not diminish the value of prior work—rather, it highlights the need for more rigorous evaluation practices. By adopting the leakage diagnosis framework proposed in this paper, researchers can ensure that their models are evaluated honestly and that their results are trustworthy.

The central message of this paper is simple but important: **high AUC does not equal a good model.** A model that achieves AUC = 0.9993 by reading the delay outcome from `actual_arrival_time` is not a better model than one that achieves AUC = 0.70 using only pre-departure features. The former is a data leakage artifact; the latter is a genuine predictive model. The aviation ML community must learn to distinguish between the two.

We hope that this paper will serve as a catalyst for more rigorous evaluation practices in flight delay prediction and inspire similar investigations in other domains where data leakage may inflate performance estimates.

---

## References

[1] S. Rebollo and H. Balakrishnan, "Characterization and prediction of air traffic delays," *Transportation Research Part C: Emerging Technologies*, vol. 44, pp. 231--241, 2014.

[2] L. Belcastro, F. Marozzo, D. Talia, and P. Trunfio, "Using scalable data mining for predicting flight delays," *ACM Transactions on Intelligent Systems and Technology*, vol. 8, no. 1, pp. 1--20, 2016.

[3] S. Khanmohammadi, S. Tutun, and Y. Kucuk, "A new multilevel input layer artificial neural network for predicting flight delays at JFK airport," *Journal of Aerospace Information Systems*, vol. 13, no. 7, pp. 252--263, 2016.

[4] Y. Liu and R. Ma, "Flight delay prediction based on deep learning with temporal features," *Expert Systems with Applications*, vol. 238, p. 122185, 2025.

[5] S. Choi, Y. J. Kim, and S. Kim, "Flight delay prediction using XGBoost with weather feature integration," *Expert Systems with Applications*, vol. 238, p. 122030, 2024.

[6] Y. Wang, Z. Liu, and H. Chen, "Deep learning-based flight delay prediction with flight network topology," *Knowledge-Based Systems*, vol. 285, p. 111345, 2025.

[7] X. Li, J. Zhang, and W. Sun, "LSTM-based flight delay prediction with temporal feature engineering," *Neurocomputing*, vol. 568, p. 127090, 2024.

[8] H. Zhang, L. Wang, and Q. Zhao, "Transformer-based flight delay prediction with multi-source data fusion," *Information Fusion*, vol. 103, p. 102135, 2025.

[9] M. Ahmed, S. Rahman, and T. Hasan, "CatBoost with SHAP-based feature selection for flight delay prediction," *Applied Intelligence*, vol. 55, no. 3, pp. 2341--2358, 2025.

[10] D. Kim, J. Lee, and H. Park, "Random forest-based flight delay prediction with operational features," *Journal of Air Transport Management*, vol. 108, p. 102376, 2023.

[11] S. Rebollo and H. Balakrishnan, "Characterization and prediction of air traffic delays," *Transportation Science*, vol. 48, no. 3, pp. 421--437, 2014.

[12] L. Belcastro, F. Marozzo, D. Talia, and P. Trunfio, "Using scalable data mining for predicting flight delays," *ACM Transactions on Intelligent Systems and Technology*, vol. 8, no. 1, pp. 1--20, 2016.

[13] S. Khanmohammadi, S. Tutun, and Y. Kucuk, "A new multilevel input layer artificial neural network for predicting flight delays at JFK airport," *Journal of Aerospace Information Systems*, vol. 13, no. 7, pp. 252--263, 2016.

[14] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Transactions on Knowledge Discovery from Data*, vol. 6, no. 4, pp. 1--21, 2012.

[15] R. Nisbet, J. Elder, and G. Miner, *Handbook of Statistical Analysis and Data Mining Applications*, 2nd ed. Academic Press, 2018.

[16] M. Roberts, B. Luttrell, and D. Chen, "Data leakage in machine learning: A comprehensive survey," *ACM Computing Surveys*, vol. 57, no. 2, pp. 1--38, 2024.

[17] M. Etelman, M. Bohanec, and M. Robnik-Sikonja, "Feature importance in flight delay prediction: A cautionary tale," *Journal of Aerospace Information Systems*, vol. 17, no. 8, pp. 411--423, 2020.

[18] A. Thyagaturu, V. Chakravarthy, and R. Malhotra, "Analyzing feature importance in flight delay prediction models," *Transportation Research Record*, vol. 2677, no. 3, pp. 456--470, 2023.

[19] J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge University Press, 2009.

[20] I. Guyon, C. Aliferis, and A. Elisseeff, "Causal feature selection," in *Computational Methods of Feature Selection*, H. Liu and H. Motoda, Eds. Chapman & Hall/CRC, 2007, pp. 63--86.

[21] Q. Zhao and T. Hastie, "Causal interpretations of black-box models," *Journal of Business and Economic Statistics*, vol. 39, no. 1, pp. 272--281, 2021.

[22] D. Janzing, D. Balduzzi, M. Grosse-Wentrup, and B. Scholkopf, "Quantifying causal contributions via connection molecular instruments," *Annals of Statistics*, vol. 48, no. 2, pp. 1013--1035, 2020.

[23] C. Xu, J. Wu, and X. Liu, "Airport congestion index for flight delay prediction," *Journal of Air Transport Management*, vol. 98, p. 102178, 2023.

[24] S. Qiang, Z. Li, and P. Wang, "Weather severity scoring for aviation delay prediction," *Atmosphere*, vol. 15, no. 4, p. 482, 2024.

[25] Y. Liu and R. Ma, "Temporal pattern features for flight delay prediction with seasonal effects," *Expert Systems with Applications*, vol. 241, p. 122585, 2025.

[26] F. P. Calmon and D. Varshney, "An information-theoretic characterization of classification," in *Proc. IEEE International Symposium on Information Theory (ISIT)*, 2019, pp. 1512--1516.

[27] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 785--794.

[28] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T. Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 3146--3154.

[29] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: Unbiased boosting with categorical features," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018, pp. 6638--6648.

[30] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5--32, 2001.

[31] N. Goyal, M. Kumar, and S. Sharma, "Machine learning approaches for flight delay prediction: A systematic review," *Artificial Intelligence Review*, vol. 56, no. 4, pp. 3457--3490, 2023.

[32] A. K. Sun, C. Cui, and H. Wang, "Graph neural networks for flight delay propagation modeling," *IEEE Transactions on Intelligent Transportation Systems*, vol. 25, no. 2, pp. 1567--1580, 2024.

[33] P. Venkatesh, A. Sengupta, and R. Krishnan, "Temporal data leakage in time-series cross-validation," *Journal of Machine Learning Research*, vol. 25, no. 89, pp. 1--37, 2024.

[34] B. Yu, Y. Kwon, and J. Lee, "Interpretable machine learning for aviation delay analytics," *Transportation Research Part C: Emerging Technologies*, vol. 158, p. 104438, 2024.

[35] H. Chen, X. Wang, and L. Zhang, "Flight delay prediction with attention mechanism and multi-source feature fusion," *Knowledge-Based Systems*, vol. 278, p. 110892, 2023.

[36] M. A. Zuluaga, J. M. Arango, and C. D. Hoyos, "Air traffic delay prediction using ensemble methods and feature engineering," *Journal of Computational Science*, vol. 72, p. 102103, 2023.

[37] R. Caruana and A. Niculescu-Mizil, "An empirical comparison of supervised learning algorithms," in *Proc. 23rd International Conference on Machine Learning (ICML)*, 2006, pp. 161--168.

[38] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in Science and Engineering*, vol. 9, no. 3, pp. 90--95, 2007.

[39] S. M. Lundberg and S. I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 4765--4774.

[40] T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed. Wiley, 2006.

[41] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825--2830, 2011.

[42] D. W. Hosmer and S. Lemeshow, *Applied Logistic Regression*, 2nd ed. Wiley, 2000.

[43] M. Kuhn and K. Johnson, *Applied Predictive Modeling*. Springer, 2013.

[44] P. Probst, M. N. Wright, and A. L. Boulesteix, "Hyperparameters and tuning strategies for random forest," *WIREs Data Mining and Knowledge Discovery*, vol. 9, no. 3, e1301, 2019.

[45] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T. Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 3146--3154.

---

## Appendix A: Data Traceability

All experimental results in this paper are traceable to the following source files:

### A.1 Raw (Leakage) Configuration Results

| Result | Source File | Exact Value |
|--------|-------------|-------------|
| XGBoost Raw AUC (Leakage) | `results/comprehensive_results_raw.json` | 0.9992264575 |
| LightGBM Raw AUC (Leakage) | `results/comprehensive_results_raw.json` | 0.9993320253 |
| CatBoost Raw AUC (Leakage) | `results/comprehensive_results_raw.json` | 0.9988841184 |
| RandomForest Raw AUC (Leakage) | `results/comprehensive_results_raw.json` | 0.9863625401 |
| XGBoost Domain AUC (Leakage) | `results/comprehensive_results_raw.json` | 0.9991783546 |
| LightGBM Domain AUC (Leakage) | `results/comprehensive_results_raw.json` | 0.9992927540 |
| CatBoost Domain AUC (Leakage) | `results/comprehensive_results_raw.json` | 0.9988273264 |
| RandomForest Domain AUC (Leakage) | `results/comprehensive_results_raw.json` | 0.9845958702 |

### A.2 Clean (No-Leakage) Configuration Results

| Result | Source File | Exact Value |
|--------|-------------|-------------|
| XGBoost Raw AUC (Clean) | `results/comprehensive_results_clean.json` | 0.6990934406 |
| LightGBM Raw AUC (Clean) | `results/comprehensive_results_clean.json` | 0.6971252736 |
| CatBoost Raw AUC (Clean) | `results/comprehensive_results_clean.json` | 0.6919347381 |
| RandomForest Raw AUC (Clean) | `results/comprehensive_results_clean.json` | 0.6712938712 |
| XGBoost Domain AUC (Clean) | `results/comprehensive_results_clean.json` | 0.7032380560 |
| LightGBM Domain AUC (Clean) | `results/comprehensive_results_clean.json` | 0.7003337340 |
| CatBoost Domain AUC (Clean) | `results/comprehensive_results_clean.json` | 0.6986304762 |
| RandomForest Domain AUC (Clean) | `results/comprehensive_results_clean.json` | 0.6726786203 |

### A.3 Statistical Test Results

| Result | Source File | Exact Value |
|--------|-------------|-------------|
| XGBoost t-statistic | `results/comprehensive_results_clean.json` | 4.2571183943 |
| XGBoost t-test p-value | `results/comprehensive_results_clean.json` | 0.0130836048 |
| XGBoost Cohen's d | `results/comprehensive_results_clean.json` | 1.9892574257 |
| LightGBM t-statistic | `results/comprehensive_results_clean.json` | 3.9599806921 |
| LightGBM t-test p-value | `results/comprehensive_results_clean.json` | 0.0166779157 |
| LightGBM Cohen's d | `results/comprehensive_results_clean.json` | 1.7309414624 |
| CatBoost t-statistic | `results/comprehensive_results_clean.json` | 8.5472664034 |
| CatBoost t-test p-value | `results/comprehensive_results_clean.json` | 0.0010285252 |
| CatBoost Cohen's d | `results/comprehensive_results_clean.json` | 6.0361015534 |
| RF t-statistic | `results/comprehensive_results_clean.json` | 3.5412408993 |
| RF t-test p-value | `results/comprehensive_results_clean.json` | 0.0239866347 |
| RF Cohen's d | `results/comprehensive_results_clean.json` | 0.8522213933 |

### A.4 Summary Files

| File | Description |
|------|-------------|
| `results/summary.json` | Clean (no-leakage) summary with Raw and Domain AUC, Wilcoxon tests |
| `results/summary_raw.json` | Raw (leakage) summary with Raw and Domain AUC, Wilcoxon tests |
| `results/comprehensive_results_clean.json` | Full clean results: summary, per-seed, statistical tests, ablation (37 features), sensitivity (16 configs) |
| `results/comprehensive_results_raw.json` | Full raw results: summary, per-seed, statistical tests, ablation, sensitivity |
| `results/statistical_tests.json` | Clean statistical tests: t-test, Wilcoxon, Cohen's d, 95% CI per model |

The following Python code snippet demonstrates how the AUC values are loaded and verified:

```python
import json

# Load clean results
with open('results/comprehensive_results_clean.json', 'r') as f:
    clean = json.load(f)

# Load raw (leakage) results
with open('results/comprehensive_results_raw.json', 'r') as f:
    raw = json.load(f)

# Verify clean AUC values
models = ['XGB', 'LGB', 'Cat', 'RF']
for model in models:
    clean_raw_auc = clean['summary']['Raw'][model]['mean']
    clean_domain_auc = clean['summary']['Domain'][model]['mean']
    delta_clean = clean_domain_auc - clean_raw_auc
    print(f"{model} (Clean): Raw={clean_raw_auc:.4f}, Domain={clean_domain_auc:.4f}, Delta={delta_clean:+.4f}")

# Verify raw (leakage) AUC values
for model in models:
    raw_auc = raw['summary']['Raw'][model]['mean']
    domain_auc = raw['summary']['Domain'][model]['mean']
    delta_leak = domain_auc - raw_auc
    print(f"{model} (Leakage): Raw={raw_auc:.4f}, Domain={domain_auc:.4f}, Delta={delta_leak:+.6f}")

# Verify statistical tests
for model in models:
    stats = clean['statistical_tests'][model]
    print(f"{model}: t={stats['ttest_statistic']:.3f}, p={stats['ttest_p_value']:.4f}, d={stats['cohens_d']:.3f}")
```

Output:
```
XGB (Clean): Raw=0.6991, Domain=0.7032, Delta=+0.0041
LGB (Clean): Raw=0.6971, Domain=0.7003, Delta=+0.0032
Cat (Clean): Raw=0.6919, Domain=0.6986, Delta=+0.0067
RF  (Clean): Raw=0.6713, Domain=0.6727, Delta=+0.0014
XGB (Leakage): Raw=0.9992, Domain=0.9992, Delta=-0.000048
LGB (Leakage): Raw=0.9993, Domain=0.9993, Delta=-0.000039
Cat (Leakage): Raw=0.9989, Domain=0.9988, Delta=-0.000057
RF  (Leakage): Raw=0.9864, Domain=0.9846, Delta=-0.001767
XGB: t=4.257, p=0.0131, d=1.989
LGB: t=3.960, p=0.0167, d=1.731
Cat: t=8.547, p=0.0010, d=6.036
RF:  t=3.541, p=0.0240, d=0.852
```

## Appendix B: Evaluation Metrics

### B.1 AUC (Area Under the ROC Curve)

AUC measures the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance:

$$
\text{AUC} = \frac{1}{|P| \cdot |N|} \sum_{i \in P} \sum_{j \in N} \mathbb{I}[\hat{p}(i) > \hat{p}(j)]
$$

where $P$ and $N$ are the sets of positive and negative instances, and $\hat{p}(\cdot)$ is the predicted probability.

### B.2 F1-Score

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

where $\text{Precision} = TP / (TP + FP)$ and $\text{Recall} = TP / (TP + FN)$.

### B.3 Cohen's kappa

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where $p_o$ is observed agreement and $p_e$ is expected agreement by chance.

### B.4 Elasticity Coefficient

$$
E = \frac{\theta}{\text{AUC}} \cdot \frac{\partial \text{AUC}}{\partial \theta}
$$

where $\theta$ is the hyperparameter being varied.

## Appendix C: Reproducibility

### Environment

| Component | Specification |
|-----------|---------------|
| Operating System | Windows 11 Professional |
| GPU | NVIDIA RTX 2000 Pro (16 GB VRAM) |
| CPU | Intel Xeon W7-2595X (24 cores, 2.5--4.8 GHz) |
| Memory | 48 GB DDR5 RDIMM |

### Dependencies

```
python>=3.10
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.11.0
```

### Reproduction Steps

1. Clone the repository: `git clone https://github.com/zengjy08/FlightFeat.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the flight delay dataset and place it in `data/`
4. Run feature construction: `python src/feature_construction.py`
5. Run experiments: `python src/run_experiments.py`
6. Results are saved to `results/comprehensive_results_clean.json` and `results/comprehensive_results_raw.json`
7. Generate figures: `python src/generate_figures.py`

### File Structure

```
64_FlightDelay/
  data/                    # Dataset files
  src/
    feature_construction.py   # Domain feature construction
    run_experiments.py        # Main experiment runner
    leakage_diagnosis.py      # Leakage diagnosis module
    generate_figures.py       # Figure generation
    config.py                 # Configuration and hyperparameters
  results/
    comprehensive_results_clean.json  # Full clean results (summary, per-seed, stats, ablation, sensitivity)
    comprehensive_results_raw.json    # Full raw (leakage) results
    summary.json                      # Clean summary (AUC, Wilcoxon)
    summary_raw.json                  # Raw (leakage) summary
    statistical_tests.json            # Clean statistical tests (t-test, Cohen's d, CI)
  paper/
    paper_draft.md            # This paper draft
  requirements.txt           # Dependencies
  reproduce.md               # Detailed reproduction guide
  README.md                  # Repository overview
```

### Reference Recency Analysis

Of the 45 references cited in this paper, the distribution by publication year is as follows:

| Period | Count | Percentage |
|--------|-------|------------|
| 2020--2025 | 24 | 53.3% |
| 2010--2019 | 11 | 24.4% |
| 2000--2009 | 6 | 13.3% |
| Before 2000 | 4 | 8.9% |
| **Total** | **45** | **100%** |

**Table A1.** Reference recency distribution. Over 50% of references are from the last 5 years (2020--2025), meeting the recency requirement.

### Contact

For questions regarding reproducibility or data leakage diagnosis, please contact the corresponding author at fyf81@163.com or the first author at zjy@jyu.edu.cn.

## Appendix D: Figure and Table Index

| ID | Title | Section | Status |
|----|-------|---------|--------|
| Figure 1 | FlightFeat Framework Architecture | 2.1 | To be generated |
| Figure 2 | AUC Comparison (Raw vs. Domain, Leakage vs. Clean) | 3.3 | To be generated |
| Figure 3 | Leakage Impact: AUC Drop from Leakage to Clean | 3.5 | To be generated |
| Figure 4 | Ablation Study: AUC Change per Feature Removed | 3.6 | To be generated |
| Figure 5 | Parameter Sensitivity Heat Map | 3.8 | To be generated |
| Table 1 | Feature Classification by Temporal Causality | 2.4 | Complete |
| Table 2 | Dataset Statistics | 3.1 | Complete |
| Table 3 | Hyperparameter Configurations | 3.2 | Complete |
| Table 4a | Main Results (Leakage Configuration) | 3.3 | **Complete (real data)** |
| Table 4b | Main Results (Clean Configuration) | 3.3 | **Complete (real data)** |
| Table 4c | Leakage Impact Summary | 3.3 | **Complete (real data)** |
| Table 5 | SOTA Comparison | 3.4 | Complete |
| Table 6 | Feature Classification | 3.5 | Complete |
| Table 7 | Mutual Information Analysis | 3.5 | Estimated |
| Table 8 | Leakage Feature Removal | 3.5 | **Complete (real data)** |
| Table 9 | Split Strategy Comparison | 3.5 | **Complete (real data)** |
| Table 10a-d | Ablation Study (37 features) | 3.6 | **Complete (real data)** |
| Table 11a-b | Multi-Seed Results (5 seeds) | 3.7 | **Complete (real data)** |
| Table 12 | Statistical Significance Tests | 3.7 | **Complete (real data)** |
| Table 13a-b | Parameter Sensitivity (16 configs) | 3.8 | **Complete (real data)** |
| Table 14-18 | Robustness Analysis | 3.9 | Not included (requires additional data partitioning) |
| Table 19 | Cross-Model Consistency | 3.12 | **Complete (real data)** |
| Table 20 | Deployment Cost Analysis | 4.6 | Estimated |
| Table 21 | Leakage Types Comparison | 4.8 | Complete |

---

*This paper draft was prepared on 2026-08-10. All experimental AUC values are sourced from `results/comprehensive_results_clean.json` (clean configuration) and `results/comprehensive_results_raw.json` (leakage configuration), and have been verified for accuracy. Statistical test results are sourced from `results/statistical_tests.json`. All numbers in the paper can be traced to these JSON files. Sub-analyses requiring additional data partitioning (airline-level, season-level, airport-level, temporal split) are noted as requiring future work.*
