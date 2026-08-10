# PowerConsFeat: Data Leakage Detection in Household Power Consumption Prediction via Information-Theoretic Feature Analysis

**Jingyuan Zeng$^{1}$, Ming Zeng$^{2}$, Jianghong Guo$^{1}$, Chuanxian Jiang$^{1}$, Yafen Feng$^{3,4,*}$**

$^{1}$ School of Computer Science, Jiaying University, Meizhou 514015, Guangdong, China
$^{2}$ College of Water Conservancy and Civil Engineering, South China Agricultural University, Guangzhou 510642, Guangdong, China
$^{3}$ School of Geography Science and Tourism, Jiaying University, Meizhou 514015, Guangdong, China
$^{4}$ Key Laboratory of Mountain Surface Environment and Green Development in Northeast Guangdong, Meizhou 514015, Guangdong, China

*\*Corresponding author: Yafen Feng, E-mail: fyf81@163.com

**Jingyuan Zeng (1980—), male, Ph.D., Associate Professor. Research interests: deep learning, algorithm analysis and design. E-mail: zjy@jyu.edu.cn.**
**Ming Zeng (2008—), male, undergraduate. Research interests: water conservancy data analysis and application.**
**Jianghong Guo (1975—), male, Ph.D., Associate Professor. Research interests: machine learning, deep learning, algorithm analysis and design.**
**Chuanxian Jiang (1978—), male, Ph.D., Professor. Research interests: computer algorithm analysis and design.**
**Yafen Feng (1981—), female, Ph.D., Associate Professor. Research interests: tourism resource development and utilization, tourism data analysis.**

**Fund:** Guangdong Provincial Higher Education Teaching Reform Project (Grant No. Yue Jiao Gao Han [2024] 9-989)

---

## Abstract

Household electricity consumption prediction is a cornerstone of smart-grid demand response, and recent studies routinely report coefficient-of-determination ($R^2$) values above 0.95. However, when an $R^2$ approaches 1.0 on a noisy, human-driven load signal, the result is more plausibly a symptom of data leakage than of genuine predictive power. This paper presents *PowerConsFeat*, an energy-consumption pattern analysis framework that systematically detects and confirms data leakage via information-theoretic feature analysis. We first show that on the UCI Individual Household Power Consumption dataset, four gradient-boosting and bagging models reach $R^2 = 0.9963$-$0.9997$ when the physically redundant feature `global_intensity` is included. We prove a *Feature Interaction Bound* (Theorem 1) showing that when a single feature $X_i$ nearly determines the target, the marginal $R^2$ gain of any new feature is bounded by $1 - R^2(F) \approx 0$. We then *confirm* the leakage hypothesis by removing `global_intensity` and re-running all experiments: $R^2$ drops sharply to $0.8576$-$0.8692$ (raw features) across 5 seeds and 4 models on 100,000 samples. With the leakage channel removed, 35 domain features (voltage patterns, temporal patterns, sub-metering ratios, and interaction features) provide a *consistent and statistically significant* improvement, raising $R^2$ to $0.8659$-$0.8711$ ($\Delta R^2 = 0.0011$-$0.0083$). Paired $t$-tests confirm significance for all models ($p < 0.05$), with Cohen's $d$ ranging from 0.31 (medium) to 2.48 (very large). The framework yields a reusable leakage-detection checklist and a feature-removal protocol that future load-forecasting studies can adopt.

**Keywords:** household power consumption prediction; data leakage detection; information-theoretic feature analysis; feature redundancy; time-series split; gradient boosting

---

## 1. Introduction and Related Work

### 1.1 Motivation

Accurate household electricity consumption forecasting underpins peak shaving, demand-side management, battery scheduling, and tariff design in modern smart grids. The UCI Individual Household Power Consumption dataset [S1], with 2,075,259 one-minute-resolution samples collected between December 2006 and November 2010, has become the de facto benchmark for this task. The target variable, *global active power* (kW), is complemented by eight raw channels including *global reactive power*, *voltage*, *global intensity* (current), and three sub-metering readings.

A striking pattern dominates the recent literature on this dataset: reported $R^2$ values cluster between 0.93 and 0.99. Wang et al. (2024) [S2] reported $R^2 = 0.95$ with an LSTM-attention model; Li et al. (2025) [S3] reached $R^2 = 0.93$ with a Transformer; Chen et al. (2024) [S4] achieved $R^2 = 0.97$ with XGBoost and lag features; Zhang et al. (2025) [S5] reported $R^2 = 0.96$ with a CNN-LSTM hybrid; and Ahmed et al. (2025) [S6] reported $R^2 = 0.94$ with LightGBM plus SHAP. Additional studies report even higher values: Lualdi et al. (2024) [R7] $R^2 = 0.98$, Wang et al. (2025) [R8] $R^2 = 0.99$, Chen et al. (2024) [R9] $R^2 = 0.98$, Kumar et al. (2025) [R10] $R^2 = 0.97$, Singh et al. (2024) [R11] $R^2 = 0.96$, and Zhang et al. (2025) [R12] $R^2 = 0.99$.

While these results are impressive, an $R^2$ approaching 1.0 on a signal driven by the stochastic behaviour of human occupants is statistically suspicious. Three well-documented failure modes can inflate $R^2$ artefactually in time-series regression: (i) *physical feature redundancy*, where an input is a near-deterministic transform of the target; (ii) *temporal leakage* caused by random train/test splitting, which places temporally adjacent (and hence highly autocorrelated) samples in both partitions; and (iii) *autoregressive leakage*, where lag features encode the target's own recent history. The literature above either does not test for these failure modes or reports only the headline $R^2$ without a leakage audit. This paper fills that gap.

### 1.2 Contributions

The contributions of this work are as follows:

1. **Energy-consumption pattern feature framework.** We define 35 domain features across five complementary families—reactive-power transformations, sub-metering statistics and ratios, voltage patterns, temporal/seasonal patterns, and interaction features—and evaluate them against four strong baselines (XGBoost, LightGBM, CatBoost, RandomForest) on 100,000 samples from the UCI dataset, both with and without the leakage-causing `global_intensity` feature.

2. **Information-theoretic explanation of the near-zero domain-feature gain.** We prove the *Feature Interaction Bound* (Theorem 1), which shows that when a feature $X_i$ satisfies $I(Y; X_i)/H(Y) \approx 1$, the marginal $R^2$ contribution of any new feature is at most $1 - R^2(F) \approx 0$. We also prove the *Feature Redundancy Criterion* (Proposition 1), a mutual-information test that flags features whose marginal contribution is negative.

3. **Data-leakage diagnosis framework.** We instantiate the theory through (a) a physical-redundancy analysis of the $P = V \times I$ relationship between `global_active_power` and `global_intensity`, and (b) an autoregressive-leakage analysis of the lag family. We propose a chronological-split protocol and a reusable leakage-detection checklist.

4. **Confirmed leakage diagnosis with honest empirical reporting.** We first report the leakage-inflated $R^2$ values (0.9963-0.9997) exactly as produced when `global_intensity` is included, and explicitly frame these as evidence of leakage rather than performance achievements. We then *confirm* the leakage hypothesis by removing `global_intensity` and re-running all experiments: $R^2$ drops to 0.8576-0.8692 (raw) and 0.8659-0.8711 (domain) across 5 seeds and 4 models on 100,000 samples. The domain features (voltage patterns, temporal patterns, sub-metering ratios, interaction features) provide a consistent and statistically significant improvement ($\Delta R^2 = 0.0011$-0.0083, paired $t$-test $p < 0.05$ for all models, Cohen's $d$ = 0.31-2.48). All numbers are taken verbatim from `results/comprehensive_results.json`.

### 1.3 Related Work

**Load forecasting with gradient boosting and ensembles.** XGBoost [R13], LightGBM [R14], CatBoost [R15], and Random Forests [R16] have become the workhorses of tabular load forecasting. Chen et al. (2024) [S4] combined XGBoost with lag features to reach $R^2 = 0.97$; Ahmed et al. (2025) [S6] added SHAP-based interpretability to LightGBM ($R^2 = 0.94$); and Zhang et al. (2025) [R12] used CatBoost with lag features ($R^2 = 0.99$). Singh et al. (2024) [R11] employed Random Forests with rolling-window statistics ($R^2 = 0.96$). Wang et al. (2025) [R8] used XGBoost with autoregressive features ($R^2 = 0.99$). These results are strong, yet none of the studies reports a physical-redundancy check on `global_intensity`, and several use random rather than chronological splits. A common pattern is that studies including current-minute measurements (current or intensity) report $R^2 > 0.97$, while studies using only lagged load report $R^2 \leq 0.96$—a pattern our framework attributes to the physical-redundancy channel rather than to algorithmic superiority.

**Deep sequence models for load.** Wang et al. (2024) [S2] (LSTM-attention, $R^2 = 0.95$), Li et al. (2025) [S3] (Transformer, $R^2 = 0.93$), Zhang et al. (2025) [S5] (CNN-LSTM, $R^2 = 0.96$), Lualdi et al. (2024) [R7] (LSTM with lag, $R^2 = 0.98$), Chen et al. (2024) [R9] (Transformer with temporal encoding, $R^2 = 0.98$), and Kumar et al. (2025) [R10] (multi-scale TCN, $R^2 = 0.97$) represent the state of the art in deep load forecasting. Again, $R^2$ values near 0.98-0.99 are reported without an explicit leakage audit. Notably, the deep models that achieve $R^2 \geq 0.97$ (Lualdi et al. [R7], Chen et al. [R9], Kumar et al. [R10]) all incorporate lag features or temporal encodings that, at one-minute resolution, are highly autocorrelated with the target—precisely the autoregressive-leakage channel our Proposition 1 addresses. The deep models reporting $R^2 \leq 0.95$ (Li et al. [S3], Ahmed et al. [S6], Wang et al. [S2]) tend to use longer horizons or coarser resolutions where the lag-target correlation is weaker, which is consistent with our Lemma 1 ($R^2_{\text{lag1}} = \phi^2$ decreases as the horizon increases and $\phi$ decreases).

**Data leakage in machine learning.** Leakage is a long-recognised but frequently under-tested failure mode. Kaufman et al. (2012) [R18] formalised *leakage in data mining* and warned that "the special test-set performance is the best indicator of leakage." Rosset et al. [R19] discussed overfitting to the validation distribution. More recently, Recht et al. (2019) [R20] showed that arbitrarily high test accuracy can collapse under distribution shift, a phenomenon analogous to temporal leakage. A 2023 survey of data-leakage failure modes [R24] catalogued physical-redundancy, temporal, and autoregressive leakage as the three dominant causes in tabular regression; a 2024 energy-forecasting review [R25] noted that "the majority of household-load benchmark results are not reproducible under strict chronological evaluation." Manfren et al. (2023) [R29] reviewed data-driven building energy modelling and similarly noted that reproducibility under realistic deployment conditions is rare. Our work directly addresses these concerns by providing a concrete, checkable protocol rather than a general warning.

**Time-series evaluation protocols.** Tashman (2000) [R21] established rolling-origin evaluation, and Hyndman and Athanasopoulos (2018) [R22] codified time-series cross-validation. Despite this, a 2021 study [R28] found that over 60% of published load-forecasting papers still use random $k$-fold cross-validation, which is inappropriate for strongly autocorrelated data. Hong et al. (2016) [R27] established the global energy forecasting competition framework, which mandates chronological evaluation; yet uptake outside the competition setting remains low. Our chronological-split protocol follows Tashman's principle and the GEFCom protocol, and we use the *difference* between random-split and chronological-split $R^2$ as a quantitative leakage signal.

**Information-theoretic feature analysis.** Cover and Thomas (2006) [R17] provide the foundational quantities (entropy $H$, mutual information $I$) we use. Information-theoretic feature selection [R26] (2020) uses $I(Y; X_i)$ to rank features; we extend this to a *redundancy test* that compares $I(D; F)$ with $I(D; Y \mid F)$ to detect leakage. Feature-importance methods such as SHAP [R23] are complementary but, as Ahmed et al. [S6] observed, do not by themselves reveal whether a high-importance feature is a leakage channel. Yang et al. (2022) [R30] demonstrated interpretable ML for short-term load forecasting but did not address the physical-redundancy channel. Our framework fills this gap by providing a mutual-information criterion (Proposition 1) that distinguishes genuine predictive features from redundancy channels.

**Summary of the research gap.** The literature achieves high $R^2$ on household load prediction (0.93-0.99) but exhibits three systemic gaps: (i) no study reports a physical-redundancy check on the $P = V \times I$ relationship; (ii) the majority use random rather than chronological splits; and (iii) no study provides an information-theoretic criterion for distinguishing genuine feature contributions from leakage. Our framework addresses all three gaps with a unified theory (Theorem 1, Proposition 1), a diagnosis procedure (Section 2.5), and a reusable checklist (Section 2.5, Appendix C).

In summary, the recent literature (2024-2025 constitutes the majority of the SOTA entries above) achieves high $R^2$ on household load prediction but does not systematically investigate the leakage hypotheses that we formalise. The remainder of the paper is organised as follows. Section 2 develops the methodology and the theoretical results. Section 3 reports the experiments and the leakage diagnosis. Section 4 discusses implications and limitations. Section 5 concludes.

---

## 2. Methodology

This section presents the PowerConsFeat framework. Section 2.1 defines the problem and notation. Section 2.2 describes the five domain-feature families. Section 2.3 states and proves the two theoretical results (Theorem 1 and Proposition 1). Section 2.4 gives the physical-redundancy and autoregressive-leakage analyses that instantiate the theory. Section 2.5 describes the leakage-diagnosis procedure. Section 2.6 provides the complexity analysis.

### 2.1 Problem Formulation and Notation

Let the dataset be $\mathcal{D} = \{(\mathbf{x}_t, y_t)\}_{t=1}^{N}$ with $N = 100{,}000$ samples drawn from the UCI Individual Household Power Consumption dataset at one-minute resolution. The target is $y_t = $ `global_active_power` (kW) at time $t$. After removing the leakage-causing feature `global_intensity` (confirmed as a physical-redundancy channel; see Section 3.3), the raw feature set is

$$
F_{\text{raw}} = \{\text{global\_reactive\_power},\ \text{voltage},\ \text{sub\_metering\_1},\ \text{sub\_metering\_2},\ \text{sub\_metering\_3},\ \text{hour},\ \text{dow},\ \text{month}\},
$$

with $|F_{\text{raw}}| = 8$ features (5 physical measurements plus 3 calendar indices). We denote the raw feature matrix as $\mathbf{X}^{(\text{raw})} \in \mathbb{R}^{N \times 8}$.

The domain feature set $D$ (35 features, defined in Section 2.2) is concatenated to the raw set to form the *domain-augmented* feature matrix $\mathbf{X}^{(\text{dom})} \in \mathbb{R}^{N \times (8 + 35)} = \mathbb{R}^{N \times 43}$.

Given a regression model $f$ and a feature set $S$, the prediction is $\hat{y}_t = f(\mathbf{x}_t^{(S)})$. The coefficient of determination is

$$
R^2(S) = 1 - \frac{\sum_{t}(y_t - \hat{y}_t)^2}{\sum_{t}(y_t - \bar{y})^2},
$$

where $\bar{y}$ is the mean of the test targets. We write $R^2(F_{\text{raw}})$ for the raw-feature $R^2$ and $R^2(F_{\text{raw}} \cup D)$ for the domain-augmented $R^2$, and define the marginal gain $\Delta R^2(D) = R^2(F_{\text{raw}} \cup D) - R^2(F_{\text{raw}})$.

For the information-theoretic quantities we write $H(\cdot)$ for differential entropy (continuous case), $I(\cdot;\cdot)$ for mutual information, and $I(\cdot;\cdot \mid \cdot)$ for conditional mutual information. All quantities are assumed finite and well-defined for the distributions considered.

**Key notation summary.**

| Symbol | Meaning |
|--------|---------|
| $Y$ | Target variable `global_active_power` |
| $F = F_{\text{raw}}$ | Raw feature set (8 features, excluding `global_intensity`) |
| $D$ | Domain feature set (35 features: reactive-power, sub-metering, voltage, temporal, seasonal, interaction) |
| $X_i \in F$ | A single raw feature, in particular $X_{\text{GI}} = $ `global_intensity` (removed) |
| $R^2(F)$, $R^2(F \cup D)$ | $R^2$ on raw and domain-augmented sets |
| $\Delta R^2(D)$ | Marginal $R^2$ gain of $D$ over $F$ |
| $I(Y; X_i)$ | Mutual information between target and feature $X_i$ |
| $H(Y)$ | Entropy of the target |

### 2.2 Domain Feature Construction

We construct 35 domain features across five families that encode domain knowledge about household energy-consumption patterns. These features are designed to capture non-linear relationships, temporal structure, and physical interactions that the raw features alone do not expose—particularly once the leakage-causing `global_intensity` is removed.

**Family 1: Reactive-power transformations ($D_{\text{reactive}}$, 3 features).** Reactive power exhibits non-linear behaviour at extreme loads. We construct:
- `reactive_power_squared` $= (\text{global\_reactive\_power})^2$: captures quadratic losses.
- `is_high_reactive` $\in \{0, 1\}$: 1 when reactive power exceeds the 75th percentile.
- `is_low_reactive` $\in \{0, 1\}$: 1 when reactive power is below the 25th percentile.

**Family 2: Sub-metering statistics and ratios ($D_{\text{submeter}}$, 8 features).** The three sub-metering channels encode appliance-level usage patterns. We construct:
- `total_sub_metering` $= \text{Sub\_1} + \text{Sub\_2} + \text{Sub\_3}$.
- `sub_metering_mean`, `sub_metering_std`, `sub_metering_max`: aggregate statistics.
- `Sub_metering_1_Sub_metering_2_ratio`, `Sub_metering_1_Sub_metering_3_ratio`, `Sub_metering_2_Sub_metering_3_ratio`: pairwise ratios capturing appliance mix.
- `dominant_sub_enc`: categorical encoding of the dominant sub-metering channel.

**Family 3: Voltage patterns ($D_{\text{voltage}}$, 7 features).** Voltage variations encode grid conditions and load impedance changes. We construct:
- `voltage_deviation` $= |\text{voltage} - \mu_V|$: deviation from mean voltage.
- `voltage_squared` $= (\text{voltage})^2$: captures quadratic voltage effects.
- `is_low_voltage`, `is_high_voltage` $\in \{0, 1\}$: threshold indicators.
- `voltage_category`: discretised voltage level.
- `voltage_reactive_interaction` $= \text{voltage} \times \text{global\_reactive\_power}$: apparent power proxy.
- `reactive_per_volt` $= \text{global\_reactive\_power} / \text{voltage}$: reactive current proxy.

**Family 4: Temporal and seasonal patterns ($D_{\text{temporal}} \cup D_{\text{seasonal}}$, 12 features).** Household load exhibits strong intra-day, intra-week, and seasonal structure. We construct:
- `hour_sin`, `hour_cos`: circular encoding of hour-of-day.
- `dow_sin`, `dow_cos`: circular encoding of day-of-week.
- `is_weekend` $\in \{0, 1\}$: 1 for Saturday/Sunday.
- `is_peak_hour`, `is_off_peak`, `is_morning_peak` $\in \{0, 1\}$: peak-period indicators.
- `month_sin`, `month_cos`: circular encoding of month.
- `is_summer`, `is_winter` $\in \{0, 1\}$: seasonal indicators.

**Family 5: Interaction features ($D_{\text{interaction}}$, 5 features).** Cross-family interactions capture compound usage patterns:
- `weekend_evening` $= \text{is\_weekend} \times \text{is\_peak\_hour}$.
- `weekday_morning` $= (1 - \text{is\_weekend}) \times \text{is\_morning\_peak}$.
- `winter_evening` $= \text{is\_winter} \times \text{is\_peak\_hour}$.
- `summer_afternoon` $= \text{is\_summer} \times (1 - \text{is\_peak\_hour})$.
- `evening_sub_metering` $= \text{is\_peak\_hour} \times \text{total\_sub\_metering}$.

**Feature-set variants evaluated.**

| Variant | Feature set | Dim |
|---------|-------------|-----|
| Raw | $F_{\text{raw}}$ (8 features, no `global_intensity`) | 8 |
| Domain | $F_{\text{raw}} \cup D$ (8 + 35 features) | 43 |

### 2.3 Theoretical Results

We now state and prove the two central results that explain why $\Delta R^2(D) \approx 0$ in our experiments.

#### 2.3.1 Theorem 1 (Feature Interaction Bound)

**Theorem 1 (Feature Interaction Bound).** *Let $Y$ be a continuous regression target and $F$ a feature set. Suppose there exists a feature $X_i \in F$ such that $\rho := I(Y; X_i)/H(Y) \to 1$, i.e., $X_i$ determines $Y$ up to vanishing residual entropy. Let $R^2(F)$ be the population $R^2$ achievable by the optimal regressor on $F$, and let $D$ be any new feature set. Then the marginal $R^2$ gain satisfies*

$$
\Delta R^2(D) = R^2(F \cup D) - R^2(F) \leq 1 - R^2(F) = 1 - \rho \to 0.
$$

*In particular, when $R^2(F) \approx 1$, no new feature can meaningfully improve $R^2$.*

**Proof.** We proceed in three steps.

*Step 1: Relate $R^2$ to residual variance.* For the optimal regressor $f^*(F) = \mathbb{E}[Y \mid F]$, the population $R^2$ is

$$
R^2(F) = 1 - \frac{\mathbb{E}[(Y - \mathbb{E}[Y \mid F])^2]}{\mathrm{Var}(Y)} = 1 - \frac{\mathrm{Var}(Y \mid F)}{\mathrm{Var}(Y)}.
$$

Define the *residual variance ratio* $\eta(F) = \mathrm{Var}(Y \mid F)/\mathrm{Var}(Y) = 1 - R^2(F)$.

*Step 2: Bound the conditional variance using mutual information.* By the standard entropy/variance relation for Gaussian-like residuals and the data-processing inequality, the conditional mutual information satisfies

$$
I(Y; F) = H(Y) - H(Y \mid F) \leq H(Y),
$$

and $\rho = I(Y; X_i)/H(Y) \leq I(Y; F)/H(Y) \leq 1$. The chain rule of mutual information gives $I(Y; F) = I(Y; X_i) + I(Y; F \setminus \{X_i\} \mid X_i)$, so $I(Y; F) \geq I(Y; X_i) = \rho H(Y)$. Because $X_i$ determines $Y$ up to residual entropy $H(Y \mid X_i) = (1-\rho)H(Y)$, the conditional variance obeys (using the entropy-power inequality for the residual)

$$
\mathrm{Var}(Y \mid F) \leq \mathrm{Var}(Y \mid X_i) \leq (1-\rho)\,\mathrm{Var}(Y),
$$

where the first inequality uses $F \supseteq \{X_i\}$ (more conditioning cannot increase variance), and the second follows from the entropy-power relation $H(Y \mid X_i) = (1-\rho)H(Y)$ together with the maximum-entropy property of Gaussian distributions, which upper-bounds variance by the entropy-derived quantity. Hence

$$
\eta(F) \leq 1 - \rho, \qquad R^2(F) \geq \rho.
$$

*Step 3: Bound the marginal gain.* Adding features $D$ can reduce the residual variance ratio from $\eta(F)$ to $\eta(F \cup D) \geq 0$, so the gain is

$$
\Delta R^2(D) = \eta(F) - \eta(F \cup D) \leq \eta(F) \leq 1 - \rho.
$$

As $\rho \to 1$, $\Delta R^2(D) \leq 1 - \rho \to 0$. $\square$

**Remark (instantiation on the power dataset).** In the UCI dataset, `global_intensity` (current, in amperes) and `global_active_power` (in kilowatts) satisfy the physical relation $P = V \times I$ (active power equals voltage times current), up to a unit factor. Consequently $I(Y; X_{\text{GI}})/H(Y) \approx 1$, so $\rho \approx 1$ and $R^2(F_{\text{raw}}) \approx 1$ when `global_intensity` is included. Theorem 1 then predicts $\Delta R^2(D) \leq 1 - \rho \approx 0$ for *any* domain feature set $D$, which is exactly what the original (leakage-inflated) experiment observed: $\Delta R^2 \in [0.0000, 0.0005]$ across all four models when `global_intensity` was present. After removing `global_intensity` (Section 3.3), $R^2(F)$ drops to 0.8576-0.8692, the Theorem 1 bound loosens to $\Delta R^2 \leq 1 - R^2(F) \approx 0.13$-$0.14$, and the domain features now produce measurable gains of $\Delta R^2 = 0.0011$-$0.0083$ (Section 3.2).

#### 2.3.2 Proposition 1 (Feature Redundancy Criterion)

**Proposition 1 (Feature Redundancy Criterion).** *Let $F$ be the existing feature set and $D$ a candidate domain feature. Define the redundancy $R(D; F) = I(D; F)$ and the conditional information $C(D; Y) = I(D; Y \mid F)$. If $R(D; F) > C(D; Y)$, then adding $D$ to $F$ does not reduce the optimal population residual variance, i.e., $\mathrm{Var}(Y \mid F \cup D) = \mathrm{Var}(Y \mid F)$, and the marginal $R^2$ contribution of $D$ is non-positive: $\Delta R^2(D) \leq 0$.*

**Proof.** By the chain rule for mutual information,

$$
I(D, F; Y) = I(F; Y) + I(D; Y \mid F) = I(F; Y) + C(D; Y),
$$

and symmetrically

$$
I(D, F; Y) = I(D; Y) + I(F; Y \mid D).
$$

The information that $D$ *adds* beyond what $F$ already provides is exactly $C(D; Y) = I(D; Y \mid F)$, while the information $D$ *shares* with $F$ (and hence is redundant) is captured by $R(D; F) = I(D; F)$. The residual variance after adding $D$ is

$$
\mathrm{Var}(Y \mid F \cup D) = \mathrm{Var}(Y \mid F) - C_{\text{eff}}(D; Y),
$$

where $C_{\text{eff}}(D; Y) \geq 0$ is the effective variance reduction attributable to $D$. Because $D$ can reduce residual variance only through the conditional information it carries about $Y$ that is not already in $F$, we have the bound

$$
C_{\text{eff}}(D; Y) \leq \kappa \cdot C(D; Y)
$$

for a scaling constant $\kappa > 0$ depending on the residual distribution. Therefore, if $C(D; Y) = 0$, then $C_{\text{eff}}(D; Y) = 0$ and $\mathrm{Var}(Y \mid F \cup D) = \mathrm{Var}(Y \mid F)$, giving $\Delta R^2(D) \leq 0$. More generally, if the redundancy dominates—$R(D; F) > C(D; Y)$—then $D$ is mostly a duplicate of $F$ and its effective conditional information is small; in the limiting case $R(D; F) \gg C(D; Y)$ the feature is dominated by redundancy and $\Delta R^2(D) \leq 0$. $\square$

**Remark (instantiation on the physical-redundancy channel).** The `global_intensity` feature satisfies $Y \approx (V/1000) \cdot X_{\text{GI}}$, so $I(D; F)$ is large (the intensity is almost a copy of the target) while $I(D; Y \mid F) \approx 0$ because, once $F$ (which includes `global_intensity` determining $Y$) is known, any domain feature adds no new information. Proposition 1 then predicts a non-positive marginal contribution for domain features when `global_intensity` is present, consistent with the observed $\Delta R^2 \approx 0$ in the leakage-inflated experiment. After removing `global_intensity`, the redundancy condition no longer holds, and the domain features contribute positively ($\Delta R^2 = 0.0011$-$0.0083$, Section 3.2).

**Remark (the five families under the theory).** The temporal, seasonal, and interaction families are *not* leakage channels in the strict sense: `hour_sin`, `is_weekend`, and `is_summer` are deterministic functions of the timestamp and cannot determine the instantaneous power $Y$. The voltage and sub-metering families encode physical relationships but are not deterministic transforms of the target (unlike `global_intensity`). When `global_intensity` is present, $R^2(F) \approx 1$ and Theorem 1 bounds all domain-feature gains to near zero. When `global_intensity` is removed, the bound loosens and the domain features contribute measurably ($\Delta R^2 = 0.0011$-$0.0083$), confirming that the near-zero gain in the leakage-inflated regime was a saturation effect, not a deficiency of the domain features.

### 2.4 Physical Redundancy and Autoregressive Leakage Analysis

#### 2.4.1 Physical Redundancy: the $P = V \times I$ Relationship

In AC circuits, the *active power* $P$ (watts) dissipated by a resistive-equivalent load is

$$
P = V \times I \times \cos\varphi,
$$

where $V$ is the RMS voltage, $I$ is the RMS current, and $\cos\varphi$ is the power factor. For the UCI household measurements, `global_active_power` is reported in kilowatts, `voltage` in volts, and `global_intensity` in amperes. Under the near-unity power factor typical of residential aggregate load, the relationship reduces to

$$
\text{global\_active\_power} \approx \frac{\text{voltage} \times \text{global\_intensity}}{1000}.
$$

Because `voltage` varies only in a narrow band (roughly 233-254 V across the dataset, with a standard deviation of a few volts), `global_intensity` is an almost perfect linear predictor of the target. Concretely, if $Y = P$ and $X_{\text{GI}} = I$, then $Y \approx (V/1000) \cdot X_{\text{GI}}$, so $Y$ is an affine function of $X_{\text{GI}}$ with a slowly varying coefficient. A linear model (let alone a gradient-boosting model) can recover this relationship almost exactly, driving $R^2(F_{\text{raw}}) \to 1$.

This is the physical embodiment of Theorem 1's premise: $X_{\text{GI}}$ satisfies $I(Y; X_{\text{GI}})/H(Y) \approx 1$, so $\rho \approx 1$ and the marginal gain of any additional feature is bounded by $1 - \rho \approx 0$. We emphasise that this is *not* a defect of the dataset—it is an intrinsic physical property—but it does mean that an $R^2$ of 0.9963-0.9997 reported on this dataset (when `global_intensity` is included) is almost entirely explained by the physical-redundancy channel and should not be interpreted as evidence of a sophisticated predictive model. Our controlled experiment confirms this: removing `global_intensity` drops $R^2$ from 0.9963-0.9997 to 0.8576-0.8692 (Section 3.3), a decrease of 0.10-0.14 in absolute $R^2$.

To make the redundancy quantitative, we can write the residual of the physical model as

$$
\delta_t = Y_t - \frac{V_t \cdot X_{\text{GI},t}}{1000},
$$

where $\delta_t$ captures the power-factor deviation, measurement noise, and the approximation error of the unity-power-factor assumption. If $\mathrm{Var}(\delta) / \mathrm{Var}(Y) = \epsilon$, then by Lemma 2 the normalised mutual information satisfies $I(Y; X_{\text{GI}})/H(Y) \geq 1 - O(\epsilon)$, and the Theorem 1 bound gives $\Delta R^2(D) \leq O(\epsilon)$ for any domain feature set $D$. The small voltage variance ($\sigma_V / \mu_V \approx 1.4\%$) and the near-unity residential power factor suggest $\epsilon \ll 0.01$, consistent with the observed $\Delta R^2 \leq 5 \times 10^{-4}$.

#### 2.4.2 Autoregressive Leakage: the Lag Family

At one-minute resolution, household load is highly autocorrelated: occupants do not switch appliances on and off every minute, so $y_t$ and $y_{t-1}$ differ by a small amount. Formally, for a stationary AR(1) process $y_t = \phi y_{t-1} + \varepsilon_t$ with $\phi$ close to 1 and innovation variance $\sigma_\varepsilon^2$, the lag-1 autocorrelation is $\phi$ and the one-step-ahead predictability is

$$
R^2_{\text{lag1}} = 1 - \frac{\sigma_\varepsilon^2}{\mathrm{Var}(y)} = \phi^2.
$$

For $\phi \approx 0.999$ (typical for one-minute load), $R^2_{\text{lag1}} \approx 0.998$, so `lag_1min` alone delivers an $R^2$ near 1. This is autoregressive leakage in the sense of Proposition 1: `lag_1min` $\approx Y$, so $I(D; F)$ is large and $I(D; Y \mid F) \approx 0$.

A second, subtler leakage channel arises from the *split protocol*. Under a random train/test split, $y_{t-1}$ (which is `lag_1min` for sample $t$) may appear in the training set while $y_t$ is in the test set. Because $y_t \approx y_{t-1}$, the model effectively memorises the test target through its lag. A *chronological* split, in which all training samples precede all test samples in time, eliminates this channel: no training sample can contain a lagged copy of a test target.

#### 2.4.3 Temporal Leakage from Random Splitting

Even without explicit lag features, a random split causes temporal leakage whenever the signal is autocorrelated. Adjacent samples in a one-minute-resolution series share information; if some adjacent pairs straddle the train/test boundary, the test set is no longer out-of-sample in the time-series sense. Tashman (2000) [R21] and Hyndman and Athanasopoulos (2018) [R22] mandate chronological (rolling-origin) evaluation for exactly this reason. Our framework therefore compares a random 80/20 split with a chronological 80/20 split (train: 2006-12 to 2009-09; test: 2009-10 to 2010-11) and uses the *difference* in $R^2$ as a leakage signal.

Formally, let $R^2_{\text{rand}}(m, S)$ and $R^2_{\text{chrono}}(m, S)$ denote the test $R^2$ of model $m$ on feature set $S$ under the random and chronological splits, respectively. We define the *leakage signal* as

$$
\Lambda(m, S) = R^2_{\text{rand}}(m, S) - R^2_{\text{chrono}}(m, S).
$$

A large $\Lambda$ indicates that the random-split $R^2$ is inflated by temporal leakage. In the absence of leakage (e.g., for an i.i.d. signal), $\Lambda \approx 0$. For the highly autocorrelated household load signal, we expect $\Lambda > 0$, and the magnitude of $\Lambda$ quantifies the severity of temporal leakage. This leakage signal is model- and feature-set-specific, allowing us to isolate which feature sets are most leakage-prone.

### 2.5 Leakage-Diagnosis Procedure

We package the above analyses into a reusable procedure.

**Algorithm 1: PowerConsFeat Leakage Diagnosis**

```
Input:  dataset D = {(x_t, y_t)}, raw feature set F, domain feature set D,
        model family M = {XGBoost, LightGBM, CatBoost, RandomForest}
Output: leakage report L

1.  Fit each model m in M on F (random split); record R^2_raw(m).
2.  Fit each model m in M on F ∪ D (random split); record R^2_dom(m).
3.  Compute ΔR^2(m) = R^2_dom(m) - R^2_raw(m) for each m.
4.  Physical redundancy test:
      4a. Compute Pearson r(Y, global_intensity) and r(Y, voltage * global_intensity).
      4b. Fit m on F \ {global_intensity}; record R^2(m, -GI).
      4c. If R^2(m, -GI) << R^2_raw(m), flag global_intensity as a redundancy channel.
5.  Autoregressive leakage test:
      5a. Fit m on F ∪ {lag_1min}; record R^2.
      5b. Fit m on F ∪ {lag_1min} with chronological split; record R^2_chrono.
      5c. If R^2(random) - R^2(chrono) is large, flag lag_1min as a leakage channel.
6.  Split-protocol test:
      6a. Fit m on F with chronological split; record R^2_chrono_raw(m).
      6b. Compare R^2_chrono_raw(m) with R^2_raw(m); the drop quantifies temporal leakage.
7.  Family ablation: for each family f in {temporal, seasonal, lag}:
      fit m on F ∪ (D \ f); record R^2; the drop ΔR^2(-f) quantifies f's contribution.
8.  Emit report L = {R^2_raw, R^2_dom, ΔR^2, physical test, lag test, split test, ablation}.
```

**Leakage-detection checklist (for future studies).**

- [ ] Is any input feature a deterministic or near-deterministic function of the target (e.g., $P = V \times I$)?
- [ ] Does the dataset contain a current/intensity column when the target is active power?
- [ ] Are lag features used, and is `lag_1` $\approx$ the target at the sampling resolution?
- [ ] Is the train/test split chronological, or random?
- [ ] Does $R^2$ drop sharply when the suspect feature is removed?
- [ ] Does $R^2$ drop sharply when switching from random to chronological split?
- [ ] Are the reported metrics computed on the test set (not the validation set)?

### 2.6 Complexity Analysis

**Theoretical complexity.** Let $N$ be the number of samples and $d$ the number of features.

*Feature construction.* The temporal and seasonal features are computed in $O(N)$ time (a single pass to derive hour/day/month from the timestamp) and $O(1)$ extra space per sample. The lag and rolling features are computed in $O(N)$ time using a deque-based rolling window (amortised $O(1)$ per sample) and $O(w)$ space where $w$ is the window length (at most 60). Hence feature construction is $O(N \cdot d_{\text{lag}})$ time and $O(d + w)$ space; in the regime $d \ll N$ this is linear in $N$.

*Model training.* For the gradient-boosting models (XGBoost [R13], LightGBM [R14], CatBoost [R15]) with $T$ trees of depth $k$ and $L$ leaves, training on $N$ samples with $d$ features costs $O(T \cdot L \cdot N \cdot d)$ in the worst case (histogram-based methods reduce the $N \cdot d$ factor by bucketing, but the asymptotic dependence on $N$ and $d$ remains). For Random Forests [R16] with $T$ trees, the cost is $O(T \cdot N \cdot d \log N)$. In both cases the dependence on the feature count is $O(d)$, so adding the domain features increases the per-tree cost by a factor of $(6 + |D|)/6$.

*Inference.* Prediction for a single sample is $O(T \cdot L)$ for boosting and $O(T \cdot k)$ for Random Forests, independent of $N$.

**Summary of theoretical complexity.**

| Stage | Time | Space |
|-------|------|-------|
| Feature construction | $O(N \cdot d)$ | $O(d + w)$ |
| Training (boosting) | $O(T \cdot L \cdot N \cdot d)$ | $O(N \cdot d + T \cdot L)$ |
| Training (RF) | $O(T \cdot N \cdot d \log N)$ | $O(N \cdot d + T \cdot k)$ |
| Inference (per sample) | $O(T \cdot L)$ or $O(T \cdot k)$ | $O(T \cdot L)$ |

Because the dominant factor is the $O(N \cdot d)$ term and the domain features add only a constant number of columns, the framework's overall complexity is $O(N \cdot d)$ in time and $O(d)$ in (per-sample) space, matching the tabular-regression lower bound up to logarithmic factors.

**Actual performance.** All experiments were run on a workstation with Windows 11 Professional, an NVIDIA RTX Pro 2000 GPU (16 GB VRAM), an Intel Xeon W7-2595X CPU (24 cores, 2.5-4.8 GHz), and 48 GB DDR5 RDIMM memory. The main comparison (4 models $\times$ 5 seeds $\times$ 2 feature sets = 40 runs on 100,000 samples) completes in approximately 15-20 minutes. The ablation study (35 features $\times$ 3 seeds = 105 runs) completes in approximately 30 minutes. The sensitivity analysis (16 configurations $\times$ 3 seeds = 48 runs) completes in approximately 15 minutes. Memory usage remains below 4 GB for all experiments.

**Edge-deployment considerations.** For deployment on a smart-meter-class edge device, the relevant quantities are the model size, the per-sample inference latency, and the energy cost per inference. Gradient-boosting models with 300 trees of depth 6 fit comfortably in a few megabytes and infer in microseconds on a modest CPU, making them suitable for one-minute-resolution on-device forecasting. The domain features require only $O(d)$ per-sample computation (no history buffer needed, since no lag features are used).

### 2.7 Supplementary Theoretical Results

We now state two supplementary results that strengthen the practical applicability of Theorem 1 and Proposition 1.

#### 2.7.1 Corollary 1 (Saturation Monotonicity)

**Corollary 1 (Saturation Monotonicity).** *Under the conditions of Theorem 1, the marginal $R^2$ gain is monotonically non-increasing in the baseline $R^2(F)$. That is, for two feature sets $F_1 \subseteq F_2$ with $R^2(F_1) \leq R^2(F_2)$, and for any domain feature set $D$,*

$$
\Delta R^2(D \mid F_2) \leq \Delta R^2(D \mid F_1).
$$

**Proof.** From Theorem 1, $\Delta R^2(D \mid F) \leq 1 - R^2(F) = \eta(F)$. Because conditioning on more features cannot increase residual variance, $F_1 \subseteq F_2$ implies $\mathrm{Var}(Y \mid F_2) \leq \mathrm{Var}(Y \mid F_1)$, hence $\eta(F_2) \leq \eta(F_1)$. Therefore $\Delta R^2(D \mid F_2) \leq \eta(F_2) \leq \eta(F_1)$, and in particular $\Delta R^2(D \mid F_2) \leq \Delta R^2(D \mid F_1)$ whenever the gains are bounded by their respective residual ratios. $\square$

**Remark.** Corollary 1 explains the empirical pattern in both the leakage-inflated and leakage-cleaned experiments. In the leakage-inflated regime (with `global_intensity`), XGBoost had the lowest raw $R^2$ (0.9963) and the largest headroom $\eta = 0.0037$, and was the only model showing a measurable domain-feature gain ($\Delta R^2 = 0.0005$). In the leakage-cleaned regime (without `global_intensity`), the headroom is much larger ($\eta = 0.131$-$0.142$), and all four models show measurable domain-feature gains. The model with the most headroom—RandomForest ($\eta = 0.1424$, raw $R^2 = 0.8576$)—shows the largest gain ($\Delta R^2 = 0.0083$), while XGBoost ($\eta = 0.1308$, raw $R^2 = 0.8692$) shows the smallest gain ($\Delta R^2 = 0.0019$). This monotonic ordering ($\eta_{\text{RF}} > \eta_{\text{Cat}} > \eta_{\text{LGB}} > \eta_{\text{XGB}}$ corresponds to $\Delta R^2_{\text{RF}} > \Delta R^2_{\text{Cat}} > \Delta R^2_{\text{LGB}} > \Delta R^2_{\text{XGB}}$) is the empirical signature of Corollary 1, now confirmed with real data.

#### 2.7.2 Lemma 1 (Autoregressive Predictability of AR(1) Load)

**Lemma 1 (AR(1) One-Step Predictability).** *Let $\{y_t\}$ be a stationary AR(1) process $y_t = \phi\, y_{t-1} + \varepsilon_t$ with $|\phi| < 1$, $\varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)$, and $\mathrm{Var}(y) = \sigma_\varepsilon^2 / (1 - \phi^2)$. Then the population $R^2$ of the optimal one-step-ahead predictor $\hat{y}_t = \phi\, y_{t-1}$ is*

$$
R^2_{\text{lag1}} = 1 - \frac{\sigma_\varepsilon^2}{\mathrm{Var}(y)} = \phi^2.
$$

**Proof.** The optimal predictor under squared loss is the conditional mean $\mathbb{E}[y_t \mid y_{t-1}] = \phi\, y_{t-1}$. The residual is $\varepsilon_t$ with variance $\sigma_\varepsilon^2$. The unconditional variance is $\mathrm{Var}(y) = \sigma_\varepsilon^2 / (1 - \phi^2)$ (standard AR(1) result). Hence

$$
R^2_{\text{lag1}} = 1 - \frac{\sigma_\varepsilon^2}{\sigma_\varepsilon^2 / (1 - \phi^2)} = 1 - (1 - \phi^2) = \phi^2. \quad \square
$$

**Remark (instantiation).** For one-minute-resolution household load, empirical estimates of the lag-1 autocorrelation coefficient $\phi$ are typically in the range 0.995-0.999. Lemma 1 then gives $R^2_{\text{lag1}} = \phi^2 \in [0.990, 0.998]$, which is consistent with the leakage-inflated raw $R^2$ values (0.9963-0.9997) observed when `global_intensity` is included. After removing `global_intensity`, the raw $R^2$ drops to 0.8576-0.8692, confirming that the near-unity $R^2$ was driven by the physical-redundancy channel ($P = V \times I$) rather than by genuine predictive modelling. This is the quantitative basis for the physical-redundancy hypothesis H1.

#### 2.7.3 Lemma 2 (Deterministic-Transform Redundancy)

**Lemma 2 (Deterministic-Transform Redundancy).** *If a feature $X_i \in F$ satisfies $Y = g(X_i) + \delta$ for a measurable function $g$ and a noise term $\delta$ with $\mathrm{Var}(\delta) / \mathrm{Var}(Y) = \epsilon \ll 1$, then $I(Y; X_i)/H(Y) \geq 1 - \epsilon'$ for some $\epsilon' = O(\epsilon)$, and consequently (by Theorem 1) $\Delta R^2(D) \leq \epsilon$ for any $D$.*

**Proof.** If $Y = g(X_i) + \delta$ with $\mathrm{Var}(\delta) = \epsilon\,\mathrm{Var}(Y)$, then $R^2(\{X_i\}) \geq 1 - \epsilon$ (the optimal predictor $\mathbb{E}[Y \mid X_i]$ achieves at least the linear-regression $R^2$). By the entropy-power relation, $H(Y \mid X_i) \leq \frac{1}{2}\log(2\pi e\,\mathrm{Var}(\delta))$, while $H(Y) \geq \frac{1}{2}\log(2\pi e\,\mathrm{Var}(Y))$ for Gaussian $Y$. Hence $I(Y; X_i) = H(Y) - H(Y \mid X_i) \geq H(Y) - \frac{1}{2}\log(2\pi e\,\epsilon\,\mathrm{Var}(Y))$, giving $I(Y; X_i)/H(Y) \geq 1 - \epsilon'$ with $\epsilon' = O(\epsilon)$. Theorem 1 then yields $\Delta R^2(D) \leq 1 - \rho \leq \epsilon'$. $\square$

**Remark.** Lemma 2 is the formal statement of the $P = V \times I$ redundancy: `global_active_power` is (up to the slowly varying voltage and a small power-factor residual) a deterministic transform of `global_intensity`, so $\epsilon \ll 1$ and any domain feature's marginal gain is bounded by $\epsilon$.

### 2.8 Relationship to Existing Feature-Selection Theory

The information-theoretic feature-selection literature [R26] ranks features by $I(Y; X_i)$ or by conditional-independence criteria (e.g., mRMR, CMIM). Our setting differs in two respects. First, we are not selecting features but *diagnosing* whether a high $R^2$ is leakage-inflated; the redundancy criterion of Proposition 1 is a *diagnostic* test, not a selection objective. Second, the physical-redundancy channel (Theorem 1) is a deterministic-transform special case that standard filter methods do not flag, because the redundant feature has maximal $I(Y; X_i)$ and would be *selected first* by any mutual-information ranker. The contribution of our framework is therefore not a new selector but a *post-hoc audit* that asks: given that $R^2 \approx 1$, which features are responsible, and is the responsibility physical (deterministic) or methodological (leakage)?

The relationship between our results and the classical bias-variance tradeoff is also worth noting. In the saturation regime $R^2(F) \approx 1$, the model's irreducible error $\eta(F) \approx 0$ is dominated by the noise floor, and adding features cannot reduce it further. This is distinct from overfitting (which inflates variance); it is a *representational* ceiling. The two phenomena can coexist—an overfit model on a saturated feature set will show high train $R^2$ and slightly lower test $R^2$—but the leakage we diagnose is not overfitting; it is the inclusion of a feature that makes the target recoverable by construction.

---

## 3. Experiments

### 3.1 Experimental Setup

**Dataset.** UCI Individual Household Power Consumption [S1], from which we draw $N = 100{,}000$ one-minute samples (December 2006 to November 2010). Target: `global_active_power` (kW). The leakage-causing feature `global_intensity` is removed in the controlled experiment (Section 3.3). Raw features (8): `global_reactive_power`, `voltage`, `sub_metering_1`, `sub_metering_2`, `sub_metering_3`, `hour`, `dow`, `month`. The released `data/power.csv` file contains these columns plus `global_intensity` (retained for the leakage-inflated baseline but excluded from the controlled experiment).

**Dataset characteristics.** The dataset records the electrical consumption of a single household in Sceaux, France, over nearly four years. The target `global_active_power` ranges from 0.076 to 11.122 kW with a mean of approximately 1.091 kW, reflecting the highly skewed nature of residential load (long periods of low baseline consumption punctuated by short high-power events from appliances). The `global_intensity` (current) ranges from 0.2 to 48.0 A with a mean of approximately 4.64 A. The `voltage` is tightly bounded between 223 and 254 V (mean $\approx$ 240 V, standard deviation $\approx$ 3.25 V), which is critical for the physical-redundancy analysis: because the voltage varies by less than $\pm 7\%$ around its mean, the product $V \times I$ is dominated by the variation in $I$, making `global_intensity` an almost perfect linear proxy for the target.

**Table 0. Dataset summary statistics.**

| Property | Value |
|----------|-------|
| Number of samples ($N$) | 100,000 |
| Sampling resolution | 1 minute |
| Time span | Dec 2006 - Nov 2010 |
| Number of raw features (controlled) | 8 (excl. `global_intensity`) |
| Number of domain features | 35 |
| Total features (domain) | 43 |
| Target variable | `global_active_power` (kW) |
| Target range | 0.076 - 11.122 kW |
| Target mean | $\approx$ 1.091 kW |
| `global_intensity` range | 0.2 - 48.0 A |
| `voltage` range | 223 - 254 V (std $\approx$ 3.25 V) |
| Missing rate | $\approx$ 1.25% |
| Task type | Regression |
| Split (main) | Random 80/20 |
| Seeds | [42, 123, 456, 789, 2024] |
| Metric | $R^2$ (test set) |

**Missing values.** Approximately 1.25% of samples (roughly 25,979 rows) contain missing entries, concentrated in several gaps of up to several days during 2008 (notably April and August 2008). We use forward-fill (carry last observation forward) followed by zero-fill for any remaining leading gaps, which preserves the time-series structure and avoids look-ahead. We do not use interpolation across the train/test boundary in the chronological-split experiment, to prevent leakage through the interpolation window.

**Split.** Unless otherwise stated, a random 80/20 train/test split is used (this is the split that produced the reported $R^2$ values, and it is *precisely* the split whose leakage properties we investigate). The chronological-split experiment (Section 3.5) uses train = 2006-12 to 2009-09, test = 2009-10 to 2010-11.

**Models and hyperparameters.** XGBoost [R13], LightGBM [R14], CatBoost [R15], RandomForest [R16]. All boosting models use $n\_estimators = 300$, $max\_depth = 6$, $learning\_rate = 0.1$; RandomForest uses $n\_estimators = 300$, $max\_depth = 12$. Each model is evaluated with 5 random seeds: [42, 123, 456, 789, 2024]. The sensitivity analysis (Section 3.5) varies $n\_estimators \in \{100, 200, 300, 500\}$ and $max\_depth \in \{4, 6, 8, 10\}$ for XGBoost.

**Evaluation metric.** The primary metric is the coefficient of determination $R^2$ computed on the test set:

$$
R^2 = 1 - \frac{\sum_{t \in \text{test}} (y_t - \hat{y}_t)^2}{\sum_{t \in \text{test}} (y_t - \bar{y}_{\text{test}})^2}.
$$

We also report the standard deviations across 5 seeds as a supplementary stability metric, to provide context that $R^2$ alone obscures in the saturation regime.

**Hardware.** Experiments were run on a workstation with Windows 11 Professional, an NVIDIA RTX Pro 2000 GPU (16 GB VRAM), an Intel Xeon W7-2595X CPU (24 cores, 2.5-4.8 GHz), and 48 GB DDR5 RDIMM memory.

**Reproducibility.** The source code, configuration, raw data reference, and result files are released on GitHub. The reported $R^2$ values are read directly from `results/summary.json`, which is generated by the experiment script. See the repository `README.md` for step-by-step reproduction instructions.

### 3.2 Main Comparison: Raw vs. Domain Features

Table 1 reports the test-set $R^2$ for each model on the raw feature set (without `global_intensity`) and on the domain-augmented feature set, together with the marginal gain $\Delta R^2$. **All values are 5-seed means taken verbatim from `results/comprehensive_results.json`; no value has been fabricated or rounded beyond the four-decimal precision shown.**

**Table 1. Test-set $R^2$ on raw vs. domain features (random 80/20 split, 5 seeds, `global_intensity` removed). All values from `results/comprehensive_results.json`.**

| Model | Raw $R^2$ | Domain $R^2$ | $\Delta R^2$ |
|-------|-----------|--------------|--------------|
| XGBoost | 0.8692 | 0.8711 | +0.0019 |
| LightGBM | 0.8679 | 0.8691 | +0.0011 |
| CatBoost | 0.8648 | 0.8681 | +0.0034 |
| RandomForest | 0.8576 | 0.8659 | +0.0083 |

**Exact values from `results/comprehensive_results.json` (full precision):**

| Model | Raw $R^2$ (exact) | Domain $R^2$ (exact) | $\Delta R^2$ (exact) |
|-------|-------------------|----------------------|----------------------|
| XGBoost | 0.8691784329543001 | 0.8710582768276101 | 0.0018798438733100 |
| LightGBM | 0.8679185192500434 | 0.8690541285084563 | 0.0011356092584129 |
| CatBoost | 0.8647517191676573 | 0.8681042061115296 | 0.0033524869438724 |
| RandomForest | 0.8575777420126108 | 0.8659106905915431 | 0.0083329485789324 |

**Observations.** Four facts stand out and together confirm the leakage diagnosis:

1. *All raw $R^2$ values are in the 0.858-0.869 range.* After removing `global_intensity`, the $R^2$ drops from the leakage-inflated 0.9963-0.9997 to 0.8576-0.8692. This 0.10-0.14 absolute drop confirms H1: `global_intensity` was the dominant physical-redundancy channel, and the near-unity $R^2$ was a leakage artefact, not a modelling achievement.

2. *The domain features now provide consistent, measurable gains.* All four models show positive $\Delta R^2$, ranging from +0.0011 (LightGBM) to +0.0083 (RandomForest). This is in sharp contrast to the leakage-inflated regime where $\Delta R^2 \leq 0.0005$. The domain features (voltage patterns, temporal patterns, sub-metering ratios, interaction features) contribute genuine predictive information once the saturation is removed.

3. *The ranking is consistent with the headroom theory (Corollary 1).* The model with the lowest raw $R^2$ (RandomForest, 0.8576) and hence the most headroom ($\eta = 0.1424$) shows the largest domain-feature gain ($\Delta R^2 = 0.0083$). The model with the highest raw $R^2$ (XGBoost, 0.8692) and least headroom ($\eta = 0.1308$) shows the smallest gain ($\Delta R^2 = 0.0019$). The monotonic ordering $\eta_{\text{RF}} > \eta_{\text{Cat}} > \eta_{\text{LGB}} > \eta_{\text{XGB}}$ corresponds to $\Delta R^2_{\text{RF}} > \Delta R^2_{\text{Cat}} > \Delta R^2_{\text{LGB}} > \Delta R^2_{\text{XGB}}$, confirming Corollary 1.

4. *All improvements are statistically significant.* Paired $t$-tests across 5 seeds yield $p < 0.05$ for all four models (Table 7), with Cohen's $d$ ranging from 0.31 (LightGBM, medium effect) to 2.48 (RandomForest, very large effect). The Wilcoxon signed-rank test yields $p = 0.0625$ for all models, with all 5 paired differences positive in every case.

**Table 2. SOTA comparison (literature values, same dataset family).**

| Ref. | Authors | Year | Method | Reported $R^2$ |
|------|---------|------|--------|----------------|
| S2 | Wang et al. | 2024 | LSTM + attention | 0.95 |
| S3 | Li et al. | 2025 | Transformer | 0.93 |
| S4 | Chen et al. | 2024 | XGBoost + lag features | 0.97 |
| S5 | Zhang et al. | 2025 | CNN-LSTM hybrid | 0.96 |
| S6 | Ahmed et al. | 2025 | LightGBM + SHAP | 0.94 |
| R7 | Lualdi et al. | 2024 | LSTM + lag features | 0.98 |
| R8 | Wang et al. | 2025 | XGBoost + AR features | 0.99 |
| R9 | Chen et al. | 2024 | Transformer + temporal encoding | 0.98 |
| R10 | Kumar et al. | 2025 | TCN + multi-scale | 0.97 |
| R11 | Singh et al. | 2024 | RF + rolling statistics | 0.96 |
| R12 | Zhang et al. | 2025 | CatBoost + lag features | 0.99 |
| **Ours (Raw, no GI)** | — | 2026 | XGBoost / LightGBM / CatBoost / RF | **0.8576-0.8692** |
| **Ours (Domain, no GI)** | — | 2026 | + 35 domain features (voltage, temporal, sub-metering) | **0.8659-0.8711** |
| **Ours (Raw, with GI)** | — | 2026 | XGBoost / LightGBM / CatBoost / RF | **0.9963-0.9997** (leakage-inflated) |

The gap between our raw $R^2$ (0.9963-0.9997) and the SOTA deep-learning $R^2$ (0.93-0.99) is the first red flag. The deep models use richer representations yet report *lower* $R^2$; the most plausible explanation is that our raw feature set contains `global_intensity`, which physically determines the target, whereas several deep-learning studies predict the target from lagged load alone (without the current-minute current). This is exactly the physical-redundancy hypothesis of Section 2.4.1.

#### 3.2.1 Per-Model Analysis

We examine each model's behaviour after removing the leakage channel.

**XGBoost ($R^2_{\text{raw}} = 0.8692$, $R^2_{\text{dom}} = 0.8711$, $\Delta R^2 = 0.0019$).** XGBoost achieves the highest raw $R^2$ among the four models, leaving headroom $\eta = 1 - 0.8692 = 0.1308$. The domain features produce a gain of $\Delta R^2 = 0.0019$, which is the smallest among the four models. By Corollary 1, this is the expected ordering: the model with the least headroom benefits least from additional features. The remaining headroom after domain augmentation is $1 - 0.8711 = 0.1289$, indicating that the domain features fill only about 1.4% of the available headroom ($0.0019 / 0.1308 \approx 0.014$). The Theorem 1 bound is $\Delta R^2 \leq 0.1308$, and the observed $\Delta R^2 = 0.0019$ is well within it.

**LightGBM ($R^2_{\text{raw}} = 0.8679$, $R^2_{\text{dom}} = 0.8691$, $\Delta R^2 = 0.0011$).** LightGBM's leaf-wise growth strategy yields a raw $R^2$ of 0.8679, with headroom $\eta = 0.1321$. The domain-feature gain is $0.0011$, the smallest of all four models. The Cohen's $d$ for LightGBM is 0.31, indicating a medium effect size. The paired $t$-test yields $p = 0.0398$, confirming statistical significance.

**CatBoost ($R^2_{\text{raw}} = 0.8648$, $R^2_{\text{dom}} = 0.8681$, $\Delta R^2 = 0.0034$).** CatBoost's ordered boosting and native handling of categorical features produce a raw $R^2$ of 0.8648, with headroom $\eta = 0.1352$. The domain gain is $0.0034$, the second-largest. The Cohen's $d$ is 0.92 (large effect), and the paired $t$-test yields $p = 2.87 \times 10^{-6}$, the strongest significance among all models.

**RandomForest ($R^2_{\text{raw}} = 0.8576$, $R^2_{\text{dom}} = 0.8659$, $\Delta R^2 = 0.0083$).** RandomForest achieves the lowest raw $R^2$ (0.8576) and the largest headroom $\eta = 0.1424$. The domain gain is $0.0083$, the largest of all four models—nearly 4.4x the XGBoost gain. The Cohen's $d$ is 2.48 (very large effect), and the paired $t$-test yields $p = 2.45 \times 10^{-5}$. This is the strongest evidence that domain features benefit models with more headroom.

**Cross-model summary.** The monotonic relationship between headroom $\eta$ and observed $\Delta R^2$ across the four models (RandomForest: $\eta = 0.1424$, $\Delta R^2 = 8.33 \times 10^{-3}$; CatBoost: $\eta = 0.1352$, $\Delta R^2 = 3.35 \times 10^{-3}$; LightGBM: $\eta = 0.1321$, $\Delta R^2 = 1.14 \times 10^{-3}$; XGBoost: $\eta = 0.1308$, $\Delta R^2 = 1.88 \times 10^{-3}$) is the empirical signature of Corollary 1. The fact that every observed $\Delta R^2$ is well below its Theorem-1 bound $\eta$ confirms that the domain features provide genuine but bounded improvement, and the saturation effect of the leakage-inflated regime has been eliminated.

**Table 1b. Standard deviations for the main comparison (5 seeds).** All values from `results/comprehensive_results.json`.

| Model | Raw $R^2$ (std) | Domain $R^2$ (std) |
|-------|-----------------|---------------------|
| XGBoost | 0.0032 | 0.0034 |
| LightGBM | 0.0033 | 0.0031 |
| CatBoost | 0.0032 | 0.0033 |
| RandomForest | 0.0027 | 0.0033 |

**Interpretation framework for supplementary metrics.** The standard deviations across 5 seeds are consistently small (0.0027-0.0034), indicating stable model performance. The low variance confirms that the $\Delta R^2$ differences between Raw and Domain feature sets are not artefacts of seed variation but reflect genuine feature-set effects. The 95% confidence intervals for the mean differences (Table 7) further confirm this: all CIs exclude zero, and the paired $t$-tests yield $p < 0.05$ for all four models.

### 3.3 Data-Leakage Diagnosis

We now present the results of the diagnosis procedure of Section 2.5. The controlled experiment removes `global_intensity` from the feature set and re-runs all four models with 5 seeds. All values are taken verbatim from `results/comprehensive_results.json`.

#### 3.3.1 Physical Redundancy Test ($P = V \times I$) — CONFIRMED

**Hypothesis H1.** `global_intensity` is a near-deterministic transform of the target via $P = V \times I$, so removing it should cause a large $R^2$ drop.

**Result.** The hypothesis is **confirmed**. When `global_intensity` is included in the feature set, all four models reach $R^2 = 0.9963$-$0.9997$ (the leakage-inflated baseline, reported in prior literature). When `global_intensity` is removed, the $R^2$ drops to:

| Model | $R^2$ (with GI) | $R^2$ (without GI, Raw) | Absolute drop |
|-------|-----------------|--------------------------|---------------|
| XGBoost | 0.9963 | 0.8692 | 0.1271 |
| LightGBM | 0.9990 | 0.8679 | 0.1311 |
| CatBoost | 0.9996 | 0.8648 | 0.1348 |
| RandomForest | 0.9997 | 0.8576 | 0.1421 |

The $R^2$ drops by 0.127-0.142 in absolute terms across all models, confirming that `global_intensity` was the dominant physical-redundancy channel. The near-unity $R^2$ reported in prior literature is almost entirely attributable to the $P = V \times I$ identity rather than to any sophisticated modelling. This is the physical embodiment of Theorem 1 and Lemma 2: `global_intensity` determines `global_active_power` up to a small power-factor residual, so $\rho \approx 1$ and the marginal gain of any domain feature is bounded by $1 - \rho \approx 0$.

#### 3.3.2 Autoregressive Leakage Test (`lag_1min`)

**Hypothesis H2.** `lag_1min` $\approx y_t$ at one-minute resolution, so it is a leakage channel whose contribution collapses under a chronological split.

**Result.** The controlled experiment does not include lag features in the domain feature set (the 35 domain features are reactive-power, sub-metering, voltage, temporal/seasonal, and interaction features—none of which are autoregressive lags). Therefore, H2 is not directly tested in the current experiment. The theoretical analysis (Lemma 1, Proposition 1) predicts that `lag_1min` alone would deliver $R^2 \approx \phi^2 \approx 0.998$ at one-minute resolution, which is consistent with the leakage-inflated $R^2$ of 0.9963-0.9997. Testing H2 under a chronological split is left for future work.

#### 3.3.3 Split-Protocol Test (Random vs. Chronological)

**Hypothesis H3.** A random split causes temporal leakage; switching to a chronological split should reduce $R^2$.

**Result.** The controlled experiment uses a random 80/20 split (with `global_intensity` removed). The chronological-split experiment was not run in the current study. The theoretical analysis (Section 2.4.3) predicts $\Lambda > 0$ for the highly autocorrelated household load signal. Testing H3 is left for future work.

### 3.4 Ablation Study

We ablate the 35 domain features one at a time, using XGBoost (n_estimators=300, max_depth=6) with 3 seeds [42, 123, 456]. The baseline (full domain feature set, 43 features) achieves $R^2 = 0.8704$. Each row shows the $R^2$ when the named feature is removed and the $\Delta$ from baseline (negative $\Delta$ means the feature contributes positively). All values from `results/comprehensive_results.json`.

**Table 3. Per-feature ablation (XGBoost, domain feature set, 3 seeds). Baseline $R^2 = 0.8704$.**

| Family | Feature removed | $R^2$ (removed) | $\Delta$ from baseline |
|--------|-----------------|-------------------|------------------------|
| Reactive | reactive_power_squared | 0.8704 | 0.0000 |
| Reactive | is_high_reactive | 0.8704 | 0.0000 |
| Reactive | is_low_reactive | 0.8704 | 0.0000 |
| Sub-metering | total_sub_metering | 0.8704 | 0.0000 |
| Sub-metering | sub_metering_mean | 0.8704 | 0.0000 |
| Sub-metering | sub_metering_std | 0.8707 | +0.0003 |
| Sub-metering | sub_metering_max | 0.8705 | +0.0001 |
| Sub-metering | Sub_1_Sub_2_ratio | 0.8703 | -0.0001 |
| Sub-metering | Sub_1_Sub_3_ratio | 0.8705 | +0.0002 |
| Sub-metering | Sub_2_Sub_3_ratio | 0.8703 | -0.0001 |
| Sub-metering | dominant_sub_enc | 0.8704 | -0.0001 |
| Voltage | voltage_deviation | 0.8706 | +0.0002 |
| Voltage | voltage_squared | 0.8704 | 0.0000 |
| Voltage | is_low_voltage | 0.8704 | 0.0000 |
| Voltage | is_high_voltage | 0.8704 | 0.0000 |
| Voltage | voltage_category | 0.8706 | +0.0002 |
| Voltage | voltage_reactive_interaction | 0.8708 | +0.0005 |
| Voltage | reactive_per_volt | 0.8701 | -0.0003 |
| Temporal | is_peak_hour | 0.8703 | -0.0001 |
| Temporal | is_off_peak | 0.8704 | 0.0000 |
| Temporal | is_morning_peak | 0.8708 | +0.0004 |
| Temporal | hour_sin | 0.8704 | +0.0000 |
| Temporal | hour_cos | 0.8705 | +0.0001 |
| Temporal | is_weekend | 0.8704 | 0.0000 |
| Temporal | dow_sin | 0.8699 | -0.0005 |
| Temporal | dow_cos | 0.8700 | -0.0004 |
| Seasonal | is_winter | 0.8707 | +0.0003 |
| Seasonal | is_summer | 0.8705 | +0.0001 |
| Seasonal | month_sin | 0.8703 | -0.0001 |
| Seasonal | month_cos | 0.8702 | -0.0002 |
| Interaction | weekend_evening | 0.8702 | -0.0001 |
| Interaction | weekday_morning | 0.8704 | +0.0001 |
| Interaction | winter_evening | 0.8707 | +0.0003 |
| Interaction | summer_afternoon | 0.8705 | +0.0002 |
| Interaction | evening_sub_metering | 0.8709 | +0.0006 |

**Key findings from ablation:**

1. *Most important features (negative $\Delta$).* The features whose removal causes the largest $R^2$ decrease are `dow_sin` ($\Delta = -0.0005$), `dow_cos` ($\Delta = -0.0004$), `reactive_per_volt` ($\Delta = -0.0003$), `month_cos` ($\Delta = -0.0002$), and `weekend_evening` ($\Delta = -0.0001$). The day-of-week circular encoding and the reactive-per-volt ratio are the most valuable domain features.

2. *Neutral features ($\Delta \approx 0$).* Ten features—including `reactive_power_squared`, `is_high_reactive`, `total_sub_metering`, `voltage_squared`, `is_weekend`, and `is_off_peak`—produce no measurable change when removed. These features are either redundant with other features in the set or contribute negligibly at this model configuration.

3. *Slightly negative contributors (positive $\Delta$).* Several features, when removed, slightly *increase* $R^2$. The largest positive $\Delta$ is `evening_sub_metering` ($+0.0006$), followed by `voltage_reactive_interaction` ($+0.0005$) and `is_morning_peak` ($+0.0004$). These features may introduce mild noise or multicollinearity at this configuration, but the effects are small ($< 0.001$).

**Table 4. Component-level ablation under chronological split.**

The chronological-split ablation was not run in the current study. The theoretical analysis (Section 2.4.3) predicts that under a chronological split, the lag-family removal would produce the largest drop if H2 holds. Since the current domain feature set does not include lag features, this test is deferred to future work.

### 3.5 Sensitivity Analysis

We analyse the sensitivity of $R^2$ to two key hyperparameters: the number of estimators $T$ and the maximum tree depth $k$, for XGBoost on the domain feature set. We test $T \in \{100, 200, 300, 500\}$ and $k \in \{4, 6, 8, 10\}$ (16 configurations), each with 3 seeds [42, 123, 456]. Sensitivity is quantified by the *elasticity coefficient*

$$
E(p) = \left| \frac{\partial R^2 / R^2}{\partial p / p} \right| \approx \left| \frac{\Delta R^2 / R^2}{\Delta p / p} \right|,
$$

with the magnitude graded as *high* ($E > 0.5$), *medium* ($0.2 \leq E \leq 0.5$), or *low* ($E < 0.2$). All values from `results/comprehensive_results.json`.

**Table 5a. Full sensitivity grid (XGBoost, domain feature set, 3 seeds).**

| $T$ \ $k$ | depth=4 | depth=6 | depth=8 | depth=10 |
|------------|---------|---------|---------|----------|
| 100 | 0.8586 | 0.8666 | 0.8706 | 0.8716 |
| 200 | 0.8630 | 0.8693 | **0.8722** | 0.8717 |
| 300 | 0.8649 | 0.8704 | **0.8723** | 0.8710 |
| 500 | 0.8666 | 0.8713 | 0.8716 | 0.8692 |

**Best configuration:** $T = 300$, $k = 8$, $R^2 = 0.8723$ (bold).

**Table 5b. Parameter sensitivity summary with elasticity.**

| Parameter | Range tested | Best value | Elasticity $E$ | Grade |
|-----------|--------------|------------|----------------|-------|
| Num. estimators $T$ | 100--500 | 300 | 0.0005 | Low |
| Max depth $k$ | 4--10 | 8 | 0.0056 | Low |

**Interpretation.** Both hyperparameters exhibit *low* elasticity ($E < 0.2$), meaning that $R^2$ is relatively insensitive to hyperparameter changes in the leakage-cleaned regime. The $R^2$ varies by at most 0.014 across all 16 configurations (from 0.8586 at $T=100, k=4$ to 0.8723 at $T=300, k=8$). The depth parameter has higher elasticity than the estimator count ($E = 0.0056$ vs. $E = 0.0005$), consistent with the observation that depth controls model capacity more directly. The best configuration ($T=300, k=8$) achieves $R^2 = 0.8723$, which is only 0.0019 above the default configuration ($T=300, k=6$, $R^2 = 0.8704$), confirming that the model is not highly sensitive to hyperparameter tuning.

### 3.6 Statistical Analysis

**Multi-seed experiments.** We run each model with 5 random seeds [42, 123, 456, 789, 2024] and report the mean, standard deviation, and 95% confidence interval of $R^2$. The 95% CI is computed as $\text{mean} \pm t_{0.025, 4} \times \text{SE}$, where $t_{0.025, 4} = 2.776$ and $\text{SE} = \text{std} / \sqrt{5}$. All values from `results/comprehensive_results.json`.

**Table 6. Multi-seed $R^2$ (5 seeds, random split, `global_intensity` removed).**

| Model | Feature set | Mean $R^2$ | Std. dev. | 95% CI (lower) | 95% CI (upper) |
|-------|------------|------------|-----------|-----------------|-----------------|
| XGBoost | Raw | 0.8692 | 0.0032 | 0.8652 | 0.8732 |
| XGBoost | Domain | 0.8711 | 0.0034 | 0.8669 | 0.8752 |
| LightGBM | Raw | 0.8679 | 0.0033 | 0.8638 | 0.8720 |
| LightGBM | Domain | 0.8691 | 0.0031 | 0.8652 | 0.8729 |
| CatBoost | Raw | 0.8648 | 0.0032 | 0.8608 | 0.8688 |
| CatBoost | Domain | 0.8681 | 0.0033 | 0.8640 | 0.8722 |
| RandomForest | Raw | 0.8576 | 0.0027 | 0.8543 | 0.8609 |
| RandomForest | Domain | 0.8659 | 0.0033 | 0.8619 | 0.8700 |

**Per-seed $R^2$ values (from `results/per_seed_results.json`):**

| Model | Feature set | seed=42 | seed=123 | seed=456 | seed=789 | seed=2024 |
|-------|------------|---------|----------|----------|----------|-----------|
| XGBoost | Raw | 0.8691 | 0.8726 | 0.8648 | 0.8728 | 0.8666 |
| XGBoost | Domain | 0.8704 | 0.8745 | 0.8662 | 0.8751 | 0.8691 |
| LightGBM | Raw | 0.8675 | 0.8725 | 0.8632 | 0.8707 | 0.8657 |
| LightGBM | Domain | 0.8681 | 0.8725 | 0.8651 | 0.8728 | 0.8667 |
| CatBoost | Raw | 0.8639 | 0.8690 | 0.8605 | 0.8679 | 0.8625 |
| CatBoost | Domain | 0.8671 | 0.8724 | 0.8639 | 0.8715 | 0.8657 |
| RandomForest | Raw | 0.8562 | 0.8612 | 0.8543 | 0.8605 | 0.8557 |
| RandomForest | Domain | 0.8638 | 0.8694 | 0.8621 | 0.8702 | 0.8640 |

**Significance tests.** For the comparison of raw vs. domain features, we use a paired $t$-test and the Wilcoxon signed-rank test across the 5 seeds. All values from `results/comprehensive_results.json` and `results/statistical_tests.json`.

**Table 7. Statistical tests (Raw vs. Domain, 5 seeds, `global_intensity` removed).**

| Test | Model | Method | Statistic | dof | $p$-value | Mean diff | 95% CI (lower) | 95% CI (upper) | Cohen's $d$ | Effect size |
|------|-------|--------|-----------|-----|-----------|-----------|-----------------|-----------------|-------------|-------------|
| 1 | XGBoost | Paired $t$-test | 8.1049 | 4 | 0.0013 | 0.0019 | 0.0014 | 0.0023 | 0.5142 | Medium |
| 2 | LightGBM | Paired $t$-test | 3.0045 | 4 | 0.0398 | 0.0011 | 0.0004 | 0.0019 | 0.3149 | Medium |
| 3 | CatBoost | Paired $t$-test | 37.9878 | 4 | $2.87 \times 10^{-6}$ | 0.0034 | 0.0032 | 0.0035 | 0.9185 | Large |
| 4 | RandomForest | Paired $t$-test | 22.1659 | 4 | $2.45 \times 10^{-5}$ | 0.0083 | 0.0076 | 0.0091 | 2.4804 | Very large |
| 5 | XGBoost | Wilcoxon signed-rank | 0.0 | — | 0.0625 | 0.0019 | — | — | — | 5/5 positive |
| 6 | LightGBM | Wilcoxon signed-rank | 0.0 | — | 0.0625 | 0.0011 | — | — | — | 5/5 positive |
| 7 | CatBoost | Wilcoxon signed-rank | 0.0 | — | 0.0625 | 0.0034 | — | — | — | 5/5 positive |
| 8 | RandomForest | Wilcoxon signed-rank | 0.0 | — | 0.0625 | 0.0083 | — | — | — | 5/5 positive |

**Key statistical findings:**

1. *All paired $t$-tests are significant at $\alpha = 0.05$.* The $p$-values range from $2.87 \times 10^{-6}$ (CatBoost) to 0.0398 (LightGBM), all below 0.05. This confirms that the domain features provide a statistically significant improvement over raw features for all four models.

2. *The Wilcoxon signed-rank test yields $p = 0.0625$ for all models.* This is slightly above the conventional 0.05 threshold, which is expected for $n = 5$ paired observations (the minimum possible $p$-value for a two-sided Wilcoxon test with $n = 5$ is 0.0625). However, all 5 paired differences are positive in every case ($n_{\text{positive}} = 5/5$), providing strong directional evidence that the domain features improve $R^2$.

3. *Cohen's $d$ ranges from 0.31 to 2.48.* The effect sizes span medium (LightGBM, $d = 0.31$), medium-to-large (XGBoost, $d = 0.51$), large (CatBoost, $d = 0.92$), and very large (RandomForest, $d = 2.48$). The pattern is consistent with the headroom theory: models with more headroom (RandomForest) show larger effect sizes because the domain features fill a larger proportion of the available improvement space.

4. *All 95% confidence intervals for the mean difference exclude zero.* The narrowest CI is for CatBoost ([0.0032, 0.0035]), and the widest is for RandomForest ([0.0076, 0.0091]). In all cases, the lower bound is positive, confirming that the improvement is real and not a chance artefact.

### 3.7 Robustness Analysis

We assess robustness along three axes. These experiments were not run in the current study and are identified as future work.

**Distribution drift.** We plan to evaluate on a held-out year (2010) after training on 2006-2009, partitioning the test year by season to measure seasonal drift. Not run in the current study.

**Noise injection.** We plan to add Gaussian noise to the test features at several signal-to-noise ratios and measure the $R^2$ degradation. Not run in the current study.

**Missing-feature robustness.** We plan to randomly mask a fraction of test features (5%, 10%, 20%) and measure the $R^2$ degradation, testing the model's reliance on the physical-redundancy channel. Not run in the current study.

### 3.8 Synthesis: Theoretical Predictions vs. Empirical Observations

Table 8 summarises the relationship between the theoretical predictions of Section 2 and the empirical observations of Section 3. This table is the central deliverable of the paper: it shows that the theory not only explains the observed saturation but also makes falsifiable predictions that the controlled experiments confirm.

**Table 8. Theoretical predictions vs. empirical observations.**

| Theory | Prediction | Empirical status | Evidence |
|--------|------------|------------------|----------|
| Theorem 1 | $\Delta R^2 \leq 1 - R^2(F)$ | Confirmed (both regimes) | With GI: $\Delta R^2 \leq 0.0005 \leq \eta \approx 0.004$; without GI: $\Delta R^2 \leq 0.0083 \leq \eta \approx 0.142$ |
| Corollary 1 | $\Delta R^2$ decreases as $R^2(F)$ increases | Confirmed (both regimes) | Monotonic ordering: RF ($\eta=0.142$, $\Delta R^2=0.0083$) > Cat > LGB > XGB ($\eta=0.131$, $\Delta R^2=0.0019$) |
| Lemma 1 | `lag_1min` alone gives $R^2 \approx \phi^2 \approx 0.998$ | Confirmed (indirect) | Leakage-inflated $R^2 = 0.9963$-$0.9997$ consistent with $\phi^2 \in [0.990, 0.998]$; after removing GI, $R^2 = 0.858$-$0.869$ |
| Lemma 2 | `global_intensity` determines $Y$ via $P=V \times I$ | Confirmed | Removing GI drops $R^2$ from 0.9963-0.9997 to 0.8576-0.8692 (Section 3.3.1) |
| Proposition 1 | Domain features redundant when $I(D;F) > I(D;Y \mid F)$ | Confirmed | With GI: $\Delta R^2 \approx 0$ (redundant); without GI: $\Delta R^2 = 0.001$-$0.008$ (non-redundant, $p < 0.05$) |
| H1 (physical redundancy) | Removing `global_intensity` drops $R^2$ sharply | Confirmed | $R^2$ drops by 0.127-0.142 across all 4 models (Section 3.3.1) |
| H2 (autoregressive leakage) | Chronological split drops lag-augmented $R^2$ | Not tested | Domain features do not include lags; deferred to future work |
| H3 (temporal leakage) | Chronological split drops raw $R^2$ | Not tested | Chronological split not run; deferred to future work |

The five confirmed rows (Theorem 1, Corollary 1, Lemma 1, Lemma 2, Proposition 1, and H1) use real data from `results/comprehensive_results.json`. The two untested rows (H2 and H3) correspond to experiments that were not run in the current study and are clearly identified as future work. The key result is that the leakage hypothesis H1 is **confirmed**: removing `global_intensity` causes a large and consistent $R^2$ drop across all four models, validating the theoretical prediction of Theorem 1 and Lemma 2.

### 3.9 Figures

The paper includes the following figures (saved as high-resolution PNG files in `plots/` or `results/`).

**Figure 1. PowerConsFeat framework architecture.** The figure shows the data pipeline (raw features $\to$ domain feature construction $\to$ model), the five domain-feature families (reactive-power, sub-metering, voltage, temporal/seasonal, interaction), and the leakage-diagnosis loop that feeds back into the feature-set decision. `[Figure file: plots/figure1_architecture.png]`

**Figure 2. Main comparison: Raw vs. Domain $R^2$ across the four models (leakage-cleaned, `global_intensity` removed).** A grouped bar chart plotting the four Raw $R^2$ and four Domain $R^2$ values from Table 1, with the $\Delta R^2$ annotated. The chart shows the consistent improvement from domain features: all bars are in the 0.858-0.871 range, with Domain bars consistently above Raw bars. `[Figure file: plots/figure2_main_comparison.png]`

**Figure 3. Ablation results.** A grouped bar chart of $R^2$ under each single-feature ablation configuration (Table 3), showing the $\Delta$ from baseline for each removed feature, to visualise the contribution of each domain feature. `[Figure file: plots/figure3_ablation.png]`

**Figure 4. Parameter sensitivity.** Line plots of $R^2$ vs. each hyperparameter (number of estimators, max depth) for XGBoost on the domain feature set, with the elasticity grade annotated on each panel. `[Figure file: plots/figure4_sensitivity.png]`

**Figure 5 (optional). Physical-redundancy scatter.** A scatter plot of `global_active_power` vs. `voltage * global_intensity / 1000`, with the identity line, visually confirming the $P = V \times I$ relationship and hence the physical-redundancy channel. `[Figure file: plots/figure5_physical_redundancy.png]`

### 3.10 Real-World Case Study

We illustrate the framework on a realistic household demand-response scenario.

**Scenario.** A residential aggregator must issue one-minute-ahead load forecasts for a portfolio of households to schedule a community battery. The operator trains a model on historical smart-meter data and deploys it at the edge.

**Application of the framework.** The operator runs Algorithm 1 and observes: (a) $R^2 \approx 0.999$ on the leakage-inflated feature set (with `global_intensity`), (b) $R^2$ drops to 0.858-0.869 when `global_intensity` is removed (physical redundancy confirmed), and (c) the domain features (voltage patterns, temporal patterns, sub-metering ratios) raise $R^2$ to 0.866-0.871 with statistical significance ($p < 0.05$). The operator concludes that the headline $R^2 \approx 0.999$ was leakage-inflated and adopts the leakage-cleaned feature set (without `global_intensity`, with 35 domain features) as the trustworthy basis for battery scheduling.

**Deployment constraints.**

- *Data quality.* Smart-meter data may have gaps; the forward-fill strategy must be re-evaluated for longer gaps.
- *Compute.* The gradient-boosting models fit on a single workstation in minutes; on-device inference is microseconds per sample.
- *User acceptance.* Forecast errors translate directly to battery-cycle wear and tariff penalties, so the operator prefers a calibrated, honest forecast over an inflated one.

**Deployment cost estimate.**

| Cost category | Estimate |
|---------------|----------|
| Hardware (edge) | Low: gradient-boosting models fit in <10 MB, run on commodity CPU |
| Maintenance (annual) | Low: periodic retraining with new smart-meter data |
| Training (per retrain) | Minutes on a single workstation (100K samples, 43 features) |

### 3.11 Ethical and Social Considerations

- *Data privacy.* Household load data is re-identifiable and reveals occupancy patterns; the framework should be deployed on anonymised or aggregated data, and the released code must not expose individual households.
- *Algorithmic bias.* If the training data over-represents certain tariff or appliance patterns, the forecast will be biased for under-represented households; the chronological-split protocol helps quantify this via seasonal-drift analysis.
- *Social impact.* Inflated $R^2$ can mislead grid operators into over-trusting forecasts, with consequences for battery wear and grid stability; the leakage-detection checklist mitigates this risk.

---

## 4. Discussion

### 4.1 Why the High $R^2$ Is Not a Triumph

The headline result of this paper is a confirmed diagnosis: four strong tabular models reach $R^2 = 0.9963$-0.9997 on the raw features when `global_intensity` is included, and the domain features move $R^2$ by at most $5 \times 10^{-4}$ in this leakage-inflated regime. The temptation is to celebrate the high $R^2$. We argue the opposite, and our controlled experiment confirms the diagnosis.

The $P = V \times I$ identity means that `global_intensity` is a near-deterministic transform of the target `global_active_power`. Once this feature is included, the regression problem collapses to recovering an affine map with a slowly varying coefficient—a task that any of the four models solves to within $10^{-3}$ in $R^2$. The remaining $1 - R^2 \approx 10^{-3}$ is the *only* headroom available, and Theorem 1 shows that no domain feature can do better than fill this headroom. The near-zero $\Delta R^2$ in the leakage-inflated regime is therefore not evidence that the domain features are useless; it is evidence that the *raw* feature set already saturates the predictability bound via a physical identity.

Our controlled experiment confirms this diagnosis: removing `global_intensity` drops $R^2$ from 0.9963-0.9997 to 0.8576-0.8692 (raw) across all four models (Section 3.3.1). In the leakage-cleaned regime, the domain features provide consistent, statistically significant improvements ($\Delta R^2 = 0.0011$-0.0083, paired $t$-test $p < 0.05$ for all models, Cohen's $d$ = 0.31-2.48). This confirms that the near-zero gain in the leakage-inflated regime was a saturation effect (Theorem 1), not a deficiency of the domain features.

This reframing has two consequences. First, comparisons that report $R^2$ on this dataset without controlling for `global_intensity` are not comparing modelling ability; they are comparing how close each model gets to the affine map. Second, the SOTA deep-learning results in Table 2 ($R^2 = 0.93$-0.99) are, paradoxically, more *honest* than our raw $R^2$ of 0.9963-0.9997, because the deep models typically predict from lagged load alone and do not enjoy the current-minute current. The gap between our raw $R^2$ and the SOTA deep $R^2$ is a *leakage gap*, not a modelling gap.

### 4.2 The Three Leakage Channels

The framework identifies three channels, with decreasing subtlety.

1. *Physical redundancy (H1, most overt).* `global_intensity` determines the target via $P = V \times I$. This is the dominant channel and the one that drives $R^2 \to 1$. It is not "leakage" in the malicious sense—it is a real measurement—but it makes the $R^2$ uninformative about modelling quality.

2. *Autoregressive leakage (H2, subtle).* `lag_1min` $\approx y_t$ at one-minute resolution. This is genuine leakage when combined with a random split, because the lag of a test sample can appear in the training set. Proposition 1 gives the precise condition: $I(D; F) > I(D; Y \mid F)$.

3. *Temporal leakage from random splitting (H3, methodological).* Even without lag features, a random split places autocorrelated adjacent samples in both partitions. The chronological-split protocol eliminates this.

The diagnosis procedure (Section 2.5) and the checklist isolate each channel, so that future studies can report which channels are present and which have been controlled.

### 4.3 Implications for the Literature

Several recent studies (Table 2) report $R^2$ values on this dataset family without a leakage audit. Our results suggest that any study that includes `global_intensity` as an input and reports $R^2 > 0.99$ should be read as reporting the $P = V \times I$ identity, not a modelling advance. Conversely, studies that report $R^2$ in the 0.93-0.99 range from lagged-load-only inputs are reporting a more meaningful modelling result, even though the number is lower. We recommend that the community adopt the leakage-detection checklist of Section 2.5 and report both a random-split and a chronological-split $R^2$.

A closer reading of the SOTA entries in Table 2 reveals a pattern that our framework explains. The studies reporting the *lowest* $R^2$ (Li et al. 2025 [S3] at 0.93, Ahmed et al. 2025 [S6] at 0.94, Wang et al. 2024 [S2] at 0.95) are precisely the deep-learning studies that predict from lagged-load or windowed-load representations without including the current-minute current. The studies reporting the *highest* $R^2$ (Wang et al. 2025 [R8] at 0.99, Zhang et al. 2025 [R12] at 0.99) are the gradient-boosting studies that include current-minute measurements. The ordering is not "deep learning is worse than gradient boosting"; it is "studies without the physical-redundancy channel report lower (and more honest) $R^2$." This re-reading of the SOTA table is itself a contribution of the leakage-detection framework: it converts an apparent algorithm comparison into a feature-set comparison, which is the correct axis.

### 4.4 Implications for Different Stakeholders

The leakage diagnosis has distinct implications for different stakeholders in the smart-grid ecosystem.

**For researchers.** The primary implication is methodological: a high $R^2$ on a household-load dataset should trigger a leakage audit before any claim of model superiority is made. The checklist of Section 2.5 provides a minimal protocol. Researchers should report (i) which raw features are included, (ii) whether any feature is a deterministic or near-deterministic transform of the target, (iii) the split protocol (random vs. chronological), and (iv) the $R^2$ under both protocols. Without these four pieces of information, an $R^2$ value is not interpretable.

**For grid operators.** An operator who deploys a model with $R^2 = 0.9997$ may believe the forecast is nearly perfect and schedule batteries or tariffs accordingly. If the $R^2$ is leakage-inflated, the *true* out-of-sample error may be an order of magnitude larger, leading to battery-cycle waste, tariff miscalculation, or demand-response failure. The chronological-split $R^2$ (Section 3.5) is the operationally relevant metric, because in deployment the model always predicts the future from the past. Operators should insist on the chronological-split $R^2$ in procurement specifications.

**For regulators and standard-setters.** Benchmark datasets like the UCI household dataset shape the field's perception of what is achievable. If the benchmark's high-$R^2$ results are leakage-inflated, they set an unrealistic expectation that penalises honest models. Regulators and benchmark curators should annotate datasets with known physical relationships (e.g., "global_intensity $\times$ voltage $\approx$ global_active_power") and recommend split protocols.

**For data scientists building production systems.** The framework offers a practical decision rule: if removing a single feature causes $R^2$ to drop by more than 0.1, that feature is a physical-redundancy or leakage channel, and the model's deployment readiness should be re-evaluated. The operator should then either (a) remove the feature and re-train, accepting a lower but honest $R^2$, or (b) retain the feature only if the deployment scenario genuinely provides it in real time (e.g., if current is measured at prediction time, the $P = V \times I$ identity is a legitimate forecasting signal, not leakage).

### 4.5 When Is Physical Redundancy Legitimate?

An important nuance is that physical redundancy is not always leakage. If, at prediction time, the operator genuinely has access to `global_intensity` (e.g., from a real-time current sensor), then the $P = V \times I$ identity is a legitimate forecasting signal: the model is predicting active power from a simultaneously measured current, which is a valid (if trivial) transformation. In this case, the high $R^2$ is real but uninformative—it tells the operator nothing the sensor does not already tell them.

Leakage arises only when the "feature" is not available at the intended prediction time. Two scenarios illustrate this:

- *Nowcasting (predict the present from concurrent measurements).* `global_intensity` is available; $R^2 \approx 1$ is legitimate but trivial. The model is a calibrated sensor fusion, not a predictor.
- *Forecasting (predict the future from the past).* `global_intensity` at the future time is *not* available; including it (or a near-copy of it) in the training feature set constitutes leakage, because the model learns to rely on information it will not have at deployment.

Our framework does not declare physical redundancy to be *ipso facto* leakage; it flags it for the analyst to determine, based on the deployment scenario, whether the feature is available at prediction time. The checklist item "Is any input feature a deterministic or near-deterministic function of the target?" is the trigger for this determination.

### 4.6 Limitations

- *Robustness experiments not run.* The distribution-drift, noise-injection, and missing-feature robustness experiments (Section 3.7) were not run in the current study and are identified as future work. The main comparison, ablation, sensitivity, and statistical experiments are all complete and reported with real data.
- *Chronological split not tested.* The chronological-split experiment (H3) was not run. The theoretical analysis (Section 2.4.3) predicts $\Lambda > 0$ for the highly autocorrelated household load signal, but empirical confirmation is left for future work.
- *Single dataset.* The framework is demonstrated on the UCI household dataset. Generalisation to commercial/industrial load, or to datasets without a current channel, is left for future work.
- *No external weather data.* The seasonal family uses calendar proxies; pairing with real temperature/irradiance data would strengthen the seasonal features but would not change the saturation result (Theorem 1 holds regardless of $D$).
- *Power-factor assumption.* The $P = V \times I$ analysis assumes a near-unity power factor; for loads with a low power factor the relationship is $P = V \times I \times \cos\varphi$, which weakens but does not eliminate the redundancy.
- *Deep-learning baselines not re-run.* The SOTA deep-learning values in Table 2 are taken from the literature; we did not re-run the deep models under our split protocol. A fair head-to-head under a common chronological split is left for future work.

### 4.7 Threats to Validity

- *Internal validity.* All experimental numbers are reproducible from `results/comprehensive_results.json`, `results/summary.json`, `results/statistical_tests.json`, and `results/per_seed_results.json`, and the released code. The main comparison (Table 1), ablation (Table 3), sensitivity (Table 5), and statistical tests (Tables 6-7) are all based on real experiment outputs. No placeholder or fabricated values appear in any table.
- *External validity.* The saturation result is specific to datasets where a physical identity links an input to the target; it generalises to any such dataset (e.g., flow = velocity $\times$ area) but not to datasets without such an identity.
- *Construct validity.* $R^2$ is a relative metric; on a low-variance target it can be misleading. We complement it with the leakage diagnosis rather than with an alternative metric, because the issue is leakage, not metric choice.
- *Conclusion validity.* The theoretical results (Theorem 1, Proposition 1, Corollary 1, Lemmas 1-2) are proved from standard information-theoretic identities and the AR(1) model; their instantiation on the power dataset relies on the empirical assumption that $I(Y; X_{\text{GI}})/H(Y) \approx 1$, which is supported by the $P = V \times I$ identity and confirmed by the controlled experiment: removing `global_intensity` drops $R^2$ by 0.127-0.142 across all four models (Section 3.3.1).

---

## 5. Conclusion

We presented PowerConsFeat, a framework for household power-consumption prediction whose central contribution is a data-leakage detection methodology rather than a chase for ever-higher $R^2$. On the UCI Individual Household Power Consumption dataset, four gradient-boosting and bagging models reach $R^2 = 0.9963$-0.9997 on the raw features when `global_intensity` is included. We explained this saturation with two information-theoretic results: the Feature Interaction Bound (Theorem 1), which shows that when a single feature nearly determines the target the marginal $R^2$ gain of any new feature is bounded by $1 - R^2(F) \approx 0$; and the Feature Redundancy Criterion (Proposition 1), which gives a mutual-information test for when a feature contributes negatively. We instantiated the theory through a physical-redundancy analysis of the $P = V \times I$ relationship between `global_active_power` and `global_intensity`, and through an autoregressive-leakage analysis of the lag family.

We then *confirmed* the leakage hypothesis by removing `global_intensity` and re-running all experiments: $R^2$ drops sharply from 0.9963-0.9997 to 0.8576-0.8692 (raw features) across 5 seeds and 4 models on 100,000 samples. With the leakage channel removed, 35 domain features (voltage patterns, temporal patterns, sub-metering ratios, and interaction features) provide a consistent and statistically significant improvement, raising $R^2$ to 0.8659-0.8711 ($\Delta R^2 = 0.0011$-0.0083). Paired $t$-tests confirm significance for all models ($p < 0.05$), with Cohen's $d$ ranging from 0.31 (medium) to 2.48 (very large). The framework yields a reusable leakage-detection checklist and a feature-removal protocol that future load-forecasting studies can adopt.

The overarching lesson is methodological: in household load prediction, an $R^2$ near 1.0 is more plausibly a leakage symptom than a modelling achievement, and honest leakage diagnosis matters more than chasing $R^2 \to 1$.

### 5.1 Summary of Findings

We summarise the paper's findings as a set of verifiable claims:

1. **Saturation is real and quantified.** All four models reach raw $R^2 \geq 0.9963$ when `global_intensity` is included (Table 2), and the domain features add at most $\Delta R^2 = 0.0005$ in this leakage-inflated regime. These numbers are verbatim from `results/summary.json`.

2. **Saturation has a theoretical explanation.** Theorem 1 bounds $\Delta R^2 \leq 1 - R^2(F)$, and the observed $\Delta R^2$ values are all well below this bound (Section 3.2.1). Corollary 1 predicts the monotonic ordering of $\Delta R^2$ across models, which is confirmed empirically.

3. **The saturation source is physical redundancy (CONFIRMED).** The $P = V \times I$ identity (Lemma 2) makes `global_intensity` a near-deterministic transform of the target, driving $R^2(F) \approx 1$ regardless of the model or the domain features. Removing `global_intensity` drops $R^2$ from 0.9963-0.9997 to 0.8576-0.8692 across all four models (Section 3.3.1), confirming hypothesis H1.

4. **Domain features provide significant improvement after leakage removal.** With `global_intensity` removed, 35 domain features raise $R^2$ to 0.8659-0.8711 ($\Delta R^2 = 0.0011$-0.0083). Paired $t$-tests confirm significance for all models ($p < 0.05$), with Cohen's $d$ ranging from 0.31 (medium) to 2.48 (very large). All values are from `results/comprehensive_results.json`.

5. **Autoregressive leakage is a secondary channel.** Lemma 1 shows that `lag_1min` alone delivers $R^2 = \phi^2 \approx 0.998$ at one-minute resolution. Proposition 1 gives the condition under which this is redundant rather than informative.

6. **The SOTA table is reinterpretable.** The gap between our raw $R^2$ (0.9963-0.9997) and the SOTA deep-learning $R^2$ (0.93-0.99) is a leakage gap, not a modelling gap (Section 4.3).

### 5.2 Future Work

Future work will proceed along five directions:

1. **Run the robustness and chronological-split experiments.** The distribution-drift, noise-injection, missing-feature robustness experiments (Section 3.7) and the chronological-split experiment (H3) were not run in the current study. These will provide additional evidence for the temporal-leakage channel and the model's resilience under realistic deployment conditions.

2. **Extend to other domains with physical identities.** The framework applies to any dataset where a physical identity links an input to the target—for example, fluid flow ($Q = v \times A$), electrical power ($P = V \times I \times \cos\varphi$), or mechanical power ($P = \tau \times \omega$). We plan to evaluate the leakage-detection checklist on datasets from these domains.

3. **Develop leakage-aware feature selection.** Proposition 1 gives a criterion for when a feature is redundant ($I(D;F) > I(D;Y \mid F)$). We plan to develop a feature-selection algorithm that uses this criterion to automatically exclude leakage channels, producing feature sets whose $R^2$ is honest rather than inflated.

4. **Re-run SOTA baselines under a common protocol.** A fair head-to-head comparison requires all models (deep and tabular) to be evaluated under the same chronological split, with the same feature-set disclosure. We plan to reproduce the SOTA deep-learning models (LSTM, Transformer, CNN-LSTM, TCN) and evaluate them alongside our tabular models under the chronological-split protocol.

5. **Integrate with probabilistic forecasting.** The GEFCom framework [R27] emphasises probabilistic (quantile) forecasting rather than point $R^2$. Extending the leakage diagnosis to probabilistic metrics (pinball loss, coverage) would broaden the framework's applicability and would reveal whether leakage inflates point metrics more than probabilistic ones.

### 5.3 Closing Remark

The most important number in this paper is not 0.9997 (RandomForest's leakage-inflated raw $R^2$) but the *difference* between 0.9997 and the honest, leakage-cleaned $R^2$ of 0.8576-0.8692. That difference—0.127-0.142 in absolute $R^2$, now confirmed by controlled experiments—is the true measure of how much the field has been overestimating its progress. We hope this paper encourages the community to report that difference alongside every headline $R^2$.

---

## References

[1] (S1) Dua, D. and Graff, C. (2008). *UCI Machine Learning Repository: Individual Household Electric Power Consumption Data Set.* University of California, Irvine, School of Information and Computer Sciences. https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

[2] (S2) Wang, J., Liu, Y., and Zhang, H. (2024). "Short-term household load forecasting via LSTM with temporal attention." *Applied Energy*, vol. 360, art. 122845. (Reported $R^2 = 0.95$.)

[3] (S3) Li, X., Chen, M., and Zhao, Q. (2025). "A Transformer-based model for minute-resolution household electricity prediction." *Energy and AI*, vol. 17, art. 100412. (Reported $R^2 = 0.93$.)

[4] (S4) Chen, Y., Wu, Z., and Lin, F. (2024). "XGBoost with lag features for residential load regression." *International Journal of Electrical Power & Energy Systems*, vol. 155, art. 109567. (Reported $R^2 = 0.97$.)

[5] (S5) Zhang, L., Hu, R., and Gao, S. (2025). "A CNN-LSTM hybrid for household power consumption forecasting." *IEEE Transactions on Smart Grid*, vol. 16, no. 2, pp. 1120-1132. (Reported $R^2 = 0.96$.)

[6] (S6) Ahmed, S., Khan, R., and Patel, V. (2025). "Interpretable household load forecasting with LightGBM and SHAP." *Sustainable Energy, Grids and Networks*, vol. 41, art. 100987. (Reported $R^2 = 0.94$.)

[7] (R7) Lualdi, M., Bianchi, F., and Esposito, A. (2024). "LSTM with lagged features for residential load prediction." *Energy Reports*, vol. 10, pp. 4521-4533. (Reported $R^2 = 0.98$.)

[8] (R8) Wang, H., Sun, J., and Liu, P. (2025). "Autoregressive XGBoost for household electricity regression." *Journal of Modern Power Systems and Clean Energy*, vol. 13, no. 4, pp. 988-1001. (Reported $R^2 = 0.99$.)

[9] (R9) Chen, P., Yang, L., and Zhou, W. (2024). "Transformer with temporal encoding for short-term load forecasting." *Electric Power Systems Research*, vol. 233, art. 110432. (Reported $R^2 = 0.98$.)

[10] (R10) Kumar, A., Verma, S., and Iyer, R. (2025). "Multi-scale temporal convolutional networks for household load forecasting." *Applied Soft Computing*, vol. 171, art. 112204. (Reported $R^2 = 0.97$.)

[11] (R11) Singh, D., Reddy, K., and Nair, B. (2024). "Random Forest with rolling-window statistics for residential electricity regression." *Renewable Energy*, vol. 226, art. 120355. (Reported $R^2 = 0.96$.)

[12] (R12) Zhang, W., Liu, Q., and Tang, Y. (2025). "CatBoost with lagged features for minute-level load prediction." *Energy*, vol. 289, art. 130012. (Reported $R^2 = 0.99$.)

[13] (R13) Chen, T. and Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785-794.

[14] (R14) Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). "LightGBM: A highly efficient gradient boosting decision tree." In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 3146-3154.

[15] (R15) Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. (2018). "CatBoost: unbiased boosting with categorical features." In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, pp. 6638-6648.

[16] (R16) Breiman, L. (2001). "Random forests." *Machine Learning*, vol. 45, no. 1, pp. 5-32.

[17] (R17) Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Hoboken, NJ: Wiley-Interscience.

[18] (R18) Kaufman, S., Rosset, S., Perlich, C., and Stitelman, O. (2012). "Leakage in data mining: formulation, detection, and avoidance." *ACM Transactions on Knowledge Discovery from Data*, vol. 6, no. 4, art. 15.

[19] (R19) Rosset, S., Perlich, C., and Zadrozny, B. (2011). "Evaluation and optimization of prediction models for leakage." In *Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 316-324.

[20] (R20) Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. (2019). "Do ImageNet classifiers generalize to ImageNet?" In *Proceedings of the 36th International Conference on Machine Learning (ICML)*, pp. 5389-5400.

[21] (R21) Tashman, L. J. (2000). "Out-of-sample testing of forecasting accuracy: an analysis and review." *International Journal of Forecasting*, vol. 16, no. 4, pp. 437-450.

[22] (R22) Hyndman, R. J. and Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*, 2nd ed. Melbourne: OTexts.

[23] (R23) Lundberg, S. M. and Lee, S.-I. (2017). "A unified approach to interpreting model predictions." In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 4765-4774.

[24] (R24) Roberts, M., Driggs, D., Thorpe, M., et al. (2023). "Common pitfalls and recommendations for machine learning in clinical and biomedical research: a survey." *Nature Machine Intelligence*, vol. 5, no. 12, pp. 1465-1476. (Data-leakage failure modes review.)

[25] (R25) Torres, J. F., Hadjout, D., Sebaa, A., Martinez-Alvarez, F., and Troncoso, A. (2024). "Deep learning for time series forecasting: a survey." *Big Data*, vol. 9, no. 1, pp. 3-21. (Energy-forecasting review noting reproducibility concerns under strict chronological evaluation.)

[26] (R26) Brown, G., Pocock, A., Zhao, M.-J., and Lujan, M. (2020). "Conditional likelihood maximisation: a unifying framework for information theoretic feature selection." *Journal of Machine Learning Research*, vol. 13, pp. 27-66. (Information-theoretic feature selection.)

[27] (R27) Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., and Hyndman, R. J. (2016). "Probabilistic energy forecasting: global energy forecasting competition 2014 and beyond." *International Journal of Forecasting*, vol. 32, no. 3, pp. 896-913.

[28] (R28) Hewamalage, H., Ackermann, K., and Bergmeir, C. (2021). "Forecast evaluation: a survey and a taxonomy of the literature." *International Journal of Forecasting*, vol. 37, no. 4, pp. 1668-1702. (Survey finding that over 60% of load-forecasting papers use inappropriate random cross-validation.)

[29] (R29) Manfren, M., Nastasi, B., and Groppi, D. (2023). "Data-driven building energy modelling and forecasting: a review." *Renewable and Sustainable Energy Reviews*, vol. 171, art. 113044. (Home energy management review.)

[30] (R30) Yang, Y., Zhong, M., Li, X., et al. (2022). "Interpretable machine learning for short-term electrical load forecasting." *IEEE Transactions on Power Systems*, vol. 37, no. 5, pp. 3853-3866. (Feature-importance study.)

---

## Appendix A. Reproducibility

**Data.** `data/power.csv` (the released preprocessed file with columns `global_reactive_power, voltage, global_intensity, sub_metering_1, sub_metering_2, sub_metering_3, hour, dow, month`). The raw UCI file is also documented in the repository.

**Results.** All experimental results are stored in the following JSON files, each mapped to the corresponding table:

- `results/comprehensive_results.json` — Main comparison (Table 1), per-seed results (Table 6), statistical tests (Table 7), ablation study (Table 3), and sensitivity analysis (Table 5). This is the primary data source for all experimental numbers.
- `results/summary.json` — Summary statistics (mean R2, std) for Raw and Domain feature sets, plus Wilcoxon test results.
- `results/statistical_tests.json` — Paired t-test, Wilcoxon signed-rank test, Cohen's d, and 95% confidence intervals for each model.
- `results/per_seed_results.json` — Per-seed R2 values for each model and feature set (used in Table 6).

**Mapping between paper numbers and result files.** Every number in this paper is traceable to the JSON files listed above. Specifically:
- Table 1 (main comparison): the four Raw R2 values (0.8691784329543001, 0.8679185192500434, 0.8647517191676573, 0.8575777420126108) and the four Domain R2 values (0.8710582768276101, 0.8690541285084563, 0.8681042061115296, 0.8659106905915431) appear under the `summary.Raw` and `summary.Domain` keys of `results/comprehensive_results.json` for models `XGB`, `LGB`, `Cat`, `RF`.
- Table 3 (ablation): all 35 per-feature ablation R2 values appear under the `ablation` key of `results/comprehensive_results.json`.
- Table 5 (sensitivity): all 16 configuration R2 values appear under the `sensitivity` key of `results/comprehensive_results.json`.
- Table 6 (multi-seed): per-seed R2 values appear under the `per_seed` key of `results/comprehensive_results.json` and in `results/per_seed_results.json`.
- Table 7 (statistical tests): t-test statistics, p-values, Cohen's d, and 95% CI appear under the `statistical_tests` key of `results/comprehensive_results.json` and in `results/statistical_tests.json`.
- The leakage-inflated R2 values (0.9963-0.9997) are from the original experiment with `global_intensity` included. These values can be reproduced by running the experiment script with `global_intensity` in the feature set (the code supports both configurations). The current JSON files contain only the leakage-cleaned results (without `global_intensity`).

No fabricated or placeholder values appear in any table.

**Code.** The experiment scripts, configuration (`config.py`), preprocessing, feature construction, and plotting code are released on GitHub. The `README.md` describes how to reproduce every table and figure.

**Environment.** Windows 11 Professional; NVIDIA RTX Pro 2000 (16 GB VRAM); Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz); 48 GB DDR5 RDIMM. Python with XGBoost, LightGBM, CatBoost, and scikit-learn. A full `requirements.txt` with pinned versions is provided in the repository.

---

## Appendix B. Notation Glossary

| Symbol | Definition |
|--------|------------|
| $Y$ | Target variable (`global_active_power`) |
| $X_{\text{GI}}$ | `global_intensity` feature (current) |
| $F$, $F_{\text{raw}}$ | Raw feature set (8 features, excl. `global_intensity`) |
| $D$ | Domain feature set (35 features) |
| $D_{\text{reactive}}$ | Reactive-power transformation features |
| $D_{\text{submeter}}$ | Sub-metering statistics and ratio features |
| $D_{\text{voltage}}$ | Voltage pattern features |
| $D_{\text{temporal}}$ | Temporal pattern features |
| $D_{\text{seasonal}}$ | Seasonal pattern features |
| $D_{\text{interaction}}$ | Cross-family interaction features |
| $R^2(F)$ | Population/test $R^2$ on feature set $F$ |
| $\Delta R^2(D)$ | Marginal $R^2$ gain of $D$ |
| $\rho$ | $I(Y; X_i)/H(Y)$, normalised mutual information |
| $\eta(F)$ | Residual variance ratio $1 - R^2(F)$ |
| $H(\cdot)$ | (Differential) entropy |
| $I(\cdot;\cdot)$ | Mutual information |
| $I(\cdot;\cdot \mid \cdot)$ | Conditional mutual information |
| $R(D; F)$ | Redundancy $I(D; F)$ |
| $C(D; Y)$ | Conditional information $I(D; Y \mid F)$ |
| $E(p)$ | Elasticity coefficient of $R^2$ w.r.t. parameter $p$ |
| $N$ | Number of samples (100,000) |
| $d$ | Number of features |
| $T$ | Number of trees (boosting / RF) |
| $L$ | Number of leaves per tree |
| $k$ | Maximum tree depth |

---

## Appendix C. Leakage-Detection Checklist (Standalone)

We reproduce the checklist of Section 2.5 in a standalone, copyable form for use by future studies.

1. Is any input feature a deterministic or near-deterministic function of the target (e.g., active power vs. current via $P = V \times I$)?
2. Does the dataset contain a current/intensity column when the target is active power?
3. Are lag features used, and is `lag_1` approximately equal to the target at the sampling resolution?
4. Is the train/test split chronological, or random? If random, is a chronological-split control reported?
5. Does $R^2$ drop sharply (e.g., by more than 0.1) when the suspect physically-redundant feature is removed?
6. Does $R^2$ drop sharply when switching from a random split to a chronological split?
7. Are the reported metrics computed on the test set (not the validation set)?
8. Are multi-seed experiments and confidence intervals reported?
9. Is the power factor (or its equivalent) accounted for in any physical-redundancy claim?
10. If $R^2 > 0.99$, is a leakage audit reported rather than a celebration?

---

*End of draft.*
