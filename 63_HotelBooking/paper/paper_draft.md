# HotelFeat: Hospitality Domain Feature Analysis for Booking Cancellation Prediction

**Jingyuan Zeng¹, Ming Zeng², Jianghong Guo¹, Chuanxian Jiang¹, Yafen Feng³,⁴,\***

¹ School of Computer Science, Jiaying University, Meizhou 514015, China
² College of Water Conservancy and Civil Engineering, South China Agricultural University, Guangzhou 510642, China
³ School of Geography Science and Tourism, Jiaying University, Meizhou 514015, China
⁴ Key Laboratory of Mountain Surface Environment and Green Development in Northeast Guangdong, Meizhou 514015, China

*Corresponding author: Yafen Feng, E-mail: fyf81@163.com

**Fund:** Guangdong Provincial Higher Education Teaching Reform Project (Grant No. Yue Jiao Gao Han [2024] 9-989)

---

## Abstract

Hotel booking cancellation prediction enables hotels to optimize overbooking strategies, staffing, and inventory allocation in revenue management. While the Hotel Booking Demand dataset provides approximately 29 features per reservation, the potential for domain-specific feature engineering to improve cancellation prediction remains underexplored. This paper proposes HotelFeat, a hospitality domain feature analysis framework that constructs four families of engineered features—guest composition, booking patterns, temporal seasonality, and pricing categories—from the Hotel Booking Demand dataset. We provide a theoretical foundation through Theorem 1 (feature interaction bound), proving that deterministic transformations yield zero informational gain, and Proposition 1 (feature redundancy), characterizing when domain features become fully redundant. The augmented features are evaluated against four models—XGBoost, LightGBM, CatBoost, and RandomForest—under raw-only and domain-augmented configurations across five random seeds. Experimental results reveal a mixed picture: domain features provide small but statistically significant AUC improvements for LightGBM (+0.0004, p=0.025) and CatBoost (+0.0010, p=0.003), a non-significant positive trend for XGBoost (+0.0004, p=0.197), but a significant performance decrease for RandomForest (-0.0057, p<0.001). Ablation analysis identifies room_mismatch, month_sin, and total_nights as the most impactful domain features. The results provide practical guidance: domain feature engineering yields model-dependent effects, and its value must be empirically validated rather than assumed.

**Keywords:** Hotel booking cancellation; Feature engineering; Gradient boosting; Hospitality analytics; Ablation analysis; Revenue management

---

## 1. Introduction and Related Work

### 1.1 Background

Hotel booking cancellations represent a significant challenge for the hospitality industry, with cancellation rates often exceeding 20–30% of total reservations. Accurate prediction of which bookings will be cancelled enables hotels to implement effective overbooking strategies, optimize staffing levels, and manage inventory more efficiently. The Hotel Booking Demand dataset [1], containing 119,390 reservation records from two hotels (a city hotel and a resort hotel) in Portugal, has become the standard benchmark for this task. The dataset includes approximately 29 features per booking, covering temporal information (lead time, arrival date), guest characteristics (adults, children, babies, repeat guest), booking details (market segment, distribution channel, deposit type, room type), and historical context (previous cancellations, previous bookings).

Despite the richness of the original feature set, the question of whether domain-specific feature engineering—constructing higher-order features that encode hospitality-specific knowledge—can improve cancellation prediction accuracy has not been systematically addressed. Domain features such as guest group composition (e.g., family vs. business traveler), booking lead time patterns (e.g., early bird vs. last-minute), seasonal effects, and pricing categories are well-known to influence cancellation behavior. However, when the original feature set already contains the constituent variables (lead time, number of adults, average daily rate, etc.), the marginal benefit of explicit domain feature construction depends on whether the models can discover these patterns automatically.

### 1.2 Related Work

**Gradient boosting methods.** Tree-based ensemble methods have become the dominant approach for tabular data classification. Friedman [2] established the theoretical foundations of gradient boosting. Chen and Guestrin [3] introduced XGBoost with regularization and sparsity-aware split finding. Ke et al. [4] developed LightGBM with GOSS and EFB for scalable training. Prokhorenkova et al. [5] proposed CatBoost with ordered boosting and oblivious trees. Breiman [6] introduced Random Forest, which remains a robust baseline. These methods are particularly well-suited for hotel cancellation prediction due to their ability to handle mixed feature types, missing values, and nonlinear interactions.

**Hotel cancellation prediction.** Several studies have addressed hotel cancellation prediction in recent years. Antonio et al. [1] introduced the Hotel Booking Demand dataset and conducted exploratory analysis of cancellation patterns. Sanchez-Medina and C-Sanchez [7] developed a machine learning approach using gradient boosting and neural networks for cancellation prediction, achieving AUC above 0.85. Chen et al. [8] proposed a deep learning framework with attention mechanisms for hotel cancellation prediction. Nair et al. [9] compared multiple classifiers including logistic regression, random forest, and XGBoost. Li et al. [10] introduced a time-series approach for aggregate cancellation forecasting. Dogru et al. [11] studied the impact of deposit policies on cancellation behavior. Wang et al. [12] proposed a multi-task learning framework for joint cancellation and no-show prediction.

**Feature engineering in hospitality.** Domain feature engineering has been explored in hospitality contexts. Zhang et al. [13] constructed temporal features (day-of-week, season, holiday proximity) for hotel demand forecasting. Huang et al. [14] proposed guest segmentation features based on booking patterns for personalized recommendation. Lado-Sestayo et al. [15] developed pricing-based features for hotel performance analysis. However, these studies typically applied feature engineering alongside other innovations (e.g., new architectures, ensemble methods), making it difficult to isolate the contribution of domain features alone.

**Feature interaction theory.** The theoretical analysis of feature interactions draws from information theory [16] and functional ANOVA decompositions [17]. A key result, formalized in our Theorem 1, is that deterministic transformations of existing features cannot increase mutual information with the target. This has direct implications for domain feature engineering: if domain features are merely recombinations of existing variables, their informational contribution is zero. The practical benefit, if any, comes from approximation efficiency—helping models discover patterns with fewer computational resources.

**SHAP and interpretability.** Lundberg and Lee [18] introduced SHAP (SHapley Additive exPlanations), a unified interpretability framework based on Shapley values. Lundberg et al. [19] developed TreeSHAP for efficient computation in tree ensembles. SHAP has been applied to hospitality analytics by several researchers [20, 21] for understanding cancellation drivers.

**Recent hospitality analytics studies.** In the past five years, several studies have advanced hotel cancellation prediction and hospitality analytics. Guizzardi et al. [22] proposed a probabilistic model for hotel cancellation forecasting with time-varying effects. Huang et al. [23] developed a transformer-based model for sequential booking cancellation prediction. Zheng et al. [24] introduced a graph neural network approach for hotel recommendation and cancellation. Bagheri et al. [25] studied the fairness implications of cancellation prediction models. Jiang et al. [26] proposed an ensemble stacking approach combining multiple base learners for cancellation prediction. Kim et al. [27] developed a real-time cancellation prediction system for hotel revenue management. Pham et al. [28] studied cross-hotel transfer learning for cancellation prediction. Almeida et al. [29] analyzed the impact of COVID-19 on hotel cancellation patterns. Schetinger et al. [30] proposed interpretable rule-based models for cancellation prediction.

### 1.3 Contributions

This paper makes the following contributions:

1. **A hospitality domain feature analysis framework (HotelFeat)** that constructs four families of domain-specific features—guest composition, booking patterns, temporal seasonality, and pricing categories—totaling 38 engineered features from standard hotel reservation attributes.
2. **A theoretical framework explaining when domain feature engineering provides no benefit**, including Theorem 1 (feature interaction bound) proving zero informational gain for deterministic transformations, and Proposition 1 (feature redundancy) characterizing redundancy conditions specific to the hotel booking domain.
3. **A comprehensive empirical evaluation** across four state-of-the-art tree-based models with five-seed statistical validation, 38-feature ablation studies, parameter sensitivity analysis, and feature importance analysis, revealing that domain features produce model-dependent effects: small but significant improvements for LightGBM and CatBoost, a non-significant trend for XGBoost, and a significant performance decrease for RandomForest.
4. **Practical guidance for hospitality data scientists**: domain feature engineering yields heterogeneous effects across model architectures, and its value must be empirically validated for each model rather than assumed to be universally beneficial or negligible.

---

## 2. Methodology

### 2.1 Problem Formulation

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ denote the Hotel Booking Demand dataset, where $n = 119{,}390$, each sample consists of a feature vector $\mathbf{x}_i \in \mathbb{R}^d$ ($d = 29$ raw features) and a binary label $y_i \in \{0, 1\}$ indicating whether booking $i$ was cancelled ($y_i = 1$) or not ($y_i = 0$). The goal is to learn a classification function $f: \mathbb{R}^d \to \{0, 1\}$ that maximizes the AUC:

$$\text{AUC}(f) = P(f(\mathbf{x}_+) > f(\mathbf{x}_-))$$

In the domain-augmented setting, we construct $\Phi(\mathbf{x}_i) \in \mathbb{R}^{d'}$ where $d' = 67$ ($d' > d$), and the augmented model $g: \mathbb{R}^{d'} \to \{0, 1\}$ is trained on $\{(\Phi(\mathbf{x}_i), y_i)\}_{i=1}^{n}$.

### 2.2 Domain Feature Engineering

We define four families of domain features derived from the raw hotel booking attributes, totaling 38 engineered features.

#### 2.2.1 Guest Composition Features (5 features)

**Total guests.** Total number of guests including adults, children, and babies:

$$\text{total\_guests}_i = \text{adults}_i + \text{children}_i + \text{babies}_i$$

**Has children.** Binary indicator for presence of children or babies:

$$\text{has\_children}_i = \mathbb{1}[\text{children}_i + \text{babies}_i > 0]$$

**Adult ratio.** Proportion of adults among total guests:

$$\text{adult\_ratio}_i = \frac{\text{adults}_i}{\text{total\_guests}_i + \epsilon}$$

**Is family.** Binary indicator for family booking:

$$\text{is\_family}_i = \mathbb{1}[\text{children}_i > 0 \text{ or } \text{babies}_i > 0]$$

**Is solo.** Binary indicator for single adult traveler:

$$\text{is\_solo}_i = \mathbb{1}[\text{adults}_i = 1 \text{ and } \text{children}_i = 0 \text{ and } \text{babies}_i = 0]$$

#### 2.2.2 Booking Pattern Features (22 features)

**Stay duration features.** Total nights, weekend ratio, and no-stay indicator:

$$\text{total\_nights}_i = \text{stays\_in\_weekend\_nights}_i + \text{stays\_in\_week\_nights}_i$$

$$\text{weekend\_ratio}_i = \frac{\text{stays\_in\_weekend\_nights}_i}{\text{total\_nights}_i + \epsilon}, \quad \text{is\_no\_stay}_i = \mathbb{1}[\text{total\_nights}_i = 0]$$

**Lead time features.** Nonlinear transformations of lead time:

$$\text{lead\_time\_squared}_i = \text{lead\_time}_i^2$$

$$\text{is\_long\_lead}_i = \mathbb{1}[\text{lead\_time}_i > 180], \quad \text{is\_short\_lead}_i = \mathbb{1}[\text{lead\_time}_i \leq 7], \quad \text{is\_same\_day}_i = \mathbb{1}[\text{lead\_time}_i = 0]$$

**Cancellation history features.** Aggregate and rate features from previous booking history:

$$\text{total\_previous}_i = \text{previous\_cancellations}_i + \text{previous\_bookings\_not\_canceled}_i$$

$$\text{cancellation\_rate}_i = \frac{\text{previous\_cancellations}_i}{\text{total\_previous}_i + \epsilon}$$

$$\text{has\_cancelled\_before}_i = \mathbb{1}[\text{previous\_cancellations}_i > 0], \quad \text{has\_booking\_history}_i = \mathbb{1}[\text{total\_previous}_i > 0]$$

**Room mismatch.** Binary indicator for reserved versus assigned room type discrepancy:

$$\text{room\_mismatch}_i = \mathbb{1}[\text{reserved\_room\_type}_i \neq \text{assigned\_room\_type}_i]$$

**Special request and booking change features.** Binary indicators and squared transforms:

$$\text{has\_special\_requests}_i = \mathbb{1}[\text{total\_of\_special\_requests}_i > 0], \quad \text{special\_requests\_squared}_i = \text{total\_of\_special\_requests}_i^2$$

$$\text{has\_changes}_i = \mathbb{1}[\text{booking\_changes}_i > 0], \quad \text{booking\_changes\_squared}_i = \text{booking\_changes}_i^2$$

**Operational features.** Parking, waiting list, and their nonlinear transforms:

$$\text{needs\_parking}_i = \mathbb{1}[\text{required\_car\_parking\_spaces}_i > 0]$$

$$\text{has\_waited}_i = \mathbb{1}[\text{days\_in\_waiting\_list}_i > 0], \quad \text{wait\_days\_squared}_i = \text{days\_in\_waiting\_list}_i^2$$

**Categorical indicators.** Binary encodings for market segment, customer type, and hotel type:

$$\text{is\_online\_ta}_i = \mathbb{1}[\text{market\_segment}_i = \text{Online TA}]$$

$$\text{is\_group}_i = \mathbb{1}[\text{customer\_type}_i = \text{Group}], \quad \text{is\_resort}_i = \mathbb{1}[\text{hotel}_i = \text{Resort Hotel}]$$

#### 2.2.3 Temporal Seasonality Features (5 features)

**Arrival month (numeric).** Numeric encoding of arrival month (1–12):

$$\text{arrival\_month\_num}_i = \text{month}(\text{arrival\_date}_i)$$

**Season indicators.** Binary indicators for peak and off seasons:

$$\text{is\_peak\_season}_i = \mathbb{1}[\text{arrival\_month}_i \in \{6, 7, 8\}]$$

$$\text{is\_off\_season}_i = \mathbb{1}[\text{arrival\_month}_i \in \{12, 1, 2\}]$$

**Cyclical encoding.** Sine and cosine transforms to capture monthly periodicity:

$$\text{month\_sin}_i = \sin\left(\frac{2\pi \cdot \text{arrival\_month}_i}{12}\right), \quad \text{month\_cos}_i = \cos\left(\frac{2\pi \cdot \text{arrival\_month}_i}{12}\right)$$

#### 2.2.4 Pricing Category Features (5 features)

**ADR squared.** Nonlinear transform of average daily rate:

$$\text{adr\_squared}_i = \text{adr}_i^2$$

**Price category indicators.** Binary indicators for price tiers:

$$\text{is\_high\_price}_i = \mathbb{1}[\text{adr}_i > 200], \quad \text{is\_low\_price}_i = \mathbb{1}[\text{adr}_i \leq 50]$$

**Deposit type indicators.** Binary encodings for deposit policy:

$$\text{is\_non\_refundable}_i = \mathbb{1}[\text{deposit\_type}_i = \text{Non Refund}]$$

$$\text{is\_refundable}_i = \mathbb{1}[\text{deposit\_type}_i = \text{Refundable}]$$

#### 2.2.5 Cross-Domain Interaction Feature (1 feature)

**Lead time–ADR interaction.** Product of lead time and average daily rate:

$$\text{lead\_adr\_interaction}_i = \text{lead\_time}_i \times \text{adr}_i$$

### 2.3 Theoretical Analysis

#### 2.3.1 Feature Interaction Bound

**Theorem 1 (Feature Interaction Bound).** *Let $X \in \mathbb{R}^d$ be the raw feature set from hotel booking data, $Z = \phi(X) \in \mathbb{R}^{d'}$ be augmented features produced by a deterministic transformation $\phi$, and $Y \in \{0, 1\}$ be the cancellation label. The marginal information gain of augmentation is:*

$$\Delta I = I(Y; X, Z) - I(Y; X) = I(Y; Z | X)$$

*where $I(Y; Z | X) = H(Z | X) - H(Z | X, Y)$ is the conditional mutual information. If $Z = \phi(X)$ is a deterministic function of $X$, then $H(Z | X) = 0$ and $\Delta I = 0$: deterministic transformations cannot increase the mutual information between features and the cancellation label.*

**Proof.** By the chain rule of mutual information:

$$I(Y; X, Z) = I(Y; X) + I(Y; Z | X)$$

Therefore $\Delta I = I(Y; Z | X)$. By the definition of conditional mutual information:

$$I(Y; Z | X) = H(Z | X) - H(Z | X, Y)$$

For a deterministic transformation $Z = \phi(X)$, $Z$ is completely determined given $X$, so $H(Z | X) = 0$ and $H(Z | X, Y) = 0$, yielding $\Delta I = 0$. $\square$

**Remark 1.** Theorem 1 has a direct implication for the Hotel Booking Demand dataset: since the domain features (guest composition, booking patterns, temporal seasonality, pricing categories) are all deterministic functions of the raw features (adults, children, lead_time, adr, arrival_month, etc.), they cannot add any new information about the cancellation label. The observed AUC of 0.9406–0.9552 on raw features represents an informational ceiling that cannot be surpassed by domain feature engineering alone in terms of mutual information. However, this does not preclude small practical improvements in AUC, as models may benefit from approximation efficiency—discovering patterns more easily through explicitly engineered features rather than learning them implicitly from raw inputs.

**Remark 2.** The only way to exceed this informational ceiling is through features that are not deterministic functions of the raw data—e.g., external data sources (weather forecasts, local event calendars, competitor pricing, macroeconomic indicators) or learned representations that incorporate information from the training distribution (e.g., embedding features from neural network pre-training). Our domain features do not fall into either category.

#### 2.3.2 Feature Redundancy

**Proposition 1 (Feature Redundancy).** *Let $Z_j$ be an augmented feature derived from raw feature subset $X_{S_j}$ via $Z_j = \phi_j(X_{S_j})$. The redundancy of $Z_j$ with respect to a trained tree ensemble $\mathcal{T}$ is:*

$$\rho(Z_j, \mathcal{T}) = \frac{\sum_{m=1}^{M} \mathbb{1}[X_{S_j} \text{ used in tree } m] \cdot \text{Gain}_m(X_{S_j})}{\text{Gain}_{\max}(Z_j)}$$

*If $\rho(Z_j, \mathcal{T}) \geq 1$, feature $Z_j$ is fully redundant. For the Hotel Booking Demand dataset, since the raw feature set contains 29 comprehensive attributes including lead_time, adr, adults, children, arrival_month, and previous_cancellations, the domain features (which are deterministic recombinations of these) will have $\rho \approx 1$, predicting minimal improvement for models capable of discovering such interactions.*

**Proof sketch.** In a gradient-boosted tree ensemble, the model partitions the feature space using axis-aligned splits. A domain feature $Z_j = \phi_j(X_{S_j})$ can be approximated by a sequence of splits on the constituent raw features $X_{S_j}$. For the HotelFeat domain features:

- **total_guests** = adults + children + babies: requires at most 2 splits to compute (sum of three values can be captured by sequential splits on each variable).
- **cancellation_rate** = previous_cancellations / (previous_cancellations + previous_bookings_not_cancelled): requires $O(\log(1/\epsilon))$ splits to approximate to error $\epsilon$.
- **weekend_ratio** = stays_in_weekend_nights / total_nights: similarly requires logarithmic splits.
- **month_sin/month_cos**: continuous transforms of arrival_month, requiring $O(\log(1/\epsilon))$ splits.

With trees of depth 6 and 300 trees, the ensemble has ample capacity to discover these patterns, yielding $\rho \approx 1$ for most domain features. $\square$

**Corollary 1.** For the Hotel Booking Demand dataset, the raw features lead_time, deposit_type, and previous_cancellations are among the top predictors of cancellation. Domain features that are functions of these (e.g., lead_time_squared, is_non_refundable, cancellation_rate) will have $\rho \approx 1$ because the ensemble has already captured their information through splits on the constituent variables.

**Corollary 2.** The categorization features (is_family, is_solo, is_high_price, is_low_price, is_peak_season) are many-to-one mappings from continuous raw features to discrete categories. These mappings necessarily lose information (by the data processing inequality [16]), meaning $\rho$ could be $< 1$ in theory. However, the loss is typically minimal because the categorization aligns with natural decision boundaries that the tree ensemble would discover anyway.

**Corollary 3.** Proposition 1 applies most directly to boosting models (XGBoost, LightGBM, CatBoost) that can selectively focus on informative features through the boosting process. For RandomForest, which uses random feature subsampling ($\sqrt{d'}$ features per split), adding redundant features increases the probability of sampling uninformative features at each split, potentially degrading performance. This predicts that domain feature augmentation may hurt RandomForest—a prediction confirmed by our experiments.

### 2.4 Model Architecture

We evaluate four tree-based models under two configurations:

**Raw configuration.** Each model is trained on the original 29 features.

**Domain configuration.** Each model is trained on the original features plus the 38 augmented domain features, yielding $d' = 67$ features.

The four models are:

1. **XGBoost** [3]: Regularized gradient boosting with second-order Taylor approximation, $\ell_1$ and $\ell_2$ regularization, sparsity-aware split finding, and the binary logistic objective.
2. **LightGBM** [4]: Gradient boosting with leaf-wise growth, GOSS for instance sampling, and EFB for feature bundling.
3. **CatBoost** [5]: Ordered boosting with oblivious (symmetric) trees and permutation-based target statistics for categorical feature handling—particularly relevant given the many categorical features in hotel booking data.
4. **RandomForest** [6]: Bootstrap-aggregated decision trees with $\sqrt{d}$ feature subsampling and majority voting.

### 2.5 Complexity Analysis

#### 2.5.1 Theoretical Complexity

Let $n$ be the number of training samples ($n \approx 95{,}512$ after 80/20 split), $d$ the number of features, $T$ the number of trees, $L$ the maximum leaves per tree, and $b$ the histogram bin count ($b \leq 255$).

**Training complexity per tree:**

- **XGBoost** (histogram-based): $O(n \cdot d \cdot b)$. With the approximate algorithm using quantile sketches: $O(n \cdot d \cdot k \log k)$ where $k$ is the number of quantile candidates.
- **LightGBM**: $O(n \cdot d_{\text{eff}} \cdot b)$ after EFB, with GOSS reducing to $O((n_{\text{top}} + n_{\text{rand}}) \cdot d_{\text{eff}} \cdot b)$.
- **CatBoost**: $O(n \cdot d \cdot b \cdot \log n)$ due to ordered boosting permutations. For $n \approx 100{,}000$, this is feasible but $\sim 17\times$ slower per tree than LightGBM.
- **RandomForest**: $O(T \cdot n \log n \cdot \sqrt{d})$ for $T$ fully grown trees.

**Domain augmentation overhead.** Feature computation: $O(n \cdot d')$, a one-time cost. The increase from $d = 29$ to $d' = 67$ increases per-tree training cost by $\sim 2.31\times$.

**Inference complexity.** Per-sample: $O(T \cdot \text{depth})$. Domain augmentation has negligible impact on inference time.

**Space complexity.** Feature matrix: $O(n \cdot d')$. For $n = 119{,}390$ and $d' = 67$, the feature matrix requires $\sim 64$ MB (float64). Tree storage: $O(T \cdot L \cdot d')$.

#### 2.5.2 Summary of Complexity

| Component | Raw | Domain | Ratio |
|-----------|-----|--------|-------|
| Feature computation | $O(n \cdot d)$ | $O(n \cdot d')$ | $\sim 2.31\times$ |
| Training (per tree) | $O(n \cdot d \cdot b)$ | $O(n \cdot d' \cdot b)$ | $\sim 2.31\times$ |
| Inference (per sample) | $O(T \cdot \text{depth})$ | $O(T \cdot \text{depth}')$ | $\sim 1.0$–$1.1\times$ |
| Space (feature matrix) | $O(n \cdot d)$ | $O(n \cdot d')$ | $\sim 2.31\times$ |

#### 2.5.3 Practical Performance Considerations

With $n = 119{,}390$ and $d' = 67$, the training cost per tree for LightGBM is approximately:

$$O(119{,}390 \times 67 \times 255) \approx 2.04 \times 10^9 \text{ operations}$$

With 300 trees, total training cost is $\sim 6.12 \times 10^{11}$ operations, which is feasible on standard hardware. The domain augmentation adds approximately 131% overhead relative to raw features, which is acceptable given that the prediction accuracy effects are model-dependent.

---

## 3. Experiments

### 3.1 Experimental Setup

**Dataset.** The Hotel Booking Demand dataset contains 119,390 reservation records from a city hotel and a resort hotel in Portugal. The dataset includes 29 raw features: hotel type, is_canceled (target), lead_time, arrival_date (year, month, week, day), stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, meal, country, market_segment, distribution_channel, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, reserved_room_type, assigned_room_type, booking_changes, deposit_type, agent, company, days_in_waiting_list, customer_type, adr, required_car_parking_spaces, total_of_special_requests, and reservation_status_date.

After removing irrelevant features (reservation_status, reservation_status_date, which leak the target), the dataset is split into 80% training (95,512 samples) and 20% testing (23,878 samples), stratified by the cancellation label.

**Domain features.** The augmented feature set includes 38 domain features across four families plus one cross-domain interaction:
- Guest composition (5): total_guests, has_children, adult_ratio, is_family, is_solo
- Booking patterns (22): total_nights, weekend_ratio, is_no_stay, lead_time_squared, is_long_lead, is_short_lead, is_same_day, total_previous, cancellation_rate, has_cancelled_before, has_booking_history, room_mismatch, has_special_requests, special_requests_squared, has_changes, booking_changes_squared, needs_parking, has_waited, wait_days_squared, is_online_ta, is_group, is_resort
- Temporal seasonality (5): arrival_month_num, is_peak_season, is_off_season, month_sin, month_cos
- Pricing categories (5): adr_squared, is_high_price, is_low_price, is_non_refundable, is_refundable
- Cross-domain interaction (1): lead_adr_interaction

Total features: 29 raw + 38 domain = 67.

**Models and hyperparameters.** Boosting models (XGBoost, LightGBM, CatBoost): n_estimators = 300, max_depth = 6, learning_rate = 0.1, binary logistic objective. RandomForest: n_estimators = 300, max_depth = 12. Categorical features are encoded with label encoding for all models. All models use default subsampling and colsample parameters.

**Evaluation metric.** AUC (Area Under the ROC Curve), computed using predict_proba on the test set.

**Reproducibility.** All experiments use 5 random seeds: [42, 123, 456, 789, 2024]. Results report mean ± standard deviation. Paired t-tests and Wilcoxon signed-rank tests assess significance. Cohen's d measures effect size. 95% confidence intervals are reported for all mean differences.

### 3.2 Main Results: Raw vs. Domain Feature Comparison

**Table 1: Main comparison results (AUC, mean ± std over 5 seeds)**

| Model | Raw AUC | Domain AUC | ΔAUC | Direction |
|-------|---------|------------|------|-----------|
| XGBoost | 0.9552±0.0011 | 0.9555±0.0010 | +0.0004 | Positive (n.s.) |
| LightGBM | 0.9544±0.0012 | 0.9548±0.0012 | +0.0004 | Positive (sig.) |
| CatBoost | 0.9511±0.0013 | 0.9521±0.0013 | +0.0010 | Positive (sig.) |
| RandomForest | 0.9406±0.0016 | 0.9348±0.0018 | -0.0057 | Negative (sig.) |

*Source: comprehensive_results.json, summary field. n.s. = not significant, sig. = significant at p < 0.05.*

**AUC values for Raw configuration:** XGBoost = 0.9552, LightGBM = 0.9544, CatBoost = 0.9511, RandomForest = 0.9406.

**AUC values for Domain configuration:** XGBoost = 0.9555, LightGBM = 0.9548, CatBoost = 0.9521, RandomForest = 0.9348.

**AUC improvement (ΔAUC):** XGBoost: ΔAUC = +0.0004 (not significant, p = 0.197), LightGBM: ΔAUC = +0.0004 (significant, p = 0.025), CatBoost: ΔAUC = +0.0010 (significant, p = 0.003), RandomForest: ΔAUC = -0.0057 (significant negative, p < 0.001).

The results reveal a mixed picture. For the three boosting models (XGBoost, LightGBM, CatBoost), domain features provide small positive improvements, with LightGBM and CatBoost reaching statistical significance. However, for RandomForest, domain features cause a substantial and highly significant performance decrease of 0.0057 AUC. This model-dependent effect is consistent with Corollary 3 of Proposition 1: RandomForest's random feature subsampling mechanism is vulnerable to noise dilution from redundant features, while boosting models can selectively ignore uninformative features through the gradient boosting process.

Among boosting models, CatBoost benefits most from domain features (ΔAUC = +0.0010, Cohen's d = 0.690, medium effect), likely because its ordered boosting and oblivious tree structure can leverage the explicitly encoded categorical indicators (is_non_refundable, is_online_ta, is_group). XGBoost shows the smallest and non-significant improvement (ΔAUC = +0.0004, p = 0.197, Cohen's d = 0.312), as its powerful regularization already captures the interaction patterns implicitly.

### 3.3 Ablation Study

We conduct component-level ablation by removing each of the 38 domain features individually from the full domain feature set, using XGBoost with 3 seeds [42, 123, 456]. The baseline (all 38 domain features) achieves AUC = 0.9556 ± 0.0012.

**Table 2: Ablation study results (top 15 features by absolute impact, XGBoost, 3 seeds)**

| Feature | Family | AUC (removed) | Δ from baseline | Effect |
|---------|--------|---------------|-----------------|--------|
| room_mismatch | Booking | 0.9550±0.0014 | -0.0006 | Helps |
| total_guests | Guest | 0.9560±0.0013 | +0.0004 | Hurts |
| month_sin | Temporal | 0.9553±0.0012 | -0.0003 | Helps |
| is_family | Guest | 0.9559±0.0009 | +0.0003 | Hurts |
| is_online_ta | Booking | 0.9554±0.0014 | -0.0002 | Helps |
| total_nights | Booking | 0.9554±0.0014 | -0.0002 | Helps |
| weekend_ratio | Booking | 0.9554±0.0014 | -0.0002 | Helps |
| is_peak_season | Temporal | 0.9554±0.0014 | -0.0002 | Helps |
| is_non_refundable | Pricing | 0.9554±0.0012 | -0.0002 | Helps |
| adr_squared | Pricing | 0.9555±0.0014 | -0.0001 | Helps |
| total_previous | Booking | 0.9555±0.0013 | -0.0001 | Helps |
| has_children | Guest | 0.9557±0.0013 | +0.0001 | Hurts |
| cancellation_rate | Booking | 0.9557±0.0016 | +0.0001 | Hurts |
| is_high_price | Pricing | 0.9557±0.0012 | +0.0001 | Hurts |
| is_off_season | Temporal | 0.9557±0.0014 | +0.0001 | Hurts |

*Source: comprehensive_results.json, ablation field. Baseline (all features) = 0.9556. "Helps" = removal decreases AUC (feature is beneficial). "Hurts" = removal increases AUC (feature is detrimental).*

The remaining 23 features show |Δ| < 0.0001, indicating neutral contribution. These include: is_no_stay, lead_time_squared, is_long_lead, is_short_lead, is_same_day, has_cancelled_before, has_booking_history, has_special_requests, special_requests_squared, has_changes, booking_changes_squared, needs_parking, has_waited, wait_days_squared, is_low_price, is_refundable, is_resort, is_solo, adult_ratio, arrival_month_num, month_cos, is_group, and lead_adr_interaction.

**Key ablation findings:**

1. **room_mismatch is the most beneficial domain feature** (Δ = -0.0006 when removed), capturing the discrepancy between reserved and assigned room types—a signal not directly available as a single raw feature.

2. **Guest composition features can hurt performance.** Removing total_guests (Δ = +0.0004) or is_family (Δ = +0.0003) actually improves AUC, suggesting these features introduce noise or redundancy that the model must work around. This is consistent with Theorem 1: total_guests = adults + children + babies is a deterministic sum that XGBoost can compute internally through sequential splits.

3. **Cyclical temporal encoding (month_sin) provides unique value** (Δ = -0.0003), as the sine transform captures monthly periodicity in a form not easily discoverable through axis-aligned splits on the raw arrival_month feature.

4. **Most domain features (23 of 38) are neutral**, confirming Proposition 1's prediction of $\rho \approx 1$ for features that are simple recombinations of raw inputs.

### 3.4 Parameter Sensitivity Analysis

We analyze sensitivity to two key hyperparameters—number of estimators ($T$) and maximum depth ($D$)—using XGBoost with domain features across 3 seeds [42, 123, 456]. The learning rate is fixed at 0.1.

**Table 3: Sensitivity analysis (AUC, mean ± std, XGBoost with domain features, 3 seeds)**

| n_est \ depth | 4 | 6 | 8 | 10 |
|---------------|-------|-------|-------|-------|
| 100 | 0.9363±0.0024 | 0.9461±0.0016 | 0.9528±0.0018 | 0.9568±0.0015 |
| 200 | 0.9448±0.0016 | 0.9528±0.0011 | 0.9579±0.0012 | 0.9610±0.0013 |
| 300 | 0.9484±0.0017 | 0.9556±0.0012 | 0.9601±0.0012 | 0.9622±0.0013 |
| 500 | 0.9519±0.0015 | 0.9583±0.0013 | 0.9618±0.0014 | 0.9633±0.0015 |

*Source: comprehensive_results.json, sensitivity field. Default configuration (n_est=300, depth=6) is shown in bold context.*

**Table 4: Parameter sensitivity summary with elasticity coefficients**

| Parameter | Range | Best Value | Best AUC | Elasticity | Sensitivity Level |
|-----------|-------|------------|----------|------------|-------------------|
| n_estimators | [100, 500] | 500 | 0.9633 | 0.0032 | Low |
| max_depth | [4, 10] | 10 | 0.9633 | 0.0097 | Low |

*Elasticity = (% change in AUC) / (% change in parameter). Low sensitivity: elasticity < 0.2.*

**Sensitivity findings:**

1. **Both parameters show low sensitivity** (elasticity < 0.01), indicating that XGBoost performance is robust to hyperparameter changes within the tested ranges.

2. **max_depth has a stronger effect than n_estimators** (elasticity 0.0097 vs. 0.0032), as deeper trees can capture more complex feature interactions.

3. **The best configuration (n_est=500, depth=10, AUC=0.9633)** achieves higher AUC than the default (n_est=300, depth=6, AUC=0.9556), suggesting that the domain features benefit from higher model capacity to exploit the additional interaction signals.

4. **Diminishing returns** are observed: increasing n_estimators from 300 to 500 at depth=10 yields only +0.0011 AUC, while increasing from 100 to 300 yields +0.0054.

### 3.5 Statistical Analysis

#### 3.5.1 Multi-Seed Experiments

**Table 5: Per-seed AUC results for all models and configurations**

| Seed | XGB Raw | XGB Dom | LGB Raw | LGB Dom | Cat Raw | Cat Dom | RF Raw | RF Dom |
|------|---------|---------|---------|---------|---------|---------|--------|--------|
| 42 | 0.9566 | 0.9572 | 0.9564 | 0.9569 | 0.9535 | 0.9543 | 0.9432 | 0.9374 |
| 123 | 0.9559 | 0.9553 | 0.9547 | 0.9547 | 0.9505 | 0.9519 | 0.9395 | 0.9340 |
| 456 | 0.9535 | 0.9543 | 0.9530 | 0.9532 | 0.9496 | 0.9504 | 0.9386 | 0.9321 |
| 789 | 0.9546 | 0.9551 | 0.9535 | 0.9542 | 0.9506 | 0.9513 | 0.9404 | 0.9349 |
| 2024 | 0.9553 | 0.9558 | 0.9545 | 0.9551 | 0.9513 | 0.9526 | 0.9410 | 0.9357 |

*Source: comprehensive_results.json, per_seed field.*

**Mean ± std AUC:** XGBoost: Raw = 0.9552±0.0011, Domain = 0.9555±0.0010; LightGBM: Raw = 0.9544±0.0012, Domain = 0.9548±0.0012; CatBoost: Raw = 0.9511±0.0013, Domain = 0.9521±0.0013; RandomForest: Raw = 0.9406±0.0016, Domain = 0.9348±0.0018.

#### 3.5.2 Paired Statistical Tests

**Table 6: Statistical test results (Domain vs. Raw, 5 seeds)**

| Model | t-statistic | p-value (t-test) | Wilcoxon p | Cohen's d | Effect Size | 95% CI [lower, upper] | Mean Diff | n_positive |
|-------|-------------|------------------|------------|-----------|-------------|----------------------|-----------|------------|
| XGBoost | 1.5441 | 0.1974 | 0.3125 | 0.3118 | Small | [-0.0001, 0.0008] | +0.0004 | 4/5 |
| LightGBM | 3.5045 | 0.0248 | 0.0625 | 0.3213 | Small | [0.0002, 0.0007] | +0.0004 | 5/5 |
| CatBoost | 6.4309 | 0.0030 | 0.0625 | 0.6897 | Medium | [0.0007, 0.0013] | +0.0010 | 5/5 |
| RandomForest | -25.9612 | 1.31×10⁻⁵ | 0.0625 | -3.0937 | Large (neg.) | [-0.0062, -0.0053] | -0.0057 | 0/5 |

*Source: comprehensive_results.json, statistical_tests field; statistical_tests.json. Degrees of freedom = 4 for all t-tests. n_positive = number of seeds where Domain > Raw.*

**Statistical findings:**

1. **XGBoost (t = 1.5441, p = 0.1974, d = 0.3118):** The improvement is not statistically significant at α = 0.05. Although 4 out of 5 seeds show positive improvement, the 95% CI [-0.0001, 0.0008] includes zero. The small effect size (d = 0.31) suggests a trend toward improvement that lacks sufficient statistical power with 5 seeds.

2. **LightGBM (t = 3.5045, p = 0.0248, d = 0.3213):** The improvement is statistically significant at α = 0.05. All 5 seeds show positive improvement (n_positive = 5/5), and the 95% CI [0.0002, 0.0007] excludes zero. However, the effect size is small (d = 0.32), indicating that the improvement, while consistent, is modest in magnitude.

3. **CatBoost (t = 6.4309, p = 0.0030, d = 0.6897):** The improvement is statistically significant at α = 0.01. All 5 seeds show positive improvement, and the 95% CI [0.0007, 0.0013] excludes zero. The medium effect size (d = 0.69) represents the largest positive benefit among all models, consistent with CatBoost's ability to leverage categorical indicator features through its ordered boosting mechanism.

4. **RandomForest (t = -25.9612, p = 1.31×10⁻⁵, d = -3.0937):** The decrease is highly significant (p < 0.001). Zero out of 5 seeds show positive improvement (n_positive = 0/5), and the 95% CI [-0.0062, -0.0053] is entirely negative. The large negative effect size (d = -3.09) indicates a substantial and consistent degradation. This result confirms Corollary 3: RandomForest's random feature subsampling is vulnerable to noise dilution from the 38 additional features, as $\sqrt{67} \approx 8.2$ features are sampled per split compared to $\sqrt{29} \approx 5.4$ for raw features, increasing the probability of selecting uninformative redundant features.

**Note on Wilcoxon tests:** The Wilcoxon signed-rank test yields p = 0.3125 for XGBoost and p = 0.0625 for the other three models. With only 5 paired observations, the Wilcoxon test has limited statistical power and cannot achieve p < 0.05 unless all differences have the same sign (which occurs for LGB, Cat, and RF, yielding p = 0.0625). The paired t-test, which uses the magnitude of differences, provides greater power and is preferred for this analysis.

### 3.6 Feature Importance Analysis

Based on the ablation study results (Section 3.3), we derive feature importance by ranking domain features according to their impact on AUC when removed.

**Table 7: Feature importance ranking (top 10 most impactful domain features)**

| Rank | Feature | Family | ΔAUC when removed | Importance |
|------|---------|--------|-------------------|------------|
| 1 | room_mismatch | Booking | -0.0006 | High |
| 2 | month_sin | Temporal | -0.0003 | Medium |
| 3 | is_online_ta | Booking | -0.0002 | Medium |
| 4 | total_nights | Booking | -0.0002 | Medium |
| 5 | weekend_ratio | Booking | -0.0002 | Medium |
| 6 | is_peak_season | Temporal | -0.0002 | Medium |
| 7 | is_non_refundable | Pricing | -0.0002 | Medium |
| 8 | adr_squared | Pricing | -0.0001 | Low |
| 9 | total_previous | Booking | -0.0001 | Low |
| 10 | adult_ratio | Guest | -0.0000 | Low |

*Source: comprehensive_results.json, ablation field. Importance based on |ΔAUC| when feature is removed from the full domain set.*

**Key observations:**

1. **room_mismatch** is the single most important domain feature, capturing the discrepancy between reserved and assigned room types. This feature encodes a relational pattern (inequality between two categorical variables) that is not trivially discoverable through axis-aligned tree splits on the individual room type features.

2. **Temporal features (month_sin, is_peak_season)** rank highly, suggesting that cyclical encoding of arrival month provides value beyond what the raw month number offers. The sine transform captures periodicity that axis-aligned splits cannot easily approximate.

3. **Booking pattern features (is_online_ta, total_nights, weekend_ratio)** provide moderate value, encoding market segment and stay duration information in forms that simplify the model's decision boundaries.

4. **Pricing features (is_non_refundable, adr_squared)** contribute modestly, with the deposit type indicator and nonlinear ADR transform offering small but consistent benefits.

5. **Guest composition features (total_guests, is_family) can hurt performance**, as their removal increases AUC. These features are simple deterministic sums and categorizations that the model can discover independently, and their explicit inclusion may introduce unnecessary splitting candidates.

### 3.7 Robustness Analysis

**Multi-seed stability.** The low standard deviations across 5 seeds (0.0010–0.0018 for all models and configurations) demonstrate high stability of the results. The coefficient of variation (std/mean) ranges from 0.10% to 0.19%, indicating that the AUC measurements are highly reproducible.

**Directional consistency.** The direction of domain feature effects is consistent across seeds:
- XGBoost: 4/5 seeds positive (80% consistency)
- LightGBM: 5/5 seeds positive (100% consistency)
- CatBoost: 5/5 seeds positive (100% consistency)
- RandomForest: 0/5 seeds positive (100% consistent negative)

This near-perfect directional consistency, particularly for CatBoost (all positive) and RandomForest (all negative), indicates that the observed effects are robust and not artifacts of specific random splits.

**Sensitivity to hyperparameters.** As shown in Section 3.4, both n_estimators and max_depth show low elasticity (< 0.01), confirming that the results are robust to hyperparameter variations within practical ranges.

### 3.8 Computational Performance

**Theoretical complexity.** As analyzed in Section 2.5, domain augmentation increases per-tree training cost by approximately 2.31× (from $d = 29$ to $d' = 67$). With 300 trees, the total training cost for XGBoost with domain features is approximately $6.12 \times 10^{11}$ operations.

**Practical considerations.** All experiments were conducted on a Windows 11 Professional system with an Intel Xeon W7-2595X CPU (24 cores, 2.5–4.8 GHz) and 48 GB DDR5 RDIMM memory. The full experimental pipeline (4 models × 5 seeds × 2 configurations = 40 runs, plus 38 ablation runs and 16 sensitivity runs) completed within practical timeframes. The 38 domain features add modest computational overhead for feature computation (one-time $O(n \cdot 38)$ cost) and increase memory usage by approximately 36 MB for the feature matrix.

**Inference performance.** Domain augmentation has negligible impact on inference time, as the tree depth and number of trees remain unchanged. Per-sample inference complexity is $O(T \cdot \text{depth})$ regardless of feature count.

### 3.9 Real-World Case Study

The Hotel Booking Demand dataset itself represents a real-world scenario: two actual hotels (a city hotel and a resort hotel) in Portugal with 119,390 genuine reservation records spanning July 2015 to August 2017.

**Practical scenario.** A revenue manager at a mid-size hotel wishes to predict which bookings are likely to be cancelled to optimize overbooking strategies. Using the raw booking attributes available in the property management system (PMS), they can achieve AUC = 0.9552 with XGBoost. Adding 38 domain-engineered features provides a marginal improvement to AUC = 0.9555—an increase of 0.0004 that is not statistically significant.

**Model selection matters more than feature engineering.** The choice of model has a larger impact on AUC than domain feature engineering: XGBoost (0.9552) outperforms RandomForest (0.9406) by 0.0146 AUC with raw features, while domain features provide at most 0.0010 AUC improvement (CatBoost). For RandomForest, domain features actually decrease performance by 0.0057 AUC.

**Deployment considerations.**
- **Data quality:** Domain features require accurate recording of all constituent raw features. Missing or erroneous values in adults, children, lead_time, or adr will propagate into domain features.
- **Computational resources:** The 38 domain features add modest overhead but require additional storage and preprocessing pipelines.
- **Maintenance cost:** Domain feature definitions must be updated if the underlying PMS schema changes, adding ongoing maintenance burden.
- **User acceptance:** The marginal improvements (≤ 0.001 AUC for boosting models) may not justify the added complexity in production systems.

**Ethical implications.** Cancellation prediction models may be used to implement differential overbooking policies. The use of domain features like is_online_ta (market segment indicator) and is_group (customer type) could introduce or amplify bias against certain customer segments. Fairness audits should be conducted before deployment.

---

## 4. Discussion

### 4.1 Key Findings

The experimental results reveal that domain feature augmentation produces model-dependent effects on hotel booking cancellation prediction, contradicting both the naive expectation of universal improvement and the theoretical prediction of zero effect.

**Mixed improvement for boosting models.** Across the three boosting models, domain features provide small positive improvements: XGBoost (+0.0004, not significant), LightGBM (+0.0004, significant, p = 0.025), and CatBoost (+0.0010, significant, p = 0.003). The effect sizes range from small (Cohen's d = 0.312 for XGBoost, d = 0.321 for LightGBM) to medium (d = 0.690 for CatBoost). These improvements, while statistically significant for LightGBM and CatBoost, are small in absolute magnitude (ΔAUC ≤ 0.001).

**Significant degradation for RandomForest.** Domain features cause a substantial and highly significant performance decrease for RandomForest (-0.0057, p < 0.001, d = -3.094). All 5 seeds show negative effects (0/5 positive), confirming that this is a robust finding, not an artifact. The 95% CI [-0.0062, -0.0053] is entirely negative, ruling out any possibility of positive effect.

**Theoretical reconciliation.** Theorem 1 predicts zero informational gain from deterministic transformations, which might suggest zero AUC improvement. However, the theorem addresses mutual information, not practical model performance. Small AUC improvements can arise from approximation efficiency: explicitly engineered features help models discover patterns with fewer splits, even though no new information is added. Conversely, for RandomForest, the additional features introduce noise through the random subsampling mechanism, degrading performance—a practical effect that the information-theoretic analysis does not directly predict but that Corollary 3 of Proposition 1 anticipates.

### 4.2 Why Domain Features Show Mixed Effects

The model-dependent effects can be explained by three converging factors:

1. **Information-theoretic ceiling (Theorem 1).** Deterministic transformations of existing features cannot increase mutual information with the target. The domain features are all functions of raw features, so $\Delta I = 0$. This constrains the maximum possible improvement to what approximation efficiency can provide—typically very small for models with sufficient capacity.

2. **Model-specific interaction with feature redundancy.**
   - **Boosting models** (XGBoost, LightGBM, CatBoost) can selectively ignore uninformative features through the gradient boosting process. When a domain feature is redundant with raw features, the boosting algorithm simply assigns it low importance and rarely splits on it. This explains why domain features do not hurt boosting models—the redundant features are effectively ignored.
   - **RandomForest** uses random feature subsampling ($\sqrt{d'}$ features per split). Adding 38 features increases the feature pool from 29 to 67, increasing $\sqrt{d'}$ from 5.4 to 8.2. This means that each split considers more features, but many of the additional features are redundant or uninformative. The probability of selecting an informative feature at each split decreases, leading to weaker individual trees and degraded ensemble performance.

3. **CatBoost's unique advantage.** CatBoost benefits most from domain features (d = 0.690, medium effect) because its ordered boosting with oblivious trees can leverage the explicitly encoded categorical indicators (is_non_refundable, is_online_ta, is_group). While CatBoost has native categorical handling, the binary indicator features simplify the split decisions and reduce the need for complex target statistics computation.

### 4.3 Comparison with Related Work

Our results are consistent with the literature, where tree-based models typically achieve AUC in the range of 0.85–0.95 on Hotel Booking Demand. The AUC values we observe (0.9348–0.9555) are at the upper end of this range, reflecting the comprehensive 29-feature raw set and well-tuned models. The mixed effects of domain features align with the observation that the dataset's original features are already well-suited for tree-based models, but our findings add nuance: the effect is not universally negligible but depends on the model architecture.

The RandomForest degradation we observe (-0.0057 AUC) is a finding not previously reported in the hospitality analytics literature, to our knowledge. This highlights the importance of evaluating domain features across multiple model architectures rather than a single model.

### 4.4 Practical Implications

For hospitality data scientists and revenue managers, our findings provide nuanced guidance:

1. **Evaluate domain features per model, not universally.** Domain features are not universally beneficial or negligible—their effect depends on the model architecture. A feature set that helps CatBoost may hurt RandomForest. Empirical validation for each target model is essential.

2. **Boosting models are preferred for domain-augmented features.** XGBoost, LightGBM, and CatBoost can safely incorporate domain features without performance degradation, with CatBoost showing the largest benefit. RandomForest should be used with raw features only.

3. **Prioritize model selection over feature engineering.** The choice of model (XGBoost at 0.9552 vs. RandomForest at 0.9406) has a 4× larger impact on AUC than domain feature engineering (at most +0.0010 for CatBoost).

4. **Select domain features judiciously.** The ablation study shows that only a few domain features (room_mismatch, month_sin, is_online_ta, total_nights) provide measurable benefits, while others (total_guests, is_family) can hurt performance. A curated subset of high-impact features may be more effective than the full set.

5. **Consider external data for meaningful improvement.** Since domain features derived from existing data provide at most +0.001 AUC improvement, meaningful gains require external information: weather forecasts, local event calendars, competitor pricing, or macroeconomic indicators.

### 4.5 Limitations

1. **Single dataset.** Results are based on the Hotel Booking Demand dataset from two Portuguese hotels. Generalization to other hotels, regions, and market segments requires validation.
2. **Binary classification.** The task is binary (cancelled vs. not cancelled). More granular prediction (e.g., cancellation timing, partial cancellation) might benefit differently from domain features.
3. **Temporal scope.** The data covers July 2015 to August 2017. Cancellation patterns may have shifted post-COVID-19, as noted by Almeida et al. [29].
4. **Feature design scope.** Our domain features are designed to be derivable from the raw data alone. Features incorporating external data could provide genuine informational gain ($\Delta I > 0$).
5. **Model scope.** We evaluate only tree-based models. Neural network architectures (e.g., TabNet, FT-Transformer) might interact differently with domain features, potentially benefiting more from explicit interaction encoding.
6. **Limited seed count.** With 5 seeds, the statistical power is limited. The XGBoost result (p = 0.197) might reach significance with more seeds, given that 4/5 seeds show positive improvement.
7. **Ablation scope.** The ablation study uses only XGBoost with 3 seeds. Feature importance may differ for other models, particularly RandomForest where domain features hurt overall performance.

### 4.6 Ethical and Social Implications

Hotel cancellation prediction has ethical dimensions that warrant discussion:

1. **Consumer fairness.** Cancellation prediction models may be used to implement differential overbooking policies that disproportionately affect certain customer segments (e.g., guests from specific countries, booking through certain channels). Domain features like is_online_ta and is_group explicitly encode market segment and customer type, which could amplify such biases. Fairness audits should be conducted to ensure equitable treatment.

2. **Transparency.** Hotels should be transparent about their cancellation prediction practices, particularly when these influence deposit requirements or booking acceptance decisions.

3. **Data privacy.** The Hotel Booking Demand dataset includes country of origin and other potentially sensitive attributes. Production systems must comply with data protection regulations (e.g., GDPR) and minimize the collection of unnecessary personal data.

4. **Economic impact.** Overbooking strategies informed by cancellation prediction can lead to denied bookings, which have real economic and emotional costs for travelers. Models should be calibrated to minimize false positives (predicting cancellation when the guest intends to arrive).

---

## 5. Conclusion

This paper presented HotelFeat, a hospitality domain feature analysis framework for hotel booking cancellation prediction on the Hotel Booking Demand dataset. We constructed four families of domain features—guest composition, booking patterns, temporal seasonality, and pricing categories—totaling 38 engineered features, and evaluated them across four tree-based models with rigorous statistical validation.

The theoretical analysis (Theorem 1 and Proposition 1) established that deterministic transformations of existing features yield zero informational gain, and that domain features become fully redundant when the original feature set is already comprehensive. Corollary 3 predicted that domain features may hurt RandomForest due to noise dilution in random feature subsampling.

The experimental results revealed a nuanced picture that partially confirms and partially extends these theoretical predictions. Domain features provided small but statistically significant AUC improvements for LightGBM (+0.0004, p = 0.025, d = 0.321) and CatBoost (+0.0010, p = 0.003, d = 0.690), a non-significant positive trend for XGBoost (+0.0004, p = 0.197, d = 0.312), and a significant performance decrease for RandomForest (-0.0057, p < 0.001, d = -3.094). The ablation study identified room_mismatch, month_sin, and total_nights as the most beneficial domain features, while total_guests and is_family were found to hurt performance. Parameter sensitivity analysis confirmed low sensitivity to n_estimators and max_depth (elasticity < 0.01).

These findings provide a clear conclusion: domain feature engineering produces model-dependent effects that must be empirically validated rather than assumed. Boosting models can safely benefit from domain features, while RandomForest is harmed by them. Future research should focus on: (1) incorporating external data sources (weather, events, competitor pricing) that can provide genuine informational gain; (2) evaluating domain features with neural network architectures that may be less capable of automatic interaction discovery; (3) extending the analysis to multi-hotel and cross-cultural settings; (4) developing fairness-aware cancellation prediction models that mitigate discrimination; and (5) investigating the interaction between domain features and operational decision-making in real-time revenue management systems.

---

## References

[1] N. Antonio, A. de Almeida, and L. Nunes, "Hotel booking demand datasets," *Data in Brief*, vol. 22, pp. 41-49, 2019.

[2] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189-1232, 2001.

[3] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, 2016, pp. 785-794.

[4] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 3146-3154.

[5] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: Unbiased boosting with categorical features," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018, pp. 6638-6648.

[6] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.

[7] A. J. Sanchez-Medina and J. C-Sanchez, "Using gradient boosting and neural networks to predict hotel booking cancellations," *International Journal of Hospitality Management*, vol. 94, art. 102865, 2021.

[8] X. Chen, Y. Wang, and Z. Liu, "Deep learning with attention mechanisms for hotel cancellation prediction," *Expert Systems with Applications*, vol. 184, art. 115462, 2021.

[9] R. Nair, A. Gupta, and S. Tewari, "A comparative study of machine learning classifiers for hotel cancellation prediction," *Journal of Hospitality and Tourism Technology*, vol. 12, no. 3, pp. 521-538, 2021.

[10] J. Li, T. Wang, and X. Zhao, "Time-series forecasting of aggregate hotel cancellations with seasonal decomposition," *Tourism Management*, vol. 85, art. 104298, 2021.

[11] T. Dogru, M. Zhang, and E. Ozdemir, "The impact of deposit policies on hotel cancellation behavior: An empirical analysis," *International Journal of Contemporary Hospitality Management*, vol. 33, no. 7, pp. 2448-2468, 2021.

[12] Y. Wang, S. Chen, and J. Liu, "Multi-task learning for joint hotel cancellation and no-show prediction," *Knowledge-Based Systems*, vol. 243, art. 108493, 2022.

[13] L. Zhang, H. Zhao, and Y. Sun, "Temporal feature engineering for hotel demand forecasting with gradient boosting," *IEEE Access*, vol. 9, pp. 134567-134580, 2021.

[14] X. Huang, J. Chen, and M. Li, "Guest segmentation features for personalized hotel recommendation," *Journal of Electronic Commerce Research*, vol. 23, no. 2, pp. 112-128, 2022.

[15] M. Lado-Sestayo, M. Vivel-Búa, and I. Otero-González, "Pricing-based features for hotel performance analysis: A machine learning approach," *Tourism Economics*, vol. 28, no. 4, pp. 987-1008, 2022.

[16] T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed. Hoboken, NJ: Wiley-Interscience, 2006.

[17] G. Hooker, "Generalized functional ANOVA diagnostics for high-dimensional functions of dependent variables," *Journal of Computational and Graphical Statistics*, vol. 16, no. 3, pp. 709-732, 2007.

[18] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 4765-4774.

[19] S. M. Lundberg, G. G. Erion, and S.-I. Lee, "Consistent individualized feature attribution for tree ensembles," *arXiv preprint arXiv:1802.03888*, 2019.

[20] L. Zhang, Q. Wang, and H. Wei, "Interpretable hotel cancellation prediction using SHAP-based analysis," *Annals of Tourism Research*, vol. 95, art. 103425, 2022.

[21] R. Ali, S. Lee, and Y. Kim, "Explainable AI for hospitality analytics: A SHAP-based approach," *International Journal of Information Management*, vol. 63, art. 102458, 2022.

[22] A. Guizzardi, G. Stanghellini, and F. Pellegrini, "Probabilistic hotel cancellation forecasting with time-varying effects," *Tourism Management*, vol. 88, art. 104413, 2022.

[23] X. Huang, J. Zhang, and L. Wang, "Transformer-based sequential models for hotel booking cancellation prediction," *Neural Networks*, vol. 160, pp. 338-351, 2023.

[24] T. Zheng, M. Li, and W. Chen, "Graph neural networks for hotel recommendation and cancellation prediction," in *Proc. ACM Conf. Recommender Systems (RecSys)*, 2023, pp. 567-575.

[25] A. Bagheri, R. Hosseini, and M. Khaleghi, "Fairness in hotel cancellation prediction: A comparative analysis," in *Proc. AAAI/ACM Conf. AI, Ethics, and Society (AIES)*, 2023, pp. 234-242.

[26] Y. Jiang, X. Li, and T. Chen, "Ensemble stacking for hotel cancellation prediction with diverse base learners," *Expert Systems with Applications*, vol. 217, art. 119535, 2023.

[27] S. Kim, J. Park, and H. Lee, "Real-time cancellation prediction system for hotel revenue management," *IEEE Transactions on Knowledge and Data Engineering*, vol. 35, no. 8, pp. 7892-7906, 2023.

[28] T. Pham, H. Nguyen, and V. Tran, "Cross-hotel transfer learning for cancellation prediction with domain adaptation," *Knowledge-Based Systems*, vol. 268, art. 110456, 2023.

[29] P. Almeida, R. Silva, and J. Costa, "The impact of COVID-19 on hotel cancellation patterns: A longitudinal analysis," *International Journal of Hospitality Management*, vol. 112, art. 103726, 2023.

[30] V. Schetinger, M. Oliveira, and E. Mansour, "Interpretable rule-based models for hotel cancellation prediction," *Decision Support Systems*, vol. 165, art. 113869, 2023.

[31] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[32] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. New York: Springer, 2009.

[33] C. R. Shalizi, *Advanced Data Analysis from an Elementary Point of View*. Cambridge: Cambridge University Press, 2019.

[34] P. Hall, N. Gill, and A. Cox, *An Introduction to Machine Learning Interpretability*, 2nd ed. Sebastopol, CA: O'Reilly Media, 2022.
