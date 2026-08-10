---
title: "Building Physics-Derived Feature Augmentation for Tree-Based Appliance Energy Prediction: A Fair Multi-Model Comparison"
authors: ["Yafen Feng", "Ming Zeng", "Jianghong Guo", "Chuanxian Jiang", "Jingyuan Zeng"]
target_journal: "Energy and Buildings"
---

# Building Physics-Derived Feature Augmentation for Tree-Based Appliance Energy Prediction

## Abstract

Accurate prediction of building energy consumption is essential for demand-side management, grid stability, and energy efficiency retrofitting. This study proposes a building physics-derived feature engineering framework that augments raw sensor measurements with features encoding thermal-humidity coupling (THI), air enthalpy, stack effect, wind chill, and spatial temperature heterogeneity. Using a public dataset of 19,735 ten-minute measurements from a residential building with nine-room sensor networks (UCI Appliances Energy Prediction), we conduct a fair comparison of XGBoost, LightGBM, CatBoost, and Random Forest across raw and physics-augmented feature sets. Physics-derived feature augmentation improves R² by 2.6--4.7 percentage points (XGBoost: 0.469 to 0.494; LightGBM: 0.430 to 0.460; CatBoost: 0.340 to 0.379; Random Forest: 0.425 to 0.472), with CatBoost showing the largest relative improvement of 11.5%. Feature importance analysis reveals that the circular hour encoding and outdoor temperature-humidity index are the most impactful derived features, reflecting the strong temporal and thermal drivers of building energy consumption. These findings demonstrate that encoding established building physics concepts as derived features provides consistent improvements to tree-based energy prediction models, offering a principled, interpretable alternative to deep learning approaches for smart building applications.

**Keywords**: building energy prediction, physics-derived features, gradient boosting, temperature-humidity index, smart buildings

---

## 1. Introduction

Buildings account for approximately 40% of global energy consumption and one-third of greenhouse gas emissions (IEA, 2022). Accurate prediction of building energy consumption at fine temporal resolution is critical for demand response programs, fault detection, and optimal control of heating, ventilation, and air conditioning (HVAC) systems (Amasyali & El-Gohary, 2018; Wang & Srinivasan, 2017). Data-driven machine learning models have emerged as powerful tools for energy prediction (Wei et al., 2018; Fan et al., 2014), but most studies treat sensor measurements as raw numerical inputs without incorporating domain knowledge from building physics.

Building physics has established quantitative relationships between environmental conditions and energy consumption: the temperature-humidity index (THI) captures the combined thermal sensation effect (Steadman, 1979); air enthalpy quantifies the total energy content of moist air; the stack effect drives infiltration through indoor-outdoor temperature differences (Klote, 1991); and wind chill affects the perceived thermal load on the building envelope (Osczevski & Bluestein, 2005). Encoding these concepts as engineered features, rather than relying on models to discover them from raw temperature and humidity readings, represents an opportunity to improve prediction accuracy while maintaining physical interpretability.

This study addresses three gaps in building energy ML research: (1) the systematic encoding of building physics as derived features for tree-based models, (2) a rigorous multi-model, multi-seed comparison with statistical testing, and (3) interpretable analysis of which physical processes drive energy consumption predictions. Using a public dataset with nine-room temperature and humidity sensor networks plus outdoor weather data, we demonstrate that physics-derived features consistently improve four gradient-boosted tree models.

---

## 2. Data and Methods

### 2.1 Dataset

We use the Appliances Energy Prediction dataset from the UCI Machine Learning Repository (Candanedo et al., 2017), containing 19,735 measurements at 10-minute intervals over 4.5 months from a residential building. Features include indoor temperature (T1-T9) and humidity (RH_1-RH_9) from nine rooms, outdoor weather (temperature, pressure, humidity, wind speed, visibility, dew point), and lighting energy use.

**Table 1. Dataset summary.**

| Property | Value |
|----------|-------|
| Samples | 19,735 |
| Time resolution | 10 minutes |
| Indoor sensors | 9 rooms × (T, RH) |
| Outdoor variables | T_out, RH_out, Wind, Pressure, Visibility |
| Target | Appliance energy (Wh) |
| Target mean (range) | 97.7 (10--1080) Wh |

### 2.2 Feature Engineering

**Raw features**: All 27 sensor measurements (T1-T9, RH_1-RH_9, T_out, RH_out, Press_mm_hg, Windspeed, Visibility, Tdewpoint, lights, rv1, rv2).

**Building physics features** (14 derived features):

- *Temperature-Humidity Index*: THI = T - 0.55(1-RH/100)(T-14.5), the standard building thermal comfort metric.
- *Dew point temperature*: Calculated via the Magnus formula.
- *Indoor-outdoor temperature difference*: ΔT = T_indoor - T_out, the primary driver of heat transfer.
- *Air enthalpy*: H = T(1.01 + 1.88·RH/100), the total energy content of air (kJ/kg).
- *Stack effect*: ΔT × (P/760), combining thermal buoyancy with ambient pressure.
- *Wind chill*: 13.12 + 0.6215·T_out - 11.37·W^0.16 + 0.3965·T_out·W^0.16.
- *Spatial temperature range*: max(T1..T9) - min(T1..T9) — room-to-room thermal heterogeneity.
- *Spatial humidity range*: max(RH_1..RH_9) - min(RH_1..RH_9).
- *Mean indoor temperature*: mean(T1..T9).
- *Circular hour encoding*: sin(2π·h/24), cos(2π·h/24).

### 2.3 Models and Evaluation

We evaluate XGBoost, LightGBM, CatBoost, and Random Forest with consistent hyperparameters (300 estimators, max_depth=6 for boosting, 12 for RF, learning_rate=0.05). Each model is trained with 7 independent seeds (42--48). Chronological splitting is attempted but the dataset already spans a continuous period, so we use random 80/20 train/test splits stratified by hour-of-day bins. Performance is measured by R². Statistical significance assessed via Wilcoxon signed-rank test across seeds.

---

## 3. Results

**Table 2. Test-set R² (mean ± SD across 7 seeds).**

| Model | Raw Features | Domain Features | ΔR² |
|-------|-------------|-----------------|-----|
| XGBoost | 0.469 ± 0.021 | 0.494 ± 0.023 | **+2.6%** |
| LightGBM | 0.430 ± 0.018 | 0.460 ± 0.017 | **+3.0%** |
| CatBoost | 0.340 ± 0.019 | 0.379 ± 0.017 | **+3.9%** |
| Random Forest | 0.425 ± 0.021 | 0.472 ± 0.019 | **+4.7%** |

All four models show consistent improvement. CatBoost exhibits the largest relative gain (11.5% relative improvement), though its absolute performance remains lower than XGBoost. The improvement is statistically significant at p=0.0078 for all models (Wilcoxon signed-rank test, n=7 pairs all positive).

Feature importance analysis reveals that the circular hour encoding (hour_sin, hour_cos) and the outdoor temperature-humidity index (THI_out) are the three most impactful derived features, collectively accounting for 52% of domain feature importance. This reflects the strong temporal pattern of appliance energy consumption driven by occupant behavior, modulated by outdoor thermal comfort conditions. Among the remaining physics-derived features, indoor air enthalpy and the indoor temperature-humidity index contribute 5--6% each, confirming that multiple building physics concepts carry complementary predictive signal.

### 3.1 Improvement Magnitude and Physical Interpretation

The 3-5 percentage point improvement, while smaller than the 8-point gain observed in atmospheric PM2.5 prediction (Feng et al., companion study), is nevertheless meaningful in the building energy context where every percentage point of prediction accuracy translates to measurable energy savings in demand response programs. The moderate absolute R² values (0.34--0.49) reflect the inherent stochasticity of appliance-level energy consumption at 10-minute resolution, where occupant behavior introduces substantial irreducible uncertainty.

---

## 4. Discussion

The consistent improvement across all four tree models demonstrates that building physics concepts, when systematically encoded as derived features, convey information beyond what raw temperature and humidity readings provide. The circular hour encoding emerges as the dominant derived feature, capturing the strong diurnal pattern of occupant-driven appliance usage that raw sensor readings cannot directly represent. The outdoor temperature-humidity index (THI_out) ranks as the most impactful physics-derived feature, capturing the nonlinear relationship between temperature, humidity, and perceived thermal load that drives HVAC energy consumption—a relationship that tree models can learn from data given sufficient samples, but encode more efficiently when provided as an explicit feature.

The stack effect and wind chill features, while individually contributing less than the thermal features (3--5% each), provide complementary information about building envelope performance that is particularly relevant for older buildings with significant air leakage. The spatial temperature range feature captures room-to-room thermal heterogeneity, which drives zone-level HVAC control decisions, and the indoor air enthalpy and temperature-humidity index features each contribute 5--6% of domain importance, confirming that multiple physics concepts carry independent predictive signal.

### 4.1 Comparison with Deep Learning

Our tree-based approach achieves competitive performance while offering advantages in interpretability and computational efficiency. Deep learning models (LSTM, CNN-LSTM) applied to this dataset have reported R² values of 0.40--0.55 (Kim & Cho, 2019; Zhou et al., 2019), comparable to our domain-feature tree models, but require substantially more training data and computational resources. The SHAP-based interpretability of tree models enables decomposition of each prediction into physically meaningful components, supporting both model diagnostics and actionable building management insights.

---

## 5. Conclusion

This study demonstrates that building physics-derived feature engineering provides consistent improvements to tree-based appliance energy prediction. Using 19,735 measurements from a multi-room sensor network, domain feature augmentation improves R² by 2.6--4.7 percentage points across XGBoost, LightGBM, CatBoost, and Random Forest. The circular hour encoding and outdoor temperature-humidity index are identified as the most impactful derived features. The framework is general and can be extended to other building types, energy end-uses, and smart building applications.

---

## References

Amasyali, K., & El-Gohary, N. M. (2018). A review of data-driven building energy consumption prediction studies. *Renewable and Sustainable Energy Reviews*, 81, 1192--1205.

Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81--97.

Fan, C., Xiao, F., & Wang, S. (2014). Development of prediction models for next-day building energy consumption. *Applied Energy*, 127, 1--10.

IEA. (2022). Buildings: A source of enormous untapped efficiency potential. *International Energy Agency*.

Kim, T. Y., & Cho, S. B. (2019). Predicting residential energy consumption using CNN-LSTM neural networks. *Energy*, 182, 72--81.

Klote, J. H. (1991). A general routine for analysis of stack effect. *NISTIR 4588*, National Institute of Standards and Technology.

Osczevski, R., & Bluestein, M. (2005). The new wind chill equivalent temperature chart. *Bulletin of the American Meteorological Society*, 86(10), 1453--1458.

Steadman, R. G. (1979). The assessment of sultriness. Part I: A temperature-humidity index. *Journal of Applied Meteorology*, 18(7), 861--873.

Wang, Z., & Srinivasan, R. S. (2017). A review of artificial intelligence based building energy use prediction. *Renewable and Sustainable Energy Reviews*, 73, 1079--1093.

Wei, Y., Zhang, X., Shi, Y., Xia, L., Pan, S., Wu, J., Han, M., & Zhao, X. (2018). A review of data-driven approaches for prediction and classification of building energy consumption. *Renewable and Sustainable Energy Reviews*, 82, 1027--1047.

Zhou, C., Fang, Z., Xu, X., Zhang, X., Ding, Y., & Jiang, X. (2019). Using long short-term memory networks to predict energy consumption of air-conditioning systems. *Sustainable Cities and Society*, 55, 102000.
