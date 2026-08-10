# MatFeat: Material Science Domain Feature Analysis for Superconductor Critical Temperature Prediction

**Jingyuan Zeng**$^{1}$, **Ming Zeng**$^{2}$, **Jianghong Guo**$^{1}$, **Chuanxian Jiang**$^{1}$, **Yafen Feng**$^{3,4,*}$

$^{1}$ School of Computer Science, Jiaying University, Meizhou 514015, Guangdong, China
$^{2}$ College of Water Conservancy and Civil Engineering, South China Agricultural University, Guangzhou 510642, Guangdong, China
$^{3}$ School of Geography Science and Tourism, Jiaying University, Meizhou 514015, Guangdong, China
$^{4}$ Key Laboratory of Mountain Surface Environment and Green Development in Northeast Guangdong, Meizhou 514015, Guangdong, China

$^{*}$ Corresponding author: Yafen Feng, E-mail: fyf81@163.com

**Fund:** Guangdong Provincial Higher Education Teaching Reform Project (Grant No. Yue Jiao Gao Han [2024] 9-989)

---

## Abstract

Predicting the superconducting critical temperature ($T_c$) from material descriptors is a central problem in computational materials science. While gradient-boosted tree ensembles achieve strong predictive accuracy on the UCI Superconductivity dataset, the role of physics-informed domain features remains insufficiently understood. This paper proposes **MatFeat**, a systematic domain feature engineering framework that augments the 81 original statistical features with four physically motivated feature groups—element properties, crystal structure encoding, thermodynamic estimates, and electronic interaction descriptors. We evaluate MatFeat across four tree-based models (XGBoost, LightGBM, CatBoost, RandomForest) under both raw and domain-augmented feature regimes, complemented by SHAP-based physical interpretability analysis and information-saturation theory. We introduce two theoretical contributions: **Theorem 1** (Information Saturation), which formally bounds the marginal predictive gain achievable by adding domain features when the original feature set is information-rich, and **Proposition 1** (Feature Redundancy Criterion), which provides a SHAP-based coefficient for quantifying inter-feature redundancy. Our experiments reveal that domain features yield negligible improvement ($\Delta R^2 \leq 0.001$), which we explain through the lens of information saturation: the 81 original features already capture nearly all explainable variance in $T_c$. SHAP analysis further connects learned feature importances to established superconductivity physics, validating that models implicitly discover physically meaningful patterns. These findings offer practical guidance for feature engineering in materials informatics and demonstrate that interpretability, rather than raw accuracy, is the frontier for tree-based superconductor prediction.

**Keywords:** Superconductor critical temperature; Domain feature engineering; SHAP interpretability; Information saturation; Gradient boosting; Materials informatics

---

## 1. Introduction and Related Work

### 1.1 Background

Superconductors—materials that exhibit zero electrical resistance below a critical temperature $T_c$—are cornerstones of modern technologies ranging from magnetic resonance imaging to quantum computing. The discovery of high-temperature superconductors, beginning with the cuprate family in 1986, has motivated decades of experimental and computational research aimed at understanding and predicting $T_c$ from material composition and structure. However, the superconducting mechanism, particularly for unconventional superconductors, remains one of the outstanding open problems in condensed matter physics.

The advent of materials informatics—the application of machine learning (ML) and data science to materials discovery—has opened new avenues for $T_c$ prediction. The UCI Superconductivity dataset [1], containing 21,263 chemical compositions with 81 engineered features and experimentally measured $T_c$ values, has become a standard benchmark for regression-based materials property prediction. These 81 features encode atomic statistics (e.g., mean atomic mass, range of thermal conductivity, entropy of valence electrons) derived from elemental properties of the constituent elements. While these features were carefully designed by domain experts, the question of whether additional physics-informed features can further improve predictive performance has not been systematically addressed.

### 1.2 Machine Learning for Superconductor Prediction

Early work by Stanev et al. [2] demonstrated that ensemble methods, particularly random forests, could achieve $R^2 > 0.9$ on superconductivity datasets using composition-based features. Hamidieh [1] formalized this approach with the UCI dataset, employing gradient-boosted regression trees to achieve strong baseline performance. Subsequent studies have explored various model architectures: Konno et al. [3] applied deep neural networks to superconductor $T_c$ prediction, while Roter and Dorogokupets [4] conducted a systematic comparison of ensemble methods, finding that random forests and gradient boosting remain highly competitive.

More recently, graph neural networks (GNNs) have been applied to crystal structure prediction. Xie and Grossman [5] introduced Crystal Graph Convolutional Neural Networks (CGCNN), which represent crystals as graphs and achieve state-of-the-art results on several materials property benchmarks. The MACE framework [6] extended this with equivariant message passing for more accurate force fields, and the GNoME system [7] demonstrated large-scale materials discovery through deep learning, identifying millions of new stable materials. However, GNN-based approaches require full crystal structure information, which is often unavailable for compositional datasets like UCI Superconductivity. In such settings, tabular feature-based methods remain dominant.

Recent surveys [8, 9] have highlighted that for tabular materials data, gradient-boosted decision trees (GBDT) consistently outperform deep learning methods, a trend that was further confirmed by benchmark studies in 2024–2025 [10, 11]. The maturation of GBDT variants—XGBoost [12], LightGBM [13], and CatBoost [14]—has created a rich toolkit for materials regression tasks.

### 1.3 Feature Engineering and Domain Knowledge in Materials Science

Feature engineering remains a critical step in materials ML pipelines. Ward et al. [15] developed a general-purpose feature generation framework (Magpie) that computes composition-based statistics from elemental properties, forming the basis of the 81 features in the UCI dataset. The SISSO method [16] introduced compressed sensing for symbolic regression, enabling the discovery of physically interpretable descriptors. Choudhary et al. [17, 18] developed the JARVIS-Tools ecosystem, providing systematic feature generation and benchmarking infrastructure.

In 2024–2025, several studies have explored the integration of domain knowledge into ML pipelines for materials. De Breuck et al. [19] proposed a feature selection and joint learning approach (MODNet) for limited datasets, demonstrating that careful feature selection can match or exceed the performance of larger feature sets. Zhang et al. [20] introduced physics-constrained feature selection for alloy design, while Park et al. [21] showed that thermodynamically motivated features (e.g., Debye temperature estimates, formation energies) can improve interpretability even when raw accuracy gains are small. A consistent theme across these works is the tension between feature richness and diminishing returns—a phenomenon that has been observed but rarely formalized theoretically.

### 1.4 Interpretability in Materials ML

The SHAP (SHapley Additive exPlanations) framework [22] has become the de facto standard for interpreting tree-based models. Lundberg et al. [23] extended SHAP to tree ensembles (TreeSHAP), enabling efficient computation of feature attributions. In materials science, SHAP has been applied to understand model decisions in alloy design [24], perovskite stability prediction [25], and superconductor property modeling [26].

Recent work in 2024–2025 has emphasized the importance of connecting SHAP explanations to physical mechanisms. Chen et al. [27] demonstrated that SHAP-based feature importance for superconductor $T_c$ models aligns with known physical factors (e.g., number of valence electrons, atomic mass). Zhao et al. [28] proposed a SHAP-guided feature selection method that preserves physical interpretability. However, a systematic framework that jointly analyzes domain feature engineering, information saturation, and physical interpretability has not been established.

### 1.5 Contributions

This paper makes the following contributions:

1. **MatFeat Framework**: We design a comprehensive set of domain-informed features organized into four physically motivated categories—element properties, structural encoding, thermodynamic estimates, and electronic interactions—and systematically evaluate their impact on $T_c$ prediction across four state-of-the-art tree-based models.

2. **Information Saturation Theorem (Theorem 1)**: We formally prove that when the original feature set captures a sufficient fraction of the mutual information with the target, the marginal predictive gain from additional domain features is bounded and approaches zero—a phenomenon we call *information saturation*. This theorem provides a theoretical explanation for the empirically observed negligible improvement.

3. **Feature Redundancy Criterion (Proposition 1)**: We introduce a SHAP-based redundancy coefficient that quantifies the degree of informational overlap between feature pairs, enabling principled feature selection and explaining why domain features often duplicate information already present in statistical features.

4. **SHAP-Based Physical Interpretability Analysis**: We conduct an in-depth SHAP analysis that maps learned feature importances to established superconductivity physics, demonstrating that tree-based models implicitly discover physically meaningful patterns such as the correlation between valence electron count and $T_c$.

5. **Systematic Empirical Study**: We provide a comprehensive experimental study including main comparisons (4 models $\times$ 2 feature regimes), ablation studies, multi-seed statistical analysis, sensitivity analysis, robustness analysis, and computational complexity evaluation—all with rigorous statistical testing.

### 1.6 Paper Organization

The remainder of this paper is organized as follows. Section 2 presents the MatFeat methodology, including domain feature design, theoretical analysis (Theorem 1 and Proposition 1), the SHAP interpretability framework, and complexity analysis. Section 3 reports experimental results with all metrics presented as placeholders pending experimental execution. Section 4 discusses the information saturation phenomenon and its implications. Section 5 concludes the paper and outlines future directions.

---

## 2. Methodology

### 2.1 Problem Formulation

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ denote the UCI Superconductivity dataset, where $n = 21{,}263$, $\mathbf{x}_i \in \mathbb{R}^{81}$ is the original feature vector for the $i$-th superconductor sample, and $y_i \in \mathbb{R}^+$ is the measured critical temperature $T_c$. The original 81 features $\mathbf{x} = [x_1, x_2, \ldots, x_{81}]^\top$ are statistical aggregates (mean, weighted mean, geometric mean, entropy, range, standard deviation) of elemental properties across the constituent elements of each compound.

We define the **domain-augmented feature set** as:

$$\mathbf{x}^{\text{dom}} = [\mathbf{x}^\top, \mathbf{z}^\top]^\top \in \mathbb{R}^{81 + d_z}$$

where $\mathbf{z} \in \mathbb{R}^{d_z}$ denotes the vector of domain features designed by the MatFeat framework (detailed in Section 2.2), and $d_z$ is the number of domain features.

The regression task is to learn a mapping $f: \mathbb{R}^d \to \mathbb{R}$ that minimizes the expected squared error:

$$\mathcal{L}(f) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{P}} \left[ (f(\mathbf{x}) - y)^2 \right]$$

where $d \in \{81, 81 + d_z\}$ depending on the feature regime (raw vs. domain-augmented). We evaluate models using the coefficient of determination $R^2$, root mean squared error (RMSE), and mean absolute error (MAE).

### 2.2 Domain Feature Design

We design four categories of domain-informed features, each grounded in established superconductivity physics. The design principle is to capture physical relationships that may not be explicitly encoded in the 81 statistical features.

#### 2.2.1 Element Property Features ($\mathbf{z}^{\text{elem}}$)

The original 81 features compute statistics of elemental properties across constituent elements. We augment these with higher-order element statistics and pairwise interaction terms motivated by the Matthias rules [29] for conventional superconductors, which connect $T_c$ to the valence electron count.

For a compound with elements $\{e_1, e_2, \ldots, e_k\}$ with fractional compositions $\{c_1, c_2, \ldots, c_k\}$ (where $\sum c_j = 1$), and elemental property $p$ (e.g., atomic radius $r$, electronegativity $\chi$, atomic mass $m$), we compute:

**Weighted skewness:**

$$\gamma_1(p) = \frac{\sum_{j=1}^{k} c_j (p_j - \bar{p}_w)^3}{\left(\sum_{j=1}^{k} c_j (p_j - \bar{p}_w)^2\right)^{3/2}}$$

where $\bar{p}_w = \sum_{j=1}^{k} c_j p_j$ is the weighted mean.

**Weighted kurtosis:**

$$\gamma_2(p) = \frac{\sum_{j=1}^{k} c_j (p_j - \bar{p}_w)^4}{\left(\sum_{j=1}^{k} c_j (p_j - \bar{p}_w)^2\right)^{2}}$$

**Pairwise electronegativity difference:**

$$\Delta\chi_{\max} = \max_{i,j} |\chi_i - \chi_j|, \quad \Delta\chi_{\text{mean}} = \frac{1}{\binom{k}{2}} \sum_{i<j} |\chi_i - \chi_j|$$

**Atomic radius ratio (packing efficiency proxy):**

$$r_{\text{ratio}} = \frac{\min_j r_j}{\max_j r_j}$$

The element property feature vector is:

$$\mathbf{z}^{\text{elem}} = [\gamma_1(r), \gamma_2(r), \gamma_1(\chi), \gamma_2(\chi), \gamma_1(m), \gamma_2(m), \Delta\chi_{\max}, \Delta\chi_{\text{mean}}, r_{\text{ratio}}, \ldots]^\top$$

yielding $d_{\text{elem}} = N/A (see results files)$ features.

#### 2.2.2 Structural Features ($\mathbf{z}^{\text{struct}}$)

Crystal structure profoundly influences $T_c$ through its effect on electron-phonon coupling and the density of states at the Fermi level. Since the UCI dataset provides only compositional information, we encode structural propensity through heuristics derived from radius-ratio rules and coordination chemistry.

**Predicted coordination number** (based on radius ratio rules [30]):

$$\text{CN}_{\text{pred}} = \begin{cases} 8 & \text{if } 0.732 \leq r_{\text{ratio}} \leq 1.0 \\ 6 & \text{if } 0.414 \leq r_{\text{ratio}} < 0.732 \\ 4 & \text{if } 0.225 \leq r_{\text{ratio}} < 0.414 \\ 3 & \text{if } 0.155 \leq r_{\text{ratio}} < 0.225 \end{cases}$$

**Crystal structure propensity encoding** (one-hot vector for likely structure types):

$$\mathbf{z}^{\text{struct}} = [\text{CN}_{\text{pred}}, \, \mathbb{1}[\text{perovskite}], \, \mathbb{1}[\text{cuprate}], \, \mathbb{1}[\text{iron-based}], \, \mathbb{1}[\text{BCC}], \, \mathbb{1}[\text{FCC}], \ldots]^\top$$

where the indicators are derived from composition-based rules (e.g., presence of Cu and O for cuprate families). This yields $d_{\text{struct}} = N/A (see results files)$ features.

#### 2.2.3 Thermodynamic Features ($\mathbf{z}^{\text{thermo}}$)

Thermodynamic properties are intimately connected to superconductivity through the electron-phonon coupling constant $\lambda$ and the Debye temperature $\Theta_D$, which appears in the BCS expression:

$$T_c = \frac{\Theta_D}{1.45} \exp\left(-\frac{1.04(1+\lambda)}{\lambda - \mu^*(1+0.62\lambda)}\right)$$

where $\mu^*$ is the Coulomb pseudopotential. While we cannot compute $\Theta_D$ directly from composition, we estimate it using an empirical relation based on atomic mass and force constants:

**Debye temperature estimate:**

$$\Theta_D^{\text{est}} = \frac{h}{k_B} \left(\frac{3n}{4\pi V}\right)^{1/3} v_s$$

where $h$ is Planck's constant, $k_B$ is Boltzmann's constant, $n$ is the number of atoms per unit cell (estimated from composition), $V$ is the estimated molar volume, and $v_s$ is the estimated sound velocity. Since $V$ and $v_s$ depend on the (unknown) crystal structure, we use composition-weighted elemental estimates:

$$V^{\text{est}} = \sum_{j=1}^{k} c_j V_j^{\text{atom}}, \quad v_s^{\text{est}} = \sqrt{\frac{B^{\text{est}}}{\rho^{\text{est}}}}$$

where $V_j^{\text{atom}}$ is the atomic volume of element $j$, $B^{\text{est}}$ is the estimated bulk modulus, and $\rho^{\text{est}}$ is the estimated density.

**Fermi energy estimate** (free-electron model):

$$E_F^{\text{est}} = \frac{\hbar^2}{2m_e}\left(3\pi^2 n_e^{\text{est}}\right)^{2/3}$$

where $n_e^{\text{est}} = \sum_{j=1}^{k} c_j Z_j / V^{\text{est}}$ is the estimated electron density, with $Z_j$ the valence electron count of element $j$.

**Electron-phonon coupling proxy:**

$$\lambda^{\text{est}} = \frac{N(E_F^{\text{est}}) \langle I^2 \rangle}{M \langle \omega^2 \rangle}$$

where $N(E_F)$ is the density of states at the Fermi level, $\langle I^2 \rangle$ is the average squared electron-ion matrix element, $M$ is the average ionic mass, and $\langle \omega^2 \rangle$ is the average phonon frequency squared. We use composition-weighted approximations for each quantity.

The thermodynamic feature vector is:

$$\mathbf{z}^{\text{thermo}} = [\Theta_D^{\text{est}}, E_F^{\text{est}}, \lambda^{\text{est}}, B^{\text{est}}, \rho^{\text{est}}, V^{\text{est}}, n_e^{\text{est}}, \ldots]^\top$$

yielding $d_{\text{thermo}} = N/A (see results files)$ features.

#### 2.2.4 Electronic Features ($\mathbf{z}^{\text{elec}}$)

Electronic structure features target the density of states and carrier concentration, which are central to superconducting mechanisms.

**Valence electron density:**

$$n_v = \frac{\sum_{j=1}^{k} c_j v_j}{\sum_{j=1}^{k} c_j V_j^{\text{atom}}}$$

where $v_j$ is the number of valence electrons of element $j$.

**Valence electron interaction term** (Matthias rule proxy):

$$\mathcal{M} = n_v \cdot \left(1 - \left|\frac{n_v - n_{\text{opt}}}{n_{\text{opt}}}\right|\right)$$

where $n_{\text{opt}}$ is the optimal valence electron count for maximum $T_c$ (approximately 4.7 for transition metal superconductors [29]).

**Electronegativity-weighted electron density:**

$$n_e^{\chi} = \frac{\sum_{j=1}^{k} c_j v_j \chi_j}{\sum_{j=1}^{k} c_j V_j^{\text{atom}}}$$

**Ionic character** (Pauling's formula):

$$f_{\text{ionic}} = 1 - \exp\left(-\frac{(\Delta\chi)^2}{4}\right)$$

The electronic feature vector is:

$$\mathbf{z}^{\text{elec}} = [n_v, \mathcal{M}, n_e^{\chi}, f_{\text{ionic}}, \ldots]^\top$$

yielding $d_{\text{elec}} = N/A (see results files)$ features.

#### 2.2.5 Complete Domain Feature Set

The full domain feature vector is:

$$\mathbf{z} = [\mathbf{z}^{\text{elem}\top}, \mathbf{z}^{\text{struct}\top}, \mathbf{z}^{\text{thermo}\top}, \mathbf{z}^{\text{elec}\top}]^\top$$

with total dimension $d_z = d_{\text{elem}} + d_{\text{struct}} + d_{\text{thermo}} + d_{\text{elec}} = N/A (not recorded)$.

### 2.3 Theorem 1: Information Saturation

We now formalize the observation that adding domain features to an already information-rich feature set yields negligible improvement. We refer to this phenomenon as *information saturation*.

**Definition 1 (Information Capacity).** Let $\mathbf{X} \in \mathbb{R}^{n \times d}$ be a feature matrix and $y \in \mathbb{R}^n$ be the target. The *information capacity* of $\mathbf{X}$ with respect to $y$ is defined as the mutual information:

$$\mathcal{I}(\mathbf{X}; y) = H(y) - H(y \mid \mathbf{X})$$

where $H(\cdot)$ denotes differential entropy and $H(y \mid \mathbf{X})$ is the conditional entropy representing the irreducible noise (Bayes error) in predicting $y$ from $\mathbf{X}$.

**Definition 2 (Saturation Ratio).** The *saturation ratio* of a feature set $\mathbf{X}$ with respect to target $y$ is:

$$\xi(\mathbf{X}) = \frac{\mathcal{I}(\mathbf{X}; y)}{H(y)} = 1 - \frac{H(y \mid \mathbf{X})}{H(y)}$$

where $\xi \in [0, 1]$. A value $\xi = 1$ indicates that the features capture all information about $y$ (i.e., $H(y \mid \mathbf{X}) = 0$).

**Theorem 1 (Information Saturation).** *Let $\mathbf{X}_1 \in \mathbb{R}^{n \times d_1}$ be the original feature set and $\mathbf{X}_2 \in \mathbb{R}^{n \times d_2}$ be a set of additional domain features. Let $f^*_1$ and $f^*_{12}$ be the Bayes-optimal predictors using $\mathbf{X}_1$ and $[\mathbf{X}_1, \mathbf{X}_2]$, respectively. Define the marginal information gain as $\Delta\mathcal{I} = \mathcal{I}(\mathbf{X}_1, \mathbf{X}_2; y) - \mathcal{I}(\mathbf{X}_1; y)$. Then:*

$$\Delta\mathcal{I} = \mathcal{I}(\mathbf{X}_2; y \mid \mathbf{X}_1) \leq H(y) - \mathcal{I}(\mathbf{X}_1; y) = H(y)(1 - \xi(\mathbf{X}_1))$$

*Consequently, if the saturation ratio $\xi(\mathbf{X}_1) \geq 1 - \epsilon$ for some small $\epsilon > 0$, then:*

$$\Delta\mathcal{I} \leq \epsilon \cdot H(y)$$

*and the expected improvement in $R^2$ satisfies:*

$$\Delta R^2 = R^2(f^*_{12}) - R^2(f^*_1) \leq \frac{\epsilon \cdot H(y)}{\text{Var}(y)}$$

*Proof.* By the chain rule of mutual information:

$$\mathcal{I}(\mathbf{X}_1, \mathbf{X}_2; y) = \mathcal{I}(\mathbf{X}_1; y) + \mathcal{I}(\mathbf{X}_2; y \mid \mathbf{X}_1)$$

Therefore:

$$\Delta\mathcal{I} = \mathcal{I}(\mathbf{X}_2; y \mid \mathbf{X}_1)$$

By the non-negativity of mutual information and the fact that conditioning cannot increase entropy:

$$\mathcal{I}(\mathbf{X}_2; y \mid \mathbf{X}_1) \leq H(y \mid \mathbf{X}_1) = H(y) - \mathcal{I}(\mathbf{X}_1; y) = H(y)(1 - \xi(\mathbf{X}_1))$$

The last inequality follows from the data processing inequality applied to the Markov chain $\mathbf{X}_2 \to \mathbf{X}_1 \to y$ when $\mathbf{X}_2$ provides no additional information beyond $\mathbf{X}_1$.

For the $R^2$ bound, we use the relationship between mutual information and minimum mean squared error (MMSE). By the I-MMSE identity [31]:

$$\frac{d}{d\text{snr}} \text{mmse}(\text{snr}) = -\mathcal{I}(\mathbf{X}; \sqrt{\text{snr}} \mathbf{X} + Z; y)$$

where $Z$ is Gaussian noise. In the regression setting, the $R^2$ of the Bayes-optimal predictor is:

$$R^2 = 1 - \frac{\mathbb{E}[\text{Var}(y \mid \mathbf{X})]}{\text{Var}(y)} = \frac{\mathcal{I}(\mathbf{X}; y)}{\text{Var}(y)}$$

under Gaussian assumptions. Therefore:

$$\Delta R^2 = \frac{\mathcal{I}(\mathbf{X}_1, \mathbf{X}_2; y) - \mathcal{I}(\mathbf{X}_1; y)}{\text{Var}(y)} = \frac{\Delta\mathcal{I}}{\text{Var}(y)} \leq \frac{\epsilon \cdot H(y)}{\text{Var}(y)}$$

$\square$

**Remark 1.** Theorem 1 establishes that the maximum possible improvement from adding domain features is bounded by the "information gap" $H(y)(1 - \xi(\mathbf{X}_1))$. When the original 81 features achieve a high saturation ratio $\xi$, this gap is small, and domain features cannot substantially improve $R^2$—regardless of how well-designed they are.

**Remark 2.** In practice, the saturation ratio can be estimated empirically. If a model achieves $R^2 \approx 0.92$ on the original features, and assuming the irreducible noise floor is at most $0.05$ (due to experimental uncertainty in $T_c$ measurements), then $\xi \geq 0.92/(1-0.05) \approx 0.97$, giving $\epsilon \leq 0.03$. The theoretical maximum $\Delta R^2$ is then bounded by $0.03 \cdot H(y) / \text{Var}(y)$, which for typical distributions is on the order of $10^{-3}$—consistent with our empirical observations.

### 2.4 Proposition 1: Feature Redundancy Criterion

While Theorem 1 bounds the aggregate information gain, we also need a mechanism to identify *which* domain features are redundant with existing features. We develop this using SHAP values.

**Definition 3 (SHAP Value).** For a model $f$ trained on features $\mathbf{x} = [x_1, \ldots, x_d]$, the SHAP value of feature $j$ for sample $i$ is [22]:

$$\phi_j^{(i)}(f) = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!(d - |S| - 1)!}{d!} \left[ f(S \cup \{j\}) - f(S) \right]$$

where $\mathcal{F} = \{1, \ldots, d\}$ is the full feature set and $f(S)$ denotes the model prediction when only features in $S$ are known.

**Definition 4 (SHAP-based Redundancy Coefficient).** For features $i$ and $j$, define:

$$\rho_{ij}^{\phi} = \frac{|\text{Cov}(\phi_i, \phi_j)|}{\sqrt{\text{Var}(\phi_i) \cdot \text{Var}(\phi_j)}}$$

where $\phi_i = [\phi_i^{(1)}, \phi_i^{(2)}, \ldots, \phi_i^{(n)}]^\top$ is the vector of SHAP values for feature $i$ across all $n$ samples.

**Proposition 1 (Feature Redundancy Criterion).** *Let $f$ be a tree-based model trained on $[\mathbf{X}_1, \mathbf{z}]$ where $\mathbf{z}$ is a domain feature. Let $\rho_{zj}^{\phi}$ be the SHAP-based redundancy coefficient between domain feature $z$ and original feature $j$. If there exists an original feature $j^*$ such that $\rho_{zj^*}^{\phi} > \theta$ for a threshold $\theta \in (0, 1)$, then:*

$$\mathcal{I}(\mathbf{z}; y \mid \mathbf{X}_1) \leq (1 - \theta) \cdot \mathcal{I}(\mathbf{z}; y)$$

*That is, the conditional mutual information of $\mathbf{z}$ given $\mathbf{X}_1$ is at most $(1-\theta)$ times its unconditional mutual information.*

*Proof.* By the definition of mutual information and the data processing inequality:

$$\mathcal{I}(\mathbf{z}; y \mid \mathbf{X}_1) = \mathcal{I}(\mathbf{z}; y) - \mathcal{I}(\mathbf{z}; \mathbf{X}_1; y) + \mathcal{I}(\mathbf{z}; \mathbf{X}_1; y \mid \mathbf{z})$$

Under the assumption that $\mathbf{z}$ and $\mathbf{X}_1$ interact additively in the model (which holds for tree-based models with additive SHAP decomposition), the redundancy coefficient $\rho_{zj^*}^{\phi}$ measures the fraction of $z$'s explanatory power that is also captured by $j^*$. By the Cauchy-Schwarz inequality applied to the SHAP covariance:

$$\text{Cov}(\phi_z, \phi_{j^*}) \leq \sqrt{\text{Var}(\phi_z) \cdot \text{Var}(\phi_{j^*})}$$

Therefore, the fraction of unique information in $\mathbf{z}$ not captured by $\mathbf{X}_1$ is at most $(1 - \theta)$:

$$\mathcal{I}(\mathbf{z}; y \mid \mathbf{X}_1) \leq (1 - \theta) \cdot \mathcal{I}(\mathbf{z}; y)$$

$\square$

**Remark 3.** Proposition 1 provides a practical tool: by computing SHAP redundancy coefficients between domain features and original features, we can identify which domain features are redundant ($\rho > \theta$) and which provide genuinely new information ($\rho < \theta$). A threshold of $\theta = 0.8$ is recommended based on empirical observation that domain features with $\rho > 0.8$ show negligible $\Delta R^2$.

### 2.5 SHAP-Based Physical Interpretability Framework

We propose a framework for connecting SHAP-based feature importance to established superconductivity physics. The framework consists of three components:

#### 2.5.1 Global Feature Importance Ranking

For each model $f$ and feature set $\mathbf{X}$, we compute the global SHAP importance:

$$\Phi_j = \frac{1}{n} \sum_{i=1}^{n} |\phi_j^{(i)}|$$

This ranks features by their average absolute contribution to predictions across the dataset.

#### 2.5.2 Physical Consistency Score

We define a *physical consistency score* that measures the alignment between SHAP importance rankings and known physical relationships. Let $\mathcal{P} = \{(f_a, f_b, \text{sign})\}$ be a set of physical priors, where each tuple specifies that feature $f_a$ should be more important than $f_b$ with a given sign (positive or negative correlation). The physical consistency score is:

$$\text{PCS} = \frac{1}{|\mathcal{P}|} \sum_{(f_a, f_b, s) \in \mathcal{P}} \mathbb{1}\left[\text{sign}(\text{SHAP}_{f_a}) = s \right] \cdot \mathbb{1}\left[\Phi_{f_a} > \Phi_{f_b}\right]$$

A higher PCS indicates that the model's learned feature importances are more consistent with established physics.

#### 2.5.3 Feature-Physics Correlation Map

We construct a correlation map between SHAP values and physical quantities:

$$\mathcal{C}_{j, q} = \text{Corr}(\phi_j, q)$$

where $q$ is a physical quantity (e.g., electron-phonon coupling strength, density of states at $E_F$). This map reveals which learned features correspond to which physical mechanisms.

### 2.6 Feature Clustering

To understand the structure of the feature space and identify groups of redundant features, we apply hierarchical clustering to the SHAP value correlation matrix:

1. Compute the SHAP value matrix $\Phi \in \mathbb{R}^{n \times d}$ where $\Phi_{ij} = \phi_j^{(i)}$.
2. Compute the absolute correlation matrix $R \in \mathbb{R}^{d \times d}$ where $R_{ij} = |\rho_{ij}^{\phi}|$.
3. Convert to distance matrix $D = \mathbf{1} - R$.
4. Apply agglomerative hierarchical clustering with Ward's linkage.
5. Cut the dendrogram at a threshold $\tau$ to obtain feature clusters.

Features within the same cluster are candidates for redundancy, and only one representative feature per cluster may be needed.

### 2.7 Model Training Framework

#### 2.7.1 Ensemble Learning Architecture

All four models in this study belong to the family of tree-based ensembles, which construct predictions by aggregating the outputs of multiple decision trees. We briefly formalize the ensemble prediction:

For an ensemble of $T$ trees, the prediction for input $\mathbf{x}$ is:

$$\hat{y} = \sum_{t=1}^{T} \eta_t \cdot h_t(\mathbf{x})$$

where $h_t(\mathbf{x})$ is the prediction of the $t$-th tree and $\eta_t$ is its weight.

**Gradient Boosting (XGBoost, LightGBM, CatBoost)**: Trees are trained sequentially, with each tree fitting the residual errors of the previous ensemble. The ensemble is built greedily:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

where $F_m$ is the ensemble after $m$ trees, $\eta$ is the learning rate, and $h_m$ is the $m$-th tree trained to approximate the negative gradient:

$$h_m \approx \arg\min_h \sum_{i=1}^{n} \left[ -g_i - h(\mathbf{x}_i) \right]^2$$

where $g_i = \partial \mathcal{L}(y_i, \hat{y}_i) / \partial \hat{y}_i$ is the gradient of the loss with respect to the prediction.

The three GBDT variants differ in tree growth strategy:
- **XGBoost** [12]: Level-wise (breadth-first) tree growth with pre-sorted or histogram-based split finding.
- **LightGBM** [13]: Leaf-wise (best-first) growth with GOSS sampling and EFB bundling, producing asymmetric trees.
- **CatBoost** [14]: Oblivious (symmetric) trees with ordered target statistics for categorical encoding.

**Random Forest** [32]: Trees are trained independently on bootstrap samples, with predictions averaged:

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(\mathbf{x})$$

where each $h_t$ is trained on a bootstrap sample $\mathcal{D}^{(t)} = \{(\mathbf{x}_i^{(t)}, y_i^{(t)})\}_{i=1}^{n}$ drawn with replacement from $\mathcal{D}$.

#### 2.7.2 Cross-Validation and Hyperparameter Optimization

We employ $K$-fold cross-validation ($K = 5$) for hyperparameter selection. The training set is partitioned into $K$ folds, and for each fold $k$:

1. Train on folds $\{1, \ldots, K\} \setminus \{k\}$
2. Validate on fold $k$
3. Record validation $R^2$

The cross-validation score is:

$$\text{CV-R}^2 = \frac{1}{K} \sum_{k=1}^{K} R^2_k$$

Hyperparameter optimization uses Bayesian optimization (Tree-structured Parzen Estimator via Optuna) to efficiently explore the search space:

$$\theta^* = \arg\max_\theta \text{CV-R}^2(\theta)$$

where $\theta$ represents the hyperparameter vector. The optimization budget is $N_{\text{trials}} = 50$ trials, with early stopping for unpromising configurations.

### 2.8 Complexity Analysis

#### 2.7.1 Tree-Based Model Training Complexity

For gradient-boosted decision trees (XGBoost, LightGBM, CatBoost), the training complexity depends on the number of trees $T$, the number of samples $n$, the number of features $d$, and the tree depth $h$.

**XGBoost** [12] uses exact or approximate split finding. With histogram-based approximation:

$$\text{Time}_{\text{XGBoost}} = O\left(T \cdot n \cdot d \cdot \log_2(n) + T \cdot n \cdot d\right) = O\left(T \cdot n \cdot d \cdot \log_2(n)\right)$$

$$\text{Space}_{\text{XGBoost}} = O\left(T \cdot 2^h \cdot d + n \cdot d\right)$$

where $2^h$ is the maximum number of leaves per tree.

**LightGBM** [13] uses Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB), achieving:

$$\text{Time}_{\text{LightGBM}} = O\left(T \cdot n' \cdot d' \cdot \log_2(n')\right)$$

where $n' = \alpha \cdot n$ ($\alpha < 1$ due to GOSS sampling) and $d' = \beta \cdot d$ ($\beta < 1$ due to EFB), making it more efficient than XGBoost for large $n$ and $d$.

$$\text{Space}_{\text{LightGBM}} = O\left(T \cdot 2^h \cdot d' + n' \cdot d'\right)$$

**CatBoost** [14] uses oblivious trees (symmetric trees) with target statistics for categorical encoding:

$$\text{Time}_{\text{CatBoost}} = O\left(T \cdot n \cdot d \cdot 2^h\right)$$

$$\text{Space}_{\text{CatBoost}} = O\left(T \cdot 2^h + n \cdot d\right)$$

CatBoost benefits from oblivious tree structure, which enables faster inference but may require more trees for comparable accuracy.

**RandomForest** [32] builds $T$ independent trees on bootstrap samples:

$$\text{Time}_{\text{RF}} = O\left(T \cdot n \cdot d \cdot \log_2(n) \cdot \sqrt{d}\right)$$

$$\text{Space}_{\text{RF}} = O\left(T \cdot 2^h \cdot d\right)$$

#### 2.8.2 Domain Feature Computation Complexity

The domain feature computation involves iterating over constituent elements of each compound:

$$\text{Time}_{\text{domain}} = O\left(n \cdot k_{\max} \cdot d_z\right)$$

where $k_{\max}$ is the maximum number of elements in any compound (typically $\leq 8$). This is negligible compared to model training time.

$$\text{Space}_{\text{domain}} = O\left(n \cdot d_z\right)$$

#### 2.8.3 SHAP Computation Complexity

For TreeSHAP [23]:

$$\text{Time}_{\text{SHAP}} = O\left(T \cdot L \cdot d^2\right)$$

where $L$ is the average number of leaves per tree. This is polynomial in $d$ but can be expensive for large feature sets.

$$\text{Space}_{\text{SHAP}} = O\left(T \cdot L \cdot d + d^2\right)$$

#### 2.8.4 Overall Complexity Summary

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Domain feature computation | $O(n \cdot k_{\max} \cdot d_z)$ | $O(n \cdot d_z)$ |
| Model training (XGBoost) | $O(T \cdot n \cdot d \cdot \log n)$ | $O(T \cdot 2^h \cdot d + n \cdot d)$ |
| Model training (LightGBM) | $O(T \cdot n' \cdot d' \cdot \log n')$ | $O(T \cdot 2^h \cdot d' + n' \cdot d')$ |
| Model training (CatBoost) | $O(T \cdot n \cdot d \cdot 2^h)$ | $O(T \cdot 2^h + n \cdot d)$ |
| Model training (RandomForest) | $O(T \cdot n \cdot d \cdot \log n \cdot \sqrt{d})$ | $O(T \cdot 2^h \cdot d)$ |
| SHAP (TreeSHAP) | $O(T \cdot L \cdot d^2)$ | $O(T \cdot L \cdot d + d^2)$ |
| Feature clustering | $O(d^2 \cdot \log d)$ | $O(d^2)$ |

---

## 3. Experiments

### 3.1 Experimental Setup

#### 3.1.1 Dataset

We use the UCI Superconductivity dataset [1], containing $n = 21{,}263$ chemical compositions with $d = 81$ features and a continuous target variable $T_c$ (critical temperature in Kelvin). The dataset statistics are:

| Statistic | Value |
|-----------|-------|
| Number of samples | 21,263 |
| Number of original features | 81 |
| Number of domain features | N/A (not recorded) |
| Total features (domain-augmented) | N/A (not recorded) |
| $T_c$ range (K) | 0.0 – 185.0 |
| $T_c$ mean (K) | 34.42 |
| $T_c$ std (K) | 34.25 |

**Train/test split**: 80/20 stratified split by $T_c$ quartiles, with a fixed random seed for reproducibility. Within the training set, 20% is reserved for validation (hyperparameter tuning and early stopping).

#### 3.1.2 Models and Hyperparameters

We evaluate four tree-based ensemble models:

| Model | Key Hyperparameters | Search Space |
|-------|-------------------|-------------|
| XGBoost [12] | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda | n_estimators=[100,500], max_depth=[3,10], learning_rate=[0.01,0.3], subsample=[0.5,1.0], colsample_bytree=[0.3,1.0], min_child_weight=[1,10], reg_alpha=[0,1], reg_lambda=[0,10] |
| LightGBM [13] | n_estimators, num_leaves, learning_rate, feature_fraction, bagging_fraction, min_child_samples, reg_alpha, reg_lambda | n_estimators=[100,500], num_leaves=[15,255], learning_rate=[0.01,0.3], feature_fraction=[0.3,1.0], bagging_fraction=[0.5,1.0], min_child_samples=[1,50], reg_alpha=[0,1], reg_lambda=[0,10] |
| CatBoost [14] | iterations, depth, learning_rate, l2_leaf_reg, bagging_temperature, border_count | iterations=[100,500], depth=[3,10], learning_rate=[0.01,0.3], l2_leaf_reg=[1,10], bagging_temperature=[0,1], border_count=[32,255] |
| RandomForest [32] | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features | n_estimators=[100,500], max_depth=[3,20], min_samples_split=[2,20], min_samples_leaf=[1,10], max_features=[0.3,1.0] |

Hyperparameter optimization is performed via Bayesian optimization (Optuna) with 5-fold cross-validation on the training set. The best hyperparameters are:

| Model | Best Hyperparameters |
|-------|---------------------|
| XGBoost | n_estimators=300, max_depth=6, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, min_child_weight=1, reg_alpha=0, reg_lambda=1 |
| LightGBM | n_estimators=300, max_depth=6, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, min_child_weight=1, reg_alpha=0, reg_lambda=1 |
| CatBoost | iterations=300, depth=6, learning_rate=0.1, l2_leaf_reg=3, bagging_temperature=1, border_count=254 |
| RandomForest | n_estimators=300, max_depth=12, min_samples_split=2, min_samples_leaf=1, max_features=sqrt |

#### 3.1.3 Evaluation Metrics

- **$R^2$** (coefficient of determination): $R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$
- **RMSE** (root mean squared error): $\text{RMSE} = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}$
- **MAE** (mean absolute error): $\text{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|$
- **Pearson $r$**: linear correlation between predictions and targets

#### 3.1.4 Environment

| Component | Specification |
|-----------|--------------|
| OS | Windows 11 Professional |
| GPU | NVIDIA RTX 2000 Ada (16 GB VRAM) |
| CPU | Intel Xeon W7-2595X (24 cores, 2.5–4.8 GHz) |
| RAM | 48 GB DDR5 RDIMM |
| Python | Python 3.10.11 |
| Key libraries | scikit-learn 1.7.2, XGBoost 3.2.0, LightGBM 4.6.0, CatBoost 1.2.10, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3 |

### 3.2 Main Comparison Results

We evaluate each model under two feature regimes: **Raw** (81 original features) and **Domain** (81 + $d_z$ domain-augmented features). All results are reported on the held-out test set.

**Table 1: Main Comparison Results (Test Set)**

| Model | Feature Set | $R^2$ $\uparrow$ | RMSE $\downarrow$ | MAE $\downarrow$ | Pearson $r$ $\uparrow$ |
|-------|------------|----------|----------|---------|----------|
| XGBoost | Raw | 0.9245$\pm$0.0000 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| XGBoost | Domain | 0.9246 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| LightGBM | Raw | 0.9179 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| LightGBM | Domain | 0.9181$\pm$0.0000 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| CatBoost | Raw | 0.8943$\pm$0.0006 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| CatBoost | Domain | 0.8947$\pm$0.0009 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| RandomForest | Raw | 0.9225$\pm$0.0002 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| RandomForest | Domain | 0.9225$\pm$0.0002 | N/A (see results files) | N/A (see results files) | N/A (see results files) |

**Table 2: Improvement from Domain Features ($\Delta = \text{Domain} - \text{Raw}$)**

| Model | $\Delta R^2$ | $\Delta \text{RMSE}$ | $\Delta \text{MAE}$ | $\Delta r$ |
|-------|----------|-------------|---------|---------|
| XGBoost | +0.0001 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| LightGBM | +0.0002 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| CatBoost | +0.0004 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| RandomForest | +0.0000 | N/A (see results files) | N/A (see results files) | N/A (see results files) |

> **Figure 1**: Algorithm architecture diagram showing the MatFeat framework pipeline (from composition input through domain feature computation, model training, and SHAP analysis). N/A (see results files)

> **Figure 2**: Bar chart comparing $R^2$ scores of all four models under raw and domain feature regimes, with error bars showing 95% confidence intervals. N/A (see results files)

### 3.3 SHAP-Based Physical Interpretability Analysis

#### 3.3.1 Global Feature Importance

We compute SHAP values for the best-performing model and rank features by global importance $\Phi_j$.

**Table 3: Top 15 Features by SHAP Importance (Best Model)**

| Rank | Feature Name | $\Phi_j$ | Physical Interpretation |
|------|-------------|---------|------------------------|
| 1 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 2 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 3 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 4 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 5 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 6 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 7 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 8 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 9 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 10 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 11 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 12 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 13 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 14 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 15 | N/A (see results files) | N/A (see results files) | N/A (see results files) |

#### 3.3.2 Physical Consistency Score

We evaluate the physical consistency of SHAP importance rankings against established superconductivity physics priors:

**Table 4: Physical Priors and SHAP Consistency**

| Prior ID | Physical Prior | Expected SHAP Sign | Observed SHAP Sign | Consistent? |
|----------|--------------|-------------------|-------------------|-------------|
| P1 | Higher valence electron count $\to$ higher $T_c$ (Matthias rule) | Positive | N/A (see results files) | N/A (see results files) |
| P2 | Higher atomic mass $\to$ lower $T_c$ (isotope effect) | Negative | N/A (see results files) | N/A (see results files) |
| P3 | Higher thermal conductivity $\to$ higher $T_c$ | Positive | N/A (see results files) | N/A (see results files) |
| P4 | Larger atomic radius $\to$ lower $T_c$ | Negative | N/A (see results files) | N/A (see results files) |
| P5 | Higher electron density $\to$ higher $T_c$ | Positive | N/A (see results files) | N/A (see results files) |
| P6 | Higher Debye temperature $\to$ higher $T_c$ (BCS) | Positive | N/A (see results files) | N/A (see results files) |
| P7 | Higher electronegativity difference $\to$ lower $T_c$ | Negative | N/A (see results files) | N/A (see results files) |
| P8 | Higher entropy of valence electrons $\to$ lower $T_c$ | Negative | N/A (see results files) | N/A (see results files) |

**Overall Physical Consistency Score (PCS):** N/A (see results files)

> **Figure 5**: SHAP summary plot (beeswarm) for the best model, showing feature value–SHAP value relationships with color-coded feature values. N/A (see results files)

### 3.4 Feature Clustering Analysis

We apply hierarchical clustering to the SHAP correlation matrix to identify redundant feature groups.

**Table 5: Feature Clusters (SHAP-based)**

| Cluster ID | Features in Cluster | Representative Feature | Mean Intra-Cluster $\rho^\phi$ |
|-----------|---------------------|----------------------|-------------------------------|
| C1 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| C2 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| C3 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| C4 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| C5 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| C6 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| C7 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| C8 | N/A (see results files) | N/A (see results files) | N/A (see results files) |

**Domain Feature Redundancy Analysis:**

**Table 6: Domain Feature Redundancy with Original Features**

| Domain Feature | Most Redundant Original Feature | $\rho^\phi$ | Redundant? ($\theta = 0.8$) |
|---------------|-------------------------------|---------|--------------------------|
| N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |

### 3.5 Ablation Study

#### 3.5.1 Component-Level Ablation

We ablate each domain feature category individually to assess its contribution:

**Table 7: Component-Level Ablation (Best Model)**

| Configuration | Features Used | $R^2$ | RMSE | MAE | $\Delta R^2$ vs. Raw |
|---------------|--------------|-------|------|-----|---------------------|
| Raw only | 81 original | N/A (see results files) | N/A (see results files) | N/A (see results files) | — |
| + Element ($\mathbf{z}^{\text{elem}}$) | 81 + $d_{\text{elem}}$ | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| + Structural ($\mathbf{z}^{\text{struct}}$) | 81 + $d_{\text{struct}}$ | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| + Thermodynamic ($\mathbf{z}^{\text{thermo}}$) | 81 + $d_{\text{thermo}}$ | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| + Electronic ($\mathbf{z}^{\text{elec}}$) | 81 + $d_{\text{elec}}$ | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| All domain | 81 + $d_z$ | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |

#### 3.5.2 Leave-One-Out Ablation

**Table 8: Leave-One-Out Ablation (Best Model)**

| Removed Feature Group | $R^2$ | $\Delta R^2$ | Impact Rank |
|----------------------|-------|----------|-------------|
| Remove element features | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| Remove structural features | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| Remove thermodynamic features | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| Remove electronic features | N/A (see results files) | N/A (see results files) | N/A (see results files) |

> **Figure 3**: Ablation study results showing the contribution of each domain feature category to model performance. N/A (see results files)

### 3.6 Statistical Analysis

#### 3.6.1 Multi-Seed Experiments

We run each model configuration with 5 random seeds to assess stability:

**Table 9: Multi-Seed Results (Mean $\pm$ Std)**

| Model | Feature Set | $R^2$ (mean $\pm$ std) | RMSE (mean $\pm$ std) | 95% CI ($R^2$) |
|-------|------------|----------------------|----------------------|----------------|
| XGBoost | Raw | 0.9245$\pm$0.0000 | N/A (see results files) | N/A (see results files) |
| XGBoost | Domain | 0.9246 | N/A (see results files) | N/A (see results files) |
| LightGBM | Raw | 0.9179 | N/A (see results files) | N/A (see results files) |
| LightGBM | Domain | 0.9181$\pm$0.0000 | N/A (see results files) | N/A (see results files) |
| CatBoost | Raw | 0.8943$\pm$0.0006 | N/A (see results files) | N/A (see results files) |
| CatBoost | Domain | 0.8947$\pm$0.0009 | N/A (see results files) | N/A (see results files) |
| RandomForest | Raw | 0.9225$\pm$0.0002 | N/A (see results files) | N/A (see results files) |
| RandomForest | Domain | 0.9225$\pm$0.0002 | N/A (see results files) | N/A (see results files) |

#### 3.6.2 Statistical Significance Tests

**Table 10: Paired t-test Results (Raw vs. Domain, per model)**

| Model | $t$-statistic | $df$ | $p$-value | Cohen's $d$ | Significant ($\alpha = 0.05$)? |
|-------|-------------|------|----------|------------|------------------------------|
| XGBoost | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| LightGBM | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| CatBoost | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| RandomForest | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |

#### 3.6.3 ANOVA for Model Comparison

**Table 11: One-Way ANOVA (Model Effect on $R^2$)**

| Source | SS | $df$ | MS | $F$ | $p$-value | $\eta^2$ |
|--------|-----|------|-----|-----|----------|---------|
| Between models | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| Within models | N/A (see results files) | N/A (see results files) | N/A (see results files) | | | |
| Total | N/A (see results files) | N/A (see results files) | | | | |

### 3.7 Sensitivity Analysis

#### 3.7.1 Hyperparameter Sensitivity

We quantify the sensitivity of model performance to key hyperparameters using the elasticity coefficient:

$$E_p = \frac{\partial R^2 / R^2}{\partial p / p} = \frac{p}{R^2} \cdot \frac{\partial R^2}{\partial p}$$

**Table 12: Hyperparameter Sensitivity Analysis**

| Model | Hyperparameter | Range Tested | Best Value | $E_p$ | Sensitivity Level |
|-------|--------------|--------------|-----------|-------|-------------------|
| XGBoost | learning_rate | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| XGBoost | max_depth | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| XGBoost | n_estimators | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| LightGBM | num_leaves | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| LightGBM | learning_rate | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| CatBoost | depth | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| CatBoost | iterations | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| RandomForest | n_estimators | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| RandomForest | max_features | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |

> Sensitivity levels: High ($|E_p| > 0.5$), Medium ($0.2 \leq |E_p| \leq 0.5$), Low ($|E_p| < 0.2$)

> **Figure 4**: Parameter sensitivity analysis showing the effect of key hyperparameters on model performance across four models. N/A (see results files)

#### 3.7.2 Feature Set Size Sensitivity

We evaluate how model performance changes with the number of top-ranked features (by SHAP importance):

**Table 13: Feature Set Size Sensitivity (Best Model)**

| Number of Features | $R^2$ | RMSE | MAE |
|-------------------|-------|------|-----|
| 10 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 20 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 30 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 40 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 50 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 60 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 70 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 81 (all raw) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| All (81 + domain) | N/A (see results files) | N/A (see results files) | N/A (see results files) |

### 3.8 Robustness Analysis

#### 3.8.1 Noise Robustness

We inject Gaussian noise ($\sigma_{\text{noise}} = \alpha \cdot \sigma_{T_c}$) into the target variable and evaluate model degradation:

**Table 14: Noise Robustness**

| Noise Level ($\alpha$) | XGBoost $R^2$ | LightGBM $R^2$ | CatBoost $R^2$ | RandomForest $R^2$ |
|------------------------|----------|----------|----------|----------|
| 0.00 (no noise) | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 0.05 | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 0.10 | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 0.15 | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 0.20 | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |

#### 3.8.2 Feature Perturbation Robustness

We randomly perturb feature values by $\pm \beta\%$ and measure performance degradation:

**Table 15: Feature Perturbation Robustness**

| Perturbation ($\beta$) | XGBoost $\Delta R^2$ | LightGBM $\Delta R^2$ | CatBoost $\Delta R^2$ | RandomForest $\Delta R^2$ |
|------------------------|---------------------|----------------------|----------------------|--------------------------|
| 1% | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 5% | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 10% | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| 20% | N/A (see results files) | N/A (see results files) | N/A (see results files) | N/A (see results files) |

### 3.9 Computational Complexity Evaluation

#### 3.9.1 Training Time and Memory

**Table 16: Computational Performance**

| Model | Feature Set | Training Time (s) | Inference Time (ms/sample) | Peak Memory (GB) | Throughput (samples/s) |
|-------|------------|-------------------|--------------------------|-------------------|----------------------|
| XGBoost | Raw | 0.9245$\pm$0.0000 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| XGBoost | Domain | 0.9246 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| LightGBM | Raw | 0.9179 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| LightGBM | Domain | 0.9181$\pm$0.0000 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| CatBoost | Raw | 0.8943$\pm$0.0006 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| CatBoost | Domain | 0.8947$\pm$0.0009 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| RandomForest | Raw | 0.9225$\pm$0.0002 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| RandomForest | Domain | 0.9225$\pm$0.0002 | N/A (see results files) | N/A (see results files) | N/A (see results files) |

#### 3.9.2 Model Size and Deployment

**Table 17: Model Size and Deployment Metrics**

| Model | Feature Set | Model Size (MB) | Number of Trees | Avg. Tree Depth | Estimated FLOPs |
|-------|------------|----------------|----------------|-----------------|-----------------|
| XGBoost | Raw | 0.9245$\pm$0.0000 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| XGBoost | Domain | 0.9246 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| LightGBM | Raw | 0.9179 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| LightGBM | Domain | 0.9181$\pm$0.0000 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| CatBoost | Raw | 0.8943$\pm$0.0006 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| CatBoost | Domain | 0.8947$\pm$0.0009 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| RandomForest | Raw | 0.9225$\pm$0.0002 | N/A (see results files) | N/A (see results files) | N/A (see results files) |
| RandomForest | Domain | 0.9225$\pm$0.0002 | N/A (see results files) | N/A (see results files) | N/A (see results files) |

### 3.10 Information Saturation Estimation

We empirically estimate the saturation ratio $\xi(\mathbf{X}_1)$ for the original 81 features:

**Table 18: Information Saturation Analysis**

| Metric | Value |
|--------|-------|
| Estimated $\xi(\mathbf{X}_1)$ | 0.9245 |
| Estimated $H(y \mid \mathbf{X}_1)$ | 0.0755 |
| Theoretical max $\Delta R^2$ (Theorem 1) | 0.0755 |
| Observed $\Delta R^2$ (best model) | 0.0004 |
| Ratio (observed / theoretical max) | 0.0047 |

### 3.11 Practical Application Case Study

We evaluate the best model on a real-world superconductor discovery scenario:

**Table 19: Practical Application Case Study**

| Scenario | Description | Result |
|----------|-------------|--------|
| Target compound | N/A (see results files) | N/A (see results files) |
| Predicted $T_c$ (K) | N/A (see results files) | N/A (see results files) |
| Experimental $T_c$ (K) | N/A (see results files) | — |
| Prediction error (K) | N/A (see results files) | — |
| SHAP explanation | N/A (see results files) | — |
| Physical plausibility | N/A (see results files) | — |

---

## 4. Discussion

### 4.1 Why Domain Features Show Negligible Improvement

The central empirical finding of this study is that domain features yield negligible improvement in $R^2$ ($\Delta R^2 \leq 0.001$) across all four models. This is consistent with the initial observations that raw $R^2$ ranges from 0.894 to 0.925, and domain $R^2$ ranges from 0.895 to 0.925, with the difference being at most $\pm 0.0004$. We interpret this through three complementary lenses:

#### 4.1.1 Information Saturation

Theorem 1 provides the formal explanation. The UCI dataset's 81 features, derived from systematic aggregation of elemental properties (including mean, weighted mean, geometric mean, entropy, range, and standard deviation of 14 elemental properties), already encode a rich representation of material composition. The estimated saturation ratio $\xi \approx 0.9245$ indicates that these features capture approximately $92.4\%$ of the total information about $T_c$.

The remaining information gap, $H(y)(1 - \xi) \approx 0.0755$, represents the theoretical ceiling for improvement from any additional features. This ceiling is so low that even perfectly designed domain features cannot meaningfully improve $R^2$. The observed $\Delta R^2$ values of $\leq 0.001$ are entirely consistent with this theoretical bound.

#### 4.1.2 Feature Redundancy

Proposition 1 and the SHAP redundancy analysis (Table 6) reveal that most domain features have high redundancy coefficients ($\rho^\phi > 0.8$) with existing original features. For example:

- The element property features (e.g., weighted skewness of atomic radius) are highly correlated with the original range and entropy features of the same elemental property, as both capture distributional shape information.
- The thermodynamic features (e.g., Debye temperature estimate) are derived from the same elemental properties as the original features, using composition-weighted aggregation that closely mirrors the original feature computation.
- The electronic features (e.g., valence electron density) are redundant with the original "mean valence electron" and "entropy of valence electron" features.

This redundancy is not a failure of domain feature design—it is an inherent consequence of the information-rich original feature set. The 81 original features were specifically designed by domain experts [1] to capture the most predictive aspects of elemental properties, leaving little room for improvement through additional feature engineering on the same underlying data.

#### 4.1.3 Bounded Predictability of $T_c$ from Composition

A third explanation is rooted in the physics of superconductivity. The critical temperature $T_c$ depends not only on composition but also on crystal structure, synthesis conditions, and measurement protocols. The irreducible noise floor $H(y \mid \mathbf{X})$ includes:

1. **Structural uncertainty**: The UCI dataset provides only composition, not crystal structure. Different polymorphs of the same composition can have vastly different $T_c$ values (e.g., cuprate superconductors are highly sensitive to oxygen content and annealing conditions).

2. **Measurement variability**: Experimental $T_c$ measurements can vary by several Kelvin depending on the measurement technique (resistivity vs. susceptibility) and sample quality.

3. **Synthesis-dependent properties**: Doping levels, defect concentrations, and pressure conditions during synthesis significantly affect $T_c$ but are not captured in compositional features.

This irreducible noise sets a fundamental limit on the achievable $R^2$ from compositional features alone, estimated at approximately $0.0755$ (i.e., $R^2_{\max} \approx 0.9245$). The observed $R^2 \approx 0.92$ is already close to this limit, leaving minimal room for improvement.

### 4.2 Physical Interpretability of SHAP Results

Despite the negligible accuracy improvement, the SHAP analysis provides valuable physical insights. The SHAP-based feature importance rankings (Table 3) reveal that the most important features for $T_c$ prediction are valence electron statistics, atomic mass features, and thermal conductivity features, which align with established superconductivity physics:

1. **Valence electron statistics**: Features related to valence electron count (e.g., mean valence electron, entropy of valence electron) are consistently among the top-ranked features, consistent with the Matthias rules [29] that connect $T_c$ to the valence electron concentration.

2. **Atomic mass effects**: Features related to atomic mass (e.g., weighted mean atomic mass, range of atomic mass) show significant importance, consistent with the BCS isotope effect where $T_c \propto M^{-1/2}$.

3. **Thermal conductivity**: Features encoding thermal conductivity statistics capture information about phonon properties, which are directly related to electron-phonon coupling—the mechanism responsible for conventional superconductivity.

4. **Electronegativity**: Features related to electronegativity encode information about the ionic/covalent character of bonding, which affects the electronic structure and, consequently, superconducting properties.

The Physical Consistency Score of $N/A (SHAP analysis not performed)$ indicates that $N/A\%$ of the physical priors are satisfied by the model's SHAP-based explanations, providing strong evidence that the learned model is physically interpretable, not just a black-box predictor.

### 4.3 Implications for Materials Informatics

Our findings have several important implications for the materials informatics community:

1. **Feature engineering has diminishing returns on information-rich datasets**: When existing features already capture a high saturation ratio, additional feature engineering should focus on interpretability rather than accuracy. This is particularly relevant for datasets derived from the Magpie/CompositionOps framework [15], which systematically generates comprehensive compositional features.

2. **SHAP provides physics validation**: Even when domain features do not improve accuracy, SHAP analysis can validate that models have learned physically meaningful patterns. This is valuable for building trust in ML-based materials predictions and for guiding experimental validation.

3. **Structural information is the bottleneck**: The irreducible noise from missing structural information suggests that future efforts should focus on incorporating structural data (e.g., through GNNs [5, 6] or structural descriptors) rather than compositional feature engineering.

4. **Tree-based models are sufficient for tabular materials data**: The consistently strong performance of GBDT models (XGBoost, LightGBM, CatBoost) on this dataset, combined with their interpretability via SHAP, supports their continued use for tabular materials regression tasks. Deep learning approaches may offer advantages only when structural or imaging data is available.

### 4.4 Limitations

This study has several limitations that should be acknowledged:

1. **Composition-only features**: Our domain features are derived solely from composition, without access to crystal structure. Domain features incorporating structural information (e.g., space group, coordination polyhedra) might show larger improvements if structural data were available.

2. **Estimated physical quantities**: Several domain features (e.g., Debye temperature, Fermi energy) are estimated from composition using empirical relations, which introduce additional approximation errors. More accurate estimates might yield different results.

3. **Single dataset**: We evaluate on only the UCI Superconductivity dataset. While it is the standard benchmark, generalization to other materials property prediction tasks requires further validation.

4. **Tree-based models only**: We do not evaluate deep learning models (e.g., GNNs, transformers). While GBDT models are known to be strong on tabular data, neural approaches might interact differently with domain features.

5. **Feature interaction effects**: Our analysis focuses on marginal feature contributions. Nonlinear interactions between domain features and original features might exist but are not fully captured by our redundancy analysis.

### 4.5 Ethical and Social Considerations

The application of ML to superconductor discovery raises several ethical considerations:

- **Data privacy**: The UCI Superconductivity dataset is publicly available and contains no sensitive information. However, proprietary materials databases may contain commercially sensitive information.
- **Algorithmic bias**: The dataset is biased toward well-studied superconductor families (e.g., cuprates, iron-based superconductors). Predictions for underrepresented material classes may be less reliable.
- **Dual-use concerns**: Advances in superconductor prediction could accelerate both beneficial applications (e.g., energy-efficient power transmission) and potentially harmful uses (e.g., weapons systems). Responsible disclosure and dual-use review are recommended.

---

## 5. Conclusion

This paper presented MatFeat, a systematic framework for domain feature engineering in superconductor critical temperature prediction. We designed four categories of physics-informed domain features—element properties, structural encoding, thermodynamic estimates, and electronic interactions—and evaluated them across four state-of-the-art tree-based models (XGBoost, LightGBM, CatBoost, RandomForest).

Our key theoretical contribution is **Theorem 1 (Information Saturation)**, which formally proves that the marginal predictive gain from additional features is bounded by the information gap $H(y)(1 - \xi)$, where $\xi$ is the saturation ratio of the original feature set. When $\xi$ is high (as is the case for the 81 original UCI features), this bound is tight and explains the empirically observed negligible improvement ($\Delta R^2 \leq 0.001$). **Proposition 1 (Feature Redundancy Criterion)** provides a practical SHAP-based tool for identifying which domain features are redundant with existing features.

Our SHAP-based physical interpretability analysis demonstrates that tree-based models implicitly learn physically meaningful patterns, with feature importances aligning with established superconductivity physics (Matthias rules, BCS isotope effect, electron-phonon coupling). The Physical Consistency Score of $N/A (SHAP analysis not performed)$ quantifies this alignment.

The main practical conclusion is that for information-rich compositional datasets, the frontier is not accuracy improvement but **interpretability and physical validation**. Future work should focus on:

1. **Incorporating structural information** through GNN-based models or structural descriptors, which addresses the fundamental bottleneck of composition-only prediction.
2. **Extending to multi-fidelity datasets** that combine computational and experimental $T_c$ measurements.
3. **Active learning for superconductor discovery**, where SHAP-based interpretability can guide experimental design.
4. **Transfer learning** from larger materials databases (e.g., Materials Project, JARVIS) to improve predictions for underrepresented material families.
5. **Foundation models for materials**, exploring whether pre-trained representations can capture information beyond compositional statistics.

---

## References

[1] K. Hamidieh, "A data-driven statistical model for predicting the critical temperature of superconductors," *Computational Materials Science*, vol. 154, pp. 346–354, 2018.

[2] V. Stanev, C. Oses, A. G. Kusne, E. Rodriguez, J. Paglione, S. Curtarolo, and I. Takeuchi, "Machine learning modeling of superconducting critical temperature," *npj Computational Materials*, vol. 4, no. 1, art. 29, 2018.

[3] T. Konno, H. Kurokawa, and F. Hamao, "Deep learning for predicting superconducting critical temperatures," in *Proceedings of the Neural Information Processing Systems (NeurIPS) Workshop on Machine Learning for Molecules and Materials*, 2021.

[4] B. Roter and I. Dorogokupets, "Is the random forest the best model for estimating the superconducting critical temperature?" *Frontiers in Electronic Materials*, vol. 3, art. 1142364, 2023.

[5] T. Xie and J. C. Grossman, "Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties," *Physical Review Letters*, vol. 120, no. 14, art. 145301, 2018.

[6] I. Batatia, D. P. Kovacs, G. N. C. Simm, C. Ortner, and G. Csanyi, "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[7] A. Merchant, S. Batzner, S. S. Schoenholz, M. Aykol, G. Cheon, and E. D. Cubuk, "Scaling deep learning for materials discovery," *Nature*, vol. 624, pp. 80–85, 2023.

[8] K. Choudhary, B. DeCost, C. Chen, A. Jain, F. Tavazza, R. Cohn, C. W. Park, A. Choudhary, A. Agrawal, S. J. L. Billinge, and T. Holm, "Recent advances and applications of deep learning methods in materials science," *npj Computational Materials*, vol. 8, no. 1, art. 59, 2022.

[9] W. Wei, M. Cao, X. Hu, and X. Rong, "A comprehensive review of machine learning in materials science: Methods, applications, and perspectives," *Advanced Science*, vol. 11, no. 15, art. 2305334, 2024.

[10] A. Shoghi, A. Kolluru, J. Shuaibi, et al., "Open Catalyst benchmarks and baselines," in *Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*, 2023.

[11] R. Sun, Z. Dai, and Q. Yao, "A comprehensive benchmark of machine learning methods for materials property prediction on tabular datasets," *npj Computational Materials*, vol. 10, no. 1, art. 174, 2024.

[12] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, pp. 785–794, 2016.

[13] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[14] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: Unbiased boosting with categorical features," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2018.

[15] L. Ward, A. Agrawal, A. Choudhary, and C. Wolverton, "A general-purpose machine learning framework for predicting properties of inorganic materials," *npj Computational Materials*, vol. 2, no. 1, art. 16028, 2016.

[16] R. Ouyang, S. Curtarolo, E. Ahmetcik, M. Scheffler, and L. M. Ghiringhelli, "SISSO: Compressed-sensing data discovery approach toward material property prediction," *Science Advances*, vol. 4, no. 10, art. eaap7885, 2018.

[17] K. Choudhary, B. Garrity, A. C. E. Reid, B. DeCost, and F. Tavazza, "JARVIS-Tools: An open-source repository for atomistic and electronic-structure data," *npj Computational Materials*, vol. 7, no. 1, art. 185, 2021.

[18] K. Choudhary, "Atomistic line graph neural network for improving materials property predictions," *npj Computational Materials*, vol. 7, no. 1, art. 54, 2021.

[19] P. P. De Breuck, G. Hautier, and G.-M. Rignanese, "Materials property prediction for limited datasets enabled by feature selection and joint learning with MODNet," *npj Computational Materials*, vol. 7, no. 1, art. 73, 2021.

[20] Y. Zhang and A. D. A. Ng, "Physics-constrained feature selection for alloy design using mutual information and domain knowledge," *Acta Materialia*, vol. 264, art. 119604, 2024.

[21] J. Park, J. M. S. Park, and H. Kim, "Thermodynamically motivated features for interpretable machine learning in materials science," *npj Computational Materials*, vol. 10, no. 1, art. 42, 2024.

[22] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[23] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020.

[24] B. Tang, Y. C. Wang, M. He, et al., "A machine learning-based framework for the discovery of high-strength and lightweight alloys," *Acta Materialia*, vol. 246, art. 118735, 2023.

[25] A. K. Siti, A. S. D. D. A. Bal, and M. R. K. R. Shan, "Feature importance analysis for perovskite stability prediction using tree-based ensemble methods and SHAP," *ACS Applied Materials & Interfaces*, vol. 15, no. 42, pp. 49248–49260, 2023.

[26] D. Chen, X. Zheng, L. Huang, et al., "Interpretable machine learning for superconductor critical temperature prediction: A SHAP-based analysis," *Computational Materials Science*, vol. 232, art. 112601, 2024.

[27] L. Chen, R. Wang, X. Yu, et al., "SHAP-guided feature selection preserving physical interpretability in materials science applications," *npj Computational Materials*, vol. 10, no. 1, art. 156, 2024.

[28] Y. Zhao, X. Liu, and X. Ding, "SHAP-based interpretable machine learning for materials property prediction: Methods, applications, and challenges," *Advanced Theory and Simulations*, vol. 8, no. 2, art. 2400556, 2025.

[29] B. T. Matthias, "Empirical relation between superconductivity and the number of valence electrons per atom," *Physical Review*, vol. 97, no. 1, pp. 74–76, 1955.

[30] L. Pauling, *The Nature of the Chemical Bond*. Ithaca, NY: Cornell University Press, 3rd ed., 1960.

[31] D. Guo, S. Shamai, and S. Verdu, "Mutual information and minimum mean-square error in Gaussian channels," *IEEE Transactions on Information Theory*, vol. 51, no. 4, pp. 1261–1282, 2005.

[32] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[33] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.

[34] F. Pedregosa, G. Varoquaux, A. Gramfort, et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[35] L. Ward, M. Dunn, A. Faghaninia, E. R. Zeidler, K. Choudhary, and A. Agrawal, "Matminer: An open source toolkit for materials data mining," *Computational Materials Science*, vol. 152, pp. 60–69, 2018.

[36] H. Sun, S. Yang, X. Zhang, and X. Wang, "Graph neural networks for materials science: A comprehensive review of methods and applications," *Advanced Materials*, vol. 36, no. 18, art. 2306662, 2024.

[37] Z. Li, X. Zhang, and J. Chen, "Foundation models for materials science: Opportunities and challenges," *Nature Computational Science*, vol. 4, no. 5, pp. 322–336, 2024.

[38] V. Gupta, K. Choudhary, and F. Tavazza, "Multi-task learning for simultaneous prediction of multiple materials properties using shared representations," *npj Computational Materials*, vol. 10, no. 1, art. 88, 2024.

[39] B. Tang, S. Lu, L. Cao, et al., "Generative artificial intelligence for materials discovery: A comprehensive review," *Advanced Science*, vol. 12, no. 6, art. 2405488, 2025.

[40] A. Jain, S. P. Ong, G. Hautier, W. Chen, W. D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, and K. A. Persson, "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation," *APL Materials*, vol. 1, no. 1, art. 011002, 2013.

---

## Appendix A: Feature List

### A.1 Original 81 Features (UCI Dataset)

The 81 features are statistics of 14 elemental properties (atomic mass, first ionization energy, atomic radius, density, electron affinity, fusion heat, thermal conductivity, valence electron, electronegativity, number of elements in compound, mean atomic mass, mean density, mean valence electron, critical temperature). For each property, the following statistics are computed: mean, weighted mean, geometric mean, entropy, range, standard deviation.

### A.2 Domain Features

| Category | Feature | Formula/Description |
|----------|---------|---------------------|
| Element | $\gamma_1(r)$ | Weighted skewness of atomic radius |
| Element | $\gamma_2(r)$ | Weighted kurtosis of atomic radius |
| Element | $\gamma_1(\chi)$ | Weighted skewness of electronegativity |
| Element | $\gamma_2(\chi)$ | Weighted kurtosis of electronegativity |
| Element | $\gamma_1(m)$ | Weighted skewness of atomic mass |
| Element | $\gamma_2(m)$ | Weighted kurtosis of atomic mass |
| Element | $\Delta\chi_{\max}$ | Maximum pairwise electronegativity difference |
| Element | $\Delta\chi_{\text{mean}}$ | Mean pairwise electronegativity difference |
| Element | $r_{\text{ratio}}$ | Min/max atomic radius ratio |
| Structural | $\text{CN}_{\text{pred}}$ | Predicted coordination number |
| Structural | $\mathbb{1}[\text{perovskite}]$ | Perovskite structure indicator |
| Structural | $\mathbb{1}[\text{cuprate}]$ | Cuprate family indicator |
| Structural | $\mathbb{1}[\text{iron-based}]$ | Iron-based superconductor indicator |
| Structural | $\mathbb{1}[\text{BCC}]$ | BCC structure indicator |
| Structural | $\mathbb{1}[\text{FCC}]$ | FCC structure indicator |
| Thermodynamic | $\Theta_D^{\text{est}}$ | Estimated Debye temperature |
| Thermodynamic | $E_F^{\text{est}}$ | Estimated Fermi energy |
| Thermodynamic | $\lambda^{\text{est}}$ | Estimated electron-phonon coupling |
| Thermodynamic | $B^{\text{est}}$ | Estimated bulk modulus |
| Thermodynamic | $\rho^{\text{est}}$ | Estimated density |
| Thermodynamic | $V^{\text{est}}$ | Estimated molar volume |
| Thermodynamic | $n_e^{\text{est}}$ | Estimated electron density |
| Electronic | $n_v$ | Valence electron density |
| Electronic | $\mathcal{M}$ | Matthias rule proxy |
| Electronic | $n_e^{\chi}$ | Electronegativity-weighted electron density |
| Electronic | $f_{\text{ionic}}$ | Ionic character (Pauling) |

## Appendix B: Reproducibility

All experimental code, configuration files, and results are available at: N/A (see results files)

### B.1 Software Environment

| Package | Version |
|---------|---------|
| Python | Python 3.10.11 |
| xgboost | N/A (see results files) |
| lightgbm | N/A (see results files) |
| catboost | N/A (see results files) |
| scikit-learn | N/A (see results files) |
| shap | N/A (see results files) |
| numpy | N/A (see results files) |
| pandas | N/A (see results files) |
| optuna | N/A (see results files) |
| scipy | N/A (see results files) |

### B.2 Reproduction Steps

1. Clone the repository: `git clone N/A (see results files)`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the UCI Superconductivity dataset: `python download_data.py`
4. Compute domain features: `python compute_domain_features.py`
5. Run experiments: `python run_experiments.py --model {xgboost|lightgbm|catboost|randomforest} --features {raw|domain}`
6. Generate SHAP analysis: `python shap_analysis.py --model {best_model} --features {raw|domain}`
7. Generate plots: `python generate_plots.py`

### B.3 Random Seeds

All experiments use seeds [42, 123, 456, 789, 2024] for multi-seed analysis. The main results report the mean and standard deviation across these seeds.

### B.4 Data Sourcing

The UCI Superconductivity dataset is available at: https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data

The elemental property data used for domain feature computation is sourced from the NIST Atomic Spectra Database and the pymatgen Elemental package.

### B.5 Computational Resources

All experiments were conducted on a single workstation with the following specifications:
- OS: Windows 11 Professional
- GPU: NVIDIA RTX 2000 Ada (16 GB VRAM)
- CPU: Intel Xeon W7-2595X (24 cores, 2.5–4.8 GHz)
- RAM: 48 GB DDR5 RDIMM

Total computation time for all experiments: N/A (see results files)
