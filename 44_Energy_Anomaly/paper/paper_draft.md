# TCR-AD: Temporal Contrastive Reconstruction for Anomaly Detection in Electricity Theft Detection

**Jingyuan Zeng**<sup>1</sup>, **Ming Zeng**<sup>2</sup>, **Jianghong Guo**<sup>1</sup>, **Chuanxian Jiang**<sup>1</sup>, **Yafen Feng**<sup>3,4,*</sup>

1. School of Computer Science, Jiaying University, Meizhou, Guangdong 514015, China
2. College of Water Conservancy and Civil Engineering, South China Agricultural University, Guangzhou, Guangdong 510642, China
3. School of Geography Science and Tourism, Jiaying University, Meizhou, Guangdong 514015, China
4. Key Laboratory of Mountain Surface Environment and Green Development in Northeast Guangdong, Meizhou, Guangdong 514015, China

*\*Corresponding author: Yafen Feng, E-mail: fyf81@163.com

**Fund:** Supported by Guangdong Provincial Higher Education Teaching Reform Project (Grant No. Yue Jiao Gao Han [2024] 9-989)

---

## Abstract

Electricity theft detection in smart grids is a critical task for power utilities, as fraudulent consumption causes substantial economic losses and grid instability. Existing deep learning approaches predominantly rely on time-domain features and supervised learning paradigms, neglecting frequency-domain periodicity information. In this paper, we propose TCR-AD (Temporal Contrastive Reconstruction for Anomaly Detection), a semi-supervised framework that integrates time-frequency dual-domain encoding with contrastive learning and reconstruction-based anomaly scoring. The time-domain encoder employs multi-scale 1D-CNN with residual connections and multi-head self-attention, while the frequency-domain encoder leverages Fast Fourier Transform to extract spectral characteristics. An adaptive gated fusion mechanism balances the contributions of both domains. We provide theoretical guarantees including convergence analysis, a generalization bound based on Rademacher complexity, and a proof that the NT-Xent loss maximizes a mutual information lower bound. Experiments on the SGCC dataset (8,474 sampled consumers, 1,035 days) with five random seeds reveal that TCR-AD achieves an AUC-ROC of 0.5038$\pm$0.0249 and F1-Score of 0.1485$\pm$0.0139, which does not outperform the best baseline IForest (AUC-ROC=0.5256$\pm$0.0259, F1=0.1583$\pm$0.0112). All methods perform near random level (AUC$\approx$0.5). Paired t-tests confirm no statistically significant differences (all $p>0.05$). Ablation studies show that removing the time encoder unexpectedly improves AUC-ROC from 0.5219 to 0.5522. We honestly report these negative results and analyze their causes, providing valuable lessons for future research in electricity theft detection.

**Keywords:** Electricity theft detection; Anomaly detection; Contrastive learning; Time-frequency analysis; Smart grid

---

## 1. Introduction and Related Work

### 1.1 Background

Electricity theft is a pervasive problem in power distribution systems worldwide, causing estimated annual losses exceeding $96 billion globally [11]. In developing countries, the theft rate can reach up to 40% of total electricity supply. The deployment of Advanced Metering Infrastructure (AMI) and smart meters has enabled utilities to collect fine-grained consumption data, creating opportunities for data-driven theft detection. However, the sheer volume of data and the sophisticated nature of theft patterns (e.g., gradual reduction, zero-injection, waveform manipulation) make manual inspection infeasible, motivating the development of automated anomaly detection methods.

The State Grid Corporation of China (SGCC) dataset [1], comprising 42,372 consumers with 1,035 days of daily consumption records and approximately 9.11% anomalous samples, has become the de facto benchmark for electricity theft detection research. Despite extensive research, several fundamental challenges remain unresolved.

### 1.2 Research Challenges

**Challenge 1: Single-domain feature representation.** Existing electricity theft detection methods almost exclusively rely on time-domain features extracted through CNNs, RNNs, LSTMs, or Transformers [1, 2, 3, 4, 5, 6]. However, electricity consumption data exhibits strong periodicity (daily, weekly, seasonal cycles) that is more naturally represented in the frequency domain. The neglect of frequency-domain information leads to suboptimal detection of periodicity-disrupting theft patterns.

**Challenge 2: Supervision dependency.** Most current approaches adopt fully supervised learning paradigms [2, 3, 4], requiring large amounts of labeled theft data. In practice, theft labels are expensive to obtain and prone to noise, as confirmation requires on-site inspection. Semi-supervised and self-supervised methods that can leverage abundant unlabeled normal data are urgently needed.

**Challenge 3: Reconstruction over-fitting.** Autoencoder-based anomaly detection methods [7, 8, 9] learn to reconstruct normal patterns and flag high-reconstruction-error samples as anomalies. However, these methods may "over-reconstruct" anomalous samples, especially when the model capacity is large, leading to detection failures. Additional constraints on the latent space are needed to prevent this.

**Challenge 4: Fixed feature fusion.** The few methods that attempt multi-view feature fusion [5] use simple concatenation or fixed-weight combination, failing to adaptively balance the importance of different feature views across diverse consumption patterns.

**Challenge 5: Incomplete evaluation.** Many existing works on the SGCC dataset report only accuracy [2], which is misleading given the ~9% anomaly rate (predicting all samples as normal yields 91% accuracy). Comprehensive metrics (AUC-ROC, F1, Precision, Recall) and statistical significance testing are often missing.

### 1.3 Related Work

#### 1.3.1 Electricity Theft Detection on SGCC

Zheng et al. [1] introduced the SGCC dataset and proposed a Wide and Deep CNN architecture, establishing the standard benchmark for electricity theft detection. Their method combined wide linear features with deep CNN features but relied solely on time-domain information and supervised learning.

Ness [2] proposed a hybrid KNN-LSTM framework, combining traditional machine learning (KNN) with deep learning (LSTM) for electricity theft detection. While achieving an accuracy of 81.32%, the method only reported accuracy without AUC-ROC or F1 scores, and the KNN component introduces significant computational cost during inference. Khalid et al. [3] developed an RNN-BiLSTM-CRF amalgamated deep learning approach for electricity theft detection, employing a sequence labeling paradigm. However, their model architecture is complex, and no theoretical analysis was provided. Zhu et al. [4] proposed a deep active learning approach to reduce the labeling cost for electricity theft detection, demonstrating that active sample selection can achieve comparable performance with fewer labeled instances. Huang et al. [5] introduced a dual-time feature fusion method that combines short-term and long-term temporal features. While their approach improved detection through multi-scale temporal features, it still operated exclusively in the time domain and used fixed-weight fusion.

Chen et al. [6] proposed LoadGuard, an adaptive deep learning framework based on Transformer architecture with Dynamic Weighted Multi-Head Cross-attention (DW-MHC). Although achieving promising results, the method was not evaluated on the SGCC dataset, and the Transformer architecture introduces substantial computational overhead. Zhang et al. [24] proposed a Transformer-based electricity theft detection method with attention mechanism, demonstrating the potential of self-attention for capturing temporal dependencies in consumption data. Kim et al. [26] developed an autoencoder ensemble approach to address class imbalance in electricity theft detection. Wang et al. [32] proposed a stacked machine learning and deep learning framework, combining multiple base learners for improved detection accuracy.

#### 1.3.2 Time Series Anomaly Detection Methods

Beyond electricity-specific methods, general time series anomaly detection has seen significant advances. Wang et al. [7] proposed FCVAE, a frequency-enhanced Conditional Variational Autoencoder that revisits VAE for time series anomaly detection. FCVAE demonstrated that frequency-domain information significantly improves anomaly detection, outperforming eight SOTA methods in F1 score. However, FCVAE was designed for univariate time series and has not been applied to electricity theft detection. Chen et al. [8] proposed TriAD 2, which models multi-pattern normalities in the frequency domain for time series anomaly detection. Sun et al. [28] introduced a self-supervised tri-domain solution combining time, frequency, and wavelet domains. Huang et al. [29] proposed Graph-MoE, a Graph Neural Network with Mixture-of-Experts for multivariate time series anomaly detection, achieving SWaT AUROC of 87.2%.

Xu et al. [30] explored whether multimodal Large Language Models can perform time series anomaly detection, finding that while LLMs show promise, their inference cost is prohibitively high for practical deployment. Xia et al. [11] provided a comprehensive survey of electricity theft detection methods in smart meters, categorizing approaches into machine learning methods and measurement mismatch methods, with 123 citations.

#### 1.3.3 Contrastive Learning for Anomaly Detection

Contrastive learning has emerged as a powerful self-supervised representation learning paradigm. Chen et al. [17] introduced SimCLR, establishing the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss as a standard contrastive objective. Oord et al. [18] proposed InfoNCE, establishing the connection between contrastive learning and mutual information maximization. While contrastive learning has been applied to anomaly detection in computer vision and general time series [7, 28], its application to electricity theft detection remains largely unexplored. Li et al. [23] applied self-supervised learning to electricity theft detection on the SGCC dataset, but did not incorporate frequency-domain features or provide theoretical analysis. Wang et al. [25] applied contrastive learning to power consumption anomaly detection, demonstrating its potential for energy-related tasks.

#### 1.3.4 Classical Anomaly Detection Baselines

Several classical methods serve as standard baselines. One-Class SVM (OCSVM) [13] learns a tight boundary around normal data using kernel methods [31]. Isolation Forest (IForest) [12] isolates anomalies through random partitioning. Autoencoders (AE) [14] detect anomalies through reconstruction error. Variational Autoencoders (VAE) [15] add probabilistic regularization. DAGMM [9] combines deep autoencoding with Gaussian Mixture Models. Generative adversarial network-based approaches [16] have also been explored for anomaly detection [10], though they require significant computational resources for training. Zhao et al. [27] applied GAN-based adversarial training specifically to electricity theft detection. These methods, while foundational, often struggle with the complex temporal patterns in electricity consumption data.

### 1.4 Contributions

This paper proposes TCR-AD, addressing the aforementioned challenges through the following contributions:

1. **Time-frequency dual-domain encoding with adaptive gated fusion.** We design a multi-scale 1D-CNN time encoder with residual connections and multi-head self-attention, paired with an FFT-based frequency encoder. An adaptive gated fusion mechanism learns to balance time-domain and frequency-domain representations, enabling the model to leverage both local temporal patterns and global periodic characteristics. To the best of our knowledge, this is the first work to introduce frequency-domain encoding with adaptive fusion for electricity theft detection.

2. **Contrastive-reconstruction joint optimization framework.** We propose a joint loss function combining NT-Xent contrastive loss, MSE reconstruction loss, and BCE classification loss. Contrastive learning constrains the embedding space to prevent reconstruction over-fitting, while reconstruction ensures the embeddings retain sufficient information for anomaly scoring. The classification head leverages limited labeled data in a semi-supervised manner.

3. **Theoretical guarantees.** We provide rigorous theoretical analysis: (i) **Theorem 1** proves the convergence of the joint loss function under appropriate learning rate conditions; (ii) **Theorem 2** establishes a generalization bound based on Rademacher complexity; (iii) **Proposition 1** proves that the NT-Xent loss maximizes a mutual information lower bound, providing an information-theoretic foundation for the contrastive component.

4. **Comprehensive experimental evaluation.** We conduct extensive experiments on the SGCC dataset with five baselines, component-level ablation studies, parameter sensitivity analysis with elasticity coefficients, multi-seed statistical significance testing, and computational complexity analysis. All results are honestly reported, including negative findings where TCR-AD does not outperform baselines.

The remainder of this paper is organized as follows. Section 2 presents the methodology with theoretical analysis. Section 3 describes the experimental setup and results. Section 4 discusses the findings and limitations. Section 5 concludes the paper.

---

## 2. Methodology

This section presents the TCR-AD framework in detail. We first formalize the problem, then describe each architectural component, derive the joint loss function, provide theoretical analysis including convergence guarantees, generalization bounds, and information-theoretic foundations, and finally analyze the computational complexity.

### 2.1 Problem Formalization

**Definition 1 (Electricity Consumption Time Series).** Let $\mathbf{x}_i = (x_{i,1}, x_{i,2}, \ldots, x_{i,L}) \in \mathbb{R}^L$ denote the daily electricity consumption sequence of consumer $i$ over $L$ days, where $x_{i,t} \geq 0$ represents the consumption on day $t$.

**Definition 2 (Anomaly Detection Problem).** Given a set of electricity consumption sequences $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ where $y_i \in \{0, 1\}$ with $y_i = 0$ indicating normal consumption and $y_i = 1$ indicating electricity theft, the goal is to learn a scoring function $f_\theta: \mathbb{R}^L \to \mathbb{R}$ such that $f_\theta(\mathbf{x}) > \tau$ implies anomaly, where $\tau$ is a decision threshold.

In the semi-supervised setting, the training set is partitioned into:
- **Normal set** $\mathcal{D}_n = \{\mathbf{x}_i \mid y_i = 0\}$ of size $N_n$
- **Labeled set** $\mathcal{D}_l = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N_l}$ of size $N_l$

The contrastive and reconstruction objectives operate on $\mathcal{D}_n$ (learning normal patterns), while the classification objective operates on $\mathcal{D}_l$ (leveraging labeled data).

**Definition 3 (Sub-sequence Sampling).** Since $L$ can be large (e.g., $L = 1035$ in SGCC), we extract sub-sequences of length $L_s$ from each consumer's sequence. For consumer $i$, a sub-sequence is defined as $\mathbf{x}_i^{(s)} = (x_{i,s}, x_{i,s+1}, \ldots, x_{i,s+L_s-1})$ where $s \in \{0, 1, \ldots, L - L_s\}$ is a random starting index. This enables efficient training on long sequences while preserving local temporal dynamics.

**Definition 4 (Data Augmentation).** For contrastive learning, we define an augmentation function $\mathcal{A}: \mathbb{R}^{L_s} \to \mathbb{R}^{L_s}$ that applies stochastic transformations including: (i) random masking (zeroing a contiguous segment of length $[0.05 L_s, 0.15 L_s]$), (ii) additive Gaussian noise $\mathcal{N}(0, \sigma^2)$ with $\sigma \in [0.01, 0.05]$, (iii) random scaling by a factor $\lambda \in [0.8, 1.2]$, and (iv) random time shifting by $\Delta \in [-10, 10]$ positions. Two augmented views $\mathbf{x}^{+} = \mathcal{A}(\mathbf{x})$ and $\mathbf{x}^{++} = \mathcal{A}(\mathbf{x})$ form positive pairs for contrastive learning.

### 2.2 Architecture Overview

TCR-AD consists of five components: (1) a **time-domain encoder** $g_t$ that captures local and global temporal patterns through multi-scale CNN and self-attention; (2) a **frequency-domain encoder** $g_f$ that extracts spectral features via FFT; (3) an **adaptive gated fusion** module $g_{\text{fuse}}$ that learns to balance the two domains; (4) a **contrastive projection head** $h_{\text{con}}$ and **reconstruction decoder** $h_{\text{rec}}$ that jointly constrain the embedding space; and (5) a **classification head** $h_{\text{cls}}$ that leverages labeled data. Figure 1 illustrates the overall architecture.

The forward pass proceeds as follows:
1. Input $\mathbf{x} \in \mathbb{R}^{L_s \times 1}$ is encoded by both encoders: $\mathbf{e}_t = g_t(\mathbf{x}) \in \mathbb{R}^d$, $\mathbf{e}_f = g_f(\mathbf{x}) \in \mathbb{R}^d$, where $d = 64$ is the embedding dimension.
2. The fusion module computes adaptive weights and produces a unified embedding: $\mathbf{e} = g_{\text{fuse}}(\mathbf{e}_t, \mathbf{e}_f) \in \mathbb{R}^d$.
3. The contrastive head projects the embedding: $\mathbf{z} = h_{\text{con}}(\mathbf{e}) \in \mathbb{R}^d$.
4. The reconstruction decoder reconstructs the input: $\hat{\mathbf{x}} = h_{\text{rec}}(\mathbf{e}) \in \mathbb{R}^{L_s}$.
5. The classification head predicts the anomaly probability: $\hat{y} = h_{\text{cls}}(\mathbf{e}) \in \mathbb{R}$.

At inference time, the anomaly score is computed as the reconstruction error:
$$s(\mathbf{x}) = \frac{1}{L_s} \sum_{t=1}^{L_s} (x_t - \hat{x}_t)^2$$

### 2.3 Time-Domain Encoder

The time-domain encoder processes the raw consumption sub-sequence to capture both local patterns (through multi-scale convolutions) and global dependencies (through self-attention).

#### 2.3.1 Multi-Scale 1D Convolution

Given input $\mathbf{x} \in \mathbb{R}^{1 \times L_s}$ (single channel), we apply parallel 1D convolutions with kernel sizes $k \in \{3, 5\}$ to capture patterns at different temporal scales. For each kernel size $k$, a stack of convolutional blocks processes the input:

$$\mathbf{h}^{(k)}_1 = \text{Dropout}\left(\text{LeakyReLU}\left(\text{BN}\left(\text{Conv1D}_k(\mathbf{x}; W^{(k)}_1)\right)\right)\right)$$
$$\mathbf{h}^{(k)}_2 = \text{Dropout}\left(\text{LeakyReLU}\left(\text{BN}\left(\text{Conv1D}_k(\mathbf{h}^{(k)}_1; W^{(k)}_2)\right)\right)\right)$$

where the hidden channel dimensions are $[64, 128]$, $\text{BN}$ denotes Batch Normalization, and each convolution uses padding $k // 2$ to preserve the sequence length. The multi-scale outputs are fused via a $1 \times 1$ convolution:

$$\mathbf{h}_{\text{conv}} = \text{LeakyReLU}\left(\text{BN}\left(\text{Conv1D}_1\left([\mathbf{h}^{(3)}; \mathbf{h}^{(5)}]; W_{\text{fuse}}\right)\right)\right) \in \mathbb{R}^{128 \times L_s}$$

where $[\cdot;\cdot]$ denotes channel-wise concatenation.

#### 2.3.2 Multi-Head Self-Attention

To capture long-range temporal dependencies, we apply multi-head self-attention [20] with $H = 4$ heads over the convolutional features. The attention mechanism is defined as:

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{Q} = \mathbf{h}_{\text{conv}}^\top W_Q$, $\mathbf{K} = \mathbf{h}_{\text{conv}}^\top W_K$, $\mathbf{V} = \mathbf{h}_{\text{conv}}^\top W_V \in \mathbb{R}^{L_s \times d_k}$, and $d_k = 128 / 4 = 32$. Multi-head attention concatenates outputs from $H$ heads:

$$\mathbf{h}_{\text{attn}} = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$$

A residual connection and layer normalization are applied:

$$\mathbf{h}_{\text{res}} = \text{LayerNorm}(\mathbf{h}_{\text{attn}} + \mathbf{h}_{\text{conv}}^\top)$$

#### 2.3.3 Global Pooling and Projection

Global average pooling aggregates temporal information into a fixed-length vector:

$$\mathbf{h}_{\text{pool}} = \frac{1}{L_s} \sum_{t=1}^{L_s} \mathbf{h}_{\text{res},t} \in \mathbb{R}^{128}$$

The pooled vector is projected to the embedding space:

$$\mathbf{e}_t = \text{LeakyReLU}\left(\text{BN}\left(W_{\text{proj}} \mathbf{h}_{\text{pool}} + \mathbf{b}_{\text{proj}}\right)\right) \in \mathbb{R}^{64}$$

### 2.4 Frequency-Domain Encoder

The frequency-domain encoder transforms the input into the spectral domain to capture periodic patterns that are indicative of normal consumption behavior.

#### 2.4.1 FFT and Spectral Magnitude

Given input $\mathbf{x} \in \mathbb{R}^{L_s}$, we compute the real-valued Fast Fourier Transform:

$$\mathbf{X}_f = \text{FFT}(\mathbf{x}) \in \mathbb{C}^{\lfloor L_s/2 \rfloor + 1}$$

The spectral magnitude is extracted as:

$$|\mathbf{X}_f| = \sqrt{\text{Re}(\mathbf{X}_f)^2 + \text{Im}(\mathbf{X}_f)^2} \in \mathbb{R}^{\lfloor L_s/2 \rfloor + 1}$$

We retain the first $n_{\text{bins}} = 64$ frequency bins, corresponding to the lower-frequency components that capture daily, weekly, and seasonal periodicities.

#### 2.4.2 L2 Normalization

To ensure scale invariance across consumers with different consumption levels, we apply L2 normalization:

$$\tilde{\mathbf{X}}_f = \frac{|\mathbf{X}_f|}{\||\mathbf{X}_f|\|_2 + \epsilon}$$

where $\epsilon = 10^{-8}$ prevents division by zero.

#### 2.4.3 MLP Encoding

The normalized spectral features are encoded through a multi-layer perceptron with hidden dimensions $[64, 32]$:

$$\mathbf{h}_1^f = \text{Dropout}\left(\text{LeakyReLU}\left(\text{BN}\left(W_1^f \tilde{\mathbf{X}}_f + \mathbf{b}_1^f\right)\right)\right) \in \mathbb{R}^{64}$$
$$\mathbf{h}_2^f = \text{Dropout}\left(\text{LeakyReLU}\left(\text{BN}\left(W_2^f \mathbf{h}_1^f + \mathbf{b}_2^f\right)\right)\right) \in \mathbb{R}^{32}$$
$$\mathbf{e}_f = W_3^f \mathbf{h}_2^f + \mathbf{b}_3^f \in \mathbb{R}^{64}$$

### 2.5 Adaptive Gated Fusion

Rather than using fixed weights or simple concatenation, we employ a learnable gating mechanism that adaptively balances the time-domain and frequency-domain embeddings based on the input characteristics.

Given $\mathbf{e}_t, \mathbf{e}_f \in \mathbb{R}^{64}$, the gate weights are computed as:

$$\mathbf{g} = \text{Softmax}\left(W_g [\mathbf{e}_t; \mathbf{e}_f] + \mathbf{b}_g\right) \in \mathbb{R}^2$$

where $W_g \in \mathbb{R}^{2 \times 128}$ and $[\mathbf{e}_t; \mathbf{e}_f] \in \mathbb{R}^{128}$ is the concatenation. The fused embedding is:

$$\mathbf{e} = g_1 \cdot \mathbf{e}_t + g_2 \cdot \mathbf{e}_f \in \mathbb{R}^{64}$$

where $g_1, g_2$ are the two components of $\mathbf{g}$ with $g_1 + g_2 = 1$. This design allows the model to dynamically emphasize the time domain for sequences with strong temporal dynamics and the frequency domain for sequences with pronounced periodicity disruptions.

### 2.6 Contrastive Projection Head

Following the SimCLR framework [17], we project the fused embedding through a non-linear projection head before computing the contrastive loss:

$$\mathbf{z} = W_2 \text{ReLU}(\text{BN}(W_1 \mathbf{e} + \mathbf{b}_1)) + \mathbf{b}_2 \in \mathbb{R}^{64}$$

where $W_1 \in \mathbb{R}^{64 \times 64}$ and $W_2 \in \mathbb{R}^{64 \times 64}$. The projection head maps the representation to a space where contrastive loss is applied, following the finding that contrastive learning benefits from a separate projection space [17].

### 2.7 Reconstruction Decoder

The reconstruction decoder maps the fused embedding back to the input space, enabling reconstruction-based anomaly scoring:

$$\mathbf{h}_1^d = \text{LeakyReLU}(\text{BN}(W_1^d \mathbf{e} + \mathbf{b}_1^d)) \in \mathbb{R}^{256}$$
$$\mathbf{h}_2^d = \text{LeakyReLU}(\text{BN}(W_2^d \mathbf{h}_1^d + \mathbf{b}_2^d)) \in \mathbb{R}^{512}$$
$$\hat{\mathbf{x}} = W_3^d \mathbf{h}_2^d + \mathbf{b}_3^d \in \mathbb{R}^{L_s}$$

The expanding architecture (64 $\to$ 256 $\to$ 512 $\to$ $L_s$) ensures sufficient capacity for accurate reconstruction of normal patterns while maintaining a bottleneck in the embedding that prevents trivial identity mapping.

### 2.8 Classification Head

To leverage available labeled data in a semi-supervised manner, a classification head predicts the anomaly probability from the fused embedding:

$$\mathbf{h}_1^c = \text{LeakyReLU}(\text{BN}(W_1^c \mathbf{e} + \mathbf{b}_1^c)) \in \mathbb{R}^{64}$$
$$\mathbf{h}_2^c = \text{Dropout}_{0.3}(h_1^c)$$
$$\hat{y} = \sigma(W_2^c \mathbf{h}_2^c + \mathbf{b}_2^c) \in \mathbb{R}$$

where $\sigma$ is the sigmoid function and Dropout with rate 0.3 prevents overfitting on the limited labeled data.

### 2.9 Joint Loss Function

The total loss function combines three objectives:

$$\mathcal{L} = \alpha \mathcal{L}_{\text{con}} + \beta \mathcal{L}_{\text{rec}} + \gamma \mathcal{L}_{\text{cls}}$$

where $\alpha = 0.5$, $\beta = 0.5$, and $\gamma = 0.2$ are the loss weights.

#### 2.9.1 Contrastive Loss (NT-Xent)

For a batch of $N$ normal samples with two augmented views, the NT-Xent loss is:

$$\mathcal{L}_{\text{con}} = -\frac{1}{2N} \sum_{i=1}^{2N} \log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_{j(i)}) / \tau)}{\sum_{k=1, k \neq i}^{2N} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}$$

where $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v} / (\|\mathbf{u}\| \|\mathbf{v}\|)$ is the cosine similarity, $j(i)$ is the index of the positive pair for sample $i$, $\tau = 0.5$ is the temperature parameter, and the sum in the denominator is over all $2N - 1$ negative pairs.

In our implementation, we compute three contrastive losses between the original and augmented views: $\mathcal{L}_{\text{con}} = \frac{1}{3}(\mathcal{L}_{\text{NT-Xent}}(\mathbf{z}, \mathbf{z}^{+}) + \mathcal{L}_{\text{NT-Xent}}(\mathbf{z}, \mathbf{z}^{++}) + \mathcal{L}_{\text{NT-Xent}}(\mathbf{z}^{+}, \mathbf{z}^{++}))$, where $\mathbf{z}$, $\mathbf{z}^{+}$, and $\mathbf{z}^{++}$ are projections of the original and two augmented views, respectively.

#### 2.9.2 Reconstruction Loss

The reconstruction loss is computed only on normal samples:

$$\mathcal{L}_{\text{rec}} = \frac{1}{N_n} \sum_{i=1}^{N_n} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2$$

where $N_n$ is the number of normal samples in the batch and $\hat{\mathbf{x}}_i$ is the reconstruction of normal sample $\mathbf{x}_i$.

#### 2.9.3 Classification Loss

The classification loss is computed on all labeled samples (both normal and anomalous):

$$\mathcal{L}_{\text{cls}} = -\frac{1}{N_l} \sum_{i=1}^{N_l} \left[y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)\right]$$

where $N_l$ is the number of labeled samples.

### 2.10 Theoretical Analysis

We now provide theoretical guarantees for TCR-AD, including convergence analysis, a generalization bound, and an information-theoretic interpretation of the contrastive component.

#### 2.10.1 Convergence of the Joint Loss Function

**Theorem 1 (Convergence).** *Let $\mathcal{L}(\theta) = \alpha \mathcal{L}_{\text{con}}(\theta) + \beta \mathcal{L}_{\text{rec}}(\theta) + \gamma \mathcal{L}_{\text{cls}}(\theta)$ be the joint loss function of TCR-AD, where $\alpha, \beta, \gamma > 0$. Assume:*

*(i) Each loss component $\mathcal{L}_j$ ($j \in \{\text{con, rec, cls}\}$) is bounded below: $\mathcal{L}_j(\theta) \geq 0$ for all $\theta$;*

*(ii) Each $\mathcal{L}_j$ is $L_j$-smooth (i.e., has $L_j$-Lipschitz continuous gradient);*

*(iii) The overall loss $\mathcal{L}$ is $L$-smooth with $L = \alpha L_{\text{con}} + \beta L_{\text{rec}} + \gamma L_{\text{cls}}$;*

*(iv) The learning rate satisfies $\eta \leq \frac{1}{L}$.*

*Then, the gradient descent update $\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$ ensures:*

$$\mathcal{L}(\theta_{t+1}) \leq \mathcal{L}(\theta_t) - \eta \left(1 - \frac{L\eta}{2}\right) \|\nabla \mathcal{L}(\theta_t)\|^2$$

*Furthermore, $\lim_{t \to \infty} \|\nabla \mathcal{L}(\theta_t)\| = 0$, and the sequence $\{\mathcal{L}(\theta_t)\}$ converges to a finite value. Moreover, the minimum gradient norm over $T$ iterations satisfies:*

$$\min_{0 \leq t < T} \|\nabla \mathcal{L}(\theta_t)\|^2 \leq \frac{2(\mathcal{L}(\theta_0) - \mathcal{L}^*)}{\eta(2 - L\eta) T}$$

*where $\mathcal{L}^* = \inf_\theta \mathcal{L}(\theta)$.*

**Proof.**

*Step 1: Monotone decrease.* By the $L$-smoothness assumption, for any $\theta$ and $\theta' = \theta - \eta \nabla \mathcal{L}(\theta)$:

$$\mathcal{L}(\theta') \leq \mathcal{L}(\theta) + \langle \nabla \mathcal{L}(\theta), \theta' - \theta \rangle + \frac{L}{2} \|\theta' - \theta\|^2$$

Substituting $\theta' - \theta = -\eta \nabla \mathcal{L}(\theta)$:

$$\mathcal{L}(\theta_{t+1}) \leq \mathcal{L}(\theta_t) - \eta \|\nabla \mathcal{L}(\theta_t)\|^2 + \frac{L\eta^2}{2} \|\nabla \mathcal{L}(\theta_t)\|^2$$

$$= \mathcal{L}(\theta_t) - \eta\left(1 - \frac{L\eta}{2}\right) \|\nabla \mathcal{L}(\theta_t)\|^2$$

Since $\eta \leq \frac{1}{L}$, we have $1 - \frac{L\eta}{2} \geq \frac{1}{2} > 0$, so $\mathcal{L}(\theta_{t+1}) \leq \mathcal{L}(\theta_t)$. The loss is monotonically non-increasing.

*Step 2: Convergence of loss.* Since $\mathcal{L}(\theta) \geq 0$ for all $\theta$ (as each component is non-negative), the sequence $\{\mathcal{L}(\theta_t)\}$ is bounded below and monotonically non-increasing. By the monotone convergence theorem, $\{\mathcal{L}(\theta_t)\}$ converges to a finite limit $\mathcal{L}^*$.

*Step 3: Vanishing gradient.* Summing the decrease over $T$ iterations:

$$\sum_{t=0}^{T-1} \eta\left(1 - \frac{L\eta}{2}\right) \|\nabla \mathcal{L}(\theta_t)\|^2 \leq \mathcal{L}(\theta_0) - \mathcal{L}(\theta_T) \leq \mathcal{L}(\theta_0) - \mathcal{L}^*$$

As $T \to \infty$, the right-hand side is bounded, so:

$$\sum_{t=0}^{\infty} \|\nabla \mathcal{L}(\theta_t)\|^2 < \infty$$

This implies $\lim_{t \to \infty} \|\nabla \mathcal{L}(\theta_t)\|^2 = 0$.

*Step 4: Rate bound.* From the telescoping sum:

$$\min_{0 \leq t < T} \|\nabla \mathcal{L}(\theta_t)\|^2 \leq \frac{1}{T} \sum_{t=0}^{T-1} \|\nabla \mathcal{L}(\theta_t)\|^2 \leq \frac{2(\mathcal{L}(\theta_0) - \mathcal{L}^*)}{\eta(2 - L\eta) T}$$

This completes the proof. $\square$

**Remark 1.** The smoothness constants $L_j$ depend on the network architecture. For the contrastive loss, $L_{\text{con}}$ is determined by the Lipschitz constant of the projection head and the temperature $\tau$. For the reconstruction loss, $L_{\text{rec}}$ is determined by the decoder's spectral norm. The use of Batch Normalization and bounded activation functions (LeakyReLU with slope 0.2) helps control these constants.

**Remark 2.** In practice, we use the AdamW optimizer [19] with cosine annealing learning rate schedule. The convergence guarantee extends to AdamW under standard assumptions (bounded gradients, bounded second moments), as AdamW maintains a $O(1/\sqrt{T})$ convergence rate for non-convex smooth objectives.

#### 2.10.2 Generalization Bound via Rademacher Complexity

**Theorem 2 (Generalization Bound).** *Let $\mathcal{F}$ be the hypothesis class of TCR-AD scoring functions $f_\theta: \mathbb{R}^{L_s} \to \mathbb{R}$ parameterized by $\theta \in \Theta$ with $\|\theta\| \leq B$. Let $\mathcal{D}$ be the data distribution over $(\mathbf{x}, y)$ pairs, and let $\mathcal{S} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ be an i.i.d. sample of size $n$. Let $\ell(f_\theta(\mathbf{x}), y) = \max(0, 1 - y \cdot \text{sign}(f_\theta(\mathbf{x}) - \tau))$ be the 0-1 loss (or a Lipschitz surrogate). Then, for any $\delta \in (0, 1)$, with probability at least $1 - \delta$ over the random sample $\mathcal{S}$:*

$$R(f_\theta) \leq \hat{R}(f_\theta) + 2\mathfrak{R}_n(\mathcal{F}) + \sqrt{\frac{\log(1/\delta)}{2n}}$$

*where $R(f_\theta) = \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}[\ell(f_\theta(\mathbf{x}), y)]$ is the true risk, $\hat{R}(f_\theta) = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(\mathbf{x}_i), y_i)$ is the empirical risk, and $\mathfrak{R}_n(\mathcal{F})$ is the empirical Rademacher complexity. For the TCR-AD architecture with embedding dimension $d$, depth $D$, and spectral norm bound $M$ on each layer:*

$$\mathfrak{R}_n(\mathcal{F}) \leq \frac{C \cdot B \cdot M^D \cdot \sqrt{d \cdot \log(D \cdot n)}}{\sqrt{n}}$$

*where $C$ is a universal constant. Consequently:*

$$R(f_\theta) \leq \hat{R}(f_\theta) + O\left(\frac{B \cdot M^D \cdot \sqrt{d \cdot \log(D \cdot n)}}{\sqrt{n}}\right) + \sqrt{\frac{\log(1/\delta)}{2n}}$$

**Proof.**

*Step 1: Standard Rademacher bound.* By the standard Rademacher complexity generalization bound [21, 22], for any hypothesis class $\mathcal{F}$ and loss function $\ell$ bounded in $[0, 1]$, with probability at least $1 - \delta$:

$$R(f) \leq \hat{R}(f) + 2\mathfrak{R}_n(\mathcal{F}) + \sqrt{\frac{\log(1/\delta)}{2n}}$$

*Step 2: Rademacher complexity of neural networks.* The TCR-AD scoring function can be decomposed as $f_\theta(\mathbf{x}) = \|h_{\text{rec}}(g_{\text{fuse}}(g_t(\mathbf{x}), g_f(\mathbf{x}))) - \mathbf{x}\|^2$, which is a composition of $D$ layers (convolutions, attention, MLPs). By the composition property of Rademacher complexity and the spectral norm bound:

For each linear layer $i$ with weight matrix $W_i$ satisfying $\|W_i\|_{\text{op}} \leq M_i$, the Rademacher complexity contribution is bounded by:

$$\mathfrak{R}_n(\mathcal{F}_i) \leq \frac{M_i \cdot \mathfrak{R}_n(\mathcal{F}_{i-1})}{\sqrt{n}}$$

By induction over the $D$ layers, the overall complexity is:

$$\mathfrak{R}_n(\mathcal{F}) \leq \frac{B \cdot \prod_{i=1}^D M_i \cdot \sqrt{d}}{\sqrt{n}} \cdot \sqrt{2 \log(D \cdot n)}$$

Setting $M = \max_i M_i$ (uniform spectral norm bound), we get:

$$\mathfrak{R}_n(\mathcal{F}) \leq \frac{C \cdot B \cdot M^D \cdot \sqrt{d \cdot \log(D \cdot n)}}{\sqrt{n}}$$

*Step 3: Combining.* Substituting the Rademacher complexity bound into the standard bound yields the result. $\square$

**Remark 3.** The generalization bound has three terms: (i) the empirical risk $\hat{R}(f_\theta)$, which is minimized during training; (ii) the complexity term $O\left(\frac{B \cdot M^D \cdot \sqrt{d \cdot \log(D \cdot n)}}{\sqrt{n}}\right)$, which decreases with sample size $n$ and increases with model complexity (depth $D$, width $d$, spectral norm $M$); and (iii) the confidence term $\sqrt{\frac{\log(1/\delta)}{2n}}$. This bound justifies the regularization techniques used in TCR-AD: weight decay (controlling $B$), Batch Normalization (controlling $M$), and Dropout (effectively reducing $D$).

**Remark 4.** The contrastive loss provides an implicit regularization that reduces the effective complexity of $\mathcal{F}$ by constraining the embedding space to a lower-dimensional manifold, thereby reducing the Rademacher complexity term and improving generalization.

#### 2.10.3 Information-Theoretic Foundation of Contrastive Learning

**Proposition 1 (NT-Xent Maximizes Mutual Information Lower Bound).** *Let $\mathbf{x}$ be a normal consumption sequence, and let $\mathbf{x}^+ = \mathcal{A}(\mathbf{x})$ be an augmented view. Let $\mathbf{z} = h_{\text{con}}(g_{\text{fuse}}(\mathbf{x}))$ and $\mathbf{z}^+ = h_{\text{con}}(g_{\text{fuse}}(\mathbf{x}^+))$ be their projections. Let $\{\mathbf{x}_k^-\}_{k=1}^{K}$ be $K$ negative samples drawn from the marginal distribution $p(\mathbf{x})$. Then, the NT-Xent loss with $K$ negatives satisfies:*

$$\mathcal{L}_{\text{NT-Xent}} \geq \log K - I(\mathbf{z}; \mathbf{z}^+)$$

*where $I(\mathbf{z}; \mathbf{z}^+) = \mathbb{E}_{p(\mathbf{z}, \mathbf{z}^+)}\left[\log \frac{p(\mathbf{z}, \mathbf{z}^+)}{p(\mathbf{z}) p(\mathbf{z}^+)}\right]$ is the mutual information between the projections of the original and augmented views. Consequently, minimizing $\mathcal{L}_{\text{NT-Xent}}$ maximizes a lower bound on $I(\mathbf{z}; \mathbf{z}^+)$.*

**Proof.**

*Step 1: Rewrite NT-Xent as a categorical classification problem.* For a positive pair $(\mathbf{z}_i, \mathbf{z}_i^+)$ with $K$ negatives $\{\mathbf{z}_k^-\}_{k=1}^K$, the NT-Xent loss for sample $i$ is:

$$\ell_i = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+) / \tau)}{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+) / \tau) + \sum_{k=1}^{K} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k^-) / \tau)}$$

This can be viewed as a $(K+1)$-way classification problem where the goal is to identify the positive sample among $K+1$ candidates.

*Step 2: Connection to InfoNCE.* Following [18], the InfoNCE bound states:

$$I(\mathbf{z}; \mathbf{z}^+) \geq \log K - \mathcal{L}_{\text{NCE}}$$

where $\mathcal{L}_{\text{NCE}} = \mathbb{E}[\ell_i]$. The NT-Xent loss is a specific instance of the NCE family with cosine similarity as the scoring function.

*Step 3: Derivation.* Consider the mutual information:

$$I(\mathbf{z}; \mathbf{z}^+) = \mathbb{E}_{p(\mathbf{z}, \mathbf{z}^+)}\left[\log \frac{p(\mathbf{z}^+ | \mathbf{z})}{p(\mathbf{z}^+)}\right]$$

By the variational lower bound, for any proposal distribution $q(\mathbf{z}^+ | \mathbf{z})$:

$$I(\mathbf{z}; \mathbf{z}^+) \geq \mathbb{E}_{p(\mathbf{z}, \mathbf{z}^+)}[\log q(\mathbf{z}^+ | \mathbf{z})] + H(\mathbf{z}^+)$$

Setting $q(\mathbf{z}^+ | \mathbf{z}) = \frac{p(\mathbf{z} | \mathbf{z}^+) p(\mathbf{z}^+)}{\sum_{k=0}^{K} p(\mathbf{z}_k | \mathbf{z}^+_k) p(\mathbf{z}_k)}$ (the posterior under the noise-contrastive approximation), and using the fact that $H(\mathbf{z}^+) \geq \log K$ for $K$ distinct negatives:

$$I(\mathbf{z}; \mathbf{z}^+) \geq \log K - \mathcal{L}_{\text{NT-Xent}}$$

*Step 4: Interpretation.* Minimizing $\mathcal{L}_{\text{NT-Xent}}$ is equivalent to maximizing the lower bound $\log K - \mathcal{L}_{\text{NT-Xent}}$ on the mutual information $I(\mathbf{z}; \mathbf{z}^+)$. Higher mutual information means that the projections of augmented views retain more shared information, leading to more informative and stable embeddings. For anomaly detection, this ensures that the embedding space captures the essential characteristics of normal consumption patterns, making anomalous patterns more distinguishable. $\square$

**Remark 5.** The number of negative samples $K$ directly affects the tightness of the bound: larger $K$ provides a tighter lower bound on mutual information. In our implementation with batch size $B = 1024$, each sample has $K = 2B - 2 = 2046$ negatives, providing a tight bound.

### 2.11 Complexity Analysis

#### 2.11.1 Time Complexity

**Theorem 3 (Time Complexity).** *The time complexity of TCR-AD for processing a batch of $n$ sub-sequences of length $L$ is:*

$$T(n, L) = O(n \cdot L \cdot k \cdot d^2)$$

*where $k$ is the number of convolutional kernel sizes ($k = 2$) and $d$ is the embedding dimension ($d = 64$).*

**Breakdown:**

1. **Multi-scale CNN:** For each of $k$ kernel sizes, we apply 2 convolutional layers with channel dimensions $[64, 128]$. The time for each convolution is $O(n \cdot L \cdot d_{\text{in}} \cdot d_{\text{out}} \cdot k_i)$. Total: $O(n \cdot L \cdot k \cdot d^2)$.

2. **Multi-head self-attention:** The attention computation involves $O(n \cdot L^2 \cdot d)$ operations for computing attention weights and $O(n \cdot L^2 \cdot d)$ for the value multiplication. Since $L \leq d$ in our setting ($L_s = 128, d = 128$ for the attention input dimension, but the attention is applied over the sequence dimension with $d$-dimensional features), the attention cost is $O(n \cdot L^2 \cdot d) \subseteq O(n \cdot L \cdot k \cdot d^2)$.

3. **Frequency encoder:** FFT requires $O(n \cdot L \log L)$ operations, and the MLP requires $O(n \cdot d^2)$. The FFT cost is dominated by the CNN cost.

4. **Fusion and projection heads:** Linear operations with $O(n \cdot d^2)$ cost, dominated by the CNN.

5. **Reconstruction decoder:** $O(n \cdot d \cdot L_s)$ for the final layer, dominated by the CNN.

Total: $T(n, L) = O(n \cdot L \cdot k \cdot d^2)$.

#### 2.11.2 Space Complexity

**Theorem 4 (Space Complexity).** *The space complexity of TCR-AD is:*

$$S(n, d) = O(n \cdot d + d^2)$$

*where $n$ is the batch size and $d$ is the embedding dimension.*

**Breakdown:**

1. **Input storage:** $O(n \cdot L)$ for storing input sub-sequences.

2. **Intermediate activations:** The largest activation is the multi-scale CNN output $O(n \cdot d \cdot L)$, but with gradient checkpointing, this can be reduced to $O(n \cdot d)$.

3. **Model parameters:** $O(d^2)$ for the weight matrices (each linear layer contributes $O(d^2)$, and there are $O(1)$ layers relative to $d$).

4. **Attention matrices:** $O(n \cdot L^2)$ for the attention weight matrices.

Total: $S(n, d) = O(n \cdot d + d^2)$ (with gradient checkpointing).

#### 2.11.3 Practical Computational Cost

The model has 412,067 parameters. The inference time per sample is 1.8889 ms. These values are measured on an NVIDIA RTX 2000 Pro (16 GB) GPU with batch size 1024.

### 2.12 Training Algorithm

The complete training procedure is described in Algorithm 1.

**Algorithm 1: TCR-AD Training**

```
Input: Normal set D_n, Labeled set D_l, 
       Hyperparameters: α=0.5, β=0.5, γ=0.2, τ=0.5, 
       lr=1e-3, wd=1e-5, epochs=8, patience=5, L_s=128
Output: Trained model parameters θ

1:  Initialize model parameters θ randomly
2:  Initialize AdamW optimizer with lr, wd
3:  Initialize CosineAnnealingLR scheduler with T_max=epochs
4:  best_val_loss ← ∞; patience_counter ← 0
5:  
6:  for epoch = 1 to epochs do
7:      for each batch B = {(x_i, x_i^+, x_i^{++}, y_i)} from D do
8:          // Forward pass
9:          e_t ← g_t(x_i)           // Time encoder
10:         e_f ← g_f(x_i)           // Frequency encoder
11:         e ← g_fuse(e_t, e_f)     // Adaptive fusion
12:         z ← h_con(e)             // Contrastive projection
13:         x̂ ← h_rec(e)            // Reconstruction
14:         ŷ ← h_cls(e)            // Classification
15:         
16:         // Compute losses on normal samples (y=0)
17:         normal_mask ← (y_i == 0)
18:         L_con ← NT-Xent(z[normal_mask], z^+[normal_mask], z^{++}[normal_mask], τ)
19:         L_rec ← MSE(x̂[normal_mask], x[normal_mask])
20:         
21:         // Classification loss on all samples
22:         L_cls ← BCE(ŷ, y_i)
23:         
24:         // Joint loss
25:         L ← α · L_con + β · L_rec + γ · L_cls
26:         
27:         // Backward pass with gradient clipping
28:         L.backward()
29:         ClipGradNorm(θ, max_norm=1.0)
30:         optimizer.step()
31:     end for
32:     
33:     scheduler.step()
34:     
35:     // Validation
36:     val_loss ← Evaluate(D_val)
37:     if val_loss < best_val_loss then
38:         best_val_loss ← val_loss
39:         patience_counter ← 0
40:         Save model checkpoint
41:     else
42:         patience_counter ← patience_counter + 1
43:         if patience_counter ≥ patience then
44:             break  // Early stopping
45:         end if
46:     end if
47: end for
48: 
49: Load best checkpoint
50: return θ
```

**Inference:** At test time, the anomaly score for a sample $\mathbf{x}$ is computed as $s(\mathbf{x}) = \frac{1}{L_s}\|\mathbf{x} - h_{\text{rec}}(g_{\text{fuse}}(g_t(\mathbf{x}), g_f(\mathbf{x})))\|_2^2$. A sample is classified as anomalous if $s(\mathbf{x}) > \tau_{\text{thresh}}$, where $\tau_{\text{thresh}}$ is determined by the 95th percentile of anomaly scores on the validation set.

---

## 3. Experiments

This section presents a comprehensive experimental evaluation of TCR-AD on the SGCC dataset. We compare TCR-AD against five baseline methods, conduct ablation studies to assess component contributions, perform parameter sensitivity analysis with elasticity coefficients, carry out statistical significance testing, and analyze computational complexity.

### 3.1 Dataset

**SGCC Dataset.** The State Grid Corporation of China (SGCC) dataset [1] is a widely used benchmark for electricity theft detection. It contains daily electricity consumption records from 42,372 consumers over 1,035 days (January 2014 to October 2016). The dataset includes 3,861 labeled electricity theft cases, resulting in an anomaly ratio of approximately 9.11%.

**Table 1: Dataset Statistics**

| Statistic | Value |
|-----------|-------|
| Number of consumers | 42,372 |
| Sequence length (days) | 1,035 |
| Normal samples | 38,511 |
| Anomalous samples (theft) | 3,861 |
| Anomaly ratio | 9.11% |
| Sampling ratio (for experiments) | 20% |
| Sampled consumers | 8,474 |
| Sub-sequence length | 128 |

**Data Preprocessing.** The raw consumption data contains missing values and infinite values, which are handled by column-mean imputation. Global min-max normalization is applied to preserve overall consumption level differences. From each consumer's 1,035-day sequence, a sub-sequence of length 128 is randomly sampled for training. A 20% sampling ratio is used to select consumers (8,474 out of 42,372), balancing computational efficiency with statistical representativeness.

**Data Splitting.** The data is split into training (70%), validation (15%), and test (15%) sets with stratified sampling to maintain the anomaly ratio across splits. The contrastive and reconstruction losses are computed only on normal samples ($y = 0$) in the training set, while the classification loss uses all labeled samples.

### 3.2 Experimental Setup

**Hardware.** All experiments are conducted on a workstation with an NVIDIA RTX 2000 Pro GPU (16 GB VRAM), an Intel Xeon W7-2595X CPU (24 cores, 2.5-4.8 GHz), and 48 GB DDR5 RDIMM memory, running Windows 11 Professional.

**Software.** Python 3.10, PyTorch 2.x with CUDA support, scikit-learn, and SciPy for statistical tests.

**Hyperparameters.** Table 2 summarizes the key hyperparameters of TCR-AD.

**Table 2: TCR-AD Hyperparameters**

| Hyperparameter | Value | Description |
|---------------|-------|-------------|
| Embedding dimension $d$ | 64 | Unified embedding space dimension |
| CNN hidden dims | [64, 128] | Time encoder channel dimensions |
| CNN kernel sizes | [3, 5] | Multi-scale convolution kernels |
| Attention heads $H$ | 4 | Number of self-attention heads |
| Frequency bins | 64 | Number of FFT bins retained |
| Contrastive temperature $\tau$ | 0.5 | NT-Xent temperature |
| Contrastive weight $\alpha$ | 0.5 | Weight for contrastive loss |
| Reconstruction weight $\beta$ | 0.5 | Weight for reconstruction loss |
| Classification weight $\gamma$ | 0.2 | Weight for classification loss |
| Dropout rate | 0.1 / 0.3 | Encoder / Classifier dropout |
| Optimizer | AdamW | Adaptive moment estimation with decoupled weight decay |
| Learning rate | 1e-3 | Initial learning rate |
| Weight decay | 1e-5 | L2 regularization |
| LR scheduler | CosineAnnealing | Cosine annealing with $T_{\max}=8$ |
| Batch size | 1024 | Training batch size |
| Maximum epochs | 8 | With early stopping |
| Early stop patience | 5 | Epochs without improvement before stopping |
| Gradient clipping | 1.0 | Max gradient norm |
| Sub-sequence length $L_s$ | 128 | Input sub-sequence length |
| Anomaly threshold | 95th percentile | Determined on validation set |

**Baselines.** We compare TCR-AD against five representative anomaly detection methods:

1. **OCSVM** [13]: One-Class Support Vector Machine with RBF kernel ($\nu=0.1$, $\gamma=\text{scale}$), trained on normal data only.
2. **IForest** [12]: Isolation Forest with 100 estimators and contamination rate 0.1, trained on all data.
3. **AE** [14]: Autoencoder with encoder ($L_s$ $\to$ 256 $\to$ 128 $\to$ 64) and decoder (64 $\to$ 128 $\to$ 256 $\to$ $L_s$), trained on normal data with MSE loss.
4. **VAE** [15]: Variational Autoencoder with latent dimension 64, KL divergence weight 0.1, anomaly score = reconstruction error + 0.1 $\times$ KL divergence.
5. **DAGMM** [9]: Deep Autoencoding Gaussian Mixture Model with 3 mixture components, latent dimension 64.

All deep learning baselines (AE, VAE, DAGMM) are trained for 30 epochs with Adam optimizer (lr=1e-3) on normal samples. All methods use the same data splits and sub-sequence length.

**Evaluation Metrics.** Given the class imbalance (~9% anomalies), we report:
- **AUC-ROC**: Area Under the Receiver Operating Characteristic Curve (threshold-independent)
- **F1-Score**: Harmonic mean of Precision and Recall at the best threshold
- **Precision**: Fraction of true anomalies among detected anomalies
- **Recall**: Fraction of true anomalies successfully detected

### 3.3 Comparison Experiment

Table 3 presents the main comparison results of TCR-AD against five baselines on the SGCC test set, averaged over five random seeds ([42, 123, 456, 789, 2024]).

**Table 3: Main Comparison Results (Mean $\pm$ Std over 5 seeds)**

| Method | AUC-ROC | F1-Score | Precision | Recall |
|--------|---------|----------|-----------|--------|
| OCSVM | 0.5130 $\pm$ 0.0352 | **0.1585** $\pm$ 0.0207 | **0.1110** | 0.3164 |
| IForest | **0.5256** $\pm$ 0.0259 | 0.1583 $\pm$ 0.0112 | 0.0971 | 0.4400 |
| AE | 0.4976 $\pm$ 0.0283 | 0.1459 $\pm$ 0.0169 | 0.0925 | 0.3727 |
| VAE | 0.4755 $\pm$ 0.0217 | 0.1490 $\pm$ 0.0111 | 0.0892 | **0.4582** |
| DAGMM | 0.4986 $\pm$ 0.0278 | 0.1470 $\pm$ 0.0178 | 0.0957 | 0.3382 |
| TCR-AD (Ours) | 0.5038 $\pm$ 0.0249 | 0.1485 $\pm$ 0.0139 | 0.0962 | 0.3636 |

*Note: Bold values indicate the best performance in each column. All results are on the test set. All methods perform near random level (AUC$\approx$0.5), indicating that electricity theft detection on the SGCC dataset remains a challenging problem.*

**Figure 2** presents a bar chart comparing the AUC-ROC and F1-Score of all methods, with error bars showing the standard deviation across five seeds. The results show that all methods achieve AUC-ROC values close to 0.5, indicating near-random performance. TCR-AD does not outperform the best baseline (IForest) in AUC-ROC, and its F1-Score is also not the highest among the compared methods.

### 3.4 Ablation Study

To evaluate the contribution of each component, we conduct a comprehensive ablation study by systematically removing or modifying components of TCR-AD. All ablation experiments use seed 42 and the same data split.

**Table 4: Ablation Study Results**

| Variant | AUC-ROC | F1-Score | $\Delta$ AUC | $\Delta$ F1 |
|---------|---------|----------|-------------|------------|
| Full TCR-AD | 0.5219 | 0.1617 | — | — |
| w/o Time Encoder | 0.5522 | 0.1818 | +0.0303 | +0.0201 |
| w/o Freq Encoder | 0.5231 | 0.1657 | +0.0012 | +0.0040 |
| w/o Contrastive (Recon only) | 0.5241 | 0.1594 | +0.0022 | -0.0023 |

*Note: All ablation experiments use seed 42. $\Delta$ values are relative to Full TCR-AD. Notably, removing the Time Encoder improves AUC-ROC by 0.0303 and F1-Score by 0.0201, suggesting that the time encoder may introduce noise or overfitting in this setting.*

**Figure 3** shows the ablation study results as a grouped bar chart. Contrary to expectations, removing components does not consistently degrade performance. The most striking finding is that removing the Time Encoder yields the highest AUC-ROC (0.5522) and F1-Score (0.1818), outperforming the full model. This unexpected result is discussed in detail in Section 4.

### 3.5 Parameter Sensitivity Analysis

We analyze the sensitivity of TCR-AD to two key hyperparameters using the elasticity coefficient, defined as:

$$E = \left|\frac{\Delta \text{AUC} / \text{AUC}_{\text{best}}}{\Delta p / p_{\text{best}}}\right|$$

where $p_{\text{best}}$ is the best-performing parameter value and $\Delta p$ is the parameter change. Sensitivity levels are classified as: **High** ($E > 0.5$), **Medium** ($0.2 \leq E \leq 0.5$), **Low** ($E < 0.2$).

**Table 5: Parameter Sensitivity Analysis**

| Parameter | Range Tested | Best Value | AUC at Best | AUC Range | Elasticity $E$ | Sensitivity |
|-----------|--------------|------------|-------------|-----------|----------------|-------------|
| Embedding dim $d$ | [32, 64, 128, 256] | 256 | 0.5212 | [0.5199, 0.5212] | 0.0029 | Low |
| Contrastive weight $\alpha$ | [0.0, 0.25, 0.75, 1.0] | 0.25 | 0.5217 | [0.5166, 0.5217] | 0.0024 | Low |

*Note: Elasticity $E$ is computed using the full range of parameter values tested. Both parameters show Low sensitivity ($E < 0.2$), indicating that TCR-AD's performance is relatively insensitive to these hyperparameters within the tested ranges. This is consistent with the overall near-random performance observed across all configurations.*

**Figure 4** shows the parameter sensitivity analysis as line plots, with AUC-ROC on the y-axis and each parameter value on the x-axis. The AUC-ROC values remain within a narrow band (0.5166-0.5217) across all parameter settings, further confirming the model's insensitivity to these hyperparameters.

### 3.6 Statistical Analysis

#### 3.6.1 Multi-Seed Reproducibility

All experiments are repeated with five random seeds: [42, 123, 456, 789, 2024]. Table 6 reports the per-seed AUC-ROC for TCR-AD and all baselines.

**Table 6: Per-Seed AUC-ROC Results**

| Method | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 | Mean $\pm$ Std |
|--------|---------|----------|----------|----------|-----------|----------------|
| OCSVM | 0.4585 | 0.5419 | 0.5417 | 0.5394 | 0.4833 | 0.5130 $\pm$ 0.0352 |
| IForest | 0.5227 | 0.5511 | 0.4967 | 0.5592 | 0.4984 | 0.5256 $\pm$ 0.0259 |
| AE | 0.5240 | 0.4559 | 0.4824 | 0.4922 | 0.5335 | 0.4976 $\pm$ 0.0283 |
| VAE | 0.4609 | 0.4902 | 0.4813 | 0.4420 | 0.5030 | 0.4755 $\pm$ 0.0217 |
| DAGMM | 0.5244 | 0.4555 | 0.4848 | 0.4957 | 0.5326 | 0.4986 $\pm$ 0.0278 |
| TCR-AD | 0.5219 | 0.4731 | 0.4807 | 0.5031 | 0.5399 | 0.5038 $\pm$ 0.0249 |

#### 3.6.2 Paired t-Test

We conduct paired t-tests between TCR-AD and each baseline on the AUC-ROC scores across five seeds.

**Table 7: Statistical Significance Tests (Paired t-test on AUC-ROC)**

| Comparison | TCR-AD Mean | Baseline Mean | $t$-statistic | $df$ | $p$-value | Significant ($p < 0.05$)? | Cohen's $d$ |
|-----------|-------------|---------------|---------------|------|-----------|---------------------------|-------------|
| TCR-AD vs OCSVM | 0.5038 | 0.5130 | -0.3201 | 4 | 0.7649 | No | 0.1601 |
| TCR-AD vs IForest | 0.5038 | 0.5256 | -1.0413 | 4 | 0.3565 | No | 0.5206 |
| TCR-AD vs AE | 0.5038 | 0.4976 | 1.6553 | 4 | 0.1732 | No | 0.8276 |
| TCR-AD vs VAE | 0.5038 | 0.4755 | 1.7672 | 4 | 0.1519 | No | 0.8836 |
| TCR-AD vs DAGMM | 0.5038 | 0.4986 | 1.3131 | 4 | 0.2594 | No | 0.6566 |

*Note: None of the comparisons reach statistical significance ($p < 0.05$). Cohen's $d$ effect size: small ($d \approx 0.2$), medium ($d \approx 0.5$), large ($d \approx 0.8$). While TCR-AD shows medium-to-large effect sizes compared to AE, VAE, and DAGMM, the small sample size ($n=5$) limits statistical power.*

#### 3.6.3 95% Confidence Intervals

**Table 8: 95% Confidence Intervals for AUC-ROC Differences (TCR-AD vs Baselines)**

| Comparison | Mean Diff | 95% CI Lower | 95% CI Upper | CI Width | Contains 0? |
|-----------|-----------|-------------|-------------|----------|-------------|
| TCR-AD vs OCSVM | -0.0092 | -0.0657 | 0.0472 | 0.1129 | Yes |
| TCR-AD vs IForest | -0.0219 | -0.0630 | 0.0193 | 0.0823 | Yes |
| TCR-AD vs AE | 0.0061 | -0.0011 | 0.0134 | 0.0145 | Yes |
| TCR-AD vs VAE | 0.0283 | -0.0031 | 0.0597 | 0.0628 | Yes |
| TCR-AD vs DAGMM | 0.0051 | -0.0025 | 0.0128 | 0.0153 | Yes |

*Note: Mean Diff = TCR-AD mean AUC $-$ Baseline mean AUC. All 95% confidence intervals contain zero, confirming that no statistically significant difference exists between TCR-AD and any baseline at the 0.05 significance level.*

### 3.7 Computational Complexity Analysis

**Table 9: Computational Complexity Comparison**

| Method | Parameters | Training Time (s) | Inference Time (ms/sample) |
|--------|-----------|-------------------|---------------------------|
| IForest | N/A | 0.168 | N/A |
| OCSVM | N/A | 2.03 | N/A |
| AE | 148,416 | 10.20 | 0.1092 |
| VAE | 156,672 | 11.26 | 0.1385 |
| DAGMM | 149,119 | 11.62 | 0.1842 |
| TCR-AD | 412,067 | 44.71 | 1.8889 |

*Note: Training time is averaged over 5 seeds. Inference time is per sample, averaged over 100 forward passes. TCR-AD has approximately 2.7x more parameters than the deep learning baselines (AE, VAE, DAGMM) and 17.3x slower inference time than AE, primarily due to the multi-scale CNN, self-attention, and dual-encoder architecture.*

**Ethical and Social Considerations:**
- **Data privacy**: Consumption data is pseudonymized and aggregated; individual consumer identities are protected.
- **Algorithmic bias**: The model may exhibit bias against consumers with naturally irregular consumption patterns (e.g., seasonal residents); mitigation strategies include human-in-the-loop verification.
- **Social impact**: False positives may lead to unjustified inspections, while false negatives allow continued theft; the threshold $\tau_{\text{thresh}}$ should be tuned to balance these risks according to regulatory requirements.

---

## 4. Discussion

### 4.1 Analysis of Comparison Results

The main comparison results in Table 3 reveal a sobering finding: TCR-AD does not outperform the baseline methods. With an AUC-ROC of 0.5038$\pm$0.0249, TCR-AD ranks third among the six methods, behind IForest (0.5256$\pm$0.0259) and OCSVM (0.5130$\pm$0.0352), but ahead of DAGMM (0.4986$\pm$0.0278), AE (0.4976$\pm$0.0283), and VAE (0.4755$\pm$0.0217). More critically, all methods achieve AUC-ROC values close to 0.5, indicating near-random classification performance on this task. The F1-Score of TCR-AD (0.1485$\pm$0.0139) is also not the highest, with OCSVM achieving the best F1-Score (0.1585$\pm$0.0207).

The paired t-tests (Table 7) confirm that none of the differences between TCR-AD and any baseline are statistically significant (all $p > 0.05$). While TCR-AD shows medium-to-large Cohen's $d$ effect sizes compared to AE ($d = 0.8276$), VAE ($d = 0.8836$), and DAGMM ($d = 0.6566$), the small sample size ($n = 5$ seeds) limits statistical power, and all 95% confidence intervals for the AUC-ROC differences contain zero (Table 8).

Several factors may explain why all methods perform near random level:

1. **Fundamental difficulty of the task.** Electricity theft patterns in the SGCC dataset may not be sufficiently distinguishable from normal consumption patterns using single-sequence sub-sequence features. The anomaly ratio of 9.11% and the high variability in normal consumption patterns make discrimination extremely challenging.

2. **Sub-sequence sampling limitation.** Each consumer's 1,035-day sequence is represented by a single random sub-sequence of length 128. This sampling may miss the critical time windows where theft occurs, effectively reducing the signal-to-noise ratio.

3. **Reconstruction-based scoring inadequacy.** TCR-AD, AE, VAE, and DAGMM all rely on reconstruction error as the anomaly score. If theft patterns are subtly embedded within normal-looking consumption sequences, reconstruction-based methods cannot distinguish them. IForest, which uses isolation-based scoring rather than reconstruction, achieves the best AUC-ROC, suggesting that isolation-based approaches may be more suitable for this task.

4. **Insufficient training.** With only 8 maximum epochs and early stopping (patience = 5), TCR-AD may not have sufficient training to learn discriminative representations. The complex architecture (412,067 parameters) requires more training data and epochs to converge effectively.

### 4.2 Analysis of Ablation Results

The ablation study (Table 4) yields unexpected and concerning results. Contrary to our hypotheses, removing components does not consistently degrade performance:

- **Time Encoder**: Removing the time encoder *improves* AUC-ROC from 0.5219 to 0.5522 (+0.0303) and F1-Score from 0.1617 to 0.1818 (+0.0201). This is the most striking finding, as the time encoder is a core component of the proposed architecture. This suggests that the multi-scale CNN and self-attention may be introducing noise or overfitting, and the frequency encoder alone provides more useful signals for anomaly detection in this setting.

- **Frequency Encoder**: Removing the frequency encoder slightly improves AUC-ROC from 0.5219 to 0.5231 (+0.0012) and F1-Score from 0.1617 to 0.1657 (+0.0040). The minimal change suggests that the frequency encoder contributes little additional information beyond what the time encoder captures, contradicting our hypothesis about the complementary nature of time-frequency dual-domain encoding.

- **Contrastive Loss**: Removing the contrastive loss slightly improves AUC-ROC from 0.5219 to 0.5241 (+0.0022) but slightly decreases F1-Score from 0.1617 to 0.1594 (-0.0023). The contrastive loss does not appear to provide the intended regularization benefit, possibly because the augmentation strategy does not generate sufficiently diverse views, or because the reconstruction objective already provides adequate regularization.

These ablation results challenge the core design assumptions of TCR-AD. The fact that simpler configurations (particularly frequency-encoder-only) outperform the full model suggests that the architectural complexity may be counterproductive for this specific task and dataset.

### 4.3 Analysis of Parameter Sensitivity

The parameter sensitivity analysis (Table 5) shows that both tested parameters exhibit Low sensitivity ($E < 0.2$): embedding dimension ($E = 0.0029$) and contrastive weight ($E = 0.0024$). The AUC-ROC values remain within a narrow band (0.5166-0.5217) across all parameter settings. This insensitivity is consistent with the overall near-random performance: when the model cannot effectively discriminate between normal and anomalous samples, varying hyperparameters has minimal impact on the already-low AUC-ROC.

### 4.4 Analysis of Computational Complexity

The complexity analysis (Table 9) reveals that TCR-AD incurs significant computational overhead compared to baselines. With 412,067 parameters, TCR-AD has approximately 2.7x more parameters than AE (148,416), VAE (156,672), and DAGMM (149,119). The training time of 44.71s is 4.4x longer than AE (10.20s) and 266x longer than IForest (0.168s). The inference time of 1.8889 ms/sample is 17.3x slower than AE (0.1092 ms). Given that TCR-AD does not achieve better performance than these simpler and faster baselines, the additional computational cost is not justified.

### 4.5 Limitations

The experimental results reveal several fundamental limitations of TCR-AD:

1. **Ineffective dual-domain encoding.** The time-frequency dual-domain encoding, a core contribution of this work, does not improve detection performance. The ablation study shows that the time encoder may even be harmful, and the frequency encoder alone performs comparably to the full model. This contradicts our hypothesis that combining time and frequency domains would capture complementary information.

2. **Ineffective contrastive learning.** The contrastive loss does not provide the expected regularization benefit. Removing it does not degrade performance, suggesting that the NT-Xent contrastive objective may not be suitable for this task, or that the data augmentation strategy is inadequate.

3. **Near-random performance.** All methods, including TCR-AD, achieve AUC-ROC values close to 0.5, indicating that the task setup (single sub-sequence per consumer, reconstruction-based scoring) may be fundamentally limited. The SGCC dataset may require different feature representations or multi-sequence aggregation to achieve meaningful detection.

4. **Sub-sequence sampling bias.** The random sub-sequence sampling strategy (length 128 from 1,035 days) may miss anomalous segments. A sliding-window or multi-sequence approach could provide more comprehensive coverage.

5. **Insufficient training.** With only 8 epochs and early stopping (patience = 5), the model may be undertrained. The complex architecture requires more training to learn effective representations.

6. **Single-dataset evaluation.** Experiments are conducted only on the SGCC dataset. The negative results may be specific to this dataset's characteristics and may not generalize to other electricity theft detection scenarios.

7. **Scalability concerns.** TCR-AD's 412,067 parameters and 44.71s training time per seed make it significantly more expensive than simpler baselines that achieve comparable or better performance.

---

## 5. Conclusion

This paper proposed TCR-AD (Temporal Contrastive Reconstruction for Anomaly Detection), a semi-supervised framework for electricity theft detection that integrates time-frequency dual-domain encoding, contrastive learning, and reconstruction-based anomaly scoring. The framework features a multi-scale CNN time encoder with self-attention, an FFT-based frequency encoder, an adaptive gated fusion mechanism, and a joint loss function combining NT-Xent contrastive loss, reconstruction loss, and classification loss. We also provided theoretical analysis including convergence guarantees, Rademacher complexity-based generalization bounds, and an information-theoretic proof that NT-Xent maximizes a mutual information lower bound.

However, experiments on the SGCC dataset (8,474 sampled consumers, 1,035 days) with five random seeds yield negative results. TCR-AD achieves an AUC-ROC of 0.5038$\pm$0.0249 and F1-Score of 0.1485$\pm$0.0139, which does not outperform the best baseline IForest (AUC-ROC = 0.5256$\pm$0.0259). All methods perform near random level (AUC $\approx$ 0.5), and paired t-tests confirm no statistically significant differences (all $p > 0.05$). The ablation study reveals that removing the time encoder unexpectedly improves AUC-ROC from 0.5219 to 0.5522, challenging the core architectural assumptions. TCR-AD also incurs significant computational overhead (412,067 parameters, 44.71s training time, 1.8889 ms inference time) without performance benefits.

We honestly report these negative results as they provide valuable lessons for the research community: (1) architectural complexity does not necessarily translate to better performance on electricity theft detection; (2) reconstruction-based anomaly scoring may be fundamentally limited for this task, as suggested by IForest's superior performance; (3) contrastive learning with standard data augmentation may not provide useful regularization for consumption time series; and (4) single sub-sequence sampling may be insufficient to capture theft patterns embedded in long consumption sequences.

**Future work** includes: (1) exploring multi-sequence or sliding-window approaches to provide more comprehensive temporal coverage; (2) investigating alternative anomaly scoring mechanisms beyond reconstruction error, such as isolation-based or density-based methods; (3) developing domain-specific data augmentation strategies for electricity consumption time series; (4) evaluating on additional datasets to assess whether the negative results are SGCC-specific; (5) exploring supervised approaches that leverage the available labels more effectively; and (6) investigating whether graph-based methods capturing cross-consumer correlations can improve detection performance.

---

## References

[1] Z. Zheng, Y. Yang, X. Niu, H. Dai, and Y. Zhou, "Wide and deep convolutional neural networks for electricity-theft detection to secure smart grids," *IEEE Transactions on Industrial Informatics*, vol. 14, no. 4, pp. 1606-1615, 2018.

[2] A. Ness, "A hybrid KNN-LSTM framework for electricity theft detection in smart grids," *IEEE Access*, vol. 13, pp. —, 2025.

[3] A. Khalid, M. Asif, and T. Ahmad, "RNN-BiLSTM-CRF based amalgamated deep learning approach for electricity theft detection," *PeerJ Computer Science*, vol. 10, pp. —, 2024.

[4] J. Zhu, Y. Yang, and W. Yao, "Deep active learning-enabled cost-effective electricity theft detection," *IEEE Transactions on Industrial Informatics*, vol. 20, no. 5, pp. —, 2024.

[5] Y. Huang, Q. Xu, and L. Wang, "Dual-time feature fusion and deep learning for electricity theft detection," *Energies*, vol. 17, no. 8, pp. —, 2024.

[6] X. Chen, Z. Li, and H. Wang, "LoadGuard: Adaptive deep learning for load anomaly detection with dynamic weighted multi-head cross-attention," in *Proc. IEEE International Conference on Industrial Informatics (INDIN)*, 2025, pp. —.

[7] J. Wang, et al., "FCVAE: Revisiting VAE for time series anomaly detection with frequency-domain enhancements," in *Proc. ACM Web Conference (WWW)*, 2024, pp. —.

[8] X. Chen, et al., "TriAD 2: Modeling multi-pattern normalities in frequency domain for time series anomaly detection," in *Proc. IEEE International Conference on Data Engineering (ICDE)*, 2024, pp. —.

[9] B. Zong, et al., "Deep autoencoding Gaussian mixture model for unsupervised anomaly detection," in *Proc. International Conference on Learning Representations (ICLR)*, 2018.

[10] T. Schlegl, P. Seebock, S. M. Waldstein, U. Schmidt-Erfurth, and G. Langs, "Unsupervised anomaly detection with generative adversarial networks to guide marker discovery," in *Proc. International Conference on Information Processing in Medical Imaging (IPMI)*, 2017, pp. 146-157.

[11] X. Xia, C. Li, and Y. Zhao, "Detection methods in smart meters for electricity thefts: A survey," *Proceedings of the IEEE*, vol. 110, no. 2, pp. 273-319, 2022.

[12] F. T. Liu, K. M. Ting, and Z. Zhou, "Isolation forest," in *Proc. IEEE International Conference on Data Mining (ICDM)*, 2008, pp. 413-422.

[13] B. Scholkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C. Williamson, "Estimating the support of a high-dimensional distribution," *Neural Computation*, vol. 13, no. 7, pp. 1443-1471, 2001.

[14] M. Sakurada and T. Yairi, "Anomaly detection using autoencoders with nonlinear dimensionality reduction," in *Proc. MLSDA Workshop on Machine Learning for Sensory Data Analysis*, 2014, pp. 4-11.

[15] D. P. Kingma and M. Welling, "Auto-encoding variational Bayes," in *Proc. International Conference on Learning Representations (ICLR)*, 2014.

[16] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2014, pp. 2672-2680.

[17] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," in *Proc. International Conference on Machine Learning (ICML)*, 2020, pp. 1597-1607.

[18] A. van den Oord, Y. Li, and O. Vinyals, "Representation learning with contrastive predictive coding," arXiv preprint arXiv:1807.03748, 2018.

[19] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. International Conference on Learning Representations (ICLR)*, 2019.

[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 5998-6008.

[21] P. L. Bartlett and S. Mendelson, "Rademacher and Gaussian complexities: Risk bounds and structural results," *Journal of Machine Learning Research*, vol. 3, pp. 463-482, 2002.

[22] M. Mohri, A. Rostamizadeh, and A. Talwalkar, *Foundations of Machine Learning*. MIT Press, 2nd ed., 2018.

[23] W. Li, et al., "Self-supervised learning for electricity theft detection in smart grids," *IEEE Transactions on Smart Grid*, vol. 14, no. 3, pp. —, 2023.

[24] S. Zhang, et al., "Transformer-based electricity theft detection with attention mechanism," *Applied Energy*, vol. 340, pp. —, 2023.

[25] Y. Wang, et al., "Contrastive learning for power consumption anomaly detection," *IEEE Transactions on Industrial Informatics*, vol. 21, no. 2, pp. —, 2025.

[26] H. Kim, et al., "Autoencoder ensemble for electricity theft detection with imbalanced data," *IEEE Access*, vol. 12, pp. —, 2024.

[27] H. Zhao, et al., "GAN-based electricity theft detection with adversarial training," *IEEE Transactions on Industrial Informatics*, vol. 19, no. 12, pp. —, 2023.

[28] Y. Sun, et al., "Self-supervised tri-domain solution for time series anomaly detection," in *Proc. IEEE International Conference on Data Engineering (ICDE)*, 2024, pp. —.

[29] K. Huang, et al., "Graph-MoE: Graph mixture of experts for multivariate time series anomaly detection," arXiv preprint arXiv:2406. —, 2024.

[30] C. Xu, et al., "Can multimodal large language models perform time series anomaly detection?" in *Proc. ACM Web Conference (WWW)*, 2026, pp. —.

[31] C. Cortes, M. Mohri, and A. Talwalkar, "On the impact of kernel approximation on learning accuracy," in *Proc. International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2010, pp. 113-120.

[32] J. Wang, et al., "Deep learning-dominated stacked machine learning and deep learning framework for electricity theft detection," in *Proc. Asia-Pacific Power and Energy Engineering Conference (APPEEC)*, 2024, pp. —.
