# 44_Energy_Anomaly 参考文献调研材料

> 调研日期：2026-08-10
> 方向：电力消费异常检测（TCR-AD: Temporal Contrastive Reconstruction for Anomaly Detection）
> 数据集：SGCC (State Grid Corporation of China) 智能电表数据

---

## 一、SGCC数据集概述

| 项目 | 内容 |
|------|------|
| 来源 | State Grid Corporation of China (SGCC) |
| 时间范围 | 2014年1月 – 2016年10月 |
| 数据类型 | 消费者用电时间序列数据（每日用电量） |
| 样本数量 | 42,372个用户 |
| 特征维度 | 1,035（每日用电量记录） |
| 异常比例 | 约9.11%（3,861个窃电用户） |
| 标签 | 二分类：0=正常, 1=窃电 |
| 引用论文 | Zheng et al. (2018), IEEE TII |

## 二、关键SOTA文献（2024-2026）

### 2.1 电力窃电检测

| 编号 | 文献 | 年份 | 来源 | 方法 | 关键指标 | 局限性 |
|------|------|------|------|------|---------|--------|
| S1 | Ness, "Hybrid KNN-LSTM Framework for Electricity Theft Detection" | 2025 | IEEE Access | KNN+LSTM混合框架 | Accuracy=81.32% | 仅报告Accuracy，无AUC/F1；KNN计算成本高 |
| S2 | Khalid et al., "RNN-BiLSTM-CRF based Amalgamated Deep Learning" | 2024 | PeerJ CS | RNN-BiLSTM-CRF组合 | 未披露具体指标 | 模型结构复杂，缺乏理论分析 |
| S3 | Zhu et al., "Deep Active Learning-Enabled Cost-Effective Electricity Theft Detection" | 2024 | IEEE TII | 深度主动学习 | 未披露具体指标 | 需要人工标注反馈，部署成本高 |
| S4 | Huang et al., "Dual-Time Feature Fusion and Deep Learning" | 2024 | Energies | 双时域特征融合 | 未披露具体指标 | 仅时域特征，未利用频域信息 |
| S5 | Chen et al., "LoadGuard: Adaptive Deep Learning" | 2025 | INDIN | Transformer+DW-MHC | 未披露具体指标 | 未使用SGCC数据集；Transformer计算开销大 |
| S6 | Wang et al., "Deep Learning-Dominated Stacked ML+DL" | 2024 | APETC | 堆叠ML+DL | 未披露具体指标 | 模型可解释性差 |

### 2.2 时间序列异常检测（通用方法）

| 编号 | 文献 | 年份 | 来源 | 方法 | 关键指标 | 局限性 |
|------|------|------|------|------|---------|--------|
| S7 | Wang et al., "FCVAE: Revisiting VAE for Time Series Anomaly Detection" | 2024 | WWW 2024 | 频率增强CVAE | 优于8个SOTA (Best F1) | 仅单变量时序；未应用于电力领域 |
| S8 | Chen et al., "TriAD 2: Multi-Pattern Normalities in Frequency Domain" | 2024 | ICDE 2024 | 频域多模式正常性 | 未披露 | 面向通用时序，未验证电力场景 |
| S9 | Sun et al., "Self-supervised Tri-domain Solution" | 2024 | ICDE 2024 | 自监督三角域 | 未披露 | 三域分解增加计算复杂度 |
| S10 | Huang et al., "Graph-MoE: Graph Mixture of Experts" | 2024 | arXiv | GNN+MoE+记忆路由 | SWaT AUROC=87.2% | 传感器关系图假设静态，不适合单变量 |
| S11 | Xu et al., "Can Multimodal LLMs Perform Time Series Anomaly Detection?" | 2026 | WWW 2026 | 多模态大模型 | 未披露 | LLM推理成本极高，不适合边缘部署 |

### 2.3 经典基线方法

| 编号 | 文献 | 年份 | 方法 | 说明 |
|------|------|------|------|------|
| B1 | Liu et al. | 2008 | Isolation Forest (IForest) | 基于隔离的异常检测，无需标签 |
| B2 | Schölkopf et al. | 2001 | One-Class SVM (OCSVM) | 基于超球面的异常检测 |
| B3 | Zong et al. | 2018 | DAGMM | 深度自编码高斯混合模型 |
| B4 | Schlegl et al. | 2017 | AnoGAN | GAN-based异常检测 |
| B5 | Sakurada & Yagiri | 2014 | Autoencoder (AE) | 基于重构误差的异常检测 |
| B6 | Kingma & Welling | 2014 | VAE | 变分自编码器异常检测 |

## 三、SGCC数据集经典论文

### 3.1 Wide and Deep CNN (Zheng et al., 2018)

| 项目 | 内容 |
|------|------|
| 标题 | Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids |
| 期刊 | IEEE Transactions on Industrial Informatics, Vol. 14, No. 4, pp. 1606-1615 |
| DOI | 10.1109/TII.2018.2839353 |
| 方法 | Wide and Deep CNN |
| 数据集 | SGCC |
| 贡献 | 首次在SGCC上提出Wide and Deep CNN架构，成为该数据集的标准基线 |

### 3.2 综述论文

| 项目 | 内容 |
|------|------|
| 标题 | Detection Methods in Smart Meters for Electricity Thefts: A Survey |
| 作者 | Xia et al. |
| 期刊 | Proceedings of the IEEE, Vol. 110, No. 2, pp. 273-319 |
| 年份 | 2022 |
| DOI | 10.1109/JPROC.2021.3139754 |
| 引用 | 123 |
| 贡献 | 系统梳理了智能电表窃电检测方法，分为机器学习方法和测量不匹配方法 |

## 四、研究趋势分析

### 4.1 当前趋势
1. **深度学习仍是主流**：CNN、RNN、BiLSTM、Transformer等模型在窃电检测中广泛应用
2. **主动学习受关注**：降低标注成本成为研究热点
3. **联邦学习进入该领域**：隐私保护下的窃电检测开始探索
4. **频域信息开始被利用**：FCVAE和TriAD 2表明频域特征可增强异常检测
5. **对比学习在异常检测中应用增加**：但专门针对电力窃电的工作极少

### 4.2 研究空白
1. **无对比学习+频域特征组合**用于电力窃电检测
2. **多数方法仅用时域特征**，频域信息被忽略
3. **极端类别不平衡**问题未被充分解决
4. **理论分析严重不足**：现有工作几乎没有收敛性分析、泛化界等理论贡献
5. **SGCC数据集上缺乏完整指标报告**：多数论文仅报告Accuracy

## 五、TCR-AD方法定位

TCR-AD (Temporal Contrastive Reconstruction for Anomaly Detection) 的核心创新点：
1. **时域-频域双编码器**：多尺度CNN+自注意力(时域) + FFT+MLP(频域)
2. **自适应门控融合**：学习时域和频域特征的最优权重
3. **对比学习+重构联合优化**：NT-Xent对比损失 + MSE重构损失
4. **半监督分类头**：利用少量标签信息指导异常检测
5. **正常样本建模**：对比学习和重构仅使用正常样本，符合异常检测范式
