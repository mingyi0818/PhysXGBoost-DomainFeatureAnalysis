# Analysis of Potential Research Directions (V2)

## Overview
This document summarizes all research directions currently being pursued or under consideration, along with their status, progress, and potential for publication.

---

## Research Directions Summary

### 方向一：表格数据统一框架 (Unified Framework for Tabular Data)
| Property | Value |
|----------|-------|
| **Folder** | [01_Tabular_Framework](file:///d:/ResearchPaperPrepare/01_Tabular_Framework) |
| **Core Innovation** | Adaptive Quantile Calibration (AQC) |
| **Datasets** | Telco-Customer-Churn, IBM-HR-Analytics, UCI-Adult, Bank-Marketing |
| **Target Journal** | Expert Systems with Applications (IF=8.5) |
| **Status** | ✅ Experiments complete |
| **Key Results** | AQC reduces MPIW by 50.6% on Telco, 22.9% on Adult |
| **Paper Draft** | [paper_draft.md](file:///d:/ResearchPaperPrepare/01_Tabular_Framework/paper/paper_draft.md) |
| **Priority** | High |

### 方向二：时序预测统一框架 (Time Series Prediction Framework)
| Property | Value |
|----------|-------|
| **Folder** | [02_TimeSeries](file:///d:/ResearchPaperPrepare/02_TimeSeries) |
| **Core Innovation** | Variational Autoregressive Probabilistic Forecasting (VARP-F) |
| **Datasets** | Daily Climate, Sunspots, Bike-Sharing |
| **Target Journal** | IEEE Transactions on Neural Networks and Learning Systems (IF=14.2) |
| **Status** | ✅ Experiments complete |
| **Key Results** | Transformer: RMSE=1.617 (Daily Climate); LSTM: RMSE=19.26 (Sunspots) |
| **Paper Draft** | [paper_draft.md](file:///d:/ResearchPaperPrepare/02_TimeSeries/paper/paper_draft.md) |
| **Priority** | High |

### 方向三：文本+LLM框架 (Text + LLM Framework)
| Property | Value |
|----------|-------|
| **Folder** | [03_Text_LLM_Framework](file:///d:/ResearchPaperPrepare/03_Text_LLM_Framework) |
| **Core Innovation** | Lightweight Contrastive Learning for Sentiment Analysis |
| **Datasets** | Twitter Sentiment, IMDB, Amazon Reviews |
| **Target Journal** | ACM Transactions on Information Systems (IF=6.5) |
| **Status** | 🔄 Code ready, experiments running |
| **Key Results** | Pending |
| **Paper Draft** | Not yet created |
| **Priority** | Medium |

### 方向四：不平衡学习 (Imbalanced Learning)
| Property | Value |
|----------|-------|
| **Folder** | [04_Imbalanced_Learning](file:///d:/ResearchPaperPrepare/04_Imbalanced_Learning) |
| **Core Innovation** | Contrastive-Guided Reweighting Loss (CGRL) |
| **Datasets** | Credit Card Fraud, Pima Diabetes, Heart Disease, Telco Churn |
| **Target Journal** | Pattern Recognition (IF=8.5) |
| **Status** | ⚠️ Baseline complete, CGRL has bug |
| **Key Results** | Random Forest: F1=0.8031 (Credit Card Fraud) |
| **Paper Draft** | [paper_draft.md](file:///d:/ResearchPaperPrepare/04_Imbalanced_Learning/paper/paper_draft.md) |
| **Priority** | Medium |

### 方向五：旅游数据智能预测 (Tourism Data Intelligent Prediction)
| Property | Value |
|----------|-------|
| **Folder** | [05_Tourism_Prediction](file:///d:/ResearchPaperPrepare/05_Tourism_Prediction) |
| **Core Innovation** | Multi-View Spatio-Temporal Fusion Transformer (MVSTFT) |
| **Datasets** | Bike-Sharing |
| **Target Journal** | Tourism Management (IF=11.3) |
| **Status** | ✅ Experiments complete |
| **Key Results** | LightGBM: RMSE=1439.49, MAE=1146.25 |
| **Paper Draft** | [paper_draft.md](file:///d:/ResearchPaperPrepare/05_Tourism_Prediction/paper/paper_draft.md) |
| **Priority** | Medium |

### 方向六：农业数据融合决策支持 (Agricultural Data Fusion for Decision Support)
| Property | Value |
|----------|-------|
| **Folder** | [06_Agriculture_Fusion](file:///d:/ResearchPaperPrepare/06_Agriculture_Fusion) |
| **Core Innovation** | Multi-Modal Agricultural Knowledge Graph Fusion (MMAKGF) |
| **Datasets** | Crop Recommendation, EuroCrops, Coffee Quality |
| **Target Journal** | Computers and Electronics in Agriculture (IF=7.7) |
| **Status** | ✅ Experiments complete |
| **Key Results** | Random Forest: Accuracy=0.987, F1=0.987 |
| **Paper Draft** | [paper_draft.md](file:///d:/ResearchPaperPrepare/06_Agriculture_Fusion/paper/paper_draft.md) |
| **Priority** | Medium |

### 方向七：农业小样本增量学习 (Few-shot Incremental Learning for Agriculture)
| Property | Value |
|----------|-------|
| **Folder** | [SA-HSIC-Net](file:///d:/ResearchPaperPrepare/SA-HSIC-Net) |
| **Core Innovation** | HSIC-Guided Feature Protection (HSIC-GFP) |
| **Datasets** | DeepWeeds |
| **Target Journal** | Computers and Electronics in Agriculture (IF=7.7) |
| **Status** | ✅ 5 seeds experiments complete |
| **Key Results** | SA-HSIC-ProtoNet: 49.92±0.31% (5-shot, 2-way) |
| **Paper Draft** | [paper_draft.md](file:///d:/ResearchPaperPrepare/SA-HSIC-Net/paper/paper_draft.md) |
| **Priority** | High |

### 方向八：表格数据小样本增量学习 (Few-shot Incremental Learning for Tabular Data)
| Property | Value |
|----------|-------|
| **Folder** | [08_Tabular_FewShot](file:///d:/ResearchPaperPrepare/08_Tabular_FewShot) |
| **Core Innovation** | Contrastive-Enhanced Prototype Memory Network (CE-PMN) |
| **Datasets** | Telco-Customer-Churn, Adult-Income |
| **Target Journal** | Machine Learning (IF=5.8) |
| **Status** | ✅ Experiments complete |
| **Key Results** | ProtoNet: 63.90%→74.26% (1→20-shot, Telco) |
| **Paper Draft** | [paper_draft.md](file:///d:/ResearchPaperPrepare/08_Tabular_FewShot/paper/paper_draft.md) |
| **Priority** | Medium |

### 方向九：HSIC解耦轻量原型网络 (HSIC-Disentangled Lightweight Prototypical Network)
| Property | Value |
|----------|-------|
| **Folder** | [09_HSIC_FDANet](file:///d:/ResearchPaperPrepare/09_HSIC_FDANet) |
| **Core Innovation** | HSIC-ProtoNet with Feature Disentanglement |
| **Datasets** | PlantVillage |
| **Target Journal** | Computers and Electronics in Agriculture (IF=7.7) |
| **Status** | ✅ Complete with real experiments |
| **Key Results** | 5-shot: 86.10±1.07% (5-way), 196K params, 0.76MB INT8 |
| **Paper Draft** | [paper_draft_v4.md](file:///d:/ResearchPaperPrepare/09_HSIC_FDANet/paper/paper_draft_v4.md) |
| **Priority** | **High** - Ready for submission |

### 方向十：AI增强时空Transformer旅游预测 (AI-Enhanced Spatial-Temporal Transformer for Tourism)
| Property | Value |
|----------|-------|
| **Folder** | [10_AI_Tourism_Forecast](file:///d:/ResearchPaperPrepare/10_AI_Tourism_Forecast) |
| **Core Innovation** | ST-Transformer with Graph Attention |
| **Datasets** | Guangdong Tourism (planned), ALANA (Canada/Mexico/USA) |
| **Target Journal** | Tourism Management (IF=11.3) |
| **Status** | 🔄 Paper draft complete, experiments needed |
| **Key Results** | Simulated results only (needs real experiments) |
| **Paper Draft** | [paper_draft.md](file:///d:/ResearchPaperPrepare/10_AI_Tourism_Forecast/paper_draft.md) |
| **Priority** | Medium |

---

## Conflict Analysis

### Confirmed Conflicts (Not Separated)
| Source | Target | Conflict Type | Resolution |
|--------|--------|---------------|------------|
| SA-HSIC-Net | 方向七 | Direct conflict (same research) | SA-HSIC-Net is the implementation of 方向七 |

### Non-Conflicting Directions (Added as New)
| New Direction | Original Folder | Distinguishing Factor |
|---------------|-----------------|----------------------|
| 方向九 | HSIC-FDANet | Uses PlantVillage (5-way), focuses on edge deployment, different methodology (HSIC-ProtoNet vs SA-HSIC-ProtoNet) |
| 方向十 | AI-Tourism-Forecast | Uses ST-Transformer with graph attention, different from MVSTFT (方向五) |

---

## Priority Action Items

### 🔴 Critical - Need Immediate Attention
1. **方向四**: Fix CGRL bug in [train.py](file:///d:/ResearchPaperPrepare/04_Imbalanced_Learning/train.py) and complete experiments
2. **方向十**: Run real experiments on tourism datasets

### 🟡 Important - Should be Completed Soon
3. **方向三**: Complete experiments and write paper
4. **方向九**: Review and polish paper for submission

### 🟢 Ongoing - Progressing Well
5. **方向一**: Paper ready for review
6. **方向二**: Paper ready for review
7. **方向五**: Paper ready for review
8. **方向六**: Paper ready for review
9. **方向七**: 5-seed experiments complete, paper updated
10. **方向八**: Experiments complete, paper updated

---

## Target Journal Distribution

| Journal | Directions |
|---------|------------|
| Expert Systems with Applications | 方向一 |
| IEEE TNNLS | 方向二 |
| ACM TOIS | 方向三 |
| Pattern Recognition | 方向四 |
| Tourism Management | 方向五, 方向十 |
| Computers and Electronics in Agriculture | 方向六, 方向七, 方向九 |
| Machine Learning | 方向八 |

---

## Dataset Summary

| Dataset | Used By |
|---------|---------|
| Telco-Customer-Churn | 方向一, 方向四, 方向八 |
| IBM-HR-Analytics | 方向一 |
| UCI-Adult | 方向一, 方向八 |
| Bank-Marketing | 方向一 |
| Daily Climate | 方向二 |
| Sunspots | 方向二 |
| Bike-Sharing | 方向二, 方向五 |
| Credit Card Fraud | 方向四 |
| Pima Diabetes | 方向四 |
| Heart Disease | 方向四 |
| PlantVillage | 方向七 (old), 方向九 |
| DeepWeeds | 方向七 |
| Crop Recommendation | 方向六 |
| EuroCrops | 方向六 |
| Coffee Quality | 方向六 |
| Guangdong Tourism | 方向十 (planned) |
| ALANA | 方向十 (planned) |

---

## Experimental Resources

### Hardware Configuration
- **GPU**: NVIDIA RTX PRO 2000 16GB
- **CPU**: Intel Xeon W7-2595X
- **Memory**: 48GB DDR5 RDIMM
- **OS**: Windows 11

### Software Stack
- Python 3.13
- PyTorch 2.x
- scikit-learn
- XGBoost / LightGBM
- Pandas / NumPy

---

## Notes
- **SA-HSIC-Net** is NOT renamed because it directly corresponds to 方向七
- **HSIC-FDANet** → **09_HSIC_FDANet**: Independent direction with different methodology and dataset
- **AI-Tourism-Forecast** → **10_AI_Tourism_Forecast**: Independent direction with different methodology
- Total research directions: **10** (8 original + 2 new)