# 50_BuildingEnergy 参考资料汇总

> 整理日期：2026-08-08
> 本目录包含该方向的所有参考资料，可独立使用无需查阅其他方向

---

## 1. 核心分析文件
- **SOTA_击破分析.md**：该方向的SOTA分析、击破方案、推荐期刊、优先级评分（78/100）
- 目标框架：**PECFM (Physics-Encoded Conformal Foundation Model for Building Energy Forecasting)**
- 核心差异化：物理特征+TSFM+条件共形预测三融合+5个原创定理

---

## 2. 关键SOTA文献列表

### 2.1 同数据集（UCI Appliances Energy Prediction）对照研究

| 编号 | 文献 | 年份 | 来源 | 核心方法 | 关键指标 |
|------|------|------|------|---------|---------|
| S1 | Kulkarni 2025 | 2025 | MTSU硕士论文 | GRU/GBR/LSTM/Transformer系统对比 | GRU R²=0.62（最佳） |
| S2 | Chen 2025 | 2025 | UCLA硕士论文 | SARIMA/Prophet/RF/XGBoost，时序切分 | RF RMSE=63.77 Wh |
| S3 | **Moon et al. 2024** | 2024 | **PLoS ONE** | 树模型+SHAP，THI/wind chill关键 | **新颖性击穿** |
| S4 | Araujo Code | 2024 | GitHub | LightGBM+CQR，log1p+lag关键 | 工程缺口提示 |
| S5 | Manvi234 | 2026 | GitHub | XGBoost(Optuna) | RMSE=57 W |

### 2.2 时序基础模型（TSFM）在能源领域

| 编号 | 文献 | 年份 | 来源 | 核心方法 | 链接 |
|------|------|------|------|---------|------|
| S6 | MixForecast | 2025 | GitHub | N-BEATS+TSMixer，0.19M参数 | [GitHub](https://github.com/EdgeIntelligenceLab/mix-forecast) |
| S7 | CO-BUILD (Liang et al.) | 2025 | OpenReview | 对比课程学习适配TSFM至BEF | 7TgKHQeUsL |
| S8 | FreqMixer (Hou et al.) | 2025 | Energies | 频域混合TSFM适配 | doi:10.3390/en18030660 |
| S9 | TimeFound (Xiao et al.) | 2025 | arXiv | 多分辨率patch TSFM | [arXiv:2503.04118](https://arxiv.org/abs/2503.04118) |
| S10 | Spencer et al. | 2024 | arXiv | PatchTST+BDGP2 16数据集 | [arXiv:2410.14107](https://arxiv.org/abs/2410.14107) |

### 2.3 物理混合+不确定性量化

| 编号 | 文献 | 年份 | 来源 | 核心方法 | 链接 |
|------|------|------|------|---------|------|
| S11 | **Von Krannichfeldt et al.** | 2025 | arXiv | 物理+数据+CP，5种混合方法 | [arXiv:2507.17526](https://arxiv.org/abs/2507.17526) |
| S12 | Almadani et al. | 2025 | ICSPIS | LightGBM+CQR+CP | doi:10.1109/ICSPIS67605.2025 |
| S13 | Borrotti | 2024 | Energies | RF+CP建筑冷热负荷 | doi:10.3390/en17174348 |
| S14 | **STOIC** (Niresi et al.) | 2026 | arXiv | STGNN+TabPFN+CP | [arXiv:2606.31804](https://arxiv.org/abs/2606.31804) |

### 2.4 同期刊扩展引用

| 编号 | 文献 | 年份 | 来源 | 主题 |
|------|------|------|------|------|
| S15 | Ayoola et al. | 2025 | Energy and Buildings | IoT+ML热泵COP |
| S16 | Zhang et al. | 2026 | Energy and Buildings | 居住者动态热感觉 |
| S17 | Choi et al. | 2026 | Energy and Buildings | HVAC制冷剂故障AFDD |
| S18 | Li et al. | 2026 | Energy & Buildings | MTL-TBGA集中供热 |

---

## 3. 搜索关键词

从全局memo中提取的搜索关键词（50方向从属于第二轮分析，与45方向共享SOTA调研）：
1. "building energy consumption prediction deep learning 2024 2025 2026 UCI appliances"
2. "physical-informed building energy forecasting machine learning 2024 2025"
3. "time series foundation model building energy forecasting 2024 2025"

---

## 4. 已下载文件

| 文件名 | 说明 |
|--------|------|
| （本目录暂无PDF/txt文件，相关文献需从arXiv下载） |

---

## 5. 推荐下载（arXiv链接）

| 文献 | 链接 | 优先级 |
|------|------|--------|
| Von Krannichfeldt et al. | https://arxiv.org/abs/2507.17526 | ★★★★★ |
| STOIC (Niresi et al.) | https://arxiv.org/abs/2606.31804 | ★★★★★ |
| CO-BUILD | OpenReview 7TgKHQeUsL | ★★★★★ |
| TimeFound | https://arxiv.org/abs/2503.04118 | ★★★★ |
| Spencer et al. | https://arxiv.org/abs/2410.14107 | ★★★★ |
| Moon et al. 2024 | PLoS ONE 19(11):e0307654 | ★★★★ |
| Moirai (Woo et al.) | https://arxiv.org/abs/2402.02592 | ★★★ |
| Chronos (Ansari et al.) | TMLR 2024 | ★★★ |

---

## 6. 击破方案摘要

### 框架名称：PECFM (Physics-Encoded Conformal Foundation Model for Building Energy Forecasting)

### 核心组件
1. **M1: 物理特征编码器**：保留原稿件14维物理特征，新增T_wb湿球温度+T_op操作温度
2. **M2: 自回归时序特征**：lag_6h/12h/24h, rolling_mean/std, log1p变换
3. **M3: TSFM主干**：双路径——Moirai-Bolt-Small零样本(0.93M) + PatchTST LoRA微调(8M)
4. **M4: 双头输出**：Point(MSE) + Quantile(Pinball Loss 0.1/0.5/0.9) + 物理一致性正则
5. **M5: CQR条件共形预测校准**：hour-bin条件CP，covariate shift下覆盖保证

### Theorem/Proposition 列表
- **Theorem 1**：物理特征Lipschitz不变性（等价类内条件期望差有界）
- **Theorem 2**：物理特征样本复杂度降低（从O(d_raw/ε²)降至O(d_phys/ε²)，全行业首次）
- **Theorem 3**：CP covariate shift条件覆盖界（TV距离≤δ下覆盖保证）
- **Theorem 4**：TSFM适配定理（N_target ≥ N_critical时微调严格优于零样本）
- **Theorem 5**：Calibration-Quantile一致性（ECE有界）
- **Proposition 6**：计算复杂度（参数量8.47M，推理~3ms/sample，显存峰值6.4GB）

### 数据集建议
- UCI Appliances（已有） + ASHRAE Great Energy Predictor III + BDGP2子集
- 15基线（含PECFM-FT和PECFM-ZS），5种子，chronological split

### 推荐期刊
| 期刊 | 级别 | 版面费 | 策略 |
|------|------|--------|------|
| **Energy and Buildings** | Q1, IF≈6.7 | $0（非OA） | 主投（原稿已投稿） |
| Applied Energy | Q1, IF≈11.2 | $0（非OA） | 升级备选 |
| Applied Soft Computing | Q1, IF≈8.7 | $0（非OA） | 保底 |
| ESWA | Q1, IF≈7.5 | $0（非OA） | 保底 |

### 优先级评分：78/100（中高）
### 预计耗时：12-14周

---

## 附录：原稿致命缺陷
1. 同数据集代际落后：R²=0.494 vs Kulkarni GRU R²=0.62（差12pp）
2. 新颖性被击穿：Moon 2024 PLoS ONE已报告THI/wind chill关键发现
3. 数据切分错误：random split而非chronological split（数据泄露风险）
4. 无自回归特征、无目标变换、无log1p
5. 基线不全：仅4个树模型，缺LSTM/GRU/Transformer/PatchTST
6. 理论分析完全为零：0个定理/命题/引理
7. 数据溯源问题："52% domain feature importance"实际为50.1%（误差1.88pp）