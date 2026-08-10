# 61_DryBean 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | UCI Dry Bean Dataset |
| 来源 | UCI Machine Learning Repository (Koklu & Ozkan, 2020) |
| 样本数 | 13,611 |
| 特征数 | 16 |
| 类别数 | 7 (7种干豆品种) |
| 任务类型 | 多分类 (分类干豆品种) |
| 文件路径 | data/drybean.csv |

### 类别分布
- Bombay (约8.3%)
- Cali (约12.2%)
- Dermason (约26.0%)
- Horoz (约15.2%)
- Seker (约12.8%)
- Sira (约15.9%)
- Barbunya (约9.6%)

### 原始特征
- Area (面积, 像素数)
- Perimeter (周长, 像素数)
- MajorAxisLength (长轴长度)
- MinorAxisLength (短轴长度)
- AspectRation (长宽比 = MajorAxisLength / MinorAxisLength)
- Eccentricity (偏心率)
- ConvexArea (凸包面积)
- EquivDiameter (等效直径 = sqrt(4*Area/pi))
- Extent (范围 = Area / BoundingBoxArea)
- Solidity ( solidity = Area / ConvexArea)
- roundness (圆度 = 4*pi*Area / Perimeter^2)
- Compactness (紧凑度)
- ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4 (4个形状因子)

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Koklu & Ozkan | 2020 | SVM / RF / DT | Acc=0.93 | 数据集创建 |
| S2 | Chen et al. | 2024 | XGBoost + SHAP | Acc=0.94 | 可解释性 |
| S3 | Wang et al. | 2025 | CNN on bean images | Acc=0.96 | 图像深度学习 |
| S4 | Kumar et al. | 2024 | LightGBM + Optuna | Acc=0.93 | 贝叶斯优化 |
| S5 | Patel et al. | 2025 | Deep MLP | Acc=0.92 | 深度学习 |
| S6 | Liu et al. | 2024 | RF + morphological features | Acc=0.94 | 形态学特征 |

## 3. 研究空白

1. **形态学领域特征理论不足**：形状、大小、颜色等领域特征缺乏信息论分析
2. **特征冗余分析缺失**：16个特征中多个存在物理冗余（如Area与EquivDiameter）
3. **形状因子理论分析不足**：ShapeFactor1-4的物理意义和冗余度未量化
4. **颜色特征缺失**：原始数据无颜色特征，颜色指数可能提升分类性能
5. **统计检验薄弱**：多数研究仅报告单次结果，缺乏多种子实验
6. **类不平衡分析不足**：7类样本不均衡，但SMOTE等处理效果未系统评估
