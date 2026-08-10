# 统计分析与一致性检查报告

## 方向05: Agriculture Fusion (已完成)

### 完成的工作
1. **补充统计分析**
   - 计算了CMRGT的95%置信区间（Accuracy: [0.5677, 0.5947], F1: [0.5551, 0.5841]）
   - 执行了One-Sample t检验（CMRGT vs 每个基线）
   - 计算了Cohen's d效果量
   - 新增Table 5（95% CI）和Table 6（统计显著性检验）

2. **一致性修复**
   - 修正作者信息（Yafen Feng^2,3, Jianghong Guo^1, Chuanxian Jiang^1, Jingyuan Zeng^1,*）
   - 统一单位编号
   - 扩充摘要至205词（符合200-250词要求）
   - 统一数据引用（58.1% mean accuracy over 5 seeds）
   - 统一PlantVillage类别数为38
   - 添加公式编号(1)-(11)

3. **数据真实性验证**
   - 所有数字均可溯源到results/目录下的CSV文件
   - 验证了关键声明：12.4 pp差异、R²=0.891等
   - 未发现无法溯源的数字

4. **投稿材料更新**
   - 重写Cover Letter（与论文内容一致）
   - 重写Highlights（5条，均<=85字符）

### 结果文件
- `results/tables/statistical_analysis.csv` - 完整统计检验结果

---

## 方向06: Tourism Prediction (已完成)

### 完成的工作
1. **补充统计分析**
   - 运行了多种子实验（3 seeds: 42, 123, 456; 30 epochs）
   - 新增Table 3: Multi-Seed Stability Analysis
   - 计算了95% CI、One-Sample t-tests、Cohen's d
   - 发现关键结果：深度学习模型方差显著高于集成方法

2. **数据真实性修正（重大）**
   - 删除无法溯源的Tourism Demand数据集（原Table 3）
   - 删除无法溯源的消融实验表格（原Table 4, 5, 6）
   - 修正Table 2，移除Tourism Demand列，标注为"Approximate p-values"
   - 修正Figure 2 caption，移除"5 random seeds"声明

3. **一致性修复**
   - 修正作者信息
   - 删除所有对Tourism Demand Dataset的引用
   - 更新摘要、Discussion、Conclusion（诚实报告稳定性问题）
   - 更新Keywords
   - 更新Section 1.2贡献点描述

4. **投稿材料更新**
   - 重写Cover Letter（与06方向论文匹配）
   - 重写Highlights（5条，均<=85字符）

### 关键发现
- **MVSTFT_v2在多种子实验中表现出高方差**（RMSE std=49.07），95% CI为[105.86, 349.67]
- **LightGBM等集成方法在不同种子下结果完全稳定**
- **主实验（100 epochs单次运行）的结果可能是特定种子下的特例**
- 这一发现支持了更诚实的论文表述

### 结果文件
- `results/quick_multi_seed.json` - 多种子实验原始结果
- `results/quick_multi_seed_summary.csv` - 统计汇总

---

## 数据真实性评分

### 05方向: 100/100
- 所有数字均可溯源到results/文件
- 统计检验基于真实多种子数据

### 06方向: 90/100（提升中）
- Table 1数据可溯源 ✓
- 新增Table 3基于真实实验 ✓
- Table 2的p值为近似估计（已标注）
- 参数敏感性分析可溯源 ✓
- 原消融实验数据已删除并标注为"未完成"

---

## 建议

### 05方向
- 已达到投稿水平，可直接投稿Computers and Electronics in Agriculture

### 06方向
- 需要进一步运行完整的多种子实验（100 epochs, >=5 seeds）以替换Table 2中的近似p值
- 需要完成消融实验并保存结果到results/目录
- 建议考虑使用paper_draft_v2.md作为基础（该版本更诚实且数据一致性更好）
