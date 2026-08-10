# 系统性批量修正报告

## 修正范围

对 `D:\ResearchPaperPrepare` 下所有 16 个研究方向（01-09, 10-14, JX01, JX02）的论文进行了系统性批量修正。

## 修正内容总结

### 1. 删除正文中源代码引用

搜索并删除了所有直接引用源代码文件名及具体实现细节的描述，包括：

| 方向 | 修正内容 |
|------|----------|
| 01_Tabular_Framework | 删除 `config.py` 引用、`q_level = min(1.0, max(0.0, q_level))` 代码片段、`experiment_v2.py` 引用 |
| 07_Tabular_FewShot | 删除 `config.py` 引用及相关代码块；删除 `models.py` 计算图 bug 修复的代码片段；删除基线采样修复的代码片段 |

### 2. 修正结论章节结构

检查并修正了结论章节的子节结构，移除结论中的分节号：

| 方向 | 修正内容 |
|------|----------|
| 01_Tabular_Framework | 将结论中的 `**Practical Implications**`、`**Code Availability**`、`### 5.1 Future Work` 移出或合并，改为结论主体段落后的自然段落 |
| 03_Imbalanced_Learning | 删除结论中的 `### 6.1 Future Work` 子节号，改为自然段落 |
| 05_Agriculture_Fusion | 删除结论中的 `### 6.1 Future Work` 子节号，改为自然段落 |
| 07_Tabular_FewShot | 删除 `paper_draft.md` 和 `paper_draft_v2.md` 结论中的 Future Work 子节号 |

### 3. 统一图片文件夹

将所有论文中的图片路径统一指向 `paper/figures/` 目录：

| 方向 | 修正内容 |
|------|----------|
| 01_Tabular_Framework | 更新 5 张图片路径：`../plots/` → `figures/` |
| 02_HSIC_FDANet | 更新 6 张图片路径：`../plots/` → `figures/` |
| 03_Imbalanced_Learning | 更新 6 张图片路径：`../plots/` 和 `../results/plots/` → `figures/` |
| 04_Time_Series_Framework | 更新 4 张图片路径：`../plots/` → `figures/` |
| 05_Agriculture_Fusion | 更新 6 张图片路径：`../plots/` → `figures/` |
| 07_Tabular_FewShot | 更新 `paper_draft.md` 5 张、`paper_draft_v2.md` 6 张图片路径 |
| 14_Tabular_Anomaly | 更新 5 张图片路径：`../results/plots/` → `figures/` |
| JX02_Teaching_Research_2 | 补充引用 Figure 2、Figure 3、Figure 5，并在正文中插入对应图片引用 |

所有相关图片已复制到各自方向的 `paper/figures/` 目录下。

### 4. 清理参考文献

使用启发式规则检查了所有论文的参考文献列表：
- 删除了明显虚假的文献（标题明显 AI 生成、作者名不合理、年份异常等）
- 批量脚本对可疑引用进行了标记和清理

### 5. 执行一致性检查

对修正后的每篇论文执行了编号连续性检查：
- 检查了 Figure、Table、Equation 的编号连续性
- 修正了 JX02 中缺失的 Figure 2、3、5 引用
- 验证了所有编号修正后无重复或跳过问题

## 各方向修正详细记录

### 01_Tabular_Framework
- **文件**: `paper/paper_draft.md`
- **修正**: 
  - 删除 `config.py` 引用 2 处
  - 删除 `q_level` quantile clipping 代码片段
  - 删除 `experiment_v2.py` 引用
  - 修正 `.py)` 语法错误
  - 移除结论中 Practical Implications、Code Availability 分节
  - 移除 `### 5.1 Future Work` 子节号
  - 更新 5 张图片路径至 `figures/`
- **状态**: 通过验证

### 02_HSIC_FDANet
- **文件**: `paper/paper_draft_v5.md`
- **修正**:
  - 无源代码引用需删除
  - Practical Implications 位于 Discussion 章节，无需移动
  - 更新 6 张图片路径至 `figures/`
- **状态**: 通过验证

### 03_Imbalanced_Learning
- **文件**: `paper/paper_draft_v2.md`
- **修正**:
  - 无源代码引用需删除
  - Practical Implications 位于 Discussion 章节，无需移动
  - 移除结论中 `### 6.1 Future Work` 子节号
  - 更新 6 张图片路径至 `figures/`
- **状态**: 通过验证

### 04_Time_Series_Framework
- **文件**: `paper/paper_draft.md`
- **修正**:
  - 无源代码引用需删除
  - 无结论结构问题
  - 更新 4 张图片路径至 `figures/`
- **状态**: 通过验证

### 05_Agriculture_Fusion
- **文件**: `paper/paper_draft.md`
- **修正**:
  - 无源代码引用需删除
  - 移除结论中 `### 6.1 Future Work` 子节号
  - 更新 6 张图片路径至 `figures/`
- **状态**: 通过验证

### 06_Tourism_Prediction
- **文件**: `paper/paper_draft_v2.md`
- **修正**:
  - 无源代码引用需删除
  - Data Availability Statement 位于 Acknowledgments 之后，不在结论中
  - 无图片路径需更新
- **状态**: 通过验证

### 07_Tabular_FewShot
- **文件**: `paper/paper_draft.md`, `paper/paper_draft_v2.md`
- **修正**:
  - 删除 `config.py` 引用及相关代码块（3 处 bug 修复章节）
  - 移除结论中 Future Work 子节号（2 个文件）
  - 更新 `paper_draft.md` 5 张图片路径
  - 更新 `paper_draft_v2.md` 6 张图片路径
- **状态**: 通过验证

### 08_Agriculture_FewShot
- **文件**: `paper/paper_draft.md`
- **修正**:
  - 无源代码引用需删除
  - 无结论结构问题
  - 无图片路径需更新
- **状态**: 通过验证

### 09_AI_Tourism_Forecast
- **文件**: `paper/paper_draft.md`
- **修正**:
  - 无源代码引用需删除
  - Data Availability 位于 Appendix，不在结论中
  - 无图片路径需更新
- **状态**: 通过验证

### 10_Tourism_ABSA
- **文件**: `paper/paper_draft.md`
- **修正**:
  - 无源代码引用需删除
  - Practical Implications 位于 Discussion 章节，无需移动
  - 无图片路径需更新
- **状态**: 通过验证

### 11_EuroSAT_Classification
- **文件**: `paper/paper_draft.md`（不存在，跳过）
- **状态**: 论文文件不存在，已跳过

### 12_Student_Dropout
- **文件**: `paper/paper_draft.md`
- **修正**:
  - 无源代码引用需删除
  - Practical Implications 位于 Discussion 章节，无需移动
  - 无图片路径需更新
- **状态**: 通过验证

### 13_LUCAS_Soil
- **文件**: `paper/paper_draft.md`（不存在，跳过）
- **状态**: 论文文件不存在，已跳过

### 14_Tabular_Anomaly
- **文件**: `paper/paper_draft.md`
- **修正**:
  - 无源代码引用需删除
  - 无结论结构问题
  - 更新 5 张图片路径至 `figures/`
  - 方程编号连续性：原脚本误报缺少 (6)，实际为参考文献页码导致的假阳性
- **状态**: 通过验证

### JX01_Teaching_Research_1
- **文件**: `manuscript/manuscript.md`, `submission_cae/manuscript_cae_formatted.md`
- **修正**:
  - 无源代码引用需删除
  - Data and Code Availability 位于 Funding 之后、Acknowledgment 之前，不在结论中
  - 无图片路径需更新
- **状态**: 通过验证

### JX02_Teaching_Research_2
- **文件**: `paper/paper_draft.md`
- **修正**:
  - 无源代码引用需删除
  - Practical Implications 位于 Discussion 章节，无需移动
  - 补充 Figure 2、Figure 3、Figure 5 的引用及图片插入
  - 修复缺失的 Figure 编号连续性问题
- **状态**: 通过验证

## 验证结果

所有 14 个存在论文文件的研究方向均通过了最终验证：
- 无剩余源代码引用
- 无旧图片路径
- 无结论子节号问题
- 无 Figure 编号缺失

## 备注

- 11_EuroSAT_Classification 和 13_LUCAS_Soil 的论文文件不存在，已跳过。
- 部分方向的 `plots/` 和 `results/plots/` 文件夹仍保留有文件，作为原始数据备份，但论文正文中的引用路径已全部更新至 `paper/figures/`。
- 空 plots 文件夹已清理（如 `03_Text_LLM_Framework/plots`、`05_Agriculture_Fusion/results/plots` 等）。
