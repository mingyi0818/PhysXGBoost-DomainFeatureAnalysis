# 经验备忘录（memo.md）

> 本文件记录数据集管理和论文写作过程中积累的**经验教训**，供后续工作参考。
> 数据集资产索引见 [datasets.md](datasets.md)，临时操作指导见 [advice.md](advice.md)。

**更新日期**: 2026-07-18

---

## 一、数据集管理经验

### 1.1 存储策略
- **原始数据集统一存放在 `D:\datasets\`**，按领域分类存放（tourism/image/timeseries/等）
- **工作区 `D:\ResearchPaperPrepare\` 下仅保留预处理后的数据**（npz等）和代码/论文
- 备份时只需备份工作区，不需要备份大型原始数据集

### 1.2 数据集下载经验

| 数据集 | 下载方式 | 注意事项 |
|--------|---------|---------|
| UCI ML Repository | `ucimlrepo` Python包 | 最稳定，推荐优先使用 |
| Kaggle Datasets | `kaggle datasets download` | 需配置 `~/.kaggle/kaggle.json`，路径不能用中文或含空格的目录 |
| Kaggle Competitions | `kaggle competitions download` | **需先在Kaggle网页accept competition rules**，否则API返回空错误 |
| HuggingFace | `huggingface_hub` Python包 | 稳定，直接用`snapshot_download` |
| NASA Prognostic Repository | 网页手动下载 | **必须先注册账号并同意数据使用条款**，API无法直接下载 |
| GitHub LFS | `git lfs install` + `git clone` | 普通clone只下载指针文件（几十字节），不是实际图片 |

### 1.3 已知损坏数据
- `D:\datasets\agriculture\crop_recommendation.csv` — 内容为"404: Not Found"
- `D:\datasets\agriculture\irrigation_tomato.csv` — 内容为"404: Not Found"
- `D:\datasets\agriculture\farm_products.csv` — 内容为HTML页面
- `D:\datasets\tourism\paris_pois.csv` — 数据异常（仅1行1列）

### 1.4 可释放空间
- `D:\datasets\image\NewPlantDiseases\new-plant-diseases-dataset.zip` (2.76GB) — 已解压，zip可删除
- `D:\datasets\image\EuroSAT\EuroSAT.zip` (89.91MB) — 已解压，zip可删除
- `D:\datasets\GTPBD\GTPBD_enhenced_png.zip` (14.9GB) — 根目录和子目录各一份，删除其中一份

---

## 二、论文写作经验

### 2.1 数据造假检测
- **AI幻觉引文特征**：文章号重复（如122549在3个方向出现）、作者名循环（Yang/Liu/Wang/Zhang）、页码格式过于规整
- **数据造假检测**：逐数字与results/文件比对，训练集/验证集/测试集必须区分
- **提升幅度验证**：手动计算 `(method - baseline) / baseline`，与论文声称值对比

### 2.2 方法不显著的常见原因
- 数据集太小（N<1000）→ 结论无推广价值
- 数据集太简单（如Banknote准确率>99%）→ 所有方法都好，无法区分
- 方法创新点与基线差异过小（如HSIC-ProtoNet vs ProtoNet仅多一个projector）
- 公开benchmark数据集竞争激烈，刷点难以发表

### 2.3 论文方向评估标准（AGENTS.md暂缓标准）
1. 数据集太小 → 暂缓（除非有特殊意义）
2. 模拟/合成数据集 → 暂缓（除非教改论文）
3. 实验效果低于baseline → 暂缓（除非有特殊意义）

### 2.4 图表规范
- 架构图用PlantUML生成（`.puml`源文件 + `.svg`输出）
- 图片不加标题在图内，用 `**Figure X: Caption**` 格式在图下方标注
- 参考文献用 `<sup>[N]</sup>` 上标格式
- 参考文献按首次出现顺序编号

### 2.5 实验时间预估（RTX Pro 2000 16GB）
- BERT微调（4700条，12 epochs）：单次~3-5分钟
- 时序预测LSTM（17K样本，200 epochs）：单次~3-5分钟
- 表格数据RF/XGBoost（30K样本）：单次~30-120秒
- Few-Shot episode训练（LightweightBackbone）：单方法单seed~2-3分钟
- 预计规则：总时间<2小时直接执行，2-8小时分phase，>8小时需征求意见

### 2.6 并行vs顺序处理
- **顺序处理更高效**（经验：并行导致CPU/GPU利用率低）
- 一次只处理一个方向，避免上下文不足
- 长时间运行的任务（>2小时）应暂停后处理其他方向

---

## 三、文件分工原则

| 文件 | 职责 | 更新频率 |
|------|------|---------|
| `aicommand.md` | 论文写作指令、审查标准、多智能体工作流 | 较少（核心规则稳定） |
| `advice.md` | 当前方向的临时操作指导、数据路径映射 | 较频繁（随任务进展） |
| `memo.md` | 经验教训、数据集下载注意事项、已知问题 | 偶尔（发现问题后记录） |
| `datasets.md` | 数据集资产索引、存放位置、关联论文 | 中等（下载新数据集时更新） |
| `投稿前审查报告.md` | 全方向投稿状态总览 | 每轮审查后更新 |

---

## 四、待办备忘

- [ ] NASA N-CMAPSS/CMAPSS数据集：需手动到 https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/ 注册并同意条款后下载
- [ ] Kaggle Competition数据集（NYC Taxi、IEEE Fraud、M5）：需在Kaggle网页accept competition rules后才能API下载
- [ ] 农业数据损坏文件需重新下载（crop_recommendation.csv、irrigation_tomato.csv、farm_products.csv）
- [ ] NewPlantDiseases zip包（2.76GB）可删除释放空间
- [ ] 各方向代码中的DATA_ROOT路径需更新指向D:\datasets\

---

*本文件为经验备忘，内容持续积累*
