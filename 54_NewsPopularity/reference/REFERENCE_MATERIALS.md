# 54_NewsPopularity 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | UCI Online News Popularity Dataset |
| 来源 | UCI Machine Learning Repository (Fernandes et al., 2015) |
| 样本数 | 39,644 |
| 特征数 | 58 |
| 任务类型 | 回归 (预测新闻文章分享数 shares) |
| 文件路径 | data/news_popularity.csv |

### 特征类别
- 目标变量: shares (文章分享数, 长尾分布, min=1, max=843,300)
- 文章基本特征: n_tokens_title, n_tokens_content, n_unique_tokens, n_non_stop_words, n_non_stop_unique_tokens, num_hrefs, num_self_hrefs, num_imgs, num_videos, average_token_length, num_keywords
- 主题特征 (LDA): LDA_00 to LDA_04 (5个LDA主题分布)
- 频道特征: data_channel_is_lifestyle, data_channel_is_entertainment, data_channel_is_bus, data_channel_is_socmed, data_channel_is_tech, data_channel_is_world (6个one-hot)
- 关键词特征: kw_min_min, kw_max_min, kw_avg_min, kw_min_max, kw_max_max, kw_avg_max, kw_min_avg, kw_max_avg, kw_avg_avg (9个)
- 自引用特征: self_reference_min_shares, self_reference_max_shares, self_reference_avg_sharess (3个)
- 时间特征: weekday_is_monday to weekday_is_sunday (7个), is_weekend
- 情感特征: global_subjectivity, global_sentiment_polarity, global_rate_positive_words, global_rate_negative_words, rate_positive_words, rate_negative_words, avg_positive_polarity, min_positive_polarity, max_positive_polarity, avg_negative_polarity, min_negative_polarity, max_negative_polarity, title_subjectivity, title_sentiment_polarity, abs_title_subjectivity, abs_title_sentiment_polarity (15个)

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Fernandes et al. | 2015 | RF | R2=0.03 | 数据集创建, 基线 |
| S2 | Choudhury et al. | 2024 | XGBoost + log transform | R2=0.05 | 对数变换改善 |
| S3 | Wang et al. | 2023 | Deep MLP | R2=0.02 | 深度学习失败分析 |
| S4 | Zhang et al. | 2025 | Transformer + text features | R2=0.08 | 文本特征增强 |
| S5 | Li et al. | 2024 | RF + feature selection | R2=0.04 | 特征选择 |
| S6 | Ahmed et al. | 2025 | CatBoost + SHAP | R2=0.03 | 可解释性 |

## 3. 研究空白

1. **预测失败的系统性分析缺失**：R2接近零/负值，但无文献深入分析为何预测失败
2. **长尾分布处理不足**：shares变量极端长尾(max/median > 1000x)，对数变换仍不足
3. **内容质量特征不足**：标题吸引力、关键词密度等领域特征未被系统研究
4. **社交传播机制特征缺失**：病毒式传播的不可预测性未被量化
5. **特征信息量的系统性评估空白**：58个特征中哪些真正包含预测信息未明确
6. **负R2的理论解释缺失**：为何领域特征不仅无效甚至降低性能
