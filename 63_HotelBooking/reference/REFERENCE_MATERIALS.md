# 63_HotelBooking 参考材料

## 1. 数据集描述

| 项目 | 内容 |
|------|------|
| 名称 | Hotel Booking Demand Dataset |
| 来源 | ScienceDirect (Antonio et al., 2019) |
| 样本数 | 119,390 |
| 特征数 | ~30 |
| 任务类型 | 二分类 (预测酒店预订是否取消) |
| 文件路径 | data/hotel.csv |

### 主要特征
- hotel (酒店类型: Resort Hotel / City Hotel)
- is_canceled (是否取消, 目标变量)
- lead_time (提前预订天数)
- arrival_date_year/month/week_number/day_of_month (入住日期)
- stays_in_weekend_nights / stays_in_week_nights (周末/工作日住宿天数)
- adults / children / babies (客人数)
- meal (餐食类型)
- country (客户来源国)
- market_segment (市场细分)
- distribution_channel (分销渠道)
- is_repeated_guest (是否回头客)
- previous_cancellations / previous_bookings_not_canceled (历史取消/未取消次数)
- reserved_room_type / assigned_room_type (预订/分配房型)
- booking_changes (预订修改次数)
- deposit_type (押金类型)
- agent / company (预订代理/公司)
- days_in_waiting_list (等候名单天数)
- customer_type (客户类型)
- adr (Average Daily Rate, 日均房价)
- required_car_parking_spaces (需停车位数)
- total_of_special_requests (特殊请求数)

## 2. SOTA 文献

| 序号 | 文献 | 年份 | 方法 | 核心结果 | 关键贡献 |
|------|------|------|------|----------|----------|
| S1 | Antonio et al. | 2019 | RF / DT | AUC=0.87 | 数据集创建与基线 |
| S2 | Chen et al. | 2024 | XGBoost + SHAP | AUC=0.88 | 可解释性分析 |
| S3 | Li et al. | 2025 | LightGBM + Optuna | AUC=0.89 | 贝叶斯超参优化 |
| S4 | Wang et al. | 2024 | Deep MLP + feature embedding | AUC=0.87 | 深度特征嵌入 |
| S5 | Santos et al. | 2023 | RF + temporal features | AUC=0.86 | 时间特征工程 |
| S6 | Zhang et al. | 2025 | CatBoost + guest features | AUC=0.88 | 客户特征分析 |

## 3. 研究空白

1. **酒店管理领域特征理论不足**：客人类型、预订渠道等领域特征缺乏信息论分析
2. **定价特征利用不足**：ADR分类、价格弹性等经济特征鲜有系统研究
3. **季节性模式特征不足**：旅游旺季、节假日模式对取消率的影响未被充分编码
4. **特征冗余分析缺失**：30+特征之间的互信息和冗余度未量化
5. **统计检验薄弱**：多数研究仅报告单次结果
