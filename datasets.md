# 数据集总览

**更新日期**: 2026-07-24

**存储策略**: 所有原始数据集统一存放于 `D:\datasets\` 按领域分类，工作区 `D:\ResearchPaperPrepare\` 下仅保留预处理后的数据(npz等)。

---

## 一、论文相关数据集（已在研/已发表）

| # | 数据集名称 | 存放路径 | 规模 | 类型 | 关联研究方向 | 论文方向编号 |
|---|-----------|---------|------|------|-------------|------------|
| 1 | UCI Bike-Sharing (hour.csv) | `D:\datasets\tourism\UCI_Bike_Sharing\` | 17,379行 x 14列 | CSV/时序 | 共享单车需求预测 | 06_Tourism_Prediction（暂缓） |
| 2 | SemEval-2014 Restaurant ABSA | `D:\datasets\nlp\SemEval-2014_ABSA\` | 4,728条(训练3,608+测试1,120) | CSV/NLP | 餐饮评论情感分析 | 10_Tourism_ABSA（暂缓） |
| 3 | QuadABSA-TourismDataset | `D:\datasets\nlp\SemEval-2014_ABSA\` | 2,345行 x 9列 | CSV/NLP | 旅游评论ABSA | 10_Tourism_ABSA（暂缓） |
| 4 | Datafiniti Hotel Reviews | `D:\datasets\nlp\SemEval-2014_ABSA\Datafiniti_Hotel_Reviews\` | ~55,912条(3个CSV) | CSV/NLP | 酒店评论(无ABSA标注) | 10_Tourism_ABSA（辅助） |
| 5 | NSL-KDD | `D:\datasets\network_security\NSL-KDD\` | 训练125,973+测试22,544 | ARFF/TXT | 网络入侵检测 | 14_Tabular_Anomaly（A级） |
| 6 | UNSW-NB15 | `D:\datasets\network_security\UNSW-NB15\` | 训练82,332+测试175,341 x 45列 | CSV | 网络入侵检测 | 14_Tabular_Anomaly（A级） |
| 7 | OULAD | `D:\datasets\education\OULAD\` | 32,593名学生 x 7表 | CSV | 学生行为/辍学预测 | 12_Student_Dropout（B+级） |
| 8 | ALANA Airbnb | `D:\datasets\tourism\ALANA_Airbnb\` | 4个区域 x 108月 | CSV/时序 | Airbnb评论预测 | 09_AI_Tourism_Forecast（C+级） |
| 9 | LUCAS Synthetic | `D:\datasets\misc\LUCAS_Synthetic\` | 19,000行 x 4,207列 | CSV(1.42GB) | 土壤光谱预测 | 13_LUCAS_Soil（暂停） |
| 10 | EuroSAT | `D:\datasets\image\EuroSAT\` | 27,000张 x 10类 | JPG/遥感 | 地物分类 | 11_EuroSAT_Classification（暂停） |
| 11 | IP102 | `D:\datasets\image\IP102\` | 75,222张 x 102类 | JPG/农业 | 作物害虫分类 | 02_HSIC_FDANet（暂缓） |
| 12 | NewPlantDiseases | `D:\datasets\image\NewPlantDiseases\` | 87,867张 x 38类 | JPG/农业 | 植物病害分类 | 02_HSIC_FDANet, 08_Agriculture_FewShot（暂缓） |
| 13 | PlantVillage | `D:\datasets\image\PlantVillage\` | ~54,000张 x 38类 | JPG/农业 | 植物病害分类 | 02_HSIC_FDANet, 08_Agriculture_FewShot（暂缓） |
| 14 | McMiner | `D:\ResearchPaperPrepare\JX02_Teaching_Research_2\data\mcminer\` | 1,021样本, 65种误解类型 | JSON/代码 | 编程误解检测 | JX02_Teaching_Research_2（A级） |
| 15 | ProgFeed + CodeBench | `D:\ResearchPaperPrepare\JX01_Teaching_Research_1\data\` | 6,693次提交 + 3学期数据 | CSV/教育 | 编程课程早期预警 | JX01_Teaching_Research_1（A级） |
| 16 | Crop_recommendation | `D:\ResearchPaperPrepare\05_Agriculture_Fusion\data\raw\` | 2,200行 x 8列 | CSV/农业 | 作物推荐 | 05_Agriculture_Fusion（暂缓） |
| 17 | Wheat_Yield | `D:\ResearchPaperPrepare\05_Agriculture_Fusion\data\raw\` | 小麦产量数据 | CSV/农业 | 小麦产量预测 | 05_Agriculture_Fusion（暂缓） |
| 18 | Yelp Dataset | `D:\datasets\tourism\Yelp_POI_Recommend\` | ~8.8GB(6个JSON+1PDF) | JSON/旅游 | POI推荐 | 43_Tourism_Recommend（A级） |
| 19 | CropAndWeed | `D:\datasets\CropAndWeed\` | 3子文件夹(原始+工具包+裁剪) | JPG/农业 | 少样本杂草分类 | 25_FewShot_Weed |
| 20 | DeepWeeds | `D:\datasets\DeepWeeds\` | 5子文件夹(17,509图片,9类) | JPG/农业 | 杂草分类 | 08_Agriculture_FewShot（暂缓） |
| 21 | WeedSense | `D:\datasets\WeedSense\` | train/val/test+cache | CSV+JPG/农业 | 杂草识别 | 25_FewShot_Weed（辅助） |

## 二、已下载待研究数据集

| # | 数据集名称 | 存放路径 | 规模 | 类型 | 预期研究方向/用途 | 是否已开始研究 |
|---|-----------|---------|------|------|----------------|-------------|
| 18 | Hotel Booking Demand | `D:\datasets\tourism\Hotel_Booking_Demand\` | 119,390行 x 32列 | CSV/旅游 | 预订取消预测、需求预测、客户分群；数字孪生+旅游方向 | 否 |
| 19 | Beijing PM2.5 | `D:\datasets\timeseries\Beijing_PM25\` | 43,824行 x 13列 | CSV/环境 | 空气质量预测、时序预测 | 否 |
| 20 | Rain in Australia | `D:\datasets\timeseries\Rain_Australia\` | 145,460行 x 23列 | CSV/气象 | 降雨预测、分类 | 否 |
| 21 | Household Electric Power | `D:\datasets\energy\Household_Electric_Power\` | 2,075,259行 x 9列 | CSV/能源 | 用电负荷预测、异常检测；数字孪生+建筑能耗 | 否 |
| 22 | PJM Energy Consumption | `D:\datasets\energy\PJM_Energy_Consumption\` | 13个CSV, PJME最大145,366行 | CSV/能源 | 电力负荷预测、时序预测 | 否 |
| 23 | Credit Card Default | `D:\datasets\tabular\Credit_Card_Default\` | 30,000行 x 25列 | CSV/金融 | 信用风险评估、分类 | 否 |
| 24 | Online Shoppers Intention | `D:\datasets\tabular\Online_Shoppers_Intention\` | 12,330行 x 18列 | CSV/电商 | 购买意向预测、用户行为分析 | 否 |

## 三、未下载/需手动获取的数据集

| # | 数据集名称 | 来源 | 规模 | 预期研究方向/用途 | 未下载原因 |
|---|-----------|------|------|----------------|-----------|
| 25 | N-CMAPSS (Turbofan Engine) | NASA Prognostic Repository | 9台发动机全寿命 | CSV/HDF5/PHM | 数字孪生+预测性维护(RUL预测) | 需在NASA网站同意使用条款后手动下载 |
| 26 | CMAPSS (原版) | NASA | FD001: 100+100台 | CSV/PHM | 发动机RUL预测 | 需在NASA网站同意使用条款后手动下载 |
| 27 | NYC Taxi Trip Duration | Kaggle Competition | 1,458,644行 | CSV/交通 | 行程时间预测 | 需先在Kaggle网页accept competition rules |
| 28 | IEEE-CIS Fraud Detection | Kaggle Competition | ~590,000行 x 400+列 | CSV/金融 | 欺诈检测、不平衡分类 | 需先在Kaggle网页accept competition rules |
| 29 | XJTU-SY Bearing | 西安交通大学 | 多工况全寿命 | 振动信号/PHM | 轴承故障诊断 | GitHub可能有，待确认 |
| 30 | M5 Forecasting | Kaggle Competition | 42,840时序(4200万行) | CSV/零售 | 多层级时序预测 | 需先在Kaggle网页accept competition rules |

## 四、其他已有数据集（非论文直接相关）

| # | 数据集名称 | 存放路径 | 规模 | 说明 |
|---|-----------|---------|------|------|
| 31 | COVID影像数据集 | `D:\datasets\COVID\` | 6个分片zip (~38.9GB) | COVID X-ray/CT影像 |
| 32 | Slope LiDAR (SLidE) | `D:\datasets\Dataset-Slope-LiDAR-Embankment-SLidE\` | ~241 MB | 边坡LiDAR点云数据 |
| 33 | GTPBD (梯田数据集) | `D:\datasets\GTPBD\` | 含14.9GB压缩包 | 梯田遥感数据集 |
| 34 | 客家话语音数据 | `D:\datasets\dataset\` + `D:\datasets\NlpStudy\` | ~12.3GB zip | 客家话ASR研究 |
| 35 | USGS旧金山湾区滑坡 | `D:\datasets\usgs旧金山湾区滑坡数据集\` | 3个滑坡点 | 滑坡水文/地质数据 |
| 36 | TexturePic | `D:\datasets\TexturePic\` | 40张TIFF (10MB) | 纹理图片集 |
| 37 | Paris POIs | `D:\datasets\tourism\paris_pois.csv` | 空/异常 | 巴黎兴趣点（数据损坏） |
| 38 | Stravl旅游偏好 | `D:\datasets\tourism\Stravl-Data\` | 21.28 MB | 旅游偏好数据 |
| 39 | 农业杂项数据 | `D:\datasets\agriculture\` | 多个小CSV | 作物推荐/产量/灌溉(部分损坏) |

## 五、工作区预处理数据（npz，不移动）

这些是代码运行时生成的预处理数据，保留在工作区不移动：

| 方向 | 文件 | 大小 | 说明 |
|------|------|------|------|
| 01_Tabular_Framework | `data/adult_processed.npz` 等4个 | ~6.6 MB | Adult/Bank/IBM_HR/Telco预处理数据 |
| 03_Imbalanced_Learning | `data/credit_card_fraud_processed.npz` 等4个 | ~68.6 MB | 信用卡欺诈/心脏病/糖尿病/电信流失 |
| 04_Time_Series_Framework | `data/air_passengers_processed.npz` 等5个 | ~3.4 MB | 航客/出生率/气候/洗发水/股票时序 |
| 10_Tourism_ABSA | `data/processed/bert_emb_*.npz` 6个 | ~406 MB | BERT嵌入向量 |

## 六、D:\datasets目录结构

```
D:\datasets\
├── tabular/
│   ├── Credit_Card_Default/
│   └── Online_Shoppers_Intention/
├── timeseries/
│   ├── Beijing_PM25/
│   ├── Rain_Australia/
│   └── NYC_Taxi_Trip/ (空，下载失败)
├── tourism/
│   ├── ALANA_Airbnb/
│   ├── Hotel_Booking_Demand/
│   ├── UCI_Bike_Sharing/
│   └── Yelp_POI_Recommend/
├── agriculture/
│   ├── crop_recommendation.csv (损坏)
│   ├── diabetes.csv (Pima)
│   ├── soybean_large.csv
│   └── yield.csv
├── nlp/
│   └── SemEval-2014_ABSA/
├── image/
│   ├── EuroSAT/
│   ├── IP102/
│   ├── NewPlantDiseases/
│   └── PlantVillage/
├── network_security/
│   ├── NSL-KDD/
│   └── UNSW-NB15/
├── education/
│   └── OULAD/
├── energy/
│   ├── Household_Electric_Power/
│   └── PJM_Energy_Consumption/
├── digital_twin/
│   └── (空，待下载N-CMAPSS/CMAPSS)
├── misc/
│   └── LUCAS_Synthetic/
├── COVID/
├── Dataset-Slope-LiDAR-Embankment-SLidE/
├── GTPBD/
├── NlpStudy/
├── tourism/ (旧目录，部分数据已迁移)
├── geography/
├── usgs旧金山湾区滑坡数据集/
├── CropAndWeed/
│   ├── CropAndWeed/ (原始数据+bboxes)
│   ├── cropandweed-dataset/ (CNW工具包)
│   └── CropAndWeed_cropped/ (裁剪后按类别)
├── DeepWeeds/
│   ├── DeepWeeds/ (原始图片+labels.csv)
│   ├── DeepWeeds_complete/ (按类别完整)
│   ├── DeepWeeds_hf/ (HuggingFace版)
│   ├── DeepWeeds_organized/ (按类别组织)
│   └── DeepWeeds_zenodo/ (Zenodo版images.zip)
└── WeedSense/ (train/val/test+cache)
```

## 七、注意事项

1. **论文代码中的数据路径**：移动后，各方向代码中的DATA_ROOT路径需要更新指向 `D:\datasets\` 对应子目录
2. **可释放空间**：`NewPlantDiseases/new-plant-diseases-dataset.zip` (2.76GB) 已解压可删除
3. **损坏数据**：农业数据目录下3个CSV文件内容为404页面，需重新下载
4. **Kaggle Competition限制**：NYC Taxi、IEEE Fraud、M5 Forecasting需先在Kaggle网页accept competition rules后才能API下载
5. **NASA数据限制**：N-CMAPSS和CMAPSS需在NASA网站注册并同意数据使用条款后才能下载

---

*本文件由系统自动生成，手动维护时请保持格式一致*
