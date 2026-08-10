#!/usr/bin/env python3
"""
Universal experiment runner for PhysXGBoost domain feature analysis.
Runs 4 tree models × 5 seeds on Raw vs Domain features.
Computes: per-seed results, Wilcoxon tests, 95% CI, Cohen's d.

Usage: python run_experiments.py --direction 51_GasTurbine
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ─── Models ───
def get_models(task: str):
    if task == 'regression':
        return {
            'XGB': lambda seed: __import__('xgboost').XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=seed, n_jobs=-1, verbosity=0
            ),
            'LGB': lambda seed: __import__('lightgbm').LGBMRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=seed, n_jobs=-1, verbose=-1
            ),
            'Cat': lambda seed: __import__('catboost').CatBoostRegressor(
                iterations=300, depth=6, learning_rate=0.1,
                random_seed=seed, verbose=0
            ),
            'RF': lambda seed: RandomForestRegressor(
                n_estimators=300, max_depth=12,
                random_state=seed, n_jobs=-1
            ),
        }
    else:
        return {
            'XGB': lambda seed: __import__('xgboost').XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=seed, n_jobs=-1, eval_metric='logloss', verbosity=0
            ),
            'LGB': lambda seed: __import__('lightgbm').LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=seed, n_jobs=-1, verbose=-1
            ),
            'Cat': lambda seed: __import__('catboost').CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.1,
                random_seed=seed, verbose=0
            ),
            'RF': lambda seed: RandomForestClassifier(
                n_estimators=300, max_depth=12,
                random_state=seed, n_jobs=-1
            ),
        }

SEEDS = [42, 123, 456, 789, 2024]

# ─── Direction configs ───
DIRECTIONS = {
    '51_GasTurbine': {
        'data_file': 'gasturbine.csv',
        'target': 'NOX',
        'task': 'regression',
        'drop_cols': ['year'],
        'domain_fn': 'gas_turbine_domain',
    },
    '52_CCPP': {
        'data_file': 'ccpp.csv',
        'target': 'PE',
        'task': 'regression',
        'drop_cols': [],
        'domain_fn': 'ccpp_domain',
    },
    '53_BikeSharing': {
        'data_file': 'bikesharing.csv',
        'target': 'cnt',
        'task': 'regression',
        'drop_cols': ['dteday', 'casual', 'registered'],
        'domain_fn': 'bike_sharing_domain',
    },
    '54_NewsPopularity': {
        'data_file': 'news_pop.csv',
        'target': 'shares',
        'task': 'regression',
        'drop_cols': [],
        'domain_fn': 'news_popularity_domain',
    },
    '60_StudentPerf': {
        'data_file': 'student.csv',
        'target': 'G3',
        'task': 'regression',
        'drop_cols': ['G1', 'G2'],
        'domain_fn': 'student_performance_domain',
    },
    '55_CalHousing': {
        'data_file': 'california_housing.csv',
        'target': 'MedHouseVal',
        'task': 'regression',
        'drop_cols': [],
        'domain_fn': 'cal_housing_domain',
    },
    '58_CDNOW': {
        'data_file': 'cdnow.csv',
        'target': 'target',
        'task': 'classification',
        'drop_cols': [],
        'domain_fn': 'health_indicators_domain',
    },
    '61_DryBean': {
        'data_file': 'drybean.csv',
        'target': 'class',
        'task': 'classification',
        'drop_cols': ['Type'],
        'domain_fn': 'tool_wear_domain',
    },
    '47_OnlineShoppers': {
        'data_file': 'online_shoppers.csv',
        'target': 'y',
        'task': 'classification',
        'drop_cols': [],
        'domain_fn': 'online_shoppers_domain',
    },
    '65_HR': {
        'data_file': 'hr_data.csv',
        'target': 'Attrition',
        'task': 'classification',
        'drop_cols': ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'],
        'domain_fn': 'hr_attrition_domain',
    },
    '59_NYCProperty': {
        'data_file': 'nyc_property_sales.csv',
        'target': 'SALE PRICE',
        'task': 'regression',
        'drop_cols': ['Unnamed: 0', 'ADDRESS', 'APARTMENT NUMBER', 'SALE DATE',
                       'EASE-MENT', 'NEIGHBORHOOD', 'BUILDING CLASS AT PRESENT',
                       'BUILDING CLASS AT TIME OF SALE', 'TAX CLASS AT PRESENT',
                       'TAX CLASS AT TIME OF SALE'],
        'domain_fn': 'nyc_property_domain',
        'data_clean': 'nyc_property_clean',
    },
}

# ─── Domain feature functions ───
def gas_turbine_domain(df):
    """Thermodynamic domain features for gas turbine NOx prediction."""
    d = pd.DataFrame(index=df.index)
    if 'AT' in df.columns and 'AP' in df.columns:
        d['air_density'] = df['AP'] / (287.05 * (df['AT'] + 273.15))
    if 'TIT' in df.columns and 'TAT' in df.columns:
        d['temp_diff_TIT_TAT'] = df['TIT'] - df['TAT']
        d['temp_ratio_TIT_TAT'] = df['TIT'] / df['TAT']
    if 'GTEP' in df.columns and 'AFDP' in df.columns:
        d['pressure_ratio'] = df['GTEP'] / df['AFDP']
    if 'TEY' in df.columns and 'CDP' in df.columns:
        d['power_per_pressure'] = df['TEY'] / df['CDP']
    if 'TIT' in df.columns and 'AT' in df.columns:
        d['thermal_efficiency_proxy'] = 1 - (df['AT'] + 273.15) / (df['TIT'] + 273.15)
    if 'CO' in df.columns and 'NOX' in df.columns:
        pass  # NOX is target, don't use CO-derived features that leak
    if 'AFDP' in df.columns and 'GTEP' in df.columns and 'TIT' in df.columns:
        d['combustion_efficiency'] = (df['GTEP'] * df['TIT']) / (df['AFDP'] * 1000)
    return d

def ccpp_domain(df):
    """Thermodynamic domain features for CCPP power prediction."""
    d = pd.DataFrame(index=df.index)
    if 'AT' in df.columns and 'V' in df.columns:
        d['temp_voltage_interaction'] = df['AT'] * df['V']
    if 'AP' in df.columns and 'RH' in df.columns:
        d['humidity_pressure'] = df['AP'] * df['RH'] / 100
    if 'AT' in df.columns and 'AP' in df.columns:
        d['air_density'] = df['AP'] / (287.05 * (df['AT'] + 273.15))
    if 'AT' in df.columns and 'RH' in df.columns:
        # Wet bulb temperature approximation
        d['wet_bulb_proxy'] = df['AT'] * np.arctan(0.151977 * np.sqrt(df['RH'] + 8.313659)) + df['AT']
    if all(c in df.columns for c in ['AT', 'V', 'AP', 'RH']):
        d['cycle_efficiency'] = 1 - (df['AT'] + 273.15) / (df['AT'] + 273.15 + 500)  # Carnot-like
    return d

def bike_sharing_domain(df):
    """Transportation domain features for bike sharing demand."""
    d = pd.DataFrame(index=df.index)
    if 'temp' in df.columns and 'atemp' in df.columns:
        d['temp_discomfort'] = (df['atemp'] - df['temp']).abs()
        d['temp_squared'] = df['temp'] ** 2
    if 'hum' in df.columns and 'windspeed' in df.columns:
        d['weather_discomfort'] = df['hum'] * (1 - df['windspeed'])
    if 'hr' in df.columns:
        d['is_rush_hour'] = ((df['hr'] >= 7) & (df['hr'] <= 9) | (df['hr'] >= 17) & (df['hr'] <= 19)).astype(int)
        d['is_night'] = ((df['hr'] >= 22) | (df['hr'] <= 5)).astype(int)
        d['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
        d['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
    if 'weekday' in df.columns:
        d['is_weekend'] = (df['weekday'] >= 5).astype(int)
    if 'season' in df.columns:
        d['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
        d['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)
    if 'temp' in df.columns and 'hum' in df.columns:
        d['heat_index_proxy'] = df['temp'] * df['hum']
    return d

def news_popularity_domain(df):
    """Content engagement domain features for news popularity prediction."""
    d = pd.DataFrame(index=df.index)
    # LDA entropy
    lda_cols = [c for c in df.columns if c.startswith('LDA_')]
    if lda_cols:
        lda_vals = df[lda_cols].clip(lower=1e-10).values
        lda_entropy = -np.sum(lda_vals * np.log(lda_vals), axis=1)
        d['LDA_entropy'] = lda_entropy
    # Keyword diversity
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    if kw_cols:
        d['keyword_diversity'] = df[kw_cols].std(axis=1)
    # Channel popularity prior
    channel_cols = [c for c in df.columns if c.startswith('data_channel_is_')]
    if channel_cols:
        d['channel_count'] = df[channel_cols].sum(axis=1)
    # Sentiment extremity
    if 'global_sentiment_polarity' in df.columns:
        d['sentiment_extremity'] = df['global_sentiment_polarity'].abs()
    if 'title_sentiment_polarity' in df.columns:
        d['title_sentiment_strength'] = df['title_sentiment_polarity'].abs()
    # Weekend boost
    if 'is_weekend' in df.columns:
        d['weekend_boost'] = df['is_weekend']
    # Title length optimal (quadratic)
    if 'n_tokens_title' in df.columns:
        d['title_length_optimal'] = (df['n_tokens_title'] - 10).abs()  # optimal ~10 words
    # Content interaction
    if 'num_hrefs' in df.columns and 'num_imgs' in df.columns:
        d['media_richness'] = df['num_hrefs'] + df['num_imgs'] * 2 + df.get('num_videos', pd.Series(0, index=df.index)) * 3
    return d

def student_performance_domain(df):
    """Educational domain features for student performance prediction."""
    d = pd.DataFrame(index=df.index)
    # Study efficiency
    if 'studytime' in df.columns and 'failures' in df.columns:
        d['study_efficiency'] = df['studytime'] / (df['failures'] + 1)
    if 'absences' in df.columns and 'studytime' in df.columns:
        d['attendance_study_ratio'] = (1 / (df['absences'] + 1)) * df['studytime']
    # Family support composite
    if 'Medu' in df.columns and 'Fedu' in df.columns:
        d['parental_education_sum'] = df['Medu'] + df['Fedu']
        d['parental_education_diff'] = (df['Medu'] - df['Fedu']).abs()
    # Social drinking score
    if 'Dalc' in df.columns and 'Walc' in df.columns:
        d['total_alcohol'] = df['Dalc'] + df['Walc']
        d['weekend_drink_ratio'] = df['Walc'] / (df['Dalc'] + df['Walc'] + 1e-10)
    # Health lifestyle composite
    if 'health' in df.columns and 'absences' in df.columns:
        d['health_attendance'] = df['health'] * (1 / (df['absences'] + 1))
    # Goout vs studytime balance
    if 'goout' in df.columns and 'studytime' in df.columns:
        d['social_study_balance'] = df['studytime'] - df['goout']
    # Family relationship quality
    if 'famrel' in df.columns and 'freetime' in df.columns:
        d['social_wellbeing'] = df['famrel'] + df['freetime']
    return d

def cal_housing_domain(df):
    """Real estate domain features for California housing price prediction."""
    d = pd.DataFrame(index=df.index)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        # Distance to major cities (approximate)
        d['dist_to_la'] = np.sqrt((df['Latitude'] - 34.05)**2 + (df['Longitude'] + 118.24)**2)
        d['dist_to_sf'] = np.sqrt((df['Latitude'] - 37.77)**2 + (df['Longitude'] + 122.42)**2)
        d['dist_to_coast'] = np.minimum(d['dist_to_la'], d['dist_to_sf'])
        # Inland indicator
        d['is_inland'] = (df['Longitude'] > -120).astype(int)
    if 'AveRooms' in df.columns and 'AveBedrms' in df.columns:
        d['room_bedroom_ratio'] = df['AveRooms'] / (df['AveBedrms'] + 1e-10)
        d['living_space_proxy'] = df['AveRooms'] - df['AveBedrms']
    if 'HouseAge' in df.columns:
        d['age_squared'] = df['HouseAge'] ** 2
        d['is_new'] = (df['HouseAge'] < 10).astype(int)
        d['is_old'] = (df['HouseAge'] > 40).astype(int)
    if 'Population' in df.columns and 'AveOccup' in df.columns:
        d['population_density'] = df['Population'] * df['AveOccup']
    if 'MedInc' in df.columns:
        d['income_squared'] = df['MedInc'] ** 2
        d['is_high_income'] = (df['MedInc'] > 8).astype(int)
    if 'MedInc' in df.columns and 'AveRooms' in df.columns:
        d['income_per_room'] = df['MedInc'] / (df['AveRooms'] + 1e-10)
    return d

def health_indicators_domain(df):
    """Health domain features for diabetes risk prediction (BRFSS dataset)."""
    d = pd.DataFrame(index=df.index)
    # Risk composite scores
    if 'HighBP' in df.columns and 'HighChol' in df.columns:
        d['cardiovascular_risk'] = df['HighBP'] + df['HighChol']
    if 'BMI' in df.columns:
        d['bmi_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
        d['is_obese'] = (df['BMI'] >= 30).astype(int)
        d['bmi_squared'] = df['BMI'] ** 2
    if 'Smoker' in df.columns and 'HvyAlcoholConsump' in df.columns:
        d['lifestyle_risk'] = df['Smoker'] + df['HvyAlcoholConsump']
    if 'Stroke' in df.columns and 'HeartDiseaseorAttack' in df.columns:
        d['vascular_history'] = df['Stroke'] + df['HeartDiseaseorAttack']
    if 'PhysHlth' in df.columns and 'MentHlth' in df.columns:
        d['total_health_issues'] = df['PhysHlth'] + df['MentHlth']
        d['physical_mental_ratio'] = df['PhysHlth'] / (df['MentHlth'] + 1e-10)
    if 'GenHlth' in df.columns and 'PhysHlth' in df.columns:
        d['health_disparity'] = df['GenHlth'] * df['PhysHlth'] / 30
    if 'Age' in df.columns and 'BMI' in df.columns:
        d['age_bmi_interaction'] = df['Age'] * df['BMI'] / 100
    if 'Education' in df.columns and 'Income' in df.columns:
        d['ses_score'] = df['Education'] + df['Income']
    if 'PhysActivity' in df.columns and 'Fruits' in df.columns and 'Veggies' in df.columns:
        d['healthy_lifestyle'] = df['PhysActivity'] + df['Fruits'] + df['Veggies']
    if 'DiffWalk' in df.columns:
        d['mobility_risk'] = df['DiffWalk']
    return d

def tool_wear_domain(df):
    """Manufacturing domain features for CNC tool wear prediction."""
    d = pd.DataFrame(index=df.index)
    if 'Air temperature' in df.columns and 'Process temperature' in df.columns:
        d['temp_difference'] = df['Process temperature'] - df['Air temperature']
        d['temp_ratio'] = df['Process temperature'] / df['Air temperature']
    if 'Rotational speed' in df.columns and 'Torque' in df.columns:
        # Power = Torque * angular velocity
        d['cutting_power'] = df['Torque'] * df['Rotational speed'] * 2 * np.pi / 60
        d['power_per_speed'] = df['Torque'] / (df['Rotational speed'] + 1e-10)
    if 'Tool wear' in df.columns:
        d['wear_squared'] = df['Tool wear'] ** 2
        d['is_high_wear'] = (df['Tool wear'] > 150).astype(int)
    if 'Torque' in df.columns and 'Tool wear' in df.columns:
        d['torque_wear_interaction'] = df['Torque'] * df['Tool wear'] / 1000
    if 'Rotational speed' in df.columns and 'Tool wear' in df.columns:
        d['speed_wear_ratio'] = df['Rotational speed'] / (df['Tool wear'] + 1)
    if all(c in df.columns for c in ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']):
        d['thermal_load'] = (df['Process temperature'] - df['Air temperature']) * df['Rotational speed'] / 1000
        d['mechanical_load'] = df['Torque'] * (df['Tool wear'] + 1) / 100
    return d

def online_shoppers_domain(df):
    """E-commerce domain features for online shopping intention prediction."""
    d = pd.DataFrame(index=df.index)
    # Session engagement intensity
    page_cols = ['Administrative', 'Informational', 'ProductRelated']
    dur_cols = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']
    existing_pages = [c for c in page_cols if c in df.columns]
    existing_durs = [c for c in dur_cols if c in df.columns]
    if existing_pages:
        d['total_pages'] = df[existing_pages].sum(axis=1)
    if existing_durs:
        d['total_duration'] = df[existing_durs].sum(axis=1)
    if 'total_pages' in d.columns and 'total_duration' in d.columns:
        d['avg_duration_per_page'] = d['total_duration'] / (d['total_pages'] + 1e-10)
    # Bounce-exit composite
    if 'BounceRates' in df.columns and 'ExitRates' in df.columns:
        d['bounce_exit_composite'] = (df['BounceRates'] + df['ExitRates']) / 2
        d['bounce_exit_ratio'] = df['BounceRates'] / (df['ExitRates'] + 1e-10)
    # Page value efficiency
    if 'PageValues' in df.columns and 'ProductRelated' in df.columns:
        d['page_value_efficiency'] = df['PageValues'] / (df['ProductRelated'] + 1)
    # Special day proximity interaction
    if 'SpecialDay' in df.columns and 'PageValues' in df.columns:
        d['special_day_page_value'] = df['SpecialDay'] * df['PageValues']
    # Weekend interaction
    if 'Weekend' in df.columns and 'PageValues' in df.columns:
        d['weekend_page_interaction'] = df['Weekend'].astype(int) * df['PageValues']
    # Product engagement depth
    if 'ProductRelated' in df.columns and 'ProductRelated_Duration' in df.columns:
        d['product_depth'] = df['ProductRelated_Duration'] / (df['ProductRelated'] + 1e-10)
    # Administrative-to-informational ratio
    if 'Administrative' in df.columns and 'Informational' in df.columns:
        d['admin_info_ratio'] = df['Administrative'] / (df['Informational'] + 1)
    return d

def hr_attrition_domain(df):
    """HR domain features for employee attrition prediction."""
    d = pd.DataFrame(index=df.index)
    # Income per year of service
    if 'MonthlyIncome' in df.columns and 'YearsAtCompany' in df.columns:
        d['income_per_year'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    # Career progression rate
    if 'JobLevel' in df.columns and 'YearsAtCompany' in df.columns:
        d['career_progression_rate'] = df['JobLevel'] / (df['YearsAtCompany'] + 1)
    # Satisfaction composite
    sat_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']
    existing_sat = [c for c in sat_cols if c in df.columns]
    if existing_sat:
        d['satisfaction_composite'] = df[existing_sat].sum(axis=1)
        d['satisfaction_mean'] = df[existing_sat].mean(axis=1)
    # Work-life stability
    if 'WorkLifeBalance' in df.columns and 'OverTime' in df.columns:
        overtime_enc = (df['OverTime'] == 'Yes').astype(int)
        d['work_life_stability'] = df['WorkLifeBalance'] * (1 - overtime_enc * 0.5)
    # Tenure stability
    if 'YearsAtCompany' in df.columns and 'Age' in df.columns:
        d['tenure_ratio'] = df['YearsAtCompany'] / (df['Age'] - 17)
    # Training investment
    if 'TrainingTimesLastYear' in df.columns and 'JobLevel' in df.columns:
        d['training_investment'] = df['TrainingTimesLastYear'] * df['JobLevel']
    # Compensation growth
    if 'PercentSalaryHike' in df.columns and 'YearsAtCompany' in df.columns:
        d['compensation_growth'] = df['PercentSalaryHike'] * df['YearsAtCompany']
    # Manager relationship
    if 'YearsWithCurrManager' in df.columns and 'YearsAtCompany' in df.columns:
        d['manager_tenure_ratio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    # Role stagnation
    if 'YearsInCurrentRole' in df.columns and 'YearsAtCompany' in df.columns:
        d['role_stagnation'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    # Promotion gap
    if 'YearsSinceLastPromotion' in df.columns and 'YearsAtCompany' in df.columns:
        d['promotion_gap_ratio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    # Distance-income interaction
    if 'DistanceFromHome' in df.columns and 'MonthlyIncome' in df.columns:
        d['commute_burden'] = df['DistanceFromHome'] / (df['MonthlyIncome'] / 1000 + 1)
    # Stock-tenure interaction
    if 'StockOptionLevel' in df.columns and 'YearsAtCompany' in df.columns:
        d['equity_retention'] = df['StockOptionLevel'] * df['YearsAtCompany']
    return d

def nyc_property_domain(df):
    """Real estate domain features for NYC property price prediction."""
    d = pd.DataFrame(index=df.index)
    # Building age (relative to 2024)
    if 'YEAR BUILT' in df.columns:
        year_built = pd.to_numeric(df['YEAR BUILT'], errors='coerce')
        d['building_age'] = 2024 - year_built
        d['building_age_squared'] = d['building_age'] ** 2
        d['is_pre_war'] = (year_built < 1945).astype(int)
        d['is_new_construction'] = (year_built > 2010).astype(int)
    # Unit density
    if 'GROSS SQUARE FEET' in df.columns:
        gsf = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
        if 'TOTAL UNITS' in df.columns:
            total_units = pd.to_numeric(df['TOTAL UNITS'], errors='coerce')
            d['unit_density'] = total_units / (gsf + 1e-10)
        if 'RESIDENTIAL UNITS' in df.columns and 'COMMERCIAL UNITS' in df.columns:
            res_units = pd.to_numeric(df['RESIDENTIAL UNITS'], errors='coerce')
            com_units = pd.to_numeric(df['COMMERCIAL UNITS'], errors='coerce')
            d['commercial_ratio'] = com_units / (total_units + 1e-10)
            d['residential_ratio'] = res_units / (total_units + 1e-10)
            d['is_mixed_use'] = ((res_units > 0) & (com_units > 0)).astype(int)
    # Land-to-gross ratio
    if 'LAND SQUARE FEET' in df.columns and 'GROSS SQUARE FEET' in df.columns:
        lsf = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
        gsf = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
        d['land_gross_ratio'] = lsf / (gsf + 1e-10)
    # Borough encoding (already numeric 1-5)
    if 'BOROUGH' in df.columns:
        borough = pd.to_numeric(df['BOROUGH'], errors='coerce')
        d['is_manhattan'] = (borough == 1).astype(int)
        d['is_bronx'] = (borough == 2).astype(int)
        d['is_brooklyn'] = (borough == 3).astype(int)
        d['is_queens'] = (borough == 4).astype(int)
        d['is_staten'] = (borough == 5).astype(int)
    # Zip code density proxy
    if 'ZIP CODE' in df.columns and 'TOTAL UNITS' in df.columns:
        d['zip_unit_density'] = pd.to_numeric(df['TOTAL UNITS'], errors='coerce')
    # Block-lot interaction
    if 'BLOCK' in df.columns and 'LOT' in df.columns:
        block = pd.to_numeric(df['BLOCK'], errors='coerce')
        lot = pd.to_numeric(df['LOT'], errors='coerce')
        d['block_lot_ratio'] = lot / (block + 1e-10)
    return d

def nyc_property_clean(df):
    """Clean NYC Property Sales data: convert non-numeric values, remove invalid sales."""
    # Convert SALE PRICE to numeric (handles ' -  ' entries)
    df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')
    # Remove rows with missing or zero sale price (likely transfers, not sales)
    df = df[df['SALE PRICE'] > 100].copy()
    # Remove extreme outliers (top/bottom 1%)
    low = df['SALE PRICE'].quantile(0.01)
    high = df['SALE PRICE'].quantile(0.99)
    df = df[(df['SALE PRICE'] >= low) & (df['SALE PRICE'] <= high)].copy()
    # Convert numeric columns
    for col in ['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS',
                'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT',
                'ZIP CODE', 'BLOCK', 'LOT', 'BOROUGH']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Fill NaN with median for numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    return df

DOMAIN_FUNCTIONS = {
    'gas_turbine_domain': gas_turbine_domain,
    'ccpp_domain': ccpp_domain,
    'bike_sharing_domain': bike_sharing_domain,
    'news_popularity_domain': news_popularity_domain,
    'student_performance_domain': student_performance_domain,
    'cal_housing_domain': cal_housing_domain,
    'health_indicators_domain': health_indicators_domain,
    'tool_wear_domain': tool_wear_domain,
    'online_shoppers_domain': online_shoppers_domain,
    'hr_attrition_domain': hr_attrition_domain,
    'nyc_property_domain': nyc_property_domain,
}

DATA_CLEANERS = {
    'nyc_property_clean': nyc_property_clean,
}

# ─── Experiment runner ───
def run_experiment(direction: str, base_dir: str = 'D:/ResearchPaperPrepare'):
    cfg = DIRECTIONS[direction]
    dir_path = Path(base_dir) / direction
    data_path = dir_path / 'data' / cfg['data_file']
    results_dir = dir_path / 'results'
    results_dir.mkdir(exist_ok=True)
    code_dir = dir_path / 'code'
    code_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Direction: {direction}")
    print(f"Data: {data_path}")
    print(f"Target: {cfg['target']}, Task: {cfg['task']}")
    print(f"{'='*60}")

    # Load data
    df = pd.read_csv(data_path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Apply data-specific cleaning if configured
    if 'data_clean' in cfg and cfg['data_clean'] in DATA_CLEANERS:
        cleaner = DATA_CLEANERS[cfg['data_clean']]
        df = cleaner(df)
        print(f"After cleaning: {len(df)} rows, {len(df.columns)} columns")

    # Drop specified columns
    for col in cfg.get('drop_cols', []):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if col != cfg['target']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Separate target
    y = df[cfg['target']]
    X_raw = df.drop(columns=[cfg['target']])

    # Create domain features
    domain_fn = DOMAIN_FUNCTIONS[cfg['domain_fn']]
    X_domain_extra = domain_fn(df)
    X_domain = pd.concat([X_raw, X_domain_extra], axis=1)

    print(f"Raw features: {X_raw.shape[1]}")
    print(f"Domain features (extra): {X_domain_extra.shape[1]}")
    print(f"Domain features (total): {X_domain.shape[1]}")

    # Handle classification target encoding
    if cfg['task'] == 'classification':
        if y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
        if len(np.unique(y)) > 2:
            print(f"Multi-class detected: {len(np.unique(y))} classes. Using macro F1.")

    models = get_models(cfg['task'])
    feature_sets = {'Raw': X_raw, 'Domain': X_domain}

    all_results = {}
    per_seed = {}

    for fs_name, X in feature_sets.items():
        all_results[fs_name] = {}
        per_seed[fs_name] = {}
        for model_name, model_factory in models.items():
            scores = []
            for seed in SEEDS:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed
                )
                model = model_factory(seed)
                model.fit(X_train, y_train)

                if cfg['task'] == 'regression':
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                else:
                    if len(np.unique(y)) == 2:
                        # Use predict_proba for AUC
                        try:
                            y_proba = model.predict_proba(X_test)[:, 1]
                            score = roc_auc_score(y_test, y_proba)
                        except Exception:
                            y_pred = model.predict(X_test)
                            score = roc_auc_score(y_test, y_pred)
                    else:
                        # Multi-class: use macro F1
                        y_pred = model.predict(X_test)
                        score = f1_score(y_test, y_pred, average='macro')

                scores.append(float(score))
                print(f"  {fs_name}/{model_name}/seed{seed}: {score:.6f}")

            per_seed[fs_name][model_name] = dict(zip(SEEDS, scores))
            all_results[fs_name][model_name] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'all_scores': scores,
                'n_seeds': len(scores),
            }

    # Statistical tests
    stat_tests = {}
    for model_name in models:
        raw_scores = np.array(list(per_seed['Raw'][model_name].values()))
        dom_scores = np.array(list(per_seed['Domain'][model_name].values()))
        diff = dom_scores - raw_scores

        # Wilcoxon signed-rank test
        try:
            if np.all(diff == 0):
                w_stat, w_p = 0.0, 1.0
            else:
                w_stat, w_p = stats.wilcoxon(dom_scores, raw_scores)
        except Exception:
            w_stat, w_p = float('nan'), float('nan')

        # Paired t-test
        try:
            t_stat, t_p = stats.ttest_rel(dom_scores, raw_scores)
        except Exception:
            t_stat, t_p = float('nan'), float('nan')

        # 95% CI of difference
        mean_diff = float(np.mean(diff))
        se_diff = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff

        # Cohen's d
        pooled_std = float(np.sqrt((np.std(raw_scores, ddof=1)**2 + np.std(dom_scores, ddof=1)**2) / 2))
        cohen_d = float(mean_diff / pooled_std) if pooled_std > 0 else 0.0

        stat_tests[model_name] = {
            'wilcoxon_statistic': float(w_stat),
            'wilcoxon_p_value': float(w_p),
            'ttest_statistic': float(t_stat),
            'ttest_p_value': float(t_p),
            'mean_diff': mean_diff,
            'se_diff': se_diff,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'cohens_d': cohen_d,
            'all_positive': bool(np.all(diff >= 0)),
            'n_positive': int(np.sum(diff > 0)),
        }

    # Ablation: remove domain features one group at a time
    # (simplified: remove each extra feature individually)
    def _compute_score(model, X_te, y_te, task, n_classes):
        if task == 'regression':
            y_pred = model.predict(X_te)
            return float(r2_score(y_te, y_pred))
        else:
            if n_classes == 2:
                try:
                    y_proba = model.predict_proba(X_te)[:, 1]
                    return float(roc_auc_score(y_te, y_proba))
                except Exception:
                    y_pred = model.predict(X_te)
                    return float(roc_auc_score(y_te, y_pred))
            else:
                y_pred = model.predict(X_te)
                return float(f1_score(y_te, y_pred, average='macro'))

    n_classes = len(np.unique(y)) if cfg['task'] == 'classification' else 0
    ablation = {}
    if X_domain_extra.shape[1] > 0:
        for feat in X_domain_extra.columns:
            X_abl = X_domain.drop(columns=[feat])
            abl_scores = []
            for seed in SEEDS[:3]:  # Use 3 seeds for ablation to save time
                X_tr, X_te, y_tr, y_te = train_test_split(X_abl, y, test_size=0.2, random_state=seed)
                model = models['XGB'](seed)
                model.fit(X_tr, y_tr)
                abl_scores.append(_compute_score(model, X_te, y_te, cfg['task'], n_classes))
            ablation[feat] = {
                'mean': float(np.mean(abl_scores)),
                'std': float(np.std(abl_scores)),
            }

    # Sensitivity analysis: vary n_estimators and max_depth
    sensitivity = {}
    for n_est in [100, 200, 300, 500]:
        for max_d in [4, 6, 8, 10]:
            key = f"n_est={n_est}_depth={max_d}"
            scores = []
            for seed in SEEDS[:3]:
                X_tr, X_te, y_tr, y_te = train_test_split(X_domain, y, test_size=0.2, random_state=seed)
                if cfg['task'] == 'regression':
                    model = __import__('xgboost').XGBRegressor(
                        n_estimators=n_est, max_depth=max_d, learning_rate=0.1,
                        random_state=seed, n_jobs=-1, verbosity=0
                    )
                else:
                    model = __import__('xgboost').XGBClassifier(
                        n_estimators=n_est, max_depth=max_d, learning_rate=0.1,
                        random_state=seed, n_jobs=-1, eval_metric='logloss', verbosity=0
                    )
                model.fit(X_tr, y_tr)
                scores.append(_compute_score(model, X_te, y_te, cfg['task'], n_classes))
            sensitivity[key] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
            }

    # Save comprehensive results
    output = {
        'direction': direction,
        'task': cfg['task'],
        'target': cfg['target'],
        'n_samples': len(df),
        'n_raw_features': X_raw.shape[1],
        'n_domain_features': X_domain_extra.shape[1],
        'n_total_features': X_domain.shape[1],
        'seeds': SEEDS,
        'metric': 'R2' if cfg['task'] == 'regression' else 'AUC',
        'summary': all_results,
        'per_seed': per_seed,
        'statistical_tests': stat_tests,
        'ablation': ablation,
        'sensitivity': sensitivity,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Save summary.json (compatible with existing format)
    summary_compat = {}
    for fs_name in ['Raw', 'Domain']:
        summary_compat[fs_name] = {}
        for model_name in models:
            summary_compat[fs_name][model_name] = {
                output['metric']: all_results[fs_name][model_name]['mean'],
                'std': all_results[fs_name][model_name]['std'],
                'n_seeds': len(SEEDS),
            }
    summary_compat['wilcoxon'] = {
        m: {
            'n_pairs': len(SEEDS),
            'statistic': stat_tests[m]['wilcoxon_statistic'],
            'p_value': stat_tests[m]['wilcoxon_p_value'],
            'all_positive': stat_tests[m]['all_positive'],
        } for m in models
    }

    # Write files
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary_compat, f, indent=2)

    with open(results_dir / 'comprehensive_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    with open(results_dir / 'per_seed_results.json', 'w') as f:
        json.dump(per_seed, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY ({direction})")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Raw Mean':<12} {'Domain Mean':<14} {'Diff':<10} {'Wilcoxon p':<12} {'Cohen d':<10}")
    print("-" * 70)
    for model_name in models:
        raw_m = all_results['Raw'][model_name]['mean']
        dom_m = all_results['Domain'][model_name]['mean']
        diff = dom_m - raw_m
        p_val = stat_tests[model_name]['wilcoxon_p_value']
        d_val = stat_tests[model_name]['cohens_d']
        print(f"{model_name:<10} {raw_m:<12.6f} {dom_m:<14.6f} {diff:<+10.6f} {p_val:<12.6f} {d_val:<10.4f}")

    print(f"\nResults saved to: {results_dir}")
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', type=str, required=True,
                        choices=list(DIRECTIONS.keys()),
                        help='Direction to run experiments for')
    parser.add_argument('--all', action='store_true',
                        help='Run all directions sequentially')
    parser.add_argument('--base_dir', type=str, default='D:/ResearchPaperPrepare')
    args = parser.parse_args()

    if args.all:
        for d in DIRECTIONS:
            try:
                run_experiment(d, args.base_dir)
            except Exception as e:
                print(f"[ERROR] {d}: {e}")
                import traceback; traceback.print_exc()
    else:
        run_experiment(args.direction, args.base_dir)
