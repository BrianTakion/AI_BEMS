#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[AI-1] MainPower_Prediction_250917.py
건물 에너지 데이터를 활용한 LightGBM 기반 전력 예측 모델
"""

import os, sys
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import datetime
from sklearn.metrics import root_mean_squared_error
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

# 현재 스크립트가 있는 디렉토리를 Python 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import Utility_260207 as Util
import Data_PreProcessing_260207 as DP

# ============================================================
# 1. Data Loading
# ============================================================
parent_dir = os.path.dirname(script_dir)
file_path = os.path.join(parent_dir, 'data_colec_h_202509091411_B0019.csv')
df_buildID = pd.read_csv(file_path)
df_buildID['colec_dt'] = pd.to_datetime(df_buildID['colec_dt']).dt.floor('min')  # 분 이하는 제거
print(f"timestamp(first~last): {df_buildID['colec_dt'].iloc[0]} ~ {df_buildID['colec_dt'].iloc[-1]}")
print(f"timestamp(min~max):    {df_buildID['colec_dt'].min()} ~ {df_buildID['colec_dt'].max()}")

start_date, end_date = pd.to_datetime('2025-03-24 17:45:00'), pd.to_datetime('2025-09-09 14:05:00')
df_raw_all = df_buildID[(df_buildID['colec_dt'] >= start_date) & (df_buildID['colec_dt'] <= end_date)]

# ============================================================
# 2. devID, tagCD extraction
# ============================================================
devID, device_name = 2001, '학교MAIN'

Util.print_tagCD(df_raw_all, devID)
tag_dict = {30001: '현재 출력'}
tag_dict = {key: f"{value}@{device_name}" for key, value in tag_dict.items()}
# dictionary를 이용해 각 태그 데이터를 선택
tags = {key: Util.select_devID_tagCD(df_raw_all, devID, tagCD=key) for key in tag_dict.keys()}

# 그래프를 그리기 위해 필요한 데이터를 리스트로 변환
tag_data = [tags[key] for key in tag_dict.keys()]
tag_names = [tag_dict[key] for key in tag_dict.keys()]
Util.plot_dfL_devID_tagCD(tag_data, tag_names, device_name, createFig=False)

# ============================================================
# 3. Step A. Feature Engineering, LightGBM Modeling
#   - LightGBM은 tree-based 모델로 scaling을 특별히 요구하지 않음
#   - Prediction model 일 때는, 과거 데이터 정보(X)로만 현재 데이터(y)를 Regression
#   - train_test_split() 에서, prediction model에서는 shuffle=False
# ============================================================
df_raw = pd.DataFrame(data=tag_data[0]['colec_val'].values, index=tag_data[0]['colec_dt'], columns=['value'])
print(f"timestamp: {df_raw.index[0]}, {df_raw.index[-1]}")

X_df, y_df, nan_counts_df, missing_ratio = DP.preprocess(df_raw, points=4, freqInterval='15min')
print(f"nan_counts max= {nan_counts_df.max()}, missing_ratio= {missing_ratio}")

# 학습 및 테스트 데이터 분할
# 시계열 데이터는 시간 순으로 되어 있어야 하고, shuffle=False로 순방향 데이터검증 보장
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=42, shuffle=False)

# LightGBM 데이터 세트 생성
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 모델 하이퍼파라미터 설정
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 50,
    'verbose': -1
}

# 모델 학습
model_a = lgb.train(params,
                    train_data,
                    valid_sets=[valid_data],
                    num_boost_round=1000,
                    valid_names=['validation'],
                    callbacks=[lgb.early_stopping(stopping_rounds=30)])

# 예측 수행
y_pred_a = model_a.predict(X_test, num_iteration=model_a.best_iteration)

# RMSE 출력
rmse_a = root_mean_squared_error(y_test, y_pred_a)
print(f'Experiment A: {rmse_a= :.2f} with {X_df.shape[1]} features')

# 예측 결과 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['value'], mode='lines', name='Actual Value'))
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Cleaned Value'))
fig.add_trace(go.Scatter(x=y_test.index, y=y_pred_a, mode='lines', name='Predicted Value'))
fig.update_layout(title='Actual vs Predicted Values',
                  xaxis_title='Date',
                  yaxis_title='Values',
                  legend_title='Legend',
                  legend=dict(x=0, y=1, xanchor='left', yanchor='top', orientation='h'))
fig.show()

# Feature Importance Visualization
feature_importances = model_a.feature_importance()
feature_names = X_df.columns

importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 12))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance Visualization')
plt.gca().invert_yaxis()
plt.show()

# ============================================================
# 4. Save the model & Run Inference
# ============================================================
script_name = os.path.splitext(os.path.basename(__file__))[0]
file_path_AI_model = os.path.join(script_dir, f"{script_name}_{devID}_{device_name}.txt")
model_a.save_model(file_path_AI_model)

# 모델 로드
model_infer = lgb.Booster(model_file=file_path_AI_model)

# 예측 수행 (로드된 모델 사용)
y_pred_a = model_infer.predict(X_test)

# 모델 평가
rmse_loaded = root_mean_squared_error(y_test, y_pred_a)
print(f'RMSE (Loaded Model): {rmse_loaded= :.2f}')
print(f'{file_path_AI_model= }')

# RMSE 평가
p_ = 4
rmse_over_time = np.sqrt((y_test - y_pred_a)**2)
daily_rmse = rmse_over_time.resample('1d').mean()
daily_rmse = rmse_over_time.rolling(window=p_ * 24).mean()

# 예측 결과 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Cleaned Value'))
fig.add_trace(go.Scatter(x=y_test.index, y=y_pred_a, mode='lines', name='Predicted Value'))
fig.update_layout(
    title='Time Series Anomaly Detection with Daily RMSE (Loaded Model)',
    xaxis_title='Date',
    yaxis_title='Values',
    yaxis=dict(range=[-10, 400]),
    legend_title='Legend',
    legend=dict(x=0.5, y=0.9, xanchor='center', yanchor='bottom', orientation='h')
)
fig.show()