#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[AI-1] MainPower_Prediction_inference_260207.py
건물 에너지 데이터를 활용한 LightGBM 기반 전력 예측 모델 - 추론용
"""

import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import plotly.graph_objects as go

# 현재 스크립트가 있는 디렉토리를 Python 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import Utility_260207 as Util
import Data_PreProcessing_260207 as DP

# ============================================================
# 설정 (Configuration)
# ============================================================
# 데이터 소스 설정 (현재: CSV 파일, 추후: SQL query 등으로 변경 가능)
file_path = os.path.join(script_dir, 'data_colec_h_202509091411_B0019.csv')
devID, device_name = 2001, '학교MAIN'
tag_dict = {30001: '현재 출력'}
start_date = pd.to_datetime('2025-03-24 17:45:00')
end_date   = pd.to_datetime('2025-09-09 14:05:00')

# 모델 파일 경로
model_file_path = os.path.join(script_dir, f"AI1-MainPower_Prediction_train_260207_{devID}_{device_name}.txt")

# ============================================================
# 1. Data Loading
# ============================================================
# --- CSV 파일에서 로드 ---
df_buildID = pd.read_csv(file_path)
df_buildID['colec_dt'] = pd.to_datetime(df_buildID['colec_dt']).dt.floor('min')

# --- 추후 SQL query로 변경 시, 아래와 같이 대체 가능 ---
# import sqlalchemy
# engine = sqlalchemy.create_engine('...')
# df_buildID = pd.read_sql('SELECT * FROM ...', engine)

print(f"timestamp(first~last): {df_buildID['colec_dt'].iloc[0]} ~ {df_buildID['colec_dt'].iloc[-1]}")
print(f"timestamp(min~max):    {df_buildID['colec_dt'].min()} ~ {df_buildID['colec_dt'].max()}")

df_raw_all = df_buildID[(df_buildID['colec_dt'] >= start_date) & (df_buildID['colec_dt'] <= end_date)]

# ============================================================
# 2. devID, tagCD extraction
# ============================================================
Util.print_tagCD(df_raw_all, devID)
tag_dict = {key: f"{value}@{device_name}" for key, value in tag_dict.items()}
tags = {key: Util.select_devID_tagCD(df_raw_all, devID, tagCD=key) for key in tag_dict.keys()}

tag_data = [tags[key] for key in tag_dict.keys()]
tag_names = [tag_dict[key] for key in tag_dict.keys()]

# ============================================================
# 3. Data Preprocessing
# ============================================================
df_raw = pd.DataFrame(data=tag_data[0]['colec_val'].values, index=tag_data[0]['colec_dt'], columns=['value'])
print(f"timestamp: {df_raw.index[0]}, {df_raw.index[-1]}")

X_df, y_df, nan_counts_df, missing_ratio = DP.preprocess(df_raw, points=4, freqInterval='15min')
print(f"nan_counts max= {nan_counts_df.max()}, missing_ratio= {missing_ratio}")

# 학습 시와 동일한 분할 (test 구간에 대해 추론)
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=42, shuffle=False)

# ============================================================
# 4. Load Model & Inference
# ============================================================
model_infer = lgb.Booster(model_file=model_file_path)
print(f'{model_file_path= }')

y_pred_a = model_infer.predict(X_test)

# ============================================================
# 5. Evaluation & Visualization
# ============================================================
# --- 평가 결과 출력 (추후 SQL DB 기록으로 변경 가능) ---
rmse_loaded = root_mean_squared_error(y_test, y_pred_a)
print(f'RMSE (Loaded Model): {rmse_loaded= :.2f}')

# --- 추후 SQL DB에 결과 기록 시, 아래와 같이 대체 가능 ---
# result_df = pd.DataFrame({'timestamp': y_test.index, 'actual': y_test.values, 'predicted': y_pred_a})
# result_df.to_sql('prediction_results', engine, if_exists='append', index=False)

# RMSE 평가
p_ = 4
rmse_over_time = np.sqrt((y_test - y_pred_a)**2)
daily_rmse = rmse_over_time.resample('D').mean()
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
