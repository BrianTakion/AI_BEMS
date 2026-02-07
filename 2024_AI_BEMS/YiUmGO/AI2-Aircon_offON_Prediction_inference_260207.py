#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[AI-2] Aircon_offON_Prediction_inference_260207.py
건물 에어컨 ON/OFF 이진분류 예측 모델 (LightGBM) - 추론용
"""

import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, accuracy_score
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

# 현재 스크립트가 있는 디렉토리를 Python 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import Utility_260207 as Util
import Aircon_PreProcessing_260207 as DP

# ============================================================
# 설정 (Configuration)
# ============================================================
# 데이터 소스 설정 (현재: CSV 파일, 추후: SQL query 등으로 변경 가능)
file_path = os.path.join(script_dir, 'data_colec_h_202509091411_B0019.csv')
devID, device_name = 6095, '교직원휴게실남'
tag_dict = {100001: '운전선택탭', 100002: '냉난방모드', 100003: '현재온도', 100004: '희망온도'}
start_date = pd.to_datetime('2025-03-24 17:45:00')
end_date   = pd.to_datetime('2025-09-09 14:05:00')

# 모델 파일 경로
model_file_path = os.path.join(script_dir, f"AI2-Aircon_offON_Prediction_train_260207_{devID}_{device_name}.txt")

# ============================================================
# 1. Data Loading
# ============================================================
df_buildID = pd.read_csv(file_path)
df_buildID['colec_dt'] = pd.to_datetime(df_buildID['colec_dt']).dt.floor('min')
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
# 3. Feature Engineering - 환경 온도 및 offON 보정
# ============================================================
df_offON_raw = pd.DataFrame(data=tag_data[0]['colec_val'].values, index=tag_data[0]['colec_dt'], columns=['value'])
df_coolHeat_raw = pd.DataFrame(data=tag_data[1]['colec_val'].values, index=tag_data[1]['colec_dt'], columns=['value'])
df_tCur_raw = pd.DataFrame(data=tag_data[2]['colec_val'].values, index=tag_data[2]['colec_dt'], columns=['value'])
df_tSet_raw = pd.DataFrame(data=tag_data[3]['colec_val'].values, index=tag_data[3]['colec_dt'], columns=['value'])

# 음수 보정 --> 음수는 결측치로 처리
df_offON_raw.loc[df_offON_raw['value'] < 0, 'value'] = np.nan
df_coolHeat_raw.loc[df_coolHeat_raw['value'] < 0, 'value'] = np.nan
df_tCur_raw.loc[df_tCur_raw['value'] < 0, 'value'] = np.nan
df_tSet_raw.loc[df_tSet_raw['value'] < 0, 'value'] = np.nan

# zero가 연속된 구간을 찾아서 리스트로 저장
df_tCur_zero, _, _, _ = DP.preprocess(df_tCur_raw, points=4, freqInterval='15min', only_cleansing=True, fill_method='zero')
zero_sequence_intervals = DP.find_zero_sequence_intervals(df_tCur_zero, min_zero_streak=4*1)

# df_offON 선형 보간
df_offON, df_offON_is_missing, df_offON_nan_counts_df, df_offON_missing_ratio = DP.preprocess(df_offON_raw, points=4, freqInterval='15min', only_cleansing=True, fill_method='time')
df_offON.loc[df_offON['value'] > 0, 'value'] = 1
df_offON['value'] = df_offON['value'].mask((df_offON['value'] == 0) & (df_offON['value'].shift(-1) == 1), 1)
df_offON['value'] = df_offON['value'].mask((df_offON['value'] == 0) & (df_offON['value'].shift(1) == 1), 1)

df_coolHeat, df_coolHeat_is_missing, df_coolHeat_nan_counts_df, df_coolHeat_missing_ratio = DP.preprocess(df_coolHeat_raw, points=4, freqInterval='15min', only_cleansing=True, fill_method='time')
df_tCur, df_tCur_is_missing, df_tCur_nan_counts_df, df_tCur_missing_ratio = DP.preprocess(df_tCur_raw, points=4, freqInterval='15min', only_cleansing=True, fill_method='time')
df_tSet, df_tSet_is_missing, df_tSet_nan_counts_df, df_tSet_missing_ratio = DP.preprocess(df_tSet_raw, points=4, freqInterval='15min', only_cleansing=True, fill_method='time')

# df_offON_virtual 생성
df_offON_virtual = DP.generate_df_offON_virtual(df_tCur, df_coolHeat, df_offON)
df_offON_adj = df_offON.copy()
df_offON_adj['value'] = ((df_offON['value'] > 0) | (df_offON_virtual['value'] > 0)).astype(int)

# Generate tEnv from df_tCur and df_offON_adj
df_tEnv = DP.generate_tEnv_from_df_tCur(df_tCur, df_offON_adj)

# zero_sequence_intervals 구간을 적용하여 보정
for start_time, end_time in zero_sequence_intervals:
    df_offON.loc[start_time:end_time, 'value'] = 0
    df_offON_virtual.loc[start_time:end_time, 'value'] = 0
    df_offON_adj.loc[start_time:end_time, 'value'] = 0
    df_tCur.loc[start_time:end_time, 'value'] = 0
    df_tEnv.loc[start_time:end_time, 'value'] = 0
    df_tSet.loc[start_time:end_time, 'value'] = 0

# ============================================================
# 4. Data Preprocessing
# ============================================================
df_raw = df_offON_adj
print(f"timestamp: {df_raw.index[0]}, {df_raw.index[-1]}")

X_df, y_df, nan_counts_df, missing_ratio = DP.preprocess(df_raw, points=4, freqInterval='15min')
print(f"nan_counts max= {nan_counts_df.max()}, missing_ratio= {missing_ratio}")

# 학습 시와 동일한 분할 (test 구간에 대해 추론)
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=42, shuffle=False)

# ============================================================
# 5. Load Model & Inference
# ============================================================
model_infer = lgb.Booster(model_file=model_file_path)
print(f'{model_file_path= }')

y_pred_a = model_infer.predict(X_test)

# ============================================================
# 6. Evaluation & Visualization
# ============================================================
y_pred_a_binary = np.where(y_pred_a < 0.5, 0, 1)
accuracy_a = accuracy_score(y_test, y_pred_a_binary)
rmse_a = root_mean_squared_error(y_test, y_pred_a)
print(f'Loaded Model: {accuracy_a= :.3f}, {rmse_a= :.3f} with {X_df.shape[1]} features')

# 예측 결과 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Cleaned Value'))
fig.add_trace(go.Scatter(x=y_test.index, y=y_pred_a_binary-0.02, mode='lines', name='Predicted Value'))
fig.update_layout(
    title=f'Aircon offON Prediction, {devID}, {device_name} (Loaded Model)',
    xaxis_title='Date',
    yaxis_title='Values',
    yaxis=dict(range=[0, 2]),
    legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top', orientation='v')
)
fig.show()
