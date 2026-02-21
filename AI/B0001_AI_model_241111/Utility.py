import os
import pandas as pd
import numpy as np
from typing import Union, Literal
import plotly.graph_objects as go

def print_tagCD(df, devID):
    df_devID = df[df['dev_id'] == devID]
    print('tagCD list: ', df_devID['tag_cd'].unique())


def select_devID_tagCD(df, devID, tagCD):
    return df[(df['dev_id'] == devID) & (df['tag_cd'] == tagCD)]


#ToDo: diff의 NaN 값 처리 필요
def calc_tagCD_diff(df_devID_tagCD):
    df_diff = df_devID_tagCD.copy()
    df_diff['colec_val'] = pd.to_numeric(df_diff['colec_val'], errors='coerce')
    df_diff['colec_val'] = df_diff['colec_val'].diff().fillna(0)

    # 최대값 인덱스를 찾아 해당 값을 zero로 변경 --- 삭제
    max_index = df_diff['colec_val'].idxmax()
    df_diff.at[max_index, 'colec_val'] = 0
    
    return df_diff


def plot_dfL_devID_tagCD(dfL, legendL, device_name=None, createFig=True, W=None, H=None, normalizing=False):
    if createFig:
        createFig = go.Figure()
        yrange_min, yrange_max = -0.5, 1.5
        for idx, df in enumerate(dfL):
            collect_dt = df['colec_dt']
            cal_min = df['colec_val'].min()
            cal_max = df['colec_val'].max()

            if normalizing:
                normalized_val = df['colec_val'] / cal_max if abs(cal_max) > 0.01 else df['colec_val'] / (cal_max + 0.01)
                createFig.add_trace(go.Scatter(x=collect_dt, y=normalized_val, 
                                               mode='markers', marker=dict(size=4), name=legendL[idx]+f', max={cal_max:.1f}'))
                yrange_min, yrange_max = -0.5, 1.5
            else:
                val = df['colec_val']
                createFig.add_trace(go.Scatter(x=collect_dt, y=val, 
                                               mode='markers', marker=dict(size=4), name=legendL[idx]+f', max={cal_max:.1f}'))
                yrange_min = min(val)*1.1-5. if min(val) < yrange_min else yrange_min
                yrange_max = max(val)*1.1 if max(val) > yrange_max else yrange_max
        
        # 레이아웃 설정
        yrange = [yrange_min, yrange_max]
        layout_params = {
            'title': dict(text=device_name, x=0.5, xanchor='center'),
            'yaxis': dict(range=yrange),
            'xaxis_title': 'Collect Date',
            'yaxis_title': 'Value',
            'legend': dict(x=0.01, y=0.98, traceorder='normal', 
                           bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=1),
        }
        
        # width와 height가 주어지면 추가
        if W is not None:
            layout_params['width'] = W
        if H is not None:
            layout_params['height'] = H
        createFig.update_layout(**layout_params)
        
        createFig.show()
    else:
        pass


def plot_dfList_devID_tagCD(dfList, legendList, device_name=None, createFig=True, W=None, H=None, normalizing=False):
    if createFig:
        createFig = go.Figure()
        yrange_min, yrange_max = -0.5, 1.5
        for idx, df in enumerate(dfList):
            collect_dt = df.index
            cal_min = df['value'].min()
            cal_max = df['value'].max()

            if normalizing:
                normalized_val = df['value'] / cal_max if abs(cal_max) > 0.01 else df['value'] / (cal_max + 0.01)
                createFig.add_trace(go.Scatter(x=collect_dt, y=normalized_val, 
                                               mode='markers', marker=dict(size=4), name=legendList[idx]+f', max={cal_max:.1f}'))
                yrange_min, yrange_max = -0.5, 1.5
            else:
                val = df['value']
                createFig.add_trace(go.Scatter(x=collect_dt, y=val, 
                                               mode='markers', marker=dict(size=4), name=legendList[idx]+f', max={cal_max:.1f}'))
                yrange_min = min(val)*1.1-5. if min(val) < yrange_min else yrange_min
                yrange_max = max(val)*1.1 if max(val) > yrange_max else yrange_max
        
        # 레이아웃 설정
        yrange = [yrange_min, yrange_max]
        layout_params = {
            'title': dict(text=device_name, x=0.5, xanchor='center'),
            'yaxis': dict(range=yrange),
            'xaxis_title': 'Collect Date',
            'yaxis_title': 'Value',
            'legend': dict(x=0.01, y=0.98, traceorder='normal', 
                           bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=1),
        }
        
        # width와 height가 주어지면 추가
        if W is not None:
            layout_params['width'] = W
        if H is not None:
            layout_params['height'] = H
        createFig.update_layout(**layout_params)
        
        createFig.show()
    else:
        pass


def resample_time_index_interpolate_NaN_df(df: pd.DataFrame, freq: str = '15min', interpol_method: str = 'linear') -> pd.DataFrame:
    """
    주어진 데이터프레임을 15분 단위로 리샘플링하고 결측값을 보간하는 함수

    Args:
        df (pd.DataFrame): 원본 데이터프레임
        freq (str): 리샘플링 주기 (기본값: '15min')
        interpol_method (str): 보간 방법 (기본값: 'linear')

    Returns:
        pd.DataFrame: 리샘플링 및 보간된 데이터프레임
    """
    df_resample = df.copy()
    
    # 15분 단위로 라운딩하여 시간 맞추고(15분+=50% time index 흔들림 허용), 중복된 시간 포인트는 제거
    df_resample.index = df_resample.index.round(freq)
    df_resample = df_resample[~df_resample.index.duplicated(keep='first')]
    # 15분 단위로 인덱스 재조정, 누락된 부분은 NaN으로 채워짐
    df_resample = df_resample.resample(freq).asfreq()

    # 누락된 값 보간
    if interpol_method == 'ffill':
        df_resample = df_resample.ffill()  # forward fill 사용
    elif interpol_method == 'bfill':
        df_resample = df_resample.bfill()  # backward fill 사용
    elif interpol_method is not None:
        df_resample = df_resample.interpolate(method=interpol_method)
    else:
        pass

    return df_resample


def plot_df_raw_interpololated_data(df_raw, df_interpol, plotType='simple', title=None, W=None, H=None):
    if plotType == 'simple':
        ax = df_raw.plot(figsize=(W, H) if (W is not None) and (H is not None) else None, x_compat=True, label='Raw Data')
        df_interpol.plot(ax=ax, label='Interpolated Data', linestyle='--')
        ax.set_title(title if title else 'Data Visualization')
        ax.legend()
    elif plotType == 'plotly':
        createFig = go.Figure()

        # Original Data
        cal_max_raw = df_raw['value'].max()
        normalized_val_raw = df_raw['value'] / cal_max_raw if abs(cal_max_raw) > 0.01 else df_raw['value'] / (cal_max_raw + 0.01)
        createFig.add_trace(
            go.Scatter(x=df_raw.index, y=normalized_val_raw, 
                       mode='markers', marker=dict(size=4, symbol='circle-open'), name=f'Raw Data, max={cal_max_raw:.1f}')
        )
        
        # Interpolated Data
        cal_max_interpol = df_interpol['value'].max()
        normalized_val_interpol = df_interpol['value'] / cal_max_interpol if abs(cal_max_interpol) > 0.01 else df_interpol['colec_val'] / (cal_max_interpol + 0.01)
        createFig.add_trace(
            go.Scatter(x=df_interpol.index, y=normalized_val_interpol, 
                       mode='markers', marker=dict(size=4), name=f'Interpolated Data, max={cal_max_interpol:.1f}')
        )
        
        # 레이아웃 설정
        layout_params = {
            'title': dict(text=title if title else 'Data Visualization', x=0.5, xanchor='center'),
            'xaxis_title': 'Collect Date',
            'yaxis_title': 'Normalized Value',
            'legend': dict(x=0.01, y=0.98, traceorder='normal', 
                           bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=1)
        }
        
        # width와 height가 주어지면 추가
        if W is not None:
            layout_params['width'] = W
        if H is not None:
            layout_params['height'] = H
        createFig.update_layout(**layout_params)
        
        createFig.show()
    else:
        pass


def plot_data(df, plotType='simple', title=None, W=None, H=None):
    if plotType == 'simple':
        ax = df.plot(figsize=(W, H) if (W is not None) and (H is not None) else None, x_compat=True)
        ax.set_title(title if title else 'Data Visualization')
    elif plotType == 'plotly':
        createFig = go.Figure()

        # Data
        createFig.add_trace(
            go.Scatter(x=df.index, y=df.iloc[:,0], 
                       mode='markers', marker=dict(size=4), name=f'Data')
        )
        
        # 레이아웃 설정
        layout_params = {
            'title': dict(text=title if title else 'Data Visualization', x=0.5, xanchor='center'),
            'xaxis_title': 'timeStep',
            'yaxis_title': 'Value',
            'legend': dict(x=0.01, y=0.98, traceorder='normal', 
                           bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=1)
        }
        
        # width와 height가 주어지면 추가
        if W is not None:
            layout_params['width'] = W
        if H is not None:
            layout_params['height'] = H  
        createFig.update_layout(**layout_params)
        
        createFig.show()
    else:
        pass


# cleansing 함수
def cleansing_data_interpolation_IQR(data: pd.Series, mode: Literal['missing', 'outlier', 'both'] = 'both') -> pd.Series:
    """
    데이터의 누락치와 이상치를 처리하는 함수
    
    Args:
        data (pd.Series): 정제할 데이터
        mode (str): 정제 모드 ('missing', 'outlier', 'both' 중 하나)
    
    Returns:
        pd.Series: 정제된 데이터
    """
    data_cleaned = data.copy()

    if mode in ['missing', 'both']:  # 누락치 처리 (보간법)
        data_cleaned = data_cleaned.interpolate(method='linear')

    if mode in ['outlier', 'both']:  # 이상치 처리 (IQR 방법)
        Q1 = data_cleaned.quantile(0.25)
        Q3 = data_cleaned.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data_cleaned = np.where((data_cleaned < lower_bound) | (data_cleaned > upper_bound), np.nan, data_cleaned)
        data_cleaned = pd.Series(data_cleaned).interpolate(method='linear')

    return data_cleaned


# cleansing 함수
def cleansing_data_Kalman_Filter(data: pd.Series, mode: Literal['missing', 'outlier', 'both'] = 'both', threshold_factor: float = 3) -> pd.Series:
    """
    칼만 필터를 사용하여 데이터의 누락치와 이상치를 처리하는 함수
    
    Args:
        data (pd.Series): 정제할 데이터
        mode (str): 정제 모드 ('missing', 'outlier', 'both' 중 하나)
        threshold_factor (float): 이상치 판별을 위한 임계값 계수
    
    Returns:
        pd.Series: 정제된 데이터
    """
    from pykalman import KalmanFilter
    data_cleaned = data.copy()

    # 누락치를 평균으로 임시 대체
    data_filled = data_cleaned.fillna(data.mean())
    # 칼만 필터 초기화
    kf = KalmanFilter(initial_state_mean=data_filled.mean(), n_dim_obs=1)
    # 데이터의 누락치를 무시하고 칼만 필터 학습
    kf = kf.em(data_filled.values.reshape(-1, 1), n_iter=len(data_filled))
    
    # 칼만 필터를 사용하여 상태 추정
    state_means, state_covariances = kf.filter(data_filled.values.reshape(-1, 1))
    
    data_cleaned = data_filled.values
    
    if mode in ['missing', 'both']:
        # 누락치 대체
        data_cleaned = np.where(np.isnan(data), state_means[:, 0], data_cleaned)
    
    if mode in ['outlier', 'both']:
        # 잔차 계산
        residuals = data_cleaned - state_means[:, 0]
        # 임계값 설정 (예: 3 표준편차)
        threshold = threshold_factor * residuals.std()
        # 이상치 판별 및 대체
        data_cleaned = np.where(np.abs(residuals) > threshold, state_means[:, 0], data_cleaned)
    
    return pd.Series(data_cleaned, index=data.index)


def print_info_about_large_csv_file():
    # CSV 파일이 너무 큰 경우, 처음과 최종 3줄을 출력하기
    import pandas as pd
    file_path = '/home/ymatics/CodingSpace2/임시/data_colec_h_202410141057_B0008.csv'

    head_data = pd.read_csv(file_path, nrows=2)
    total_rows = sum(1 for row in open(file_path))
    skip_rows = max(1, total_rows - 2)
    tail_data = pd.read_csv(file_path, skiprows=skip_rows)

    print(file_path.split('/')[-1])
    print(head_data.to_csv(index=False))
    print('.............')
    print(tail_data.to_csv(index=False))


def split_large_csv_file_into_multiple_bldg_id():
    file_path = '/home/ymatics/CodingSpace/2024_AI_BEMS/data_colec_h_202410110905_19GB.csv'
    output_dir = '/home/ymatics/CodingSpace/2024_AI_BEMS/buildingID_data'  # 출력 파일을 저장할 디렉토리

    ## 데이터가 ~20GB로 너무 커서 'bldg_id' 별로 분리해서 저장
    # 출력 디렉토리가 존재하지 않으면 생성합니다.
    os.makedirs(output_dir, exist_ok=True)
    chunksize = 1_000_000  # 한번에 읽을 행 수

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # chunk를 'bldg_id' 별로 그룹화하여 처리합니다.
        grouped = chunk.groupby('bldg_id')
        for bldg_id, group in grouped:
            output_file = os.path.join(output_dir, f'buildingID_{bldg_id}.csv')
            if os.path.exists(output_file):
                # 이미 존재하는 파일에 추가 모드로 저장합니다.
                group.to_csv(output_file, mode='a', header=False, index=False)
            else:
                # 새로운 파일로 저장합니다.
                group.to_csv(output_file, mode='w', header=True, index=False)


def check_large_csv_file_into_multiple_bldg_id():
    # 잘 분리되었는지, 각 빌딩별 기록 시작일과 종료일을 출력해 본다.
    output_dir = '/home/ymatics/CodingSpace/2024_AI_BEMS/building_data(old)'
    output_dir = '/home/ymatics/CodingSpace/2024_AI_BEMS/buildingID_data'
    file_colec_dt = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(output_dir, filename)
            df = pd.read_csv(file_path)
            # 'colec_dt' 처음행과 마지막 행을 추출합니다.
            first_colec_dt = df['colec_dt'].iloc[1]
            last_colec_dt = df['colec_dt'].iloc[-1]
            len_df = len(df)
            # 파일 이름과 'colec_dt' 처음행과 마지막 행을 리스트에 추가합니다.
            file_colec_dt.append((filename, first_colec_dt, last_colec_dt, len_df))
            print(f'{filename}, {first_colec_dt}, {last_colec_dt}, {len_df=}')

    # 정렬된 파일 이름과 'colec_dt' 처음행과 마지막 행을 출력합니다.
    file_colec_dt.sort()
    for filename, first_colec_dt, last_colec_dt, len_df in file_colec_dt:
        print(f'{filename}, {first_colec_dt}, {last_colec_dt}, {len_df=}')


def convert_buildingID_devID_list_into_buildID_devID_map():
    # buildingID_devID list 메타데이터를 buildingID_devID_map.csv 파일로 변환

    file_path = r"/home/ymatics/CodingSpace/2024_AI_BEMS/(buildingID_devID list)dev_m_202410101733_5003.csv"
    df = pd.read_csv(file_path)
    buildingID_df = df[['bldg_id', 'dev_id', 'dev_nm', 'dev_dtl_desc']].rename(columns={'bldg_id': 'buildingID', 'dev_id': 'devID', 'dev_nm': 'devName', 'dev_dtl_desc': 'devType'})
    buildingID_df['devType'] = buildingID_df['devType'].fillna('인버터')  # 신축 인버터를 추가하는 경우, devType이 누락됨
    print("NaN 이 있는 행을 출력\n", buildingID_df[buildingID_df.isna().any(axis=1)])
    buildingID_df.to_csv("buildingID_devID_map.csv", index=False)

    # devID별 tagCD 메타데이터를 devID_tagCD_map.csv 파일로 변환
    data = {
        "tagCD": [90001, 90002, 90003, 90004, 90005, 90006],
        "tagName": ["공급온도", "환수온도", "순시유량", "누적유량", "순시열량", "누적열량"]
    }
    df = pd.DataFrame(data)
    df["devID_range"] = 'devID 4001~4999'
    df["devID_description"] = 'heat_meter 열량계'
    df = df[["devID_range", "devID_description", "tagCD", "tagName"]]
    df.to_csv("devID_tagCD_map.csv", index=False)
    

def plot_peak_time_distribution(peak_time_freq, peak_time_max_values, start_date, end_date):
    # Plotly 그래프 생성
    fig = go.Figure()

    # 빈도에 대한 막대 그래프 추가 (왼쪽 y축)
    fig.add_trace(go.Bar(
        x=peak_time_freq.index,
        y=peak_time_freq.values,
        name='Frequency',
        text=peak_time_freq.values,  # 막대 끝에 빈도 표시
        textposition='outside',
        textfont=dict(size=10, color='red', family="Arial Black"),  # 텍스트 스타일
        marker_color='orange',  # 막대 색상
        marker=dict(opacity=0.3),  # 투명도 적용
        yaxis='y'  # 왼쪽 y축에 할당
    ))

    # 최대 피크 값에 대한 막대 그래프 추가 (오른쪽 y축)
    fig.add_trace(go.Bar(
        x=peak_time_max_values.index,
        y=peak_time_max_values.values,
        name='Max Peak Value',
        text=peak_time_max_values.values,  # 막대 끝에 최대 피크 값 표시
        textposition='outside',
        textfont=dict(size=10, color='red', family="Arial Black"),  # 텍스트 스타일
        marker_color='blue',  # 막대 색상
        marker=dict(opacity=0.3),  # 투명도 적용
        yaxis='y2'  # 오른쪽 y축에 할당
    ))

    # 레이아웃 업데이트 (이중 y축 설정)
    fig.update_layout(
        title=f'Distribution of Daily Peak Power Times ({start_date} ~ {end_date})',
        xaxis=dict(
            title='Time of Day',
            tickangle=-45  # x축 레이블 45도 회전
        ),
        yaxis=dict(
            title='Frequency',
            side='left',
            range=[0, 30]  # y축 범위 설정
        ),
        yaxis2=dict(
            title='Max Peak Value',
            side='right',
            overlaying='y',  # y2를 y1 위에 겹치게 설정
            range=[0, 600]  # y2축 범위 설정
        ),
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h'
        )
    )

    fig.show()
