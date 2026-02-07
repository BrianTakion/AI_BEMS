import pandas as pd
import numpy as np
import re
import holidays
from pandas.tseries.offsets import CustomBusinessDay
import datetime
from scipy.fftpack import fft
import pywt
from scipy.signal import find_peaks
from scipy import stats

def preprocess(raw_df, points=4, freqInterval='15min', only_cleansing=False, fill_method='zero'):
    df = raw_df.copy()

    p_ = points # 시간당 4 points 샘플링, 15min 간격
    freq = freqInterval
    deltaT = int(re.findall(r'\d+', freqInterval)[0])

    # 1. 누락여부 피쳐 생성 및 보간
    df.index = df.index.round(freq)  # freq 단위로 라운딩하여 시간 맞추고(15분±50% time index 흔들림 허용)
    df = df[~df.index.duplicated(keep='first')]  # 중복된 시간 포인트는 제거
    df = df.resample(freq).asfreq()  # freq 단위로 인덱스 리샘플링, asfreq()로 빈 시간대는 NaN으로 채움

    # 음수 값은 0으로 보정
    #df['value'] = df['value'].clip(lower=0)

    df['is_missing'] = df['value'].isna().astype(int)
    if fill_method == 'zero':
        df.fillna(0, inplace=True)  # 결측치는 0으로 채움
    elif fill_method == 'ffill':
        df.ffill(inplace=True)  # 예측 모델인 경우, 미래 데이터를 사용하지 않도록 보간
        df.bfill(inplace=True)  # 예측 모델인 경우, 만약 첫 번째 값이 결측치인 경우를 대비
    elif fill_method == 'time':
        df['value'] = df['value'].interpolate(method='time')  # 회귀 모델인 경우, 대안으로 선택 가능

    # only cleansing operation
    if only_cleansing:
        df_interpol = df[['value']].copy()
        df_is_missing = df[['is_missing']].copy()
        nan_counts_df = df.isna().sum()  # 각 열마다 NaN의 갯수 출력
        #for idx, item in enumerate(nan_counts_df):
        #    print(f"{df.columns[idx]= }, NaN Count: {item}")

        missing_ratio = round(df['is_missing'].sum() / len(df), 1)
        return df_interpol, df_is_missing, nan_counts_df, missing_ratio

    # 예측모델일 때는 과거데이터만으로 현재 regression, 회귀모델일 때는 현재데이터로 regression
    df_value1p = df['value'].shift(1)  # Shifted values to use past data only
    # df_value1p = df['value']

    # 2. 시간 기반 피쳐 생성 (대한민국의 주말 및 공휴일 특징 반영)
    features_dict = {}
    
    kr_holidays = holidays.KR()
    #kr_business_day = CustomBusinessDay(holidays=kr_holidays)
    features_dict['hour'] = df.index.hour
    ###features_dict['day'] = df.index.day
    features_dict['month'] = df.index.month
    ###features_dict['quarter'] = df.index.quarter
    features_dict['weekday'] = df.index.weekday
    ####features_dict['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)  # 주말 여부
    features_dict['is_holiday'] = pd.Series(0, index=df.index)
    #df.index.isin(kr_holidays).astype(int)  # 대한민국 공휴일 여부

    # 2. 명절 연휴 처리(설날, 추석은 전일과 후일이 연휴임)
    major_holiday_dates = pd.to_datetime([date for date in kr_holidays if kr_holidays[date] in ['설날', '추석']])
    holiday_series = features_dict['is_holiday'].copy()
    holiday_series.loc[df.index.isin(major_holiday_dates - pd.Timedelta(days=1))] = 1  #pd.DateOffset(days=1))] = 1
    holiday_series.loc[df.index.isin(major_holiday_dates + pd.Timedelta(days=1))] = 1  #pd.DateOffset(days=1))] = 1
    features_dict['is_holiday'] = holiday_series

    # 2. 대체공휴일 보정
    for date in kr_holidays:
        if date.weekday() in [5, 6]:
            replacement_date = date + datetime.timedelta(days=1)
            while replacement_date.weekday() in [5, 6] or replacement_date in kr_holidays:
                replacement_date += datetime.timedelta(days=1)
            if replacement_date in df.index:
                features_dict['is_holiday'].loc[replacement_date] = 1

    # 3. 계절성 피처 생성
    features_dict['sin_month'] = np.sin(2 * np.pi * features_dict['month'] / 12)
    features_dict['cos_month'] = np.cos(2 * np.pi * features_dict['month'] / 12)

    # 3. 주기성 특성
    features_dict['sine_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    features_dict['cosine_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    # 3. 추가적인 계절성 특징 생성
    features_dict['sin_hour'] = np.sin(2 * np.pi * features_dict['hour'] / 24)
    features_dict['cos_hour'] = np.cos(2 * np.pi * features_dict['hour'] / 24)

    # 4. 시간 지연 피쳐 생성
    ##############
    for lag in [1, 2, 3]:  # 15min delayed pattern
        features_dict[f'lag_{lag}p'] = df['value'].shift(lag)
    for lag in [0]:  # 1일 = 4p_ x 24h/p_ 지연 패턴
        features_dict[f'lag_1d_{lag}p'] = df['value'].shift(p_ * 24 + lag)
    for lag in [0]:  #(0, 5) # 1주 = 4p_ x 24h/p_ x 7일 지연 패턴
        features_dict[f'lag_1w_{lag}p'] = df['value'].shift(p_ * 24 * 7 + lag)

    # 5. 변동률 및 변동률의 변동률 피쳐 생성
    shifted = df['value'].shift(1)  # 과거 데이터 사용
    features_dict['rate'] = shifted.diff() / deltaT
    features_dict['rate_rate'] = features_dict['rate'].diff() / deltaT

    shifted = df['value'].shift(p_ * 24)  # 하루 전의 동일 시간대 변동률의 변동률
    features_dict['rate_1d'] = shifted.diff() / deltaT
    features_dict['rate_rate_1d'] = features_dict['rate_1d'].diff() / deltaT

    # 6. 윈도우 통계 피처 생성
    window_sizes = [1]  # 시간 윈도우 통계
    for window in window_sizes:
        features_dict[f'ma_{window}h'] = df_value1p.rolling(window=p_ * window).mean()
        features_dict[f'max_{window}h'] = df_value1p.rolling(window=p_ * window).max()
        features_dict[f'min_{window}h'] = df_value1p.rolling(window=p_ * window).min()
        features_dict[f'std_{window}h'] = df_value1p.rolling(window=p_ * window).std()
        features_dict[f'skew_{window}h'] = df_value1p.rolling(window=p_ * window).skew()
        features_dict[f'kurt_{window}h'] = df_value1p.rolling(window=p_ * window).kurt()

    window_sizes = [1]  # 일 윈도우 통계
    for window in window_sizes:
        features_dict[f'ma_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).mean()
        features_dict[f'max_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).max()
        features_dict[f'min_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).min()
        features_dict[f'std_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).std()
        features_dict[f'skew_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).skew()
        features_dict[f'kurt_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).kurt()

    features_dict['p1d_ma_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).mean()  # 1일전 동시간대 1시간 통계
    features_dict['p1d_max_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).max()
    features_dict['p1d_min_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).min()
    features_dict['p1d_std_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).std()
    features_dict['p1d_skew_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).skew()
    features_dict['p1d_kurt_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).kurt()

    features_dict['p1w_ma_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).mean()  # 1주일전 동시간대 1시간 통계
    features_dict['p1w_max_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).max()
    features_dict['p1w_min_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).min()
    features_dict['p1w_std_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).std()
    features_dict['p1w_skew_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).skew()
    features_dict['p1w_kurt_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).kurt()

    # 7. 이동 평균의 변화율 특징 생성
    shifted = features_dict['ma_1h']
    features_dict['rate_ma_1h'] = shifted.diff() / deltaT
    features_dict['rate_rate_ma_1h'] = features_dict['rate_ma_1h'].diff() / deltaT
    
    shifted = features_dict['p1d_ma_1d']
    features_dict['rate_p1d_ma_1d'] = shifted.diff() / deltaT
    features_dict['rate_rate_p1d_ma_1d'] = features_dict['rate_p1d_ma_1d'].diff() / deltaT
    
    shifted = features_dict['p1w_ma_1d']
    features_dict['rate_p1w_ma_1d'] = shifted.diff() / deltaT
    features_dict['rate_rate_p1w_ma_1d'] = features_dict['rate_p1w_ma_1d'].diff() / deltaT

    # 8. 추세 및 계절성 분포
    features_dict['trend'] = np.arange(len(df))
    features_dict['season_ma_1h'] = df_value1p - features_dict['ma_1h']
    features_dict['season_ma_1d'] = df_value1p - features_dict['ma_1d']

    # 9. Fourier Transform, wavelet 특징 생성
    """fft_features = fft(df_value1p.fillna(0).values)
    features_dict['fft_real'] = np.real(fft_features)  # 실수 부분
    features_dict['fft_imag'] = np.imag(fft_features)  # 허수 부분"""

    wavelet = 'db1'
    level = 3  # 분해 레벨
    coeffs = pywt.wavedec(np.array(df_value1p, copy=True), wavelet, level=level)
    # 각 레벨별 특징 추출
    for i in range(level + 1):
        if i == 0:
            # Approximation coefficients
            features_dict[f'wavelet_approx'] = np.interp(np.linspace(0, len(coeffs[i]) - 1, len(df_value1p)), np.arange(len(coeffs[i])), coeffs[i])
        else:
            # Detail coefficients
            features_dict[f'wavelet_detail_{i}'] = np.interp(np.linspace(0, len(coeffs[i]) - 1, len(df_value1p)), np.arange(len(coeffs[i])), coeffs[i])
    
    # 10. 지수 이동 평균 (Exponential Moving Average)
    ###features_dict['ema_1d'] = df_value1p.ewm(span=p_ * 24, adjust=False).mean()  # 이전 1일 지수 이동 평균

    # 11. 누적 합계 및 변화율
    features_dict['cum_sum'] = df_value1p.cumsum()  # 누적 합계
    features_dict['rate_cum_sum'] = features_dict['cum_sum'].diff() / deltaT  # 누적 합계의 증감율
    features_dict['rate_rate_cum_sum'] = features_dict['rate_cum_sum'].diff() / deltaT  # 누적 합계의 증감율

    # 12.이벤트 탐지
    peaks, _ = find_peaks(df_value1p)
    features_dict['is_peak'] = np.zeros(len(df_value1p), dtype=int)  # 모든 값을 0으로 초기화
    features_dict['is_peak'][peaks] = 1  # 피크 지점에 플래그 추가

    """features_dict['z_score'] = stats.zscore(df_value1p, nan_policy='omit')  # z-score 계산"""


    df = pd.concat([df, pd.DataFrame(features_dict, index=df.index)], axis=1)

    nan_counts_df = df.isna().sum()  # 각 열마다 NaN의 갯수 출력
    #for idx, item in enumerate(nan_counts_df):
    #    print(f"{df.columns[idx]= }, NaN Count: {item}")

    # 결측값 처리 (피처 생성시 시간 지연으로 인해 발생)
    df.dropna(inplace=True)

    # 피처벡터와 타겟 분리
    X_df = df.drop('value', axis=1)
    y_df = df['value']

    missing_ratio = round(df['is_missing'].sum() / len(df), 1)

    return X_df, y_df, nan_counts_df, missing_ratio