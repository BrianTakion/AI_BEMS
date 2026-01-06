import pandas as pd
import numpy as np
import re
import holidays
from pandas.tseries.offsets import CustomBusinessDay
import datetime
from scipy.fftpack import fft

def preprocess(raw_df, points=4, freqInterval='15min'):
    df = raw_df.copy()

    p_ = points # 시간당 4 points 샘플링, 15min 간격
    freq = freqInterval
    deltaT = int(re.findall(r'\d+', freqInterval)[0])

    # 1. 누락여부 피쳐 생성 및 보간
    df.index = df.index.round(freq)  # freq 단위로 라운딩하여 시간 맞추고(15분±50% time index 흔들림 허용)
    df = df[~df.index.duplicated(keep='first')]  # 중복된 시간 포인트는 제거
    df = df.resample(freq).asfreq()  # freq 단위로 인덱스 리샘플링, asfreq()로 빈 시간대는 NaN으로 채움
    df['is_missing'] = df['value'].isna().astype(int)
    df.ffill(inplace=True)  # 예측 모델인 경우, 미래 데이터를 사용하지 않도록 보간
    df.bfill(inplace=True)  # 예측 모델인 경우, 만약 첫 번째 값이 결측치인 경우를 대비
    # df['value'] = df['value'].interpolate(method='time')  # 회귀 모델인 경우, 대안으로 선택 가능

    # 예측모델일 때는 과거데이터만으로 현재 regression, 회귀모델일 때는 현재데이터로 regression
    df_value1p = df['value'].shift(1)  # Shifted values to use past data only
    # df_value1p = df['value']

    #ToDo 메모리 효율성을 위하여 피처 딕셔너리로 담아둔 후 pd.concat으로 피처 생성할 것**
    # features_dict = {}
    # ...
    # df = pd.concat([df, pd.DataFrame(features_dict, index=df.index)], axis=1)

    # 2. 시간 기반 피쳐 생성 (대한민국의 주말 및 공휴일 특징 반영)
    kr_holidays = holidays.KR()
    #kr_business_day = CustomBusinessDay(holidays=kr_holidays)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)  # 주말 여부
    df['is_holiday'] = df.index.isin(kr_holidays).astype(int)  # 대한민국 공휴일 여부

    # 2. 명절 연휴 처리(설날, 추석은 전일과 후일이 연휴임)
    major_holiday_dates = pd.to_datetime([date for date in kr_holidays if kr_holidays[date] in ['설날', '추석']])
    df.loc[df.index.isin(major_holiday_dates - pd.DateOffset(days=1)), 'is_holiday'] = 1
    df.loc[df.index.isin(major_holiday_dates + pd.DateOffset(days=1)), 'is_holiday'] = 1

    # 2. 대체공휴일 보정
    for date in kr_holidays:
        if date.weekday() in [5, 6]:
            replacement_date = date + datetime.timedelta(days=1)
            while replacement_date.weekday() in [5, 6] or replacement_date in kr_holidays:
                replacement_date += datetime.timedelta(days=1)
            df.loc[replacement_date, 'is_holiday'] = 1

    # 3. 계절성 피처 생성
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    # 3. 주기성 특성
    df['sine_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cosine_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    # 3. 추가적인 계절성 특징 생성
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    # 4. 시간 지연 피쳐 생성
    for lag in range(1, 9):  # 15min delayed pattern ---> 2h 정도만 고려
        df[f'lag_{lag}p'] = df['value'].shift(lag)

    for lag in range(5):  # 1일 = 4p_ x 24h/p_ 지연 패턴
        df[f'lag_1d_{lag}p'] = df['value'].shift(p_ * 24 + lag)  

    for lag in range(5):  # 1주 = 4p_ x 24h/p_ x 7일 지연 패턴
        df[f'lag_1w_{lag}p'] = df['value'].shift(p_ * 24 * 7 + lag)

    # 5. 변동률 및 변동률의 변동률 피쳐 생성
    #ToDo (y1 - y0) / y0 또는 pct_change() 대신에 (y1 - y0) / deltaT 사용할 것
    epsilon = 1e-3
    shifted = df['value'].shift(1)  # 과거 데이터 사용
    divisor = np.where(np.abs(shifted) > epsilon, shifted, np.sign(shifted) * epsilon)
    df['rate'] = (shifted - shifted.shift(1)) / divisor
    df['diff_rate'] = df['rate'].diff()

    shifted = df['value'].shift(p_ * 24)  # 하루 전의 동일 시간대 변동률의 변동률
    divisor = np.where(np.abs(shifted) > epsilon, shifted, np.sign(shifted) * epsilon)
    df['rate_1d'] = (shifted - shifted.shift(1)) / divisor
    df['diff_rate_1d'] = df['rate_1d'].diff()

    # 6. 윈도우 통계 피처 생성
    window_sizes = [1, 2, 3, 4, 5, 6]  # 1시간, 2시간, 3시간 윈도우 통계 ----> 6시간 통계 고려
    for window in window_sizes:
        df[f'ma_{window}h'] = df_value1p.rolling(window=p_ * window).mean()
        df[f'max_{window}h'] = df_value1p.rolling(window=p_ * window).max()
        df[f'min_{window}h'] = df_value1p.rolling(window=p_ * window).min()
        df[f'std_{window}h'] = df_value1p.rolling(window=p_ * window).std()
        df[f'skew_{window}h'] = df_value1p.rolling(window=p_ * window).skew()
        df[f'kurt_{window}h'] = df_value1p.rolling(window=p_ * window).kurt()

    window_sizes = [1, 2, 3, 4, 5, 6]  # 1일, 2일 윈도우 통계 ---> 6일 통계 고려
    for window in window_sizes:
        df[f'ma_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).mean()
        df[f'max_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).max()
        df[f'min_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).min()
        df[f'std_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).std()
        df[f'skew_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).skew()
        df[f'kurt_{window}d'] = df_value1p.rolling(window=p_ * 24 * window).kurt()

    df['p1d_ma_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).mean()  # 1일전 동시간대 1시간 통계
    df['p1d_max_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).max()
    df['p1d_min_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).min()
    df['p1d_std_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).std()
    df['p1d_skew_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).skew()
    df['p1d_kurt_1d'] = df['value'].shift(p_ * 24).rolling(window=p_).kurt()

    df['p1w_ma_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).mean()  # 1주일전 동시간대 1시간 통계
    df['p1w_max_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).max()
    df['p1w_min_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).min()
    df['p1w_std_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).std()
    df['p1w_skew_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).skew()
    df['p1w_kurt_1d'] = df['value'].shift(p_ * 24 * 7).rolling(window=p_).kurt()

    # 7. 이동 평균의 변화율 특징 생성
    epsilon = 1e-3
    shifted = df['ma_1h']
    divisor = np.where(np.abs(shifted) > epsilon, shifted, np.sign(shifted) * epsilon)
    df['rate_ma_1h'] = (shifted - shifted.shift(1)) / divisor

    shifted = df['ma_2h']
    divisor = np.where(np.abs(shifted) > epsilon, shifted, np.sign(shifted) * epsilon)
    # df['rate_ma_2h'] = (shifted - shifted.shift(1)) / divisor

    shifted = df['ma_1d']
    divisor = np.where(np.abs(shifted) > epsilon, shifted, np.sign(shifted) * epsilon)
    # df['rate_ma_1d'] = (shifted - shifted.shift(1)) / divisor

    # 8. 추세 및 계절성 분포
    df['trend'] = np.arange(len(df))
    df['season'] = df_value1p - df['ma_1d']  # 과거 데이터 사용

    # 9. Fourier Transform 특징 생성
    fft_features = fft(df_value1p.fillna(0).values)
    df['fft_real'] = np.real(fft_features)  # 실수 부분
    df['fft_imag'] = np.imag(fft_features)  # 허수 부분

    # 10. 지수 이동 평균 (Exponential Moving Average)
    df['ema_1d'] = df_value1p.ewm(span=p_ * 24, adjust=False).mean()  # 이전 1일 지수 이동 평균

    # 11. 누적 합계 및 변화율
    df['cum_sum'] = df_value1p.cumsum()  # 누적 합계
    df['cum_pct'] = df['cum_sum'].pct_change()  # 누적 합계의 증감율

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