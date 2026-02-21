#!/usr/bin/env python3
"""
예측 제어 스크립트
PostgreSQL에서 데이터를 읽어 LightGBM 모델로 예측하고 제어 신호 생성
"""

import os
import psycopg2
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import pickle

# DB 연결 설정
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'mydb'),
    'user': os.getenv('POSTGRES_USER', 'user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'password')
}

MODEL_PATH = '/workspace/models/control_model.pkl'

def load_data_from_db():
    """PostgreSQL에서 최근 데이터 로드"""
    try:
        """
        conn = psycopg2.connect(**DB_CONFIG)
        query = '''
            SELECT 
                timestamp,
                sensor_value,
                temperature,
                pressure,
                humidity,
                flow_rate
            FROM sensor_data
            WHERE timestamp > NOW() - INTERVAL '30 minutes'
            ORDER BY timestamp DESC
            LIMIT 100
        '''
        df = pd.read_sql(query, conn)
        conn.close()
        """
        
        # 테스트용 임의 데이터 생성
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'sensor_value': np.random.normal(50, 10, 100),
            'temperature': np.random.normal(25, 5, 100),
            'pressure': np.random.normal(1013, 20, 100),
            'humidity': np.random.normal(60, 15, 100),
            'flow_rate': np.random.normal(100, 25, 100)
        })
        return df
    except Exception as e:
        print(f"[ERROR] DB 연결 실패: {e}")
        return None

def load_or_create_model():
    """모델 로드 또는 더미 모델 생성"""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"[INFO] 모델 로드 완료: {MODEL_PATH}")
        return model
    else:
        # 더미 학습 데이터로 간단한 모델 생성
        print("[WARN] 모델 파일 없음. 더미 모델 생성")
        X_train = np.random.rand(100, 4)
        y_train = np.random.rand(100)
        
        model = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        # 모델 저장
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        
        return model

def predict_control_signal(df, model):
    """예측 및 제어 신호 생성"""
    if df is None or len(df) == 0:
        print("[WARN] 데이터 부족")
        return None
    
    # 특성 생성
    features = ['sensor_value', 'temperature', 'pressure', 'humidity']
    X = df[features].fillna(df[features].mean()).tail(1)
    
    # 예측
    prediction = model.predict(X)[0]
    
    # 제어 신호 생성 (예측값 기반 임계값 판단)
    if prediction > 0.7:
        control_signal = 'INCREASE'
    elif prediction < 0.3:
        control_signal = 'DECREASE'
    else:
        control_signal = 'MAINTAIN'
    
    return {
        'prediction': prediction,
        'control_signal': control_signal,
        'timestamp': datetime.now()
    }

def save_control_signal(result):
    """제어 신호를 DB에 저장"""
    if result is None:
        return
    
    try:
        """
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO control_signals (timestamp, prediction_value, control_action)
            VALUES (%s, %s, %s)
        ''', (result['timestamp'], result['prediction'], result['control_signal']))
        
        conn.commit()
        cursor.close()
        conn.close()
        """
        
        # 테스트용 출력
        print(f"[TEST] DB 저장 대신 출력:")
        print(f"  - timestamp: {result['timestamp']}")
        print(f"  - prediction_value: {result['prediction']:.3f}")
        print(f"  - control_action: {result['control_signal']}")
        
        print(f"[{datetime.now()}] 제어신호 저장: {result['control_signal']} (예측값: {result['prediction']:.3f})")
    except Exception as e:
        print(f"[ERROR] 제어신호 저장 실패: {e}")

if __name__ == "__main__":
    print(f"[{datetime.now()}] 예측 제어 시작")
    
    # 데이터 로드
    df = load_data_from_db()
    
    # 모델 로드
    model = load_or_create_model()
    
    # 예측 및 제어
    result = predict_control_signal(df, model)
    
    # 결과 저장
    save_control_signal(result)
    
    print(f"[{datetime.now()}] 예측 제어 완료--- 3분마다 체크하는 것을 15분마다 체크하는 것으로 반영해야 함.")