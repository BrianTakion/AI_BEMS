#!/usr/bin/env python3
"""
이상치 탐지 스크립트
PostgreSQL에서 데이터를 읽어 이상치를 탐지하고 결과를 저장
"""

import os
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine

# DB 연결 설정 (환경변수 사용 권장)
# DATABASE_URL=postgresql://app:secret@localhost:5432/appdb
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'appdb'),
    'user': os.getenv('POSTGRES_USER', 'app'),
    'password': os.getenv('POSTGRES_PASSWORD', 'secret')
}

def get_db_url():
    """SQLAlchemy용 데이터베이스 URL 생성"""
    return f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

def load_data_from_db():
    """PostgreSQL에서 데이터 로드 (SQLAlchemy 사용)"""
    try:
        engine = create_engine(get_db_url())
        query = '''
            SELECT 
                id,
                timestamp,
                sensor_value,
                temperature,
                pressure
            FROM sensor_data
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            ORDER BY timestamp DESC
            LIMIT 1000
        '''
        df = pd.read_sql(query, engine)
        engine.dispose()
        print(f"[INFO] DB에서 {len(df)}건의 데이터 로드 완료")
        return df
    except Exception as e:
        print(f"[ERROR] DB 연결 실패: {e}")
        print(f"[INFO] 테스트용 임의 데이터로 대체")
        
        # DB 연결 실패 시 테스트용 임의 데이터 생성
        df = pd.DataFrame({
            'id': range(1, 101),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'sensor_value': np.random.normal(50, 10, 100),
            'temperature': np.random.normal(25, 5, 100),
            'pressure': np.random.normal(1013, 20, 100)
        })
        return df

def detect_anomalies(df):
    """Isolation Forest를 이용한 이상치 탐지"""
    if df is None or len(df) < 10:
        print("[WARN] 데이터 부족으로 이상치 탐지 생략")
        return None
    
    # 특성 선택
    features = ['sensor_value', 'temperature', 'pressure']
    X = df[features].fillna(df[features].mean())
    
    # 모델 학습 및 예측
    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict(X)
    df['anomaly_score'] = model.score_samples(X)
    
    # 이상치만 필터링 (-1이 이상치)
    anomalies = df[df['anomaly'] == -1]
    
    return anomalies

def save_anomalies(anomalies):
    """이상치 결과를 DB에 저장"""
    if anomalies is None or len(anomalies) == 0:
        print(f"[{datetime.now()}] 이상치 없음")
        return
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 이상치 로그 출력
        print(f"[INFO] 감지된 이상치:")
        for _, row in anomalies.iterrows():
            print(f"  - sensor_id: {row['id']}, score: {row['anomaly_score']:.3f}, value: {row['sensor_value']:.2f}")
        
        # DB에 저장
        for _, row in anomalies.iterrows():
            cursor.execute('''
                INSERT INTO anomaly_log (sensor_id, detected_at, anomaly_score, details)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            ''', (row['id'], datetime.now(), row['anomaly_score'], 
                  f"sensor_value: {row['sensor_value']}"))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"[{datetime.now()}] 이상치 {len(anomalies)}건 감지 및 DB 저장 완료")
    except Exception as e:
        print(f"[ERROR] 이상치 저장 실패: {e}")
        print(f"[INFO] DB 저장 실패 - 이상치 정보는 위 로그 참조")

if __name__ == "__main__":
    print(f"[{datetime.now()}] 이상치 탐지 시작")
    
    # 데이터 로드
    df = load_data_from_db()
    
    # 이상치 탐지
    anomalies = detect_anomalies(df)
    
    # 결과 저장
    save_anomalies(anomalies)
    
    print(f"[{datetime.now()}] 이상치 탐지 작업 완료")