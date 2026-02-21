-- ========================================
-- anomaly_detection.py용 DB 초기화 스크립트
-- ========================================
-- 용도: 이상치 탐지 모듈을 위한 테이블 생성
-- 테이블: sensor_data, anomaly_log
-- 사용법: docker exec -i pg psql -U app -d appdb < init_db_anomaly.sql

-- 1. sensor_data 테이블 생성 (이상치 탐지용 센서 데이터)
CREATE TABLE IF NOT EXISTS sensor_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    sensor_value FLOAT NOT NULL,
    temperature FLOAT,
    pressure FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 2. anomaly_log 테이블 생성 (이상치 로그)
CREATE TABLE IF NOT EXISTS anomaly_log (
    id SERIAL PRIMARY KEY,
    sensor_id INTEGER NOT NULL,
    detected_at TIMESTAMP NOT NULL,
    anomaly_score FLOAT NOT NULL,
    details TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(sensor_id, detected_at)  -- 중복 방지
);

-- 3. 인덱스 생성 (성능 향상)
CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp ON sensor_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_log_detected_at ON anomaly_log(detected_at DESC);

-- 4. 테스트용 샘플 데이터 삽입 (선택사항)
INSERT INTO sensor_data (timestamp, sensor_value, temperature, pressure)
SELECT 
    NOW() - INTERVAL '1 minute' * generate_series,
    50 + random() * 20 - 10,  -- sensor_value: 40~60 범위
    25 + random() * 10 - 5,   -- temperature: 20~30 범위
    1013 + random() * 40 - 20 -- pressure: 993~1033 범위
FROM generate_series(1, 100);

-- 5. 이상치 포함 데이터 추가 (테스트용)
INSERT INTO sensor_data (timestamp, sensor_value, temperature, pressure)
VALUES 
    (NOW(), 150, 60, 1200),  -- 명백한 이상치 1
    (NOW(), -50, -20, 500);  -- 명백한 이상치 2

COMMENT ON TABLE sensor_data IS '센서 데이터 테이블 (이상치 탐지용)';
COMMENT ON TABLE anomaly_log IS '감지된 이상치 로그 테이블';

