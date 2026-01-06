-- ========================================
-- model_control.py용 DB 초기화 스크립트
-- ========================================
-- 용도: 예측 제어 모듈을 위한 테이블/컬럼 추가
-- 테이블: control_signals (신규), sensor_data (컬럼 확장)
-- 사용법: docker exec -i pg psql -U app -d appdb < init_db_control.sql
-- 주의: init_db_anomaly.sql을 먼저 실행해야 함

-- 1. sensor_data 테이블에 humidity, flow_rate 컬럼 추가
ALTER TABLE sensor_data 
ADD COLUMN IF NOT EXISTS humidity FLOAT,
ADD COLUMN IF NOT EXISTS flow_rate FLOAT;

-- 2. control_signals 테이블 생성 (제어 신호 저장용)
CREATE TABLE IF NOT EXISTS control_signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    prediction_value FLOAT NOT NULL,
    control_action VARCHAR(20) NOT NULL CHECK (control_action IN ('INCREASE', 'DECREASE', 'MAINTAIN')),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3. 인덱스 생성 (성능 향상)
CREATE INDEX IF NOT EXISTS idx_control_signals_timestamp ON control_signals(timestamp DESC);

-- 4. 기존 데이터에 humidity, flow_rate 값 업데이트 (NULL인 경우)
UPDATE sensor_data 
SET 
    humidity = 60 + random() * 30 - 15,    -- 45~75 범위
    flow_rate = 100 + random() * 50 - 25   -- 75~125 범위
WHERE humidity IS NULL OR flow_rate IS NULL;

-- 5. 새로운 테스트 데이터 추가 (humidity, flow_rate 포함)
INSERT INTO sensor_data (timestamp, sensor_value, temperature, pressure, humidity, flow_rate)
SELECT 
    NOW() - INTERVAL '1 minute' * generate_series,
    50 + random() * 20 - 10,   -- sensor_value: 40~60
    25 + random() * 10 - 5,    -- temperature: 20~30
    1013 + random() * 40 - 20, -- pressure: 993~1033
    60 + random() * 30 - 15,   -- humidity: 45~75
    100 + random() * 50 - 25   -- flow_rate: 75~125
FROM generate_series(1, 50);

COMMENT ON TABLE control_signals IS '예측 기반 제어 신호 저장 테이블';
COMMENT ON COLUMN sensor_data.humidity IS '습도 (%)';
COMMENT ON COLUMN sensor_data.flow_rate IS '유량 (L/min)';

