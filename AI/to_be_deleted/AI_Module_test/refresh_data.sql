-- ========================================
-- 데이터 갱신 스크립트 (현재 시간 기준)
-- ========================================
-- 용도: 오래된 데이터를 삭제하고 현재 시간 기준으로 새로운 데이터 삽입
-- 사용법: docker exec -i pg psql -U app -d appdb < refresh_data.sql
-- 주의: 기존 sensor_data 삭제 후 재생성

-- 1. 기존 sensor_data 삭제 (anomaly_log는 유지)
DELETE FROM sensor_data;

-- 2. 현재 시간 기준으로 최근 2시간 데이터 삽입 (초 단위로 더 조밀하게)
INSERT INTO sensor_data (timestamp, sensor_value, temperature, pressure, humidity, flow_rate)
SELECT 
    NOW() - INTERVAL '1 second' * generate_series,
    50 + random() * 20 - 10,   -- sensor_value: 40~60
    25 + random() * 10 - 5,    -- temperature: 20~30
    1013 + random() * 40 - 20, -- pressure: 993~1033
    60 + random() * 30 - 15,   -- humidity: 45~75
    100 + random() * 50 - 25   -- flow_rate: 75~125
FROM generate_series(1, 7200);  -- 최근 2시간 (7200초)

-- 3. 이상치 포함 데이터 추가 (현재 시간)
INSERT INTO sensor_data (timestamp, sensor_value, temperature, pressure, humidity, flow_rate)
VALUES 
    (NOW(), 150, 60, 1200, 80, 200),      -- 명백한 이상치 1
    (NOW(), -50, -20, 500, 20, 10),       -- 명백한 이상치 2
    (NOW() - INTERVAL '5 minutes', 140, 55, 1150, 75, 180),  -- 5분 전 이상치
    (NOW() - INTERVAL '15 minutes', 145, 58, 1180, 78, 190); -- 15분 전 이상치

-- 4. 데이터 통계 확인
SELECT 
    COUNT(*) as total_records,
    MIN(timestamp) as oldest_record,
    MAX(timestamp) as newest_record,
    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '30 minutes') as recent_30min,
    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 hour') as recent_1hour
FROM sensor_data;

