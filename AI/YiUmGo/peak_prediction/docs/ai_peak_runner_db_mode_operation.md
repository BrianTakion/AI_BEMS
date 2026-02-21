# ai_peak_runner.py DB 모드 동작 절차

- **작성일**: 2026-02-21
- **대상 명령**: `python3 peak_prediction/ai_peak_runner.py` (DB 모드)
- **참고**: CSV 모드는 `--csv` 옵션 추가 시 동작 (개발/검증용)

---

## 1. 전체 동작 흐름

```mermaid
flowchart TD
    CLI["python3 ai_peak_runner.py\n(--csv 없음 → DB 모드)"]
    step1["main()\nconfig['data_source'] = 'db'"]
    step2["create_data_source()\nSQLAlchemy 엔진 생성\nPostgreSQL 연결"]
    step3["read_enabled_devices()\nconfig_peak_devices.csv 읽기\nbldg_id=B0019, dev_id=2001"]
    step4["process_device()\n단일 호출"]
    step4a["load_model()\nmodels/2001_power.txt\nmodels/2001_time.txt"]
    step4b["read_sensor_data()\nSELECT FROM bems.data_colec_h\n최근 336h"]
    step4c["preprocess_for_inference()\n~40개 피처 생성"]
    step4d["predict_peak()\nLightGBM 추론"]
    step4e["format_peak_result()\n'207.33@13:15' 포맷"]
    step4f["write_peak_result()\nDELETE+INSERT\nbems.max_dmand_fcst_h"]
    webui["BEMS 웹 UI\n최대수요-AI예측 컬럼 갱신"]

    CLI --> step1 --> step2 --> step3 --> step4
    step4 --> step4a --> step4b --> step4c --> step4d --> step4e --> step4f
    step4f -->|"오늘 날짜 1건 갱신"| webui
```

---

## 2. 단계별 소스 수준 설명

### [1] `main()` — 모드 결정

**파일**: `ai_peak_runner.py`, 98번째 줄

```python
config["data_source"] = "csv" if args.csv else "db"
```

`--csv` 인수가 없으면 `config["data_source"] = "db"` 로 설정됩니다.

---

### [2] `create_data_source()` — PostgreSQL 연결 엔진 생성

**파일**: `data_source.py`

```python
schema = db_cfg.get("schema", "public")          # "bems"
encoded_password = quote_plus(db_cfg["password"]) # "ai123!@#" → URL 인코딩
url = f"postgresql://ai:{encoded_password}@localhost:5432/bemsdb"
engine = create_engine(
    url,
    connect_args={"options": "-c search_path=bems"},
    pool_pre_ping=True,
)
```

- `quote_plus("ai123!@#")` — 비밀번호 특수문자 URL 충돌 방지
- `search_path=bems` — 이후 모든 SQL에서 `bems.` 접두어 불필요
- `pool_pre_ping=True` — cron 간헐적 실행 시 연결 유효성 사전 확인
- 반환값: SQLAlchemy Engine 객체 (`source` 변수)

---

### [3] `read_enabled_devices()` — 처리 대상 장치 결정

**파일**: `data_source.py`

```python
if mode == "db":
    # dev_use_purp_rel_r 쿼리 대신 config_peak_devices.csv 에서 읽음
    # (추후 DB 쿼리 방식으로 전환 시 이 블록을 교체)
    csv_path = config["csv"].get("config_peak_devices_path")
    ...
    df = pd.read_csv(abs_path, ...)
    df = df[df["PEAK_PRCV_YN"] == "Y"]
    # → [{"bldg_id": "B0019", "dev_id": 2001}]
```

`config_peak_devices.csv` 내용:

```
BLDG_ID,DEV_ID,PEAK_PRCV_YN
B0019,2001,Y
```

- DB 쿼리(`SELECT FROM dev_use_purp_rel_r`) 없이 로컬 CSV 파일에서 직접 읽음
- `mode` 분기 구조는 유지 — 추후 DB 쿼리 방식으로 정책 변경 시 `if mode == "db":` 블록만 교체
- 결과: `devices = [{"bldg_id": "B0019", "dev_id": 2001}]`

`main()`에서 `devices`가 비어있으면 즉시 종료:

```python
if not devices:
    logger.error("No enabled devices in config_peak_devices.csv -- abort")
    return
```

장치 루프 없이 `devices[0]` 단일 호출:

```python
device = devices[0]   # B0019 / 2001
bldg_id = device["bldg_id"]
dev_id  = device["dev_id"]
```

---

### [4] `process_device()` — 단일 장치 처리

#### [4-1] `load_model()` — LightGBM 모델 로딩

```python
power_model_path = "peak_prediction/models/2001_power.txt"
time_model_path  = "peak_prediction/models/2001_time.txt"
model_power = infer_peak.load_model(power_model_path)
model_time  = infer_peak.load_model(time_model_path)
```

로컬 파일에서 로딩 (DB 접근 없음).

#### [4-2] `read_sensor_data()` — DB에서 센서 데이터 조회

실행 SQL:

```sql
SELECT colec_dt, colec_val
FROM bems.data_colec_h
WHERE bldg_id = 'B0019'
  AND dev_id  = '2001'
  AND tag_cd  = '30001'
  AND colec_dt >= NOW() - INTERVAL '336 hours'
ORDER BY colec_dt ASC
```

- 최근 14일(336시간) 15분 간격 전력 데이터 조회 → 약 1,344행

#### [4-3] `preprocess_for_inference()` — 피처 생성

Python 메모리 내 처리 (DB 접근 없음):
- 15분 데이터 → 1시간 리샘플링
- 일별 피크 추출 (과거 14일)
- ~40개 피처 벡터 생성 (요일, 월, 이전 날 피크 통계, 오늘 부분 데이터 등)

#### [4-4] `predict_peak()` — LightGBM 추론

```python
power_pred = model_power.predict(X_df)  # 예: 207.33 kW
time_pred  = model_time.predict(X_df)   # 예: 53 (슬롯) → 13:15
peak_slot  = int(np.clip(np.round(time_pred[-1]), 0, 95))
```

#### [4-5] `format_peak_result()` — 결과 포맷

```python
# 슬롯 53 → 53 × 15분 = 795분 = 13:15
return f"{peak_power:.2f}@{hours:02d}:{minutes:02d}"
# → "207.33@13:15"
```

#### [4-6] `write_peak_result()` — DB에 결과 저장

실행 SQL (트랜잭션):

```sql
-- 오늘 날짜 기존 예측값 삭제
DELETE FROM bems.max_dmand_fcst_h
WHERE bldg_id = 'B0019'
  AND DATE(use_dt) = DATE('2026-02-21 13:00:00');

-- 최신 예측값 INSERT
INSERT INTO bems.max_dmand_fcst_h (use_dt, bldg_id, dly_max_dmand_fcst_inf)
VALUES ('2026-02-21 13:00:00', 'B0019', '207.33@13:15');
```

- `source.begin()` 컨텍스트 매니저로 DELETE → INSERT 원자적 실행
- 오늘 날짜 기준 1건 유지 — 매시간 cron 실행 시 최신 예측으로 덮어씀

---

### [5] 완료 로그

```python
status = "OK" if result else "FAILED"
logger.info("=== Done: %s in %.1fs ===", status, elapsed)
```

---

## 3. BEMS 웹 UI 반영

`bems.max_dmand_fcst_h`에 오늘 날짜 기준 1건이 갱신되면, 웹 UI가 아래 구조로 조회하여 **최대수요-AI예측(kW) @HH:MM** 열에 표시합니다:

```sql
SELECT
    DATE(d.colec_dt)           AS 일자,
    SUM(d.colec_val) * 0.25    AS 사용량_kWh,
    MAX(d.colec_val)           AS 최대수요_kW,
    f.dly_max_dmand_fcst_inf   AS AI예측
FROM bems.data_colec_h d
LEFT JOIN bems.max_dmand_fcst_h f
       ON d.bldg_id = f.bldg_id
      AND DATE(d.colec_dt) = DATE(f.use_dt)
WHERE d.bldg_id = 'B0019'
  AND d.dev_id  = '2001'
  AND d.tag_cd  = '30001'
GROUP BY DATE(d.colec_dt), f.dly_max_dmand_fcst_inf
ORDER BY 일자 DESC
```

`dly_max_dmand_fcst_inf = "207.33@13:15"` 가 `@` 기준으로 파싱되어 kW 수치와 시각으로 분리 표시됩니다.

---

## 4. CSV 모드 vs DB 모드 차이

| 단계 | `--csv` (CSV 모드) | (DB 모드) |
|------|-------------------|-----------|
| `create_data_source()` | config dict 반환 | SQLAlchemy 엔진 반환 |
| `read_enabled_devices()` | `config_peak_devices.csv` 읽기 | `config_peak_devices.csv` 읽기 (동일) |
| `read_sensor_data()` | 로컬 CSV 파일 청크 읽기 | `bems.data_colec_h` SELECT |
| `write_peak_result()` | `output/peak_results.csv` 추가 | `bems.max_dmand_fcst_h` DELETE+INSERT |
| 웹 UI 반영 | 없음 | 즉시 반영 |

> `read_enabled_devices()`는 두 모드 모두 `config_peak_devices.csv`를 읽습니다. `mode` 분기 구조는 유지되어 있으므로, 추후 DB 쿼리 방식으로 전환 시 `if mode == "db":` 블록 내부만 교체하면 됩니다.

---

## 5. cron 등록 예시

```cron
# 매시간 0분 실행 (production DB 모드)
0 * * * * cd /home/sems/AI/YiUmGo && /path/to/venv/bin/python3 peak_prediction/ai_peak_runner.py >> /var/log/ai_peak.log 2>&1
```
