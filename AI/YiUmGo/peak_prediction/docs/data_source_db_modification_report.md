# peak_prediction data_source.py DB 모드 수정 리포트

- **작성일**: 2026-02-21
- **수정 파일**: `peak_prediction/data_source.py`, `peak_prediction/_config.json`
- **참고 문서**: `anomaly_detection/docs/data_source_db_modification_report.md`

---

## 1. 수정 배경

`peak_prediction` 모듈은 `anomaly_detection` 모듈과 동일한 구조의 `data_source.py` 및 `_config.json`을 기반으로 작성되었습니다. CSV 모드는 정상 동작 중이나, DB 모드에서 `anomaly_detection`과 동일한 PostgreSQL 연동 문제가 존재하였습니다.

`anomaly_detection/docs/data_source_db_modification_report.md`에 문서화된 수정 사례를 참고하여 동일한 문제를 `peak_prediction`에도 적용하였습니다.

---

## 2. 발견된 문제 (6종)

| # | 구분 | 심각도 | 문제 설명 |
|---|------|--------|----------|
| P1 | `_config.json` | Critical | DB 접속 정보 오류 (database명, user, password) |
| P2 | `create_data_source()` | Critical | 비밀번호 특수문자 URL 인코딩 누락 (`ai123!@#`의 `@`, `#`) |
| P3 | `create_data_source()` | Critical | `search_path` 미지정 — `bems` 스키마 테이블 참조 불가 |
| P4 | `read_enabled_devices()` | Critical | SQL 컬럼명 대문자+큰따옴표 — PostgreSQL 소문자 컬럼과 불일치 |
| P5 | `read_sensor_data()` | Critical | SQL 컬럼명 대문자+큰따옴표 — 양 분기 모두 동일 문제 |
| P6 | `write_peak_result()` | High | `to_sql()` 사용 — 스키마 미지정, 대문자 컬럼명, 하루 중복 행 발생 |

---

## 3. 수정 내용

### 3-1. `_config.json` — 7항목

| # | 항목 | 수정 전 | 수정 후 | 관련 문제 |
|---|------|---------|---------|----------|
| C1 | `db.database` | `"bems"` | `"bemsdb"` | P1 |
| C2 | `db.user` | `"ai_user"` | `"ai"` | P1 |
| C3 | `db.password` | `"changeme"` | `"ai123!@#"` | P1 |
| C4 | `db.schema` | (없음) | `"bems"` 신규 추가 | P3 |
| C5 | `data.collection_table` | `"DATA_COLEC_H"` | `"data_colec_h"` | P4, P5 |
| C6 | `peak.config_table` | `"DEV_USE_PURP_REL_R"` | `"dev_use_purp_rel_r"` | P4 |
| C7 | `peak.result_table` | `"MAX_DMAND_FCST_H"` | `"max_dmand_fcst_h"` | P6 |

수정 후 `db` 섹션:

```json
"db": {
    "host": "localhost",
    "port": 5432,
    "database": "bemsdb",
    "schema": "bems",
    "user": "ai",
    "password": "ai123!@#"
}
```

---

### 3-2. `create_data_source()` — 3항목 (D1~D3)

**수정 전:**

```python
url = (
    f"postgresql://{db_cfg['user']}:{db_cfg['password']}"
    f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}"
)
engine = create_engine(url)
```

**수정 후:**

```python
from urllib.parse import quote_plus

schema = db_cfg.get("schema", "public")
encoded_password = quote_plus(db_cfg["password"])
url = (
    f"postgresql://{db_cfg['user']}:{encoded_password}"
    f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}"
)
engine = create_engine(
    url,
    connect_args={"options": f"-c search_path={schema}"},
    pool_pre_ping=True,
)
```

- **(D1)** `quote_plus()` — 비밀번호 `ai123!@#`의 `@`, `#` 문자가 URL 구문과 충돌하는 문제 방지
- **(D2)** `connect_args={"options": "-c search_path=bems"}` — 연결 시 `bems` 스키마 자동 설정, SQL에서 `bems.table_name` 접두어 불필요
- **(D3)** `pool_pre_ping=True` — cron 간헐적 실행 환경에서 연결 유효성 사전 확인

---

### 3-3. `read_enabled_devices()` — 2항목 (D4~D5)

**수정 전:**

```python
query = (
    f'SELECT "BLDG_ID" AS bldg_id, "DEV_ID" AS dev_id '
    f'FROM "{table}" '
    f"WHERE \"FALT_PRCV_YN\" = 'Y'"
)
df = pd.read_sql(query, source)
peak_dev_id = str(config["peak"]["dev_id"])
df = df[df["dev_id"].astype(str) == peak_dev_id]
devices = df.to_dict(orient="records")
```

**수정 후:**

```python
query = (
    f"SELECT DISTINCT bldg_id, dev_id "
    f"FROM {table} "
    f"WHERE falt_prcv_yn = 'Y'"
)
df = pd.read_sql(query, source)
peak_dev_id = int(config["peak"]["dev_id"])
df = df[df["dev_id"].astype(int) == peak_dev_id]
devices = [
    {"bldg_id": r["bldg_id"], "dev_id": int(r["dev_id"])}
    for r in df.to_dict(orient="records")
]
```

- **(D4)** 컬럼명/테이블명 소문자 + 큰따옴표 제거 — PostgreSQL은 따옴표 없는 식별자를 소문자로 처리
- **(D5)** `DISTINCT` 추가 — `dev_use_purp_rel_r` PK가 `(bldg_id, use_purp_id, dev_id)`이므로 같은 `(bldg_id, dev_id)`가 중복 존재 가능; `dev_id`를 `int()`로 타입 정규화 (DB: varchar → 코드: int)

---

### 3-4. `read_sensor_data()` — 2항목 (D6~D7)

**수정 전 (두 분기 공통):**

```python
f'SELECT "COLEC_DT" AS colec_dt, "COLEC_VAL" AS colec_val '
f'FROM "{table}" '
f'WHERE "BLDG_ID" = %(bldg_id)s '
f'  AND "DEV_ID" = %(dev_id)s '
f'  AND "TAG_CD" = %(tag_cd)s '
f'  AND "COLEC_DT" >= %(cutoff)s '
f'ORDER BY "COLEC_DT" ASC'
```

**수정 후 (fetch_hours > 0 분기):**

```python
f"SELECT colec_dt, colec_val "
f"FROM {table} "
f"WHERE bldg_id = %(bldg_id)s "
f"  AND dev_id = %(dev_id)s "
f"  AND tag_cd = %(tag_cd)s "
f"  AND colec_dt >= %(cutoff)s "
f"ORDER BY colec_dt ASC"
```

**수정 후 (fetch_hours == 0 분기):**

```python
f"SELECT colec_dt, colec_val "
f"FROM {table} "
f"WHERE bldg_id = %(bldg_id)s "
f"  AND dev_id = %(dev_id)s "
f"  AND tag_cd = %(tag_cd)s "
f"ORDER BY colec_dt ASC"
```

- **(D6)** fetch_hours > 0 분기: 모든 식별자 소문자화, 큰따옴표 제거, `AS` 별칭 제거 (컬럼명이 이미 소문자이므로 불필요)
- **(D7)** fetch_hours == 0 분기: 동일 적용

---

### 3-5. `write_peak_result()` — 1항목 (D8)

**수정 전:**

```python
table = config["peak"]["result_table"]  # MAX_DMAND_FCST_H
row = pd.DataFrame([{
    "USE_DT": now,
    "BLDG_ID": bldg_id,
    "DLY_MAX_DMAND_FCST_INF": peak_info,
}])
row.to_sql(table, source, if_exists="append", index=False)
```

**수정 후:**

```python
from sqlalchemy import text

table = config["peak"]["result_table"]  # max_dmand_fcst_h
params = {
    "use_dt":    now,
    "bldg_id":   bldg_id,
    "peak_info": peak_info,
}
delete_sql = text(
    f"DELETE FROM {table} "
    f"WHERE bldg_id = :bldg_id AND DATE(use_dt) = DATE(:use_dt)"
)
insert_sql = text(
    f"INSERT INTO {table} (use_dt, bldg_id, dly_max_dmand_fcst_inf) "
    f"VALUES (:use_dt, :bldg_id, :peak_info)"
)
with source.begin() as conn:
    conn.execute(delete_sql, params)
    conn.execute(insert_sql, params)
```

- **(D8)** `pd.DataFrame.to_sql()` → `sqlalchemy.text()` raw SQL (DELETE + INSERT) 교체
- `to_sql()`의 문제: 스키마 미지정(search_path 설정이 있어도 `to_sql`은 별도 처리 필요), 대문자 컬럼명 사용, 매시간 실행 시 하루에 24건의 중복 행 누적
- **DELETE 조건**: `bldg_id + DATE(use_dt) = DATE(:use_dt)` — 오늘 날짜 기준 기존 예측값 삭제 후 최신값 INSERT → 웹 UI의 일별 1건 표시 구조에 대응
- `source.begin()` 컨텍스트 매니저로 트랜잭션 보장 (DELETE/INSERT 원자성)

---

## 4. anomaly_detection과의 차이점

`anomaly_detection`과 `peak_prediction`의 수정 패턴은 대부분 동일하나, `write` 함수의 DELETE 조건이 다릅니다.

| 항목 | anomaly_detection | peak_prediction |
|------|-------------------|-----------------|
| 결과 테이블 | `falt_prcv_fcst` | `max_dmand_fcst_h` |
| 결과 테이블 PK | `(use_dt, bldg_id, dev_id)` | `(use_dt, bldg_id)` |
| DELETE 조건 | `bldg_id = :bldg_id AND dev_id = :dev_id` | `bldg_id = :bldg_id AND DATE(use_dt) = DATE(:use_dt)` |
| 유지 단위 | 설비당 영구 1건 (이상감지 상태 최신화) | 건물당 날짜별 1건 (매시간 예측 갱신) |
| 쓰기 컬럼 | `ad_score`, `ad_desc` | `dly_max_dmand_fcst_inf` |
| 설정 플래그 | `falt_prcv_yn` | `falt_prcv_yn` (별도 peak 플래그 없음) |

> **DELETE 조건 설계 근거**: `max_dmand_fcst_h`는 이력 테이블(`_h` 접미어)이지만, 웹 UI가 일별 1건의 AI 예측값을 표시합니다. `ai_peak_runner.py`는 매시간 cron으로 실행되어 하루에 24건이 쌓일 수 있으므로, 오늘 날짜(`DATE(use_dt)`)에 해당하는 기존 행을 삭제하고 최신 예측값으로 교체합니다.

---

## 5. 수정 결과 체크리스트

### 5-1. `_config.json` — 7항목

| # | 항목 | 상태 |
|---|------|------|
| C1 | `db.database`: `"bems"` → `"bemsdb"` | 완료 |
| C2 | `db.user`: `"ai_user"` → `"ai"` | 완료 |
| C3 | `db.password`: `"changeme"` → `"ai123!@#"` | 완료 |
| C4 | `db.schema`: `"bems"` 신규 추가 | 완료 |
| C5 | `data.collection_table`: `"DATA_COLEC_H"` → `"data_colec_h"` | 완료 |
| C6 | `peak.config_table`: `"DEV_USE_PURP_REL_R"` → `"dev_use_purp_rel_r"` | 완료 |
| C7 | `peak.result_table`: `"MAX_DMAND_FCST_H"` → `"max_dmand_fcst_h"` | 완료 |

### 5-2. `data_source.py` — 4개 함수, 8항목

| # | 함수 | 수정 내용 | 상태 |
|---|------|----------|------|
| D1 | `create_data_source()` | `quote_plus()` 비밀번호 인코딩 | 완료 |
| D2 | `create_data_source()` | `connect_args` search_path=bems 추가 | 완료 |
| D3 | `create_data_source()` | `pool_pre_ping=True` 추가 | 완료 |
| D4 | `read_enabled_devices()` | SQL 소문자화 + 큰따옴표 제거 + DISTINCT | 완료 |
| D5 | `read_enabled_devices()` | `dev_id` → `int()` 타입 정규화 | 완료 |
| D6 | `read_sensor_data()` | SQL 소문자화 (fetch_hours > 0 분기) | 완료 |
| D7 | `read_sensor_data()` | SQL 소문자화 (fetch_hours == 0 분기) | 완료 |
| D8 | `write_peak_result()` | `to_sql()` → DELETE+INSERT raw SQL | 완료 |
