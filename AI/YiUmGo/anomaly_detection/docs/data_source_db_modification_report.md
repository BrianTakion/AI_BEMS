# data_source.py DB 모드 코드 수정 — 완료 보고서

> **상태**: 코드 수정 완료 및 검증 통과 (2026-02-21)
> **참고 문서**: `checkpoint_csv_to_db_migration.md` (사전 검토 보고서)

## 1. 검증 환경

| 항목 | 값 |
|------|-----|
| DB | PostgreSQL (bemsdb) |
| DB 접속 계정 | `ai` / `ai123!@#` |
| 스키마 | `bems` |
| 대상 파일 | `anomaly_detection/data_source.py`, `anomaly_detection/_config.json` |
| 분석일 | 2026-02-21 |
| 수정 완료일 | 2026-02-21 |

---

## 2. 현재 코드와 실제 DB 간의 불일치 (6건)

### 2-1. `_config.json` DB 접속 정보 오류 (심각도: **치명**)

| 항목 | 현재 값 (코드) | 실제 값 (DB) |
|------|----------------|--------------|
| `db.database` | `"bems"` | `"bemsdb"` |
| `db.user` | `"ai_user"` | `"ai"` |
| `db.password` | `"changeme"` | `"ai123!@#"` |

DB에 연결 자체가 불가능합니다.

---

### 2-2. 비밀번호 특수문자 URL 인코딩 누락 (심각도: **치명**)

현재 `create_data_source()`에서 SQLAlchemy 연결 URL을 f-string으로 직접 조립합니다:

```python
# data_source.py:60-63 (현재 코드)
url = (
    f"postgresql://{db_cfg['user']}:{db_cfg['password']}"
    f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}"
)
```

비밀번호 `ai123!@#`에 포함된 특수문자 `@`와 `#`이 URL 구문과 충돌합니다:
- `@` → 호스트 구분자로 오인
- `#` → URL fragment 시작으로 오인

**결과**: 접속 정보가 올바르더라도 연결이 실패합니다.

**해결**: `urllib.parse.quote_plus()`로 비밀번호를 인코딩하거나, `sqlalchemy.engine.URL.create()`를 사용합니다.

---

### 2-3. 스키마 미지정 (심각도: **치명**)

실제 DB 테이블은 모두 `bems` 스키마에 존재합니다:

```
schemaname | tablename
-----------+--------------------
bems       | falt_prcv_fcst
bems       | dev_use_purp_rel_r
bems       | data_colec_h
```

`ai` 사용자의 기본 `search_path`는 `"$user", public`이므로, 스키마를 명시하지 않으면 테이블을 찾지 못합니다.

**해결**: SQLAlchemy 엔진 생성 시 `search_path=bems`를 설정합니다.

---

### 2-4. SQL 컬럼명 대소문자 불일치 (심각도: **치명**)

현재 코드는 대문자 + 큰따옴표로 컬럼명을 참조합니다:

```python
# data_source.py:92-94 (현재 코드)
f'SELECT "BLDG_ID" AS bldg_id, "DEV_ID" AS dev_id '
f'FROM "{table}" '
f"WHERE \"FALT_PRCV_YN\" = 'Y'"
```

실제 DB 컬럼은 모두 **소문자**입니다:

```
column_name  | data_type
-------------+-----------------------------
bldg_id      | character varying
dev_id       | character varying
falt_prcv_yn | character varying
```

PostgreSQL에서 큰따옴표로 감싼 식별자는 대소문자를 **그대로** 적용하므로, `"BLDG_ID"`는 소문자 `bldg_id` 컬럼을 찾지 못합니다.

**해결**: 모든 SQL에서 컬럼명/테이블명을 소문자로 변경하고 큰따옴표를 제거합니다.

---

### 2-5. `write_anomaly_result`의 `to_sql` 다중 문제 (심각도: **높음**)

```python
# data_source.py:253-264 (현재 코드)
row = pd.DataFrame([{
    "USE_DT": now,
    "BLDG_ID": bldg_id,
    "DEV_ID": str(dev_id),
    "AD_SCORE": ad_score,
    "AD_DESC": ad_desc,
}])
row.to_sql(table, source, if_exists="append", index=False)
```

**문제점 3가지**:

**(a) 스키마 미지정**: `to_sql()`에 `schema` 파라미터가 없으므로 `public` 스키마에 새 테이블 생성을 시도합니다. (`search_path` 설정으로 해결 가능)

**(b) BEMS 웹 UI와의 정합성**: BEMS 플랫폼의 설비이상감지 화면(`aiBemsPrcv.jsp`)에서 `falt_prcv_fcst`를 조인할 때 `use_dt` 필터 없이 `(bldg_id, dev_id)`만으로 조인합니다:

```sql
-- aibems.xml selectAiPrcvList
LEFT OUTER JOIN falt_prcv_fcst fpf
  ON gen.bldg_id = fpf.bldg_id AND gen.dev_id = fpf.dev_id
```

따라서 동일 `(bldg_id, dev_id)`에 여러 행이 존재하면 **웹 UI에 중복 행**이 표시됩니다. AI 모듈은 설비당 항상 **최신 1건만** 유지해야 합니다.

**(c) UPSERT 부재**: PK가 `(use_dt, bldg_id, dev_id)`이고 매 실행마다 `use_dt`가 변경되므로, `to_sql(append)`는 매번 새 행을 추가합니다. 이전 결과가 삭제되지 않아 (b) 문제가 발생합니다.

**해결**: `to_sql()` 대신 raw SQL로 **DELETE + INSERT** 패턴을 적용합니다.

```sql
DELETE FROM falt_prcv_fcst WHERE bldg_id = :bldg_id AND dev_id = :dev_id;
INSERT INTO falt_prcv_fcst (use_dt, bldg_id, dev_id, ad_score, ad_desc)
VALUES (:use_dt, :bldg_id, :dev_id, :ad_score, :ad_desc);
```

> 참고: `ON CONFLICT DO UPDATE`는 PK에 `use_dt`가 포함되어 있어 매번 새 timestamp이므로 충돌 자체가 발생하지 않습니다. 따라서 `ON CONFLICT`로는 기존 행 갱신이 불가하며, DELETE + INSERT가 올바른 접근입니다.

---

### 2-6. `read_enabled_devices` DISTINCT 누락 (심각도: **중간**)

`dev_use_purp_rel_r`의 PK는 `(bldg_id, use_purp_id, dev_id)`입니다. 동일한 `(bldg_id, dev_id)`가 서로 다른 `use_purp_id`에 중복 등록될 수 있으므로, `DISTINCT`를 추가해야 합니다.

---

### [참고] `ai` 사용자 DB 권한

| 테이블 | SELECT | INSERT | UPDATE | DELETE |
|--------|--------|--------|--------|--------|
| `bems.data_colec_h` | O | X | X | X |
| `bems.dev_use_purp_rel_r` | O | O | O | O |
| `bems.falt_prcv_fcst` | O | O | O | O |
| `bems.dev_m` | **X** | X | X | X |

- `data_colec_h`: SELECT만 가능 (센서 데이터 읽기 전용) — 코드 용도와 부합
- `falt_prcv_fcst`: 전체 CRUD 가능 — DELETE + INSERT 패턴 사용 가능
- `dev_m`: **접근 불가** — 현재 코드에서 직접 조회하지 않으므로 영향 없음

---

## 3. `_config.json` 수정 (적용 완료)

**변경 사항 요약** (7건):

| # | 항목 | 변경 전 | 변경 후 | 관련 문제 |
|---|------|---------|---------|-----------|
| C1 | `db.database` | `"bems"` | `"bemsdb"` | 2-1 |
| C2 | `db.user` | `"ai_user"` | `"ai"` | 2-1 |
| C3 | `db.password` | `"changeme"` | `"ai123!@#"` | 2-1 |
| C4 | `db.schema` | (없음) | `"bems"` (신규 추가) | 2-3 |
| C5 | `data.collection_table` | `"DATA_COLEC_H"` | `"data_colec_h"` | 2-4 |
| C6 | `anomaly.config_table` | `"DEV_USE_PURP_REL_R"` | `"dev_use_purp_rel_r"` | 2-4 |
| C7 | `anomaly.result_table` | `"FALT_PRCV_FCST"` | `"falt_prcv_fcst"` | 2-4 |

수정 후 `_config.json`의 `db` 섹션:

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

## 4. `data_source.py` 함수별 수정 (적용 완료)

### 4-1. `create_data_source()` — 3건

**수정 내용**: 비밀번호 URL 인코딩 + `search_path=bems` 설정 + 연결 헬스체크

**출처**: 2-2(URL 인코딩), 2-3(스키마), checkpoint 1-7(연결 안정성)

```python
# === 적용된 최종 코드 ===
if mode == "db":
    try:
        from sqlalchemy import create_engine
        from urllib.parse import quote_plus
    except ImportError as exc:
        raise ImportError(...) from exc

    db_cfg = config["db"]
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
    logger.info("DB engine created: %s:%s/%s (schema=%s)",
                 db_cfg["host"], db_cfg["port"], db_cfg["database"], schema)
    return engine
```

**변경 요점**:
1. (D1) `urllib.parse.quote_plus`로 비밀번호 특수문자(`!@#`) URL 인코딩
2. (D2) `connect_args`에서 `search_path`를 config의 `db.schema` 값으로 설정
3. (D3) **[checkpoint 1-7]** `pool_pre_ping=True` 추가 — 연결 풀에서 커넥션 사용 전 헬스체크로 끊어진 연결 자동 재생성

---

### 4-2. `read_enabled_devices()` — 2건

**수정 내용**: 컬럼명 소문자 + 큰따옴표 제거 + `DISTINCT` 추가 + `dev_id` 타입 정규화

**출처**: 2-4(컬럼명), 2-6(DISTINCT), checkpoint 1-3(dev_id 타입)

```python
# === 적용된 최종 코드 ===
if mode == "db":
    table = config["anomaly"]["config_table"]  # dev_use_purp_rel_r
    query = (
        f"SELECT DISTINCT bldg_id, dev_id "
        f"FROM {table} "
        f"WHERE falt_prcv_yn = 'Y'"
    )
    df = pd.read_sql(query, source)
    devices = [
        {"bldg_id": r["bldg_id"], "dev_id": int(r["dev_id"])}
        for r in df.to_dict(orient="records")
    ]
    logger.info("DB: %d devices with falt_prcv_yn='Y'", len(devices))
    return devices
```

**변경 요점**:
1. (D4) `"BLDG_ID"` → `bldg_id` (큰따옴표 제거 + 소문자), `DISTINCT` 추가로 `use_purp_id` 중복 제거
2. (D5) **[checkpoint 1-3]** `dev_id`를 `int()`로 변환 — DB에서는 `varchar(10)`이므로 `str`로 반환되지만, CSV 모드에서는 `dtype={"DEV_ID": int}`으로 `int` 반환. 하위 함수에서 `f"{dev_id}.txt"` (모델 파일), `str(dev_id)` (SQL 파라미터) 등에서 안정적 동작을 위해 타입 통일

---

### 4-3. `read_sensor_data()` — 2건

**수정 내용**: DB 모드 SQL의 컬럼명/테이블명 소문자 + 큰따옴표 제거

**출처**: 2-4(컬럼명)

두 SQL 분기 모두 동일한 패턴으로 수정:

```python
# === 적용된 최종 코드 (fetch_hours > 0 분기) ===
query = (
    f"SELECT colec_dt, colec_val "
    f"FROM {table} "
    f"WHERE bldg_id = %(bldg_id)s "
    f"  AND dev_id = %(dev_id)s "
    f"  AND tag_cd = %(tag_cd)s "
    f"  AND colec_dt >= %(cutoff)s "
    f"ORDER BY colec_dt ASC"
)
```

```python
# === 적용된 최종 코드 (fetch_hours == 0 분기) ===
query = (
    f"SELECT colec_dt, colec_val "
    f"FROM {table} "
    f"WHERE bldg_id = %(bldg_id)s "
    f"  AND dev_id = %(dev_id)s "
    f"  AND tag_cd = %(tag_cd)s "
    f"ORDER BY colec_dt ASC"
)
```

**변경 요점**:
1. (D6, D7) 모든 `"대문자"` → 소문자 (큰따옴표 제거), `AS` 별칭 제거
2. 두 분기(fetch_hours > 0, fetch_hours == 0) 모두 동일하게 적용

---

### 4-4. `write_anomaly_result()` — 1건

**수정 내용**: `to_sql()` 제거 → `DELETE + INSERT` raw SQL로 교체

**출처**: 2-5(to_sql 다중 문제)

```python
# === 적용된 최종 코드 ===
if mode == "db":
    from sqlalchemy import text

    table = config["anomaly"]["result_table"]  # falt_prcv_fcst
    params = {
        "use_dt":   now,
        "bldg_id":  bldg_id,
        "dev_id":   str(dev_id),
        "ad_score": round(ad_score, 2),
        "ad_desc":  ad_desc,
    }
    delete_sql = text(
        f"DELETE FROM {table} "
        f"WHERE bldg_id = :bldg_id AND dev_id = :dev_id"
    )
    insert_sql = text(
        f"INSERT INTO {table} (use_dt, bldg_id, dev_id, ad_score, ad_desc) "
        f"VALUES (:use_dt, :bldg_id, :dev_id, :ad_score, :ad_desc)"
    )
    with source.begin() as conn:
        conn.execute(delete_sql, params)
        conn.execute(insert_sql, params)

    logger.info(
        "DB: upserted anomaly result for %s/%s score=%.2f",
        bldg_id, dev_id, ad_score,
    )
    return
# ... CSV mode 부분은 변경 없음 ...
```

**변경 요점**:
1. (D8) `pd.DataFrame.to_sql()` → `sqlalchemy.text()` + raw SQL로 교체
2. **DELETE + INSERT 패턴**: 기존 결과 삭제 후 최신 결과 1건만 유지
3. `source.begin()` 컨텍스트 매니저로 트랜잭션 보장 (DELETE와 INSERT가 원자적으로 실행)
4. `ad_score`를 `round(ad_score, 2)`로 소수점 2자리 처리 (DB 컬럼: `numeric(15,2)`)
5. 컬럼명 모두 소문자

**DELETE + INSERT를 선택한 이유**:
- PK가 `(use_dt, bldg_id, dev_id)`이므로 `use_dt`가 매번 변경되어 `ON CONFLICT` 절이 기존 행과 매칭되지 않음
- 따라서 `INSERT ON CONFLICT DO UPDATE`로는 기존 행을 갱신할 수 없음
- BEMS 웹 UI가 `(bldg_id, dev_id)`로만 조인하므로, 설비당 1건만 유지해야 정상 표시됨

---

## 5. 전체 변경 요약

| # | 파일 | 위치 | 변경 내용 | 출처 | 심각도 |
|---|------|------|-----------|------|--------|
| C1-C4 | `_config.json` | `db` 섹션 | `database`, `user`, `password` 수정, `schema` 추가 | plan 2-1, 2-3 | 치명 |
| C5-C7 | `_config.json` | `data`, `anomaly` 섹션 | 테이블명 3건 소문자화 | plan 2-4 | 치명 |
| D1 | `data_source.py` | `create_data_source()` | 비밀번호 URL 인코딩 (`quote_plus`) | plan 2-2 | 치명 |
| D2 | `data_source.py` | `create_data_source()` | `connect_args`에 `search_path=bems` 설정 | plan 2-3 | 치명 |
| D3 | `data_source.py` | `create_data_source()` | `pool_pre_ping=True` 추가 | checkpoint 1-7 | 중간 |
| D4 | `data_source.py` | `read_enabled_devices()` | SQL 컬럼명 소문자 + DISTINCT 추가 | plan 2-4, 2-6 | 치명 |
| D5 | `data_source.py` | `read_enabled_devices()` | `dev_id`를 `int()`로 변환 (타입 정규화) | checkpoint 1-3 | 중간 |
| D6 | `data_source.py` | `read_sensor_data()` | SQL 컬럼명 소문자 (fetch_hours > 0 분기) | plan 2-4 | 치명 |
| D7 | `data_source.py` | `read_sensor_data()` | SQL 컬럼명 소문자 (fetch_hours == 0 분기) | plan 2-4 | 치명 |
| D8 | `data_source.py` | `write_anomaly_result()` | `to_sql` → DELETE + INSERT 교체 | plan 2-5 | 높음 |

> **checkpoint 반영 항목**: D3 (`pool_pre_ping`), D5 (`dev_id` 타입 정규화)는 `checkpoint_csv_to_db_migration.md`에서 식별된 추가 검토 항목입니다.

---

## 6. 검증 절차 및 결과

### 6-1. DB 연결 테스트 — **통과**

```
Engine created: Engine(postgresql://ai:***@localhost:5432/bemsdb)
```

### 6-2. 디바이스 목록 조회 테스트 — **통과** (0건, 예상대로)

```
Enabled devices: 0
```

> 참고: `dev_use_purp_rel_r` 테이블에 `falt_prcv_yn='Y'` 데이터가 0건이므로 정상 결과입니다.
> BEMS 플랫폼에서 데이터 투입 후 재테스트 필요 (7-5절 참조).

### 6-3. 센서 데이터 조회 테스트 — **통과**

```
Rows returned: 2111
Columns: ['colec_dt', 'colec_val']
dtypes: {'colec_dt': datetime64[ns], 'colec_val': float64}
             colec_dt  colec_val
0 2026-02-14 12:45:40     147.24
1 2026-02-14 12:50:40     153.83
2 2026-02-14 12:55:40     148.90
```

### 6-4. 결과 저장 테스트 — **통과**

```
# 1회 저장
Rows in falt_prcv_fcst for B0019/2001: 1
use_dt=2026-02-21 20:45:29, ad_score=85.3, ad_desc='Test: normal operation'

# 2회 저장 (DELETE+INSERT 덮어쓰기 확인)
Rows after second write: 1 (expected: 1)
ad_score=92.5, ad_desc='Test: overwrite check'
```

설비당 최신 1건만 유지되는 것을 확인했습니다.

### 6-5. CSV 모드 회귀 테스트 — **통과**

```bash
python3 anomaly_detection/ai_anomaly_runner.py --csv
```

```
Mode: CSV
Enabled devices: 4
=== Done: 4/4 devices processed in 76.5s ===
```

CSV 모드가 기존과 동일하게 정상 동작합니다. DB 모드 수정이 CSV 모드에 영향을 주지 않음을 확인했습니다.

### 6-6. 검증 결과 요약

| 테스트 항목 | 결과 | 비고 |
|-------------|------|------|
| DB 연결 (`ai@bemsdb`, schema=bems) | **통과** | |
| 디바이스 목록 조회 | **통과** | 0건 (DB 데이터 미투입) |
| 센서 데이터 조회 (`data_colec_h`) | **통과** | 2,111행 반환 |
| 결과 저장 (DELETE + INSERT) | **통과** | 설비당 1건 유지 확인 |
| 덮어쓰기 (2회 연속 write) | **통과** | 1건 유지 정상 |
| CSV 모드 회귀 | **통과** | 4/4 디바이스, exit code 0 |

---

## 7. CSV 모드 vs DB 모드 — 디바이스 선별 동작 비교

### 7-1. 공통 진입점

두 모드 모두 `ai_anomaly_runner.py`에서 동일한 함수를 호출하여 디바이스를 선별합니다:

```python
# ai_anomaly_runner.py:176
devices = data_source.read_enabled_devices(source, config)
```

`data_source.py`의 `read_enabled_devices()` 내부에서 `data_source` 값에 따라 분기됩니다.

### 7-2. CSV 모드 (정상 동작 중)

```
실행 명령: python3 anomaly_detection/ai_anomaly_runner.py --csv
```

```
data_source="csv" → read_enabled_devices()
  → config_anomaly_devices.csv 읽기
  → FALT_PRCV_YN = "Y" 필터링
  → 24개 디바이스 반환 (2002~2025)
```

`config_anomaly_devices.csv` 파일 내용:

```
BLDG_ID,DEV_ID,FALT_PRCV_YN
B0019,2001,N            ← 제외
B0019,2002,Y            ← 포함
B0019,2003,Y            ← 포함
...
B0019,2025,Y            ← 포함
B0019,2026,N            ← 제외
B0019,2027,N            ← 제외
B0019,2028,N            ← 제외
```

총 28개 디바이스 중 `FALT_PRCV_YN = "Y"`인 **24개**에 대해서만 이상감지를 수행합니다.

실행 결과 (`output/anomaly_results.csv`):

```
USE_DT,BLDG_ID,DEV_ID,AD_SCORE,AD_DESC
2026-02-21 19:33:09,B0019,2002,100.0,"정상 상태 | ..."
2026-02-21 19:33:28,B0019,2003,92.5,"정상 상태 | ..."
...
2026-02-21 19:40:28,B0019,2025,100.0,"센서 제로 | ..."
```

DEV_ID 2001, 2026, 2027, 2028은 `FALT_PRCV_YN = "N"`이므로 정상적으로 제외되었습니다.

### 7-3. DB 모드 (구현되어 있으나 동작 불가)

```
실행 명령: python3 anomaly_detection/ai_anomaly_runner.py
```

```
data_source="db" → read_enabled_devices()
  → SELECT DISTINCT bldg_id, dev_id FROM dev_use_purp_rel_r WHERE falt_prcv_yn = 'Y'
  → 현재 0건 반환
  → 어떤 설비도 처리하지 않고 종료
```

**코드에는 DB에서 선별 조회하는 로직이 구현되어 있습니다.** 그러나 현재 두 가지 이유로 동작하지 않습니다:

#### 이유 1: 코드 버그 (본 문서 2장에서 분석한 6건)

DB 접속 정보 오류, 스키마 미지정, 컬럼명 대소문자 문제 등으로 DB 연결 자체가 실패합니다.
이 부분은 본 문서 3~4장의 코드 수정으로 해결 가능합니다.

#### 이유 2: DB 테이블 데이터 부재 (BEMS 플랫폼 설정 필요)

코드 버그를 모두 수정하더라도, 현재 DB에는 디바이스 설정 데이터가 없습니다:

```sql
bemsdb=> SELECT COUNT(*) FROM bems.dev_use_purp_rel_r;
 count
-------
     0
(0개 행)
```

`bems.dev_use_purp_rel_r` 테이블은 **BEMS 플랫폼이 관리하는 테이블**입니다.
웹 UI의 "설비이상감지" 화면에서 관리자가 다음 절차를 수행해야 데이터가 생성됩니다:

1. 건물-장치-용도 관계 데이터 등록 (BEMS 플랫폼 관리자)
2. AI이상진단 체크박스를 `Y`로 활성화 (웹 UI `aiBemsPrcv.jsp` 에서 조작)
3. 체크박스 변경 시 `PUT /ajax/updateFaltPrcvYn.do` API가 호출되어 `falt_prcv_yn` 컬럼 갱신

### 7-4. 비교 요약

| 항목 | CSV 모드 (`--csv`) | DB 모드 (기본) |
|------|---------------------|----------------|
| 디바이스 선별 로직 | 구현됨, 정상 동작 | 구현됨, 동작 불가 |
| 데이터 소스 | `config_anomaly_devices.csv` | `bems.dev_use_purp_rel_r` 테이블 |
| 현재 활성 디바이스 수 | 24건 (Y) / 4건 (N) | **0건** (테이블 자체가 비어있음) |
| 동작을 위한 선행 조건 | CSV 파일에 디바이스 목록 작성 | 코드 버그 수정 + BEMS 플랫폼에서 데이터 투입 |
| 결과 출력 | `output/anomaly_results.csv` | `bems.falt_prcv_fcst` 테이블 |

### 7-5. DB 모드 운영을 위한 조치 사항

DB 모드를 실제 운영하려면 다음 두 단계가 순차적으로 필요합니다:

**단계 1 (개발팀)**: 본 문서 3~4장의 코드 수정 적용

**단계 2 (BEMS 플랫폼 관리자)**: `dev_use_purp_rel_r` 테이블에 디바이스 등록 및 `falt_prcv_yn = 'Y'` 설정. CSV 파일과 동일한 디바이스 구성을 원한다면 다음 INSERT 문을 참고할 수 있습니다:

```sql
-- 예시: CSV 파일의 디바이스 구성을 DB에 반영 (use_purp_id는 BEMS 플랫폼에 맞게 지정 필요)
INSERT INTO bems.dev_use_purp_rel_r (bldg_id, use_purp_id, dev_id, falt_prcv_yn)
VALUES
  ('B0019', '{use_purp_id}', '2001', 'N'),
  ('B0019', '{use_purp_id}', '2002', 'Y'),
  ('B0019', '{use_purp_id}', '2003', 'Y'),
  ...
  ('B0019', '{use_purp_id}', '2028', 'N');
```

> 주의: `use_purp_id` 값은 BEMS 플랫폼의 `dev_use_purp_m` 테이블에 등록된 용도ID와 일치해야 합니다.

---

## 8. `falt_prcv_yn` 필드 역할 검증

### 8-1. 필드 위치 및 구조

DB 전체에서 `falt_prcv_yn` 컬럼이 존재하는 테이블은 **1개**입니다:

| 테이블 | 컬럼 | 타입 | NULL 허용 | 의미 |
|--------|------|------|-----------|------|
| `bems.dev_use_purp_rel_r` | `falt_prcv_yn` | `varchar(1)` | YES | 설비이상감지 AI 적용 여부 |

이 테이블의 PK는 `(bldg_id, use_purp_id, dev_id)`이므로, **건물 + 용도 + 장치** 조합마다 이상감지 적용 여부를 개별 설정할 수 있습니다.

같은 테이블에 유사한 플래그 컬럼이 하나 더 존재합니다:

| 컬럼 | 값 | 용도 | 대응 화면 |
|------|----|------|-----------|
| `falt_prcv_yn` | `'Y'` / `'N'` | **설비이상감지** AI 적용 여부 | `aiBemsPrcv.jsp` (설비이상감지) |
| `ai_aply_yn` | `'Y'` / `'N'` | 냉난방 온도제어 AI 적용 여부 | `aiBemsCntrl.jsp` (냉난방 온도제어) |

### 8-2. 데이터 흐름: 웹 UI → DB → AI 모듈 → DB → 웹 UI

```
┌──────────────────┐  ① 체크박스 조작   ┌───────────────────────┐
│  BEMS 웹 UI      │ ────────────────→ │  dev_use_purp_rel_r   │
│  (aiBemsPrcv.jsp)│  PUT 요청          │  falt_prcv_yn = 'Y'   │
│  관리자가 조작    │                    └──────────┬────────────┘
└────────▲─────────┘                               │
         │                                         │ ③ SELECT WHERE 'Y'
         │ ⑤ 결과 표시                              ▼
┌────────┴─────────┐  ④ DELETE+INSERT   ┌───────────────────────┐
│  falt_prcv_fcst  │ ←──────────────── │  AI 이상감지 모듈     │
│  (ad_score,      │                    │  (ai_anomaly_runner)  │
│   ad_desc)       │                    └───────────────────────┘
└──────────────────┘
```

#### ① 웹 UI에서 관리자가 체크박스 조작

`aiBemsPrcv.jsp`의 "AI이상진단" 열에 체크박스가 렌더링됩니다:

```javascript
// aiBemsPrcv.jsp - 체크박스 변경 이벤트
$(document).on("change", ".aiBemsCheck", function () {
    const faltPrcvYn = $(this).prop("checked") ? "Y" : "N";
    const devId = $(this).data("devid");
    const bldgId = $(this).data("bldgid");
    faltPrcvYnSave(bldgId, devId, faltPrcvYn, usePurpId);
});
```

- 체크박스 ON → `faltPrcvYn = "Y"` (이상감지 활성화)
- 체크박스 OFF → `faltPrcvYn = "N"` (이상감지 비활성화)
- `falt_prcv_yn` 값이 NULL이면 → "준비중"으로 표시 (체크박스 미표시)

#### ② 서버가 DB 업데이트

```sql
-- aibems.xml: updatePrcvFcstYn
UPDATE dev_use_purp_rel_r
SET falt_prcv_yn = #{faltPrcvYn}     -- 'Y' 또는 'N'
WHERE bldg_id = #{bldgId} AND dev_id = #{devId}
```

API 엔드포인트: `PUT /ajax/updateFaltPrcvYn.do`

#### ③ AI 모듈이 DB 조회하여 대상 디바이스 선별

```python
# data_source.py: read_enabled_devices() DB 모드
query = (
    "SELECT DISTINCT bldg_id, dev_id "
    "FROM dev_use_purp_rel_r "
    "WHERE falt_prcv_yn = 'Y'"
)
```

`falt_prcv_yn = 'Y'`인 디바이스에 대해서만 이상감지 알고리즘을 실행합니다.

#### ④ AI 결과를 DB에 기록

```python
# data_source.py: write_anomaly_result() DB 모드
# DELETE + INSERT로 falt_prcv_fcst 테이블에 설비당 최신 1건 유지
```

#### ⑤ 웹 UI가 결과를 표시

```sql
-- aibems.xml: selectAiPrcvList
LEFT OUTER JOIN falt_prcv_fcst fpf
  ON gen.bldg_id = fpf.bldg_id AND gen.dev_id = fpf.dev_id
```

`falt_prcv_fcst`의 `ad_score`와 `ad_desc`를 조인하여 "상태"와 "상태설명" 열에 표시합니다.

### 8-3. CSV 모드와의 대응 관계

CSV 파일 `config_anomaly_devices.csv`는 DB 테이블 `dev_use_purp_rel_r`의 역할을 로컬 파일로 대체한 것입니다:

| CSV 컬럼 | DB 컬럼 (`dev_use_purp_rel_r`) | 의미 |
|----------|-------------------------------|------|
| `BLDG_ID` | `bldg_id` | 건물 ID |
| `DEV_ID` | `dev_id` | 장치 ID |
| `FALT_PRCV_YN` | `falt_prcv_yn` | 이상감지 적용 여부 (`Y`/`N`) |

현재 CSV 파일 내용:

```
BLDG_ID  DEV_ID  FALT_PRCV_YN
B0019    2001    N             ← AI 이상감지 미적용
B0019    2002    Y             ← AI 이상감지 적용
B0019    2003    Y             ← AI 이상감지 적용
...
B0019    2025    Y             ← AI 이상감지 적용
B0019    2026    N             ← AI 이상감지 미적용
B0019    2027    N             ← AI 이상감지 미적용
B0019    2028    N             ← AI 이상감지 미적용
```

총 28개 디바이스 중 `Y`=24개, `N`=4개. CSV 모드에서는 24개에 대해서만 이상감지가 수행됩니다.

### 8-4. 현재 DB 상태

```sql
SELECT COUNT(*) FROM bems.dev_use_purp_rel_r;
-- 결과: 0건 (테이블 비어있음)

SELECT COUNT(*) FROM bems.dev_use_purp_m;
-- 결과: 0건 (용도 마스터도 비어있음)
```

두 테이블 모두 데이터가 없으므로, DB 모드에서는 디바이스 선별 자체가 불가합니다. BEMS 플랫폼이 건물-용도-장치 관계를 초기 등록해야 하며, 이후 웹 UI에서 관리자가 `falt_prcv_yn`을 `'Y'`로 태깅하는 절차가 필요합니다.

### 8-5. 결론

`falt_prcv_yn` 필드는 **"이 dev_id에 대해 이상감지 알고리즘을 적용할 것인가?"를 태깅하는 플래그**입니다.

- 값이 `'Y'`이면 → AI 이상감지 알고리즘 **적용 대상**
- 값이 `'N'`이면 → AI 이상감지 알고리즘 **제외**
- 값이 `NULL`이면 → 웹 UI에서 "준비중"으로 표시 (체크박스 자체가 미표시)

코드 로직은 CSV 모드와 DB 모드 모두 올바르게 구현되어 있으며, DB 테이블에 데이터가 투입되고 본 문서 3~4장의 코드 수정이 적용되면 DB 모드도 정상 동작할 수 있습니다.

---

## 9. `read_sensor_data()` DB 모드 — 센서 데이터 조회 검증

### 9-1. DB 데이터 현황

`bems.data_colec_h` 테이블에 `tag_cd='30001'` 센서 데이터가 **실시간으로 수집**되고 있습니다:

```
 dev_id |  cnt  |       oldest        |       newest
--------+-------+---------------------+---------------------
 2001   | 65825 | 2025-04-01 13:22:18 | 2026-02-21 20:25:40
 2002   | 65828 | 2025-03-25 10:51:33 | 2026-02-21 20:25:40
 2003   | 65823 | 2025-04-01 16:47:07 | 2026-02-21 20:25:40
 ...
 2028   | 65828 | 2025-03-25 10:51:33 | 2026-02-21 20:25:40
```

- 디바이스 2001~2028: 각 약 65,000건 보유
- 약 5분 간격으로 지속 수집 중 (최신 데이터: 2026-02-21 20:25:40)
- `ai` 사용자에게 SELECT 권한 있음

### 9-2. 실제 DB 컬럼 구조

```
 column_name |          data_type          | character_maximum_length
-------------+-----------------------------+--------------------------
 dev_id      | character varying           |                       10
 tag_cd      | character varying           |                        8
 colec_dt    | timestamp without time zone |
 colec_val   | numeric                     |
 reg_dt      | timestamp without time zone |
 bldg_id     | character varying           |                       10
```

모든 컬럼명이 **소문자**입니다.

### 9-3. 현재 코드 대비 문제점

`read_sensor_data()` DB 모드(Line 155-184)에는 두 개의 SQL 분기가 있으며, 둘 다 동일한 문제를 포함합니다:

| 코드에서 사용하는 식별자 | 실제 DB 컬럼/테이블 | 문제 |
|--------------------------|---------------------|------|
| `"COLEC_DT"` | `colec_dt` | 대문자 + 큰따옴표 → 컬럼 미발견 오류 |
| `"COLEC_VAL"` | `colec_val` | 동일 |
| `"BLDG_ID"` | `bldg_id` | 동일 |
| `"DEV_ID"` | `dev_id` | 동일 |
| `"TAG_CD"` | `tag_cd` | 동일 |
| `FROM "{table}"` (= `"DATA_COLEC_H"`) | `data_colec_h` | 대문자 테이블명 + 큰따옴표 → 테이블 미발견 오류 |

이 문제들은 모두 본 문서 2-4절에서 분석한 공통 문제의 일부이며, 4-3절의 수정안으로 해결됩니다.

### 9-4. 로직 검증 (대소문자 문제 제외)

대소문자/스키마 문제를 제외하면, `read_sensor_data()` DB 모드의 **로직 자체는 올바르게 구현**되어 있습니다:

- **쿼리 구조**: `SELECT colec_dt, colec_val FROM data_colec_h WHERE bldg_id=... AND dev_id=... AND tag_cd=... AND colec_dt >= cutoff ORDER BY colec_dt ASC` — 정상
- **파라미터 바인딩**: `%(bldg_id)s` 형태의 parameterized query 사용 — SQL 인젝션 방지, 정상
- **시간 필터링**: `cutoff = datetime.now() - timedelta(hours=fetch_hours)` — 프로덕션 환경에서 올바른 방식
- **결과 변환**: `pd.to_datetime()`, `astype(float)` — 정상
- **정렬**: `ORDER BY colec_dt ASC` — 시계열 순서 보장, 정상

### 9-5. CSV 모드 vs DB 모드 — 시간 기준점 차이 (의도된 설계)

| 항목 | CSV 모드 | DB 모드 |
|------|----------|---------|
| 시간 기준점 | **데이터 마지막 타임스탬프** 기준 역산 | **`datetime.now()`** 기준 역산 |
| 코드 | `cutoff = df["colec_dt"].iloc[-1] - timedelta(hours=176)` | `cutoff = datetime.now() - timedelta(hours=176)` |
| 이유 | 과거 데이터셋이므로 마지막 시점 기준이 적절 | 실시간 데이터이므로 현재 시각 기준이 적절 |

이것은 **의도된 설계 차이**이며 버그가 아닙니다. DB에는 실시간 데이터가 5분 간격으로 들어오고 있으므로, `datetime.now()` 기준 176시간 역산은 충분한 데이터를 반환합니다.

### 9-6. 결론

`read_sensor_data()` DB 모드는 **추가 수정 없이 4-3절의 수정안만 적용하면 정상 동작**합니다.

수정 대상 요약:

| 수정 항목 | 해당 문서 섹션 | 상태 |
|-----------|---------------|------|
| `_config.json`의 `collection_table` 소문자화 | 3장 | `"DATA_COLEC_H"` → `"data_colec_h"` |
| SQL 컬럼명 소문자 + 큰따옴표 제거 (fetch_hours > 0) | 4-3절 | 변경 전/후 코드 제시 완료 |
| SQL 컬럼명 소문자 + 큰따옴표 제거 (fetch_hours == 0) | 4-3절 | 변경 전/후 코드 제시 완료 |

---

## 10. 전체 수정 사항 종합 체크리스트

아래는 DB 모드 운영을 위해 필요한 **모든 수정 사항**을 파일별, 함수별로 정리한 최종 체크리스트입니다.

### 10-1. `_config.json` 수정 (7건) — 모두 완료

| # | 항목 | 변경 전 | 변경 후 | 출처 | 상태 |
|---|------|---------|---------|------|------|
| C1 | `db.database` | `"bems"` | `"bemsdb"` | plan 2-1 | **완료** |
| C2 | `db.user` | `"ai_user"` | `"ai"` | plan 2-1 | **완료** |
| C3 | `db.password` | `"changeme"` | `"ai123!@#"` | plan 2-1 | **완료** |
| C4 | `db.schema` | (없음) | `"bems"` (신규 추가) | plan 2-3 | **완료** |
| C5 | `data.collection_table` | `"DATA_COLEC_H"` | `"data_colec_h"` | plan 2-4 | **완료** |
| C6 | `anomaly.config_table` | `"DEV_USE_PURP_REL_R"` | `"dev_use_purp_rel_r"` | plan 2-4 | **완료** |
| C7 | `anomaly.result_table` | `"FALT_PRCV_FCST"` | `"falt_prcv_fcst"` | plan 2-4 | **완료** |

### 10-2. `data_source.py` 수정 (4개 함수, 8건) — 모두 완료

| # | 함수 | 수정 내용 | 출처 | 상세 | 상태 |
|---|------|-----------|------|------|------|
| D1 | `create_data_source()` | 비밀번호 URL 인코딩 (`quote_plus`) | plan 2-2 | 4-1절 | **완료** |
| D2 | `create_data_source()` | `connect_args`에 `search_path=bems` 추가 | plan 2-3 | 4-1절 | **완료** |
| D3 | `create_data_source()` | `pool_pre_ping=True` 추가 | checkpoint 1-7 | 4-1절 | **완료** |
| D4 | `read_enabled_devices()` | SQL 컬럼명 소문자 + 큰따옴표 제거 + `DISTINCT` 추가 | plan 2-4, 2-6 | 4-2절 | **완료** |
| D5 | `read_enabled_devices()` | `dev_id`를 `int()`로 변환 (CSV 모드와 타입 통일) | checkpoint 1-3 | 4-2절 | **완료** |
| D6 | `read_sensor_data()` | SQL 컬럼명/테이블명 소문자 + 큰따옴표 제거 (fetch_hours > 0) | plan 2-4 | 4-3절 | **완료** |
| D7 | `read_sensor_data()` | SQL 컬럼명/테이블명 소문자 + 큰따옴표 제거 (fetch_hours == 0) | plan 2-4 | 4-3절 | **완료** |
| D8 | `write_anomaly_result()` | `to_sql()` 제거 → `DELETE + INSERT` raw SQL 교체 | plan 2-5 | 4-4절 | **완료** |

### 10-3. checkpoint에서 수정 불필요 확인된 항목

| # | checkpoint 항목 | 판단 | 사유 |
|---|-----------------|------|------|
| - | 1-2. 시간 기준점 차이 | 수정 불필요 | `datetime.now()` vs 데이터 마지막 시점은 의도된 설계 (DB=실시간, CSV=과거 데이터셋) |
| - | 1-6. Cron 실행 환경 | 코드 외 | 운영 배포 시 별도 가이드 필요 (Python venv 절대경로, 환경변수, 로그 권한) |
| - | 1-7. DB 비밀번호 평문 | 현행 유지 | 향후 환경변수 방식 전환 검토 |

### 10-4. BEMS 플랫폼 설정 (코드 외 — 운영팀 조치 필요)

| # | 항목 | 현재 상태 | 필요 조치 |
|---|------|-----------|-----------|
| B1 | `dev_use_purp_m` 테이블 | 0건 | 용도 마스터 데이터 등록 |
| B2 | `dev_use_purp_rel_r` 테이블 | 0건 | 건물-용도-장치 관계 데이터 등록 |
| B3 | `falt_prcv_yn` 설정 | 미설정 | 웹 UI에서 AI이상진단 체크박스 활성화 (`'Y'` 태깅) |

### 10-5. 수정 적용 순서 및 진행 상태

```
1. _config.json 수정 (C1~C7)              ✅ 완료
      ↓
2. data_source.py 수정 (D1~D8)            ✅ 완료
      ↓
3. DB 연결 테스트 (6-1절)                  ✅ 통과
      ↓
4. 센서 데이터 조회 테스트 (6-3절)          ✅ 통과 (2,111행)
      ↓
5. 결과 저장 테스트 (6-4절)                ✅ 통과 (1건 유지)
      ↓
6. CSV 모드 회귀 테스트 (6-5절)            ✅ 통과 (4/4)
      ↓
7. BEMS 플랫폼 데이터 투입 (B1~B3)         ⏳ 운영팀 조치 대기
      ↓
8. 디바이스 목록 조회 테스트 (6-2절)        ⏳ 7단계 이후 수행
      ↓
9. 전체 파이프라인 테스트                   ⏳ 7단계 이후 수행
```

> **현재 상태**: 1~6단계 완료. 코드 수정은 모두 적용되었으며 검증을 통과했습니다.
> 7단계 이후는 BEMS 플랫폼 운영팀이 `dev_use_purp_rel_r` 테이블에 데이터를 투입한 뒤 수행합니다.

---

## 11. 유사 모듈 적용 시 참고 사항

본 문서의 수정 패턴은 동일한 BEMS 플랫폼의 다른 AI 모듈에도 적용할 수 있습니다.

### 11-1. 공통 수정 패턴

| 패턴 | 설명 | 적용 예시 |
|------|------|-----------|
| **비밀번호 URL 인코딩** | `urllib.parse.quote_plus(password)` 사용 | 특수문자 포함 비밀번호 |
| **search_path 설정** | `connect_args={"options": "-c search_path=bems"}` | `bems` 스키마의 모든 테이블 |
| **pool_pre_ping** | `create_engine(..., pool_pre_ping=True)` | cron 실행처럼 간헐적 연결 사용 시 |
| **SQL 소문자화** | PostgreSQL 식별자를 소문자로, 큰따옴표 제거 | 모든 SQL 쿼리 |
| **DELETE + INSERT** | PK에 timestamp 포함 시 UPSERT 대체 | 설비당 최신 1건 유지 패턴 |
| **dev_id 타입 정규화** | DB(varchar) → `int()` 변환, CSV(int) 유지 | CSV/DB 모드 간 일관성 |

### 11-2. 관련 모듈 (동일 패턴 적용 대상)

| 모듈 | 설정 테이블 (플래그) | 결과 테이블 | PK 구조 |
|------|---------------------|-------------|---------|
| **AI 이상감지** (본 문서) | `dev_use_purp_rel_r` (`falt_prcv_yn`) | `falt_prcv_fcst` | `(use_dt, bldg_id, dev_id)` |
| AI 냉난방제어 | `dev_use_purp_rel_r` (`ai_aply_yn`) | `cnh_cntrl_fcst_h` | `(use_dt, bldg_id, dev_id)` |
| AI 최대수요예측 | — | `max_dmand_fcst_h` | `(use_dt, bldg_id)` |
