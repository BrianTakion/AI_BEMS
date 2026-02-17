# CSV → PostgreSQL DB 전환 사전 검토 보고서

## 1. PostgreSQL DB 모드로 전환 시 사전 검토 사항

코드 전체를 분석한 결과, CSV → DB 전환 시 다음과 같은 잠재적 문제들이 식별됩니다.

### 1-1. `to_sql` 컬럼명 대소문자 문제 (심각도: **높음**)

`write_anomaly_result()` DB 모드에서 `pandas.to_sql()`을 사용합니다.

```python
# data_source.py, write_anomaly_result() DB 모드
if mode == "db":
    table = config["anomaly"]["result_table"]  # FALT_PRCV_FCST
    row = pd.DataFrame(
        [
            {
                "USE_DT": now,
                "BLDG_ID": bldg_id,
                "DEV_ID": str(dev_id),
                "AD_SCORE": ad_score,
                "AD_DESC": ad_desc,
            }
        ]
    )
    row.to_sql(table, source, if_exists="append", index=False)
```

**문제**: PostgreSQL은 따옴표 없는 식별자를 **소문자로 정규화**합니다. `to_sql()`은 기본적으로 따옴표를 붙이지 않으므로:
- DataFrame 컬럼 `"USE_DT"`, `"BLDG_ID"` 등이 PostgreSQL에서 `use_dt`, `bldg_id`로 해석됩니다.
- 실제 DB 테이블이 `"USE_DT"` (대문자 쌍따옴표)로 생성되어 있다면 **컬럼을 찾지 못해 에러**가 발생합니다.
- 반대로 테이블이 없는 상태에서 `to_sql`이 자동 생성하면, 소문자 컬럼명으로 테이블이 만들어져 다른 시스템(BEMS 플랫폼)과 호환되지 않습니다.

**해결 방향**: SQL INSERT 문을 직접 작성하여 컬럼명을 쌍따옴표로 감싸거나, DataFrame 컬럼명을 소문자로 통일하고 DB 테이블도 소문자로 생성하는 방식 중 하나를 선택해야 합니다.

### 1-2. 시간 기준점 차이 (심각도: **높음**)

```python
# data_source.py, read_sensor_data() DB 모드
if mode == "db":
    if fetch_hours > 0:
        cutoff = datetime.now() - timedelta(hours=fetch_hours)

# data_source.py, read_sensor_data() CSV 모드
if fetch_hours > 0:
    cutoff = df["colec_dt"].iloc[-1] - timedelta(hours=fetch_hours)
```

**문제**:
- **CSV 모드**: 데이터의 **마지막 타임스탬프** 기준으로 176시간 역산 → 항상 충분한 데이터가 확보됨
- **DB 모드**: `datetime.now()` 기준으로 176시간 역산 → 데이터 수집 지연이 있으면 최근 데이터가 부족하거나, 서버 시간대 설정이 다르면 엉뚱한 범위를 조회할 수 있음

특히 **서버 timezone**과 **DB의 COLEC_DT timezone**이 불일치하면, 의도한 것보다 더 적거나 많은 데이터가 조회될 수 있습니다.

### 1-3. `dev_id` 타입 불일치 (심각도: **중간**)

```python
# data_source.py, read_enabled_devices() CSV 모드
df = pd.read_csv(abs_path, dtype={"BLDG_ID": str, "DEV_ID": int, "FALT_PRCV_YN": str})
devices = [{"bldg_id": row["BLDG_ID"], "dev_id": row["DEV_ID"]} for _, row in df.iterrows()]
```

- CSV 모드: `dev_id`가 **int** (2001)
- DB 모드: `DEV_ID`가 `VARCHAR2(10)` → `pd.read_sql`로 읽으면 **str** ("2001")

현재 코드에서 `f"{dev_id}.txt"` 등으로 사용할 때는 문제없지만, 향후 타입 비교나 로직 확장 시 문제가 될 수 있습니다.

### 1-4. `DATA_COLEC_H` 테이블 스키마 부재 (심각도: **중간**)

`docs/PostgreSQL_DB_design.tsv`에 `FALT_PRCV_FCST`, `DEV_USE_PURP_REL_R` 등은 정의되어 있지만, **`DATA_COLEC_H` 테이블 스키마가 없습니다**. 코드에서는 `COLEC_DT`, `COLEC_VAL`, `BLDG_ID`, `DEV_ID`, `TAG_CD` 컬럼을 사용하지만, 실제 DB 테이블의 컬럼명과 데이터 타입이 일치하는지 확인이 필요합니다.

### 1-5. PK 중복 위험 (심각도: **중간**)

DB 스키마에서 `FALT_PRCV_FCST` 테이블의 PK는 `(USE_DT, BLDG_ID, DEV_ID)`입니다.

```python
# data_source.py, write_anomaly_result() DB 모드
row = pd.DataFrame(
    [
        {
            "USE_DT": now,  # datetime.now()
            ...
        }
    ]
)
row.to_sql(table, source, if_exists="append", index=False)
```

`USE_DT`에 `datetime.now()`를 사용하므로 마이크로초까지 포함되어 PK 충돌 가능성은 낮지만:
- cron이 겹쳐서 동시 실행되거나
- DB가 TIMESTAMP를 초 단위로 절삭하는 경우
- 수동 재실행 시

**PK 위반 에러**가 발생할 수 있으며, 이 경우 전체 에러로 처리됩니다 (UPSERT 로직 없음).

### 1-6. Cron 실행 환경 (심각도: **중간**)

README의 cron 설정:
```bash
0 * * * * cd /path/to/YiUmGoV2 && python ai_anomaly_runner.py >> /var/log/ai_anomaly.log 2>&1
```

- `python`이 시스템 파이썬을 가리킬 수 있음 → 가상환경의 절대경로 필요 (예: `/home/user/venv/bin/python`)
- cron은 최소한의 환경변수만 제공 → `PATH`, `LD_LIBRARY_PATH` 등이 누락될 수 있음
- `/var/log/ai_anomaly.log`에 대한 쓰기 권한 확인 필요

### 1-7. DB 연결 보안 및 안정성 (심각도: **낮음~중간**)

- `_config.json`에 DB 비밀번호가 평문 저장됨 → 환경변수 또는 시크릿 관리자 사용 권장
- DB 연결 실패 시 재시도 로직이 없음 → 네트워크 불안정 시 전체 실행 실패
- SQLAlchemy 엔진 생성 후 명시적 `dispose()` 호출 없음 (프로세스 종료 시 자동 해제되므로 cron에서는 큰 문제 아님)

### 1-8. `read_enabled_devices` 쿼리에 누락된 조건 (심각도: **낮음**)

```python
# data_source.py, read_enabled_devices() DB 모드
query = (
    f'SELECT "BLDG_ID" AS bldg_id, "DEV_ID" AS dev_id '
    f'FROM "{table}" '
    f"WHERE \"FALT_PRCV_YN\" = 'Y'"
)
```

DB 스키마의 `DEV_USE_PURP_REL_R` 테이블에는 `USE_PURP_ID`도 PK에 포함됩니다. 동일한 `(BLDG_ID, DEV_ID)` 조합이 다른 `USE_PURP_ID`로 여러 행 존재할 수 있으므로, **중복 디바이스가 반환**될 가능성이 있습니다. `DISTINCT` 추가를 고려해야 합니다.

---

## 2. `--csv` 성공 시 DB 모드 성공 보장 여부

**결론: 보장되지 않습니다.**

두 모드의 실행 흐름을 비교하면:

| 단계 | `--csv` (CSV 모드) | DB 모드 (플래그 없음) | 공유 여부 |
|------|--------------------|-----------------------|-----------|
| Config 로드 | 동일 | 동일 | **공유** |
| 데이터 소스 생성 | `config` 반환 | SQLAlchemy 엔진 생성 | **분리** |
| 디바이스 목록 조회 | CSV 파일 읽기 | SQL SELECT | **분리** |
| 센서 데이터 조회 | CSV chunked 읽기 | SQL SELECT | **분리** |
| 시간 기준점 | 데이터 마지막 시점 기준 | `datetime.now()` 기준 | **분리** |
| 전처리 | `preprocess()` | `preprocess()` | **공유** |
| 추론 | `run_inference()` | `run_inference()` | **공유** |
| 점수 계산 | `compute_ad_score()` | `compute_ad_score()` | **공유** |
| 결과 저장 | CSV 파일 append | `to_sql()` INSERT | **분리** |

**핵심 로직 (전처리 → 추론 → 점수 계산)은 완전히 동일한 코드를 공유**합니다. `data_preprocessing.preprocess()`, `infer_anomaly.run_inference()`, `infer_anomaly.compute_ad_score()`는 모드에 관계없이 같은 함수가 호출됩니다.

그러나 **데이터 액세스 레이어(`data_source.py`)는 완전히 다른 코드 경로**를 타므로, CSV 성공이 DB 성공을 보장하지 않습니다. 구체적으로:

1. **DB 연결 자체가 실패**할 수 있음 (잘못된 접속 정보, 네트워크, 패키지 미설치)
2. **SQL 쿼리 오류**가 발생할 수 있음 (테이블/컬럼 부재, 대소문자 문제)
3. **반환 데이터 형식 차이** — CSV에서는 `dev_id`가 int, DB에서는 str
4. **시간 기준점 차이**로 인해 데이터 부족 → `Insufficient data` 스킵 가능
5. **결과 저장 실패** — `to_sql()` 대소문자 문제, PK 중복 등

---

## 3. 권장 검증 단계

DB 모드 전환 전에 다음을 순차적으로 검증하는 것이 좋습니다:

1. **단위 DB 연결 테스트**: `create_data_source()` → 엔진 생성 성공 확인
2. **디바이스 조회 테스트**: `read_enabled_devices()` → 올바른 목록 반환 확인
3. **센서 데이터 조회 테스트**: `read_sensor_data()` → 컬럼명/타입/데이터량 확인
4. **결과 저장 테스트**: `write_anomaly_result()` → INSERT 성공 확인 (특히 컬럼명 대소문자)
5. **전체 파이프라인 테스트**: `python ai_anomaly_runner.py` → 1개 디바이스로 end-to-end 확인

특히 **1-1 (to_sql 대소문자 문제)**과 **1-2 (시간 기준점 차이)**는 프로덕션에서 거의 확실하게 문제를 일으킬 부분이므로, DB 준비 시점에 코드 수정이 필요합니다.
