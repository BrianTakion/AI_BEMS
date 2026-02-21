# BEMS 웹 UI "최대수요-AI예측(kW) @HH:MM" 미표시 원인 분석 및 조치 요청

- **작성일**: 2026-02-21
- **작성자**: AI 개발팀
- **대상**: DB / BEMS 플랫폼 운영팀
- **긴급도**: 높음 (AI 모듈은 정상 동작 중이나 웹 UI에 결과가 표시되지 않음)

---

## 1. 증상

`ai_peak_runner.py` (최대수요 예측 AI 모듈)를 실행하면 DB에 예측 결과가 정상 저장되지만, BEMS 웹 UI의 **"최대수요-AI예측(kW) @HH:MM"** 열에 값이 표시되지 않습니다.

---

## 2. AI 모듈 실행 결과: 정상

2026-02-21 22:30 실행 로그:

```
2026-02-21 22:30:23 [INFO] ai_peak_runner - === AI Peak Prediction Start ===
2026-02-21 22:30:23 [INFO] ai_peak_runner - Mode: DB
2026-02-21 22:30:23 [INFO] data_source - DB engine created: localhost:5432/bemsdb (schema=bems)
2026-02-21 22:30:23 [INFO] data_source - DB: 4030 rows for dev_id=2001 (336h)
2026-02-21 22:30:23 [INFO] ai_peak_runner - dev_id=2001 => 226.17@13:15
2026-02-21 22:30:23 [INFO] data_source - DB: upserted peak result for B0019: 226.17@13:15
2026-02-21 22:30:23 [INFO] ai_peak_runner - === Done: OK in 0.2s ===
```

---

## 3. DB 저장 상태 확인

### 3-1. `max_dmand_fcst_h` (예측 결과 테이블): 정상

```sql
SELECT * FROM bems.max_dmand_fcst_h WHERE bldg_id = 'B0019';
```

| use_dt | bldg_id | dly_max_dmand_fcst_inf |
|--------|---------|------------------------|
| 2026-02-21 22:30:23.78072 | B0019 | 226.17@13:15 |

데이터 1건이 정상적으로 저장되어 있습니다.

### 3-2. `dev_use_purp_m` (장치사용용도기본): 비어있음

```sql
SELECT COUNT(*) FROM bems.dev_use_purp_m;
-- 결과: 0건 (테이블 자체가 비어있음, 0 bytes)
```

### 3-3. `dev_use_purp_rel_r` (장치사용용도관계): 비어있음

```sql
SELECT COUNT(*) FROM bems.dev_use_purp_rel_r;
-- 결과: 0건 (테이블 자체가 비어있음, 0 bytes)
```

---

## 4. 근본 원인

### `dev_use_purp_m` 과 `dev_use_purp_rel_r` 테이블에 데이터가 없습니다.

BEMS 웹 UI는 대시보드를 렌더링할 때 `dev_use_purp_rel_r` 테이블을 기준으로 **표시할 건물/장치 목록을 먼저 결정**합니다. 이 테이블이 비어있으면:

1. 웹 UI가 표시할 장치 목록 자체가 0건
2. `max_dmand_fcst_h`와의 JOIN도 결과가 0건
3. 화면의 "최대수요-AI예측" 열에 아무 값도 표시되지 않음

```
┌─────────────────────┐     JOIN      ┌──────────────────────────┐
│ dev_use_purp_rel_r  │ ───────────── │ max_dmand_fcst_h         │
│ (0건 - 비어있음!)     │              │ (1건 - 226.17@13:15)     │
└─────────────────────┘              └──────────────────────────┘
         ↓                                      ↓
   장치 목록 0건                         JOIN 대상 없음
         ↓                                      ↓
         └──────────── 웹 UI 표시: 없음 ─────────┘
```

---

## 5. 정상 동작을 위한 데이터 흐름

```
[선행 조건 - 현재 미완료]
   BEMS 관리자 → 웹 UI 설정 화면
     → dev_use_purp_m: 용도 마스터 등록
     → dev_use_purp_rel_r: 장치-용도 관계 등록 (ai_aply_yn = 'Y')

[AI 모듈 실행 - 현재 정상 동작 중]
   ai_peak_runner.py
     → data_colec_h에서 센서 데이터 SELECT
     → max_dmand_fcst_h에 예측 결과 DELETE+INSERT

[웹 UI 표시 - 현재 실패]
   aiBemsPrcv.jsp
     → dev_use_purp_rel_r에서 장치 목록 조회 (0건!)
     → max_dmand_fcst_h LEFT JOIN (대상 없음!)
     → "최대수요-AI예측" 열 비어있음
```

---

## 6. 요청 사항 (조치 필요)

### 6-1. `dev_use_purp_m` 테이블에 용도 마스터 등록

```sql
-- use_purp_id, use_purp_nm 값은 BEMS 플랫폼 용도ID 체계에 맞게 지정
INSERT INTO bems.dev_use_purp_m (bldg_id, use_purp_id, use_purp_nm, ai_aply_desc)
VALUES ('B0019', '{use_purp_id}', '{용도명}', 'AI 최대수요 예측');
```

### 6-2. `dev_use_purp_rel_r` 테이블에 장치-용도 관계 등록

```sql
-- ai_aply_yn = 'Y' 로 설정하여 AI 최대수요예측 활성화
INSERT INTO bems.dev_use_purp_rel_r
  (bldg_id, use_purp_id, dev_id, ai_aply_yn, falt_prcv_yn)
VALUES
  ('B0019', '{use_purp_id}', '2001', 'Y', 'N');
```

### 6-3. 확인 사항

| 항목 | 확인 필요 내용 |
|------|---------------|
| `use_purp_id` | BEMS 플랫폼에서 사용하는 용도ID 체계의 올바른 값 (웹 UI가 기대하는 값) |
| `use_purp_nm` | 용도명 (예: "최대수요예측", "전력피크관리" 등) |
| `ai_aply_yn` | 웹 UI에서 AI예측 컬럼 표시를 위해 `'Y'`로 설정 필요 여부 확인 |

---

## 7. 현재 상태 요약

| 항목 | 상태 | 비고 |
|------|------|------|
| AI 모듈 실행 | **정상** | 0.2초, OK |
| `bems.max_dmand_fcst_h` | **정상** | 226.17@13:15 저장됨 |
| `bems.dev_use_purp_m` | **0건** | 용도 마스터 미등록 |
| `bems.dev_use_purp_rel_r` | **0건** | 장치-용도 관계 미등록 |
| BEMS 웹 UI 표시 | **실패** | JOIN 대상 테이블이 비어있음 |

---

## 8. 참고: 테이블 스키마

### `dev_use_purp_m` (장치사용용도기본)

| 컬럼 | 타입 | PK | NULL | 설명 |
|------|------|-----|------|------|
| `bldg_id` | varchar(10) | PK | NOT NULL | 건물 ID |
| `use_purp_id` | varchar(10) | PK | NOT NULL | 사용용도 ID |
| `use_purp_nm` | varchar(50) | | NULL | 사용용도명 |
| `ai_aply_desc` | varchar(1000) | | NULL | AI적용설명 |

### `dev_use_purp_rel_r` (장치사용용도관계)

| 컬럼 | 타입 | PK/FK | NULL | 설명 |
|------|------|-------|------|------|
| `bldg_id` | varchar(10) | PK, FK | NOT NULL | 건물 ID |
| `use_purp_id` | varchar(10) | PK, FK | NOT NULL | 사용용도 ID |
| `dev_id` | varchar(10) | PK | NOT NULL | 장치 ID |
| `forec_base_qnt` | numeric(15,2) | | NULL | 예상소비량 |
| `forec_rduc_qnt` | numeric(15,2) | | NULL | 예상절감량 |
| `falt_prcv_yn` | varchar(1) | | NULL | 이상감지 여부 |
| `ai_aply_yn` | varchar(1) | | NULL | AI적용 여부 |
| `lnch_st_time` | varchar(5) | | NULL | 점심시작시간 |
| `lnch_ed_time` | varchar(5) | | NULL | 점심종료시간 |
| `cool_base_tempr` | numeric(10,2) | | NULL | 냉방기준온도 |
| `heat_base_tempr` | numeric(10,2) | | NULL | 난방기준온도 |

FK: `(bldg_id, use_purp_id)` → `dev_use_purp_m (bldg_id, use_purp_id)`

### `max_dmand_fcst_h` (최대수요예측이력)

| 컬럼 | 타입 | PK | NULL | 설명 |
|------|------|-----|------|------|
| `use_dt` | timestamp | PK | NOT NULL | 사용일시 |
| `bldg_id` | varchar(10) | PK | NOT NULL | 건물 ID |
| `dly_max_dmand_fcst_inf` | varchar(30) | | NULL | 일일 최대수요 예측정보 (예: "226.17@13:15") |
