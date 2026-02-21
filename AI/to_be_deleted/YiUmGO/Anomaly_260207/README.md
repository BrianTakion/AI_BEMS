# Anomaly_260207 - LightGBM 기반 전력 예측 및 이상 탐지

건물 에너지 관리 시스템(BEMS)의 전력 데이터를 활용하여 LightGBM 모델로 전력 사용량을 예측하고, 실측값과의 RMSE 기반 이상 탐지를 수행합니다.

## 파일 구성

| 파일 | 설명 |
|------|------|
| `Power_Prediction_train.py` | LightGBM 모델 학습 스크립트 |
| `Power_Prediction_inference.py` | 학습된 모델을 로드하여 추론 수행 |
| `Power_Prediction_train_<devID>_<장치명>.txt` | 학습 완료된 LightGBM 모델 파일 |
| `train_result.html` | 학습 결과 시각화 (Plotly) |
| `inference_result.html` | 추론 결과 시각화 (Plotly) |

## 의존 모듈 (상위 디렉토리)

| 모듈 | 설명 |
|------|------|
| `Utility_260207.py` | 장치명 조회, 태그 데이터 추출, 시각화 유틸리티 |
| `Data_PreProcessing_260207.py` | 시계열 피처 엔지니어링 및 결측값 처리 |

## 사용법

### 학습

```bash
python Power_Prediction_train.py --devID 2001
python Power_Prediction_train.py --devID 2002
```

- 지정한 `devID`의 전력 데이터를 학습하여 모델 파일(`.txt`)을 생성합니다.
- 학습 결과 시각화는 `train_result.html`로 저장됩니다.

### 추론

```bash
python Power_Prediction_inference.py --devID 2001
```

- 학습된 모델 파일을 로드하여 추론을 수행합니다.
- 추론 결과 시각화는 `inference_result.html`로 저장됩니다.

### 도움말

```bash
python Power_Prediction_train.py --help
python Power_Prediction_inference.py --help
```

## 처리 흐름

```
1. Data Loading       : CSV 파일에서 건물 에너지 데이터 로드
2. Data Extraction    : devID/tagCD 기준으로 대상 장치 데이터 추출
3. Feature Engineering: 시계열 피처 생성 (lag, 시간 정보 등)
4. Train / Inference  : LightGBM 학습 또는 모델 로드 후 추론
5. Evaluation         : RMSE 평가 및 Plotly HTML 시각화 저장
```

## 데이터 파일 (상위 디렉토리)

- `data_colec_h_202509091411_B0019.csv` : 건물 에너지 수집 데이터
- `dev_m_202509151816_B0019.csv` : 장치 매핑 정보 (devID - 장치명)
