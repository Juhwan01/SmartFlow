# SmartFlow 설치 및 모델 학습 가이드

## 1. 패키지 설치

```bash
pip install -r requirements.txt
```

주요 패키지:
- `xgboost==2.1.3` - ML 모델
- `scikit-learn` - 전처리 및 평가
- `pandas` - 데이터 처리

## 2. 데이터 확인

데이터가 다음 위치에 있는지 확인:
```
src/data/continuous_factory_process.csv
```

확인 방법:
```bash
python src/data/data_preprocessing.py
```

## 3. 모델 학습

```bash
python scripts/train_model.py
```

학습 결과:
- `models/quality_predictor.pkl` - 학습된 XGBoost 모델
- `models/scaler.pkl` - 데이터 정규화 Scaler
- `models/metrics.json` - 성능 지표 (R², MAE, MAPE)
- `models/variable_mapping.json` - 변수 매핑 정보

## 4. 학습 완료 후

모델이 학습되면 다음 작업 가능:
1. 메인 시스템 실행: `python main.py`
2. 대시보드 실행: `streamlit run src/dashboard/app.py`

## 5. 예상 성능

목표 지표:
- **R² Score**: >0.90 (92% 정확도로 품질 예측)
- **MAE**: <1.0
- **MAPE**: <5%

## 6. 문제 해결

### 데이터 파일이 없는 경우
```bash
# data 폴더 확인
ls src/data/
```

### 모델 재학습
```bash
# 기존 모델 삭제 후 재학습
rm -rf models/*
python scripts/train_model.py
```

### 메모리 부족
`train_model.py`에서 `n_estimators`를 줄이세요 (150 → 100)

---

**다음 단계**: 모델 학습 후 README.md의 실행 가이드를 따라 시스템을 실행하세요!
