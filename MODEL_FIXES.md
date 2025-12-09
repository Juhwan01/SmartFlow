# ML 모델 수정 사항 및 재학습 가이드

## 🔴 발견된 문제

### 1. Data Leakage 발생!
**문제**: Feature에 Stage2 Output 변수(`welding_measurement1`, `welding_measurement2`)가 포함되어 있었습니다.
- Feature Importance에서 `welding_measurement2`가 0.45로 가장 높음
- 이는 **미래 정보를 사용해서 예측**하는 것과 같음 (부정행위)
- Stage2의 출력으로 Stage2의 다른 출력을 예측하는 것은 논리적으로 불가능

**영향**:
- R² Score가 실제보다 부풀려짐
- 실제 운영 환경에서는 이 변수들을 사용할 수 없음

### 2. MAPE 계산 오류
**문제**: Target 값에 0이 포함되어 있어 division by zero 발생
```
MAPE: inf%
RuntimeWarning: divide by zero encountered in divide
```

### 3. 낮은 R² Score
**결과**: Test R² = 0.5569 (목표: >0.90)
- Data Leakage 변수 제거 후 재학습 필요

### 4. Deprecated Warning
```
FutureWarning: DataFrame.fillna with 'method' is deprecated
```

---

## ✅ 수정 사항

### 1. Data Leakage 제거
**파일**: `src/data/data_preprocessing.py`

**변경 전 (12개 변수)**:
- press_thickness, press_measurement1, press_measurement2
- welding_temp1, welding_temp2, welding_pressure, welding_temp3
- welding_control1, welding_control2
- **welding_measurement1, welding_measurement2** ❌ (Stage2 Output)
- welding_strength (Target)

**변경 후 (10개 변수)**:
- press_thickness, press_measurement1, press_measurement2
- welding_temp1, welding_temp2, welding_pressure, welding_temp3
- welding_control1, welding_control2
- welding_strength (Target)

**Feature 개수**: 11개 → **9개**

### 2. MAPE 계산 수정 (2차 개선)
**파일**: `scripts/train_model.py`

**변경 전**:
```python
"mape": np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
```

**1차 수정 (0 제외)**:
```python
mask = y_true != 0
```

**2차 수정 (0과 0에 가까운 값 제외)** ⭐:
```python
def _calculate_mape(self, y_true, y_pred):
    """MAPE 계산 (0과 0에 가까운 값 제외)"""
    threshold = 0.1  # 절댓값이 0.1 이상인 값만 사용
    mask = np.abs(y_true) > threshold

    if mask.sum() == 0:
        return 0.0

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape
```

**이유**: Target에 0.001 같은 0에 가까운 값이 있으면 MAPE가 폭발적으로 증가

### 3. fillna 메서드 업데이트
**파일**: `src/data/data_preprocessing.py`

**변경 전**:
```python
df_mapped = df_mapped.fillna(method='ffill').fillna(method='bfill')
```

**변경 후**:
```python
df_mapped = df_mapped.ffill().bfill()
```

### 4. LLM 모델 변경
**파일**: `.env.example`, `config/settings.py`

**변경 전**:
```python
LLM_MODEL=gpt-4
```

**변경 후**:
```python
LLM_MODEL=gpt-4o
```

### 5. ML Predictor Feature 수정
**파일**: `src/agents/ml_quality_predictor.py`

Features에서 `welding_measurement1`, `welding_measurement2` 제거 (9개로 변경)

---

## 🚀 재학습 방법

### 1. 모델 재학습
```bash
python scripts/train_model.py
```

### 2. 예상 결과
- ✅ MAPE가 정상 숫자로 표시됨 (inf 아님)
- ✅ Deprecated warning 사라짐
- ⚠️  R² Score는 **낮아질 가능성 높음** (Data Leakage 변수 제거했으므로)

### 3. R² Score 개선 방법

현재 Feature만으로 목표(>0.90)를 달성하기 어려울 수 있습니다. 다음 방법 시도:

#### Option 1: Feature Engineering
추가 Feature 생성:
```python
# 예시
- press_thickness_squared = press_thickness ** 2
- temp_ratio = welding_temp1 / welding_temp3
- interaction_features = press_thickness * welding_pressure
```

#### Option 2: 하이퍼파라미터 튜닝 ⭐ **이미 적용됨**
```python
# train_model.py의 main()에서 (현재 설정)
model, metrics = trainer.train_xgboost(
    n_estimators=300,    # 150 → 300 (더 많은 트리)
    max_depth=12,        # 8 → 12 (더 깊은 학습)
    learning_rate=0.03   # 0.05 → 0.03 (더 느린 학습)
)
```

**효과**: R² Score 0.53 → 0.70~0.85 예상

#### Option 3: 다른 알고리즘 시도
- Random Forest
- Gradient Boosting
- LightGBM

#### Option 4: 더 많은 Stage1/Machine4-5 변수 탐색
Kaggle 데이터셋에서 사용하지 않은 다른 컬럼 확인:
```python
# data_preprocessing.py에 추가
"machine4_variable_x": "Machine4.OtherVariable.C.Actual"
```

---

## 📊 성능 목표 재검토

### 현실적인 목표
Data Leakage 제거 후:
- **R² Score**: 0.70 ~ 0.85 (달성 가능)
- **MAE**: <1.0 (달성 가능)
- **MAPE**: 5~10% (달성 가능)

### 해커톤 시연 전략
1. **정직성 강조**: "Data Leakage를 방지하기 위해 Stage2 Output을 Feature에서 제거했습니다"
2. **실용성 강조**: "실제 운영 환경에서는 Stage1과 Machine 제어 변수만 사용 가능합니다"
3. **개선 가능성 제시**: "추가 Feature Engineering으로 성능 개선 가능합니다"

---

## 🎯 다음 단계

1. **재학습 실행**
   ```bash
   python scripts/train_model.py
   ```

2. **결과 확인**
   - R² Score가 0.70 이상이면 OK
   - 0.70 미만이면 Feature Engineering 시도

3. **대시보드 테스트**
   ```bash
   streamlit run src/dashboard/app.py
   ```
   - "평가지표" 탭에서 metrics.json 로드 확인

4. **필요시 Feature Engineering**
   - `src/data/data_preprocessing.py`에서 추가 Feature 생성

---

## 📝 참고 사항

- **Data Leakage**: ML에서 가장 흔한 실수 중 하나
- **실제 산업 환경**: Stage2 Output은 용접 후에만 측정 가능하므로 사전 예측에 사용 불가
- **모델 성능 vs 실용성**: 낮더라도 정직한 모델이 더 가치 있음

---

**Good Luck! 🍀**
