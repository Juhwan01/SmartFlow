import pandas as pd
import pickle
import numpy as np
from config import settings

# 데이터 로드
df = pd.read_csv('data/test_set.csv')
X = df.drop('welding_strength', axis=1)
y = df['welding_strength']

# 모델 로드
model = pickle.load(open('models/quality_predictor.pkl', 'rb'))
y_pred = model.predict(X)

# 품질 기준
lsl = settings.welding_strength_lsl
usl = settings.welding_strength_usl
margin = settings.anomaly_safety_margin_pct * lsl

# 불량 계산
defects_actual = (y < lsl) | (y > usl)
defects_pred = (y_pred < lsl) | (y_pred > usl)
anomaly_pred = (y_pred < lsl) | (y_pred > usl) | (y_pred < lsl+margin) | (y_pred > usl-margin)

print('=== 재학습 모델 예측 범위 ===')
print(f'예측 최소: {y_pred.min():.2f}')
print(f'예측 최대: {y_pred.max():.2f}')
print(f'실제 최소: {y.min():.2f}')
print(f'실제 최대: {y.max():.2f}')
print()
print('=== 불량 감지 ===')
print(f'실제 불량: {defects_actual.sum()}개')
print(f'예측 불량(LSL/USL): {defects_pred.sum()}개')
print(f'이상 감지(마진포함): {anomaly_pred.sum()}개')
print(f'정확 감지: {(defects_actual & defects_pred).sum()}개')
print(f'놓친 불량: {(defects_actual & ~defects_pred).sum()}개')
print(f'오탐: {(~defects_actual & defects_pred).sum()}개')
