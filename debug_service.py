import pandas as pd
import pickle
import numpy as np
from config import settings
from src.agents.process_monitor import ProcessMonitorAgent

# 데이터 로드
df = pd.read_csv('data/test_set.csv')
X = df.drop('welding_strength', axis=1)
y = df['welding_strength']

# 모델 & 스케일러 로드
model = pickle.load(open('models/quality_predictor.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# ProcessMonitor 초기화
monitor = ProcessMonitorAgent()

# 예측 (이미 스케일링된 데이터)
y_pred = model.predict(X)

# 이상 감지
lsl = settings.welding_strength_lsl
usl = settings.welding_strength_usl
margin = settings.anomaly_safety_margin_pct * lsl

anomalies = []
for pred in y_pred:
    if monitor.is_anomaly_detected(pred):
        anomalies.append(True)
    else:
        anomalies.append(False)

anomaly_count = sum(anomalies)

print(f'=== 서비스 로직 시뮬레이션 ===')
print(f'Total 샘플: {len(y_pred)}')
print(f'이상 감지: {anomaly_count}개')
print(f'예측 범위: {y_pred.min():.2f} ~ {y_pred.max():.2f}')
print(f'LSL: {lsl}, USL: {usl}, 마진: {margin:.4f}')
print(f'하한 경계: {lsl + margin:.2f}')
print(f'상한 경계: {usl - margin:.2f}')
