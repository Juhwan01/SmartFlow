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
lsl_buffer = settings.lsl_safety_buffer
usl_buffer = settings.usl_safety_buffer

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
print(f'LSL: {lsl}, USL: {usl}, 하한 마진: {lsl_buffer:.4f}, 상한 마진: {usl_buffer:.4f}')
print(f'하한 경계: {lsl + lsl_buffer:.2f}')
print(f'상한 경계: {usl - usl_buffer:.2f}')
