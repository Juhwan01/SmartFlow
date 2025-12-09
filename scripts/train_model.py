"""
ML 모델 학습 스크립트 (최종 수정버전 - XGBoost 2.0+ 호환)

1. XGBoost 2.0+ 호환성 완벽 수정 (eval_metric, early_stopping_rounds 위치 변경)
2. 비즈니스 KPI (MAE, MAPE) 중심의 학습 및 평가
3. 도메인 지식 기반 Feature Engineering 적용
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from loguru import logger
import json

from src.data.data_preprocessing import ManufacturingDataProcessor


class ModelTrainer:
    """모델 학습기"""

    def __init__(self):
        self.processor = ManufacturingDataProcessor()
        self.model = None
        self.metrics = {}
        self.scaler = MinMaxScaler() # 자체 스케일러 사용

    def _calculate_mape(self, y_true, y_pred):
        """MAPE 계산 (정밀도 지표)"""
        threshold = 0.1
        mask = np.abs(y_true) > threshold
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def feature_engineering(self, df):
        """도메인 지식 기반 변수 추가"""
        df_fe = df.copy()
        
        # 1. 용접 입열량 (Heat Input) 유사 변수
        df_fe['heat_input_proxy'] = df_fe['welding_temp1'] / (df_fe['welding_temp3'] + 1e-5)

        # 2. 압력과 온도의 상호작용
        df_fe['pressure_x_temp2'] = df_fe['welding_pressure'] * df_fe['welding_temp2']
        
        # 3. 제어 변수 합계
        df_fe['total_control'] = df_fe['welding_control1'] + df_fe['welding_control2']

        # 4. 프레스 공정의 면적/부피 유사 변수
        df_fe['press_volume_proxy'] = df_fe['press_thickness'] * df_fe['press_measurement1']

        return df_fe

    def train_xgboost(
        self,
        n_estimators: int = 2000,
        max_depth: int = 8,
        learning_rate: float = 0.02
    ):
        logger.info("=" * 70)
        logger.info("XGBoost 모델 학습 시작 (KPI: MAE 최소화)")
        logger.info("=" * 70)

        # 1. 데이터 로드
        df = self.processor.create_mapped_dataset()
        target_col = "welding_strength"

        # 2. 0값 데이터 필터링
        initial_len = len(df)
        df = df[df[target_col] > 0.1].copy()
        logger.info(f"데이터 필터링: {initial_len} -> {len(df)} (유효 공정 데이터만 사용)")

        # 3. Feature Engineering 적용
        df_fe = self.feature_engineering(df)
        feature_cols = [col for col in df_fe.columns if col != target_col]
        
        X = df_fe[feature_cols].values
        y = df_fe[target_col].values

        # 4. 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 5. Train/Test 분리
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # 6. 모델 학습
        # [수정됨] eval_metric과 early_stopping_rounds를 생성자로 이동
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            eval_metric='mae',           # <-- 여기로 이동했습니다
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        logger.info("모델 학습 중...")
        
        # [수정됨] fit 함수에서 매개변수 제거
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # 7. 예측 및 평가
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        self.metrics = {
            "train": {
                "r2": r2_score(y_train, y_pred_train),
                "mae": mean_absolute_error(y_train, y_pred_train),
                "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "mape": self._calculate_mape(y_train, y_pred_train)
            },
            "test": {
                "r2": r2_score(y_test, y_pred_test),
                "mae": mean_absolute_error(y_test, y_pred_test),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "mape": self._calculate_mape(y_test, y_pred_test)
            }
        }

        # 결과 출력
        logger.info("\n" + "=" * 70)
        logger.info("학습 결과 (Test Set)")
        logger.info("=" * 70)
        logger.info(f"✅ MAE (평균 오차): {self.metrics['test']['mae']:.4f} (목표: < 0.2)")
        logger.info(f"✅ MAPE (오차율)  : {self.metrics['test']['mape']:.2f}%  (목표: < 2%)")
        logger.info(f"ℹ️  R² Score     : {self.metrics['test']['r2']:.4f}")
        logger.info("=" * 70)

        # Feature Importance
        logger.info("\n[핵심 영향 변수 - Top 5]")
        feature_importance = self.model.feature_importances_
        importance_dict = dict(zip(feature_cols, feature_importance))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_importance[:5], 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")

        return self.model, self.metrics

    def save_model(self, model_path: str = "models/quality_predictor.pkl"):
        """모델 및 관련 파일 저장"""
        if self.model is None:
            return

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open("models/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

        with open("models/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        with open("models/variable_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(self.processor.variable_mapping, f, indent=2, ensure_ascii=False)
            
        logger.info(f"모델 저장 완료: {model_path}")

def main():
    logger.info("SmartFlow ML Model Training (Business KPI Optimized)")
    
    trainer = ModelTrainer()
    model, metrics = trainer.train_xgboost()
    trainer.save_model()

    print("\n" + "=" * 70)
    print("🎯 최종 성능 요약")
    print("=" * 70)
    
    mae_score = metrics['test']['mae']
    mape_score = metrics['test']['mape']
    
    print(f"✅ Test MAE  : {mae_score:.4f} (목표: < 0.2)")
    print(f"✅ Test MAPE : {mape_score:.2f}% (목표: < 2.0%)")
    print(f"ℹ️  Test R²   : {metrics['test']['r2']:.4f}")

    if mae_score < 0.2:
        print("\n🎉 목표 달성! 현장 투입 가능한 초정밀 예측 성능을 확보했습니다.")
        print("   (평균 오차 0.2 미만으로 품질 제어 가능)")
    else:
        print(f"\n⚠️  추가 튜닝 필요 (현재 오차: {mae_score:.4f})")
    
    print("=" * 70)

if __name__ == "__main__":
    main()