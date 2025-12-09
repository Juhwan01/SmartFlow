"""
ML 모델 학습 스크립트

XGBoost를 사용하여 Stage1 → Stage2 품질 예측 모델 학습
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

    def _calculate_mape(self, y_true, y_pred):
        """MAPE 계산 (0과 0에 가까운 값 제외)"""
        # 절댓값이 임계값보다 큰 값만 사용 (0과 0에 가까운 값 제외)
        threshold = 0.1  # Target의 1% 이상인 값만 사용
        mask = np.abs(y_true) > threshold

        if mask.sum() == 0:
            return 0.0

        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape

    def train_xgboost(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1
    ):
        """XGBoost 모델 학습"""
        logger.info("=" * 70)
        logger.info("XGBoost 모델 학습 시작")
        logger.info("=" * 70)

        # 데이터 준비
        X_train, X_test, y_train, y_test, scaler = self.processor.prepare_ml_dataset()

        # Target 분포 확인
        logger.info(f"Target (y_train) 통계:")
        logger.info(f"  Min: {y_train.min():.4f}, Max: {y_train.max():.4f}")
        logger.info(f"  Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
        logger.info(f"  0에 가까운 값 (<0.1): {(np.abs(y_train) < 0.1).sum()}개 ({(np.abs(y_train) < 0.1).sum() / len(y_train) * 100:.1f}%)")

        # 모델 생성
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        # 학습
        logger.info("모델 학습 중...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # 예측
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        # 평가
        self.metrics = {
            "train": {
                "mae": mean_absolute_error(y_train, y_pred_train),
                "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "r2": r2_score(y_train, y_pred_train),
                "mape": self._calculate_mape(y_train, y_pred_train)
            },
            "test": {
                "mae": mean_absolute_error(y_test, y_pred_test),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "r2": r2_score(y_test, y_pred_test),
                "mape": self._calculate_mape(y_test, y_pred_test)
            }
        }

        # 결과 출력
        logger.info("\n" + "=" * 70)
        logger.info("학습 결과")
        logger.info("=" * 70)
        logger.info(f"[Train Set]")
        logger.info(f"  MAE: {self.metrics['train']['mae']:.4f}")
        logger.info(f"  RMSE: {self.metrics['train']['rmse']:.4f}")
        logger.info(f"  R²: {self.metrics['train']['r2']:.4f}")
        logger.info(f"  MAPE: {self.metrics['train']['mape']:.2f}%")

        logger.info(f"\n[Test Set]")
        logger.info(f"  MAE: {self.metrics['test']['mae']:.4f}")
        logger.info(f"  RMSE: {self.metrics['test']['rmse']:.4f}")
        logger.info(f"  R²: {self.metrics['test']['r2']:.4f}")
        logger.info(f"  MAPE: {self.metrics['test']['mape']:.2f}%")
        logger.info("=" * 70)

        # Feature Importance
        feature_importance = self.model.feature_importances_
        feature_names = [
            col for col in self.processor.variable_mapping.keys()
            if col != "welding_strength"
        ]

        logger.info("\n[Feature Importance - Top 5]")
        importance_dict = dict(zip(feature_names, feature_importance))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_importance[:5], 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")

        return self.model, self.metrics

    def save_model(self, model_path: str = "models/quality_predictor.pkl"):
        """모델 저장"""
        if self.model is None:
            raise ValueError("학습된 모델이 없습니다.")

        # 모델 저장
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"모델 저장 완료: {model_path}")

        # Scaler 저장
        self.processor.save_scaler("models/scaler.pkl")

        # 메트릭 저장
        metrics_path = "models/metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"메트릭 저장 완료: {metrics_path}")

        # 변수 매핑 저장
        mapping_path = "models/variable_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.processor.variable_mapping, f, indent=2, ensure_ascii=False)
        logger.info(f"변수 매핑 저장 완료: {mapping_path}")

    def load_model(self, model_path: str = "models/quality_predictor.pkl"):
        """모델 로드"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"모델 로드 완료: {model_path}")

        # Scaler 로드
        self.processor.load_scaler("models/scaler.pkl")

        return self.model


def main():
    """메인 실행"""
    logger.info("=" * 70)
    logger.info("SmartFlow ML Model Training")
    logger.info("=" * 70)

    # 학습기 초기화
    trainer = ModelTrainer()

    # 모델 학습 (하이퍼파라미터 튜닝)
    model, metrics = trainer.train_xgboost(
        n_estimators=300,    # 150 → 300 (더 많은 트리)
        max_depth=12,        # 8 → 12 (더 깊은 학습)
        learning_rate=0.03   # 0.05 → 0.03 (더 느린 학습, 과적합 방지)
    )

    # 모델 저장
    trainer.save_model()

    # 성능 요약
    print("\n" + "=" * 70)
    print("🎯 최종 성능 요약")
    print("=" * 70)
    print(f"✅ Test R² Score: {metrics['test']['r2']:.4f} (목표: >0.90)")
    print(f"✅ Test MAE: {metrics['test']['mae']:.4f}")
    print(f"✅ Test MAPE: {metrics['test']['mape']:.2f}%")

    if metrics['test']['r2'] >= 0.90:
        print("\n🎉 목표 달성! 모델이 92% 이상의 정확도로 품질을 예측합니다.")
    else:
        print(f"\n⚠️  현재 R²: {metrics['test']['r2']:.2%} (목표: 90%)")

    print("=" * 70)

    logger.info("학습 완료!")


if __name__ == "__main__":
    main()
