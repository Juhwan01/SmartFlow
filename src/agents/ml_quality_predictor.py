"""
ML 기반 Quality Cascade Predictor

학습된 XGBoost 모델을 사용하여 실제 품질 예측
"""
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
import pickle
from pathlib import Path
from loguru import logger

from src.data.sensor_simulator import PressSensorData
from config import settings


@dataclass
class MLQualityPrediction:
    """ML 품질 예측 결과"""
    predicted_strength: float  # 예측된 용접 강도
    predicted_quality_score: float  # 품질 점수 (0~1)
    strength_degradation_pct: float  # 강도 저하율 (%)
    confidence: float  # 예측 신뢰도 (R² 기반)
    risk_level: str  # "low", "medium", "high", "critical"
    baseline_strength: float  # 기준 강도
    model_used: str  # 사용된 모델 이름
    recommendation: str


class MLQualityCascadePredictor:
    """학습된 ML 모델 기반 품질 예측 에이전트"""

    def __init__(
        self,
        model_path: str = "models/quality_predictor.pkl",
        scaler_path: str = "models/scaler.pkl"
    ):
        """
        Args:
            model_path: 학습된 모델 파일 경로
            scaler_path: Scaler 파일 경로
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.model_loaded = False

        # 품질 스펙 (config에서 로드)
        self.lsl = settings.welding_strength_lsl
        self.usl = settings.welding_strength_usl
        self.target = settings.welding_strength_target
        self.baseline_strength = self.target
        self.strength_std = 5.0  # 표준편차 (나중에 metrics에서 로드 가능)

        # 모델 로드 시도
        self._load_model()

    def _load_model(self):
        """모델 및 Scaler 로드"""
        try:
            # 모델 로드
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

            # Scaler 로드
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            self.model_loaded = True
            logger.info(f"ML 모델 로드 완료: {self.model_path}")

            # 기준값 업데이트 (모델 메타데이터가 있다면)
            try:
                import json
                metrics_path = "models/metrics.json"
                if Path(metrics_path).exists():
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    logger.info(f"모델 성능 - R²: {metrics['test']['r2']:.4f}, "
                               f"MAE: {metrics['test']['mae']:.4f}")
            except Exception:
                pass

        except FileNotFoundError:
            logger.warning(
                f"모델 파일을 찾을 수 없습니다: {self.model_path}\n"
                "먼저 'python scripts/train_model.py'로 모델을 학습하세요."
            )
            self.model_loaded = False
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            self.model_loaded = False

    def _prepare_features(
        self,
        press_data: PressSensorData,
        welding_params: Dict[str, float]
    ) -> np.ndarray:
        """
        예측을 위한 Feature 준비

        Args:
            press_data: 프레스 센서 데이터
            welding_params: 용접 파라미터

        Returns:
            정규화된 Feature 배열
        """
        # Feature 순서 (학습 시와 동일해야 함)
        # 변수 매핑에 따라 구성 (9개 features)
        features = [
            press_data.thickness,  # press_thickness
            0.0,  # press_measurement1 (임시)
            0.0,  # press_measurement2 (임시)
            welding_params.get("welding_temp1", 300.0),
            welding_params.get("welding_temp2", 290.0),
            welding_params.get("welding_pressure", 20.0),
            welding_params.get("welding_temp3", 270.0),
            welding_params.get("welding_control1", 310.0),
            welding_params.get("welding_control2", 290.0),
            # welding_measurement1, welding_measurement2 제거 (Data Leakage 방지)
        ]

        features_array = np.array(features).reshape(1, -1)

        # 정규화
        if self.scaler:
            features_scaled = self.scaler.transform(features_array)
        else:
            features_scaled = features_array

        return features_scaled

    def predict_from_press_data(
        self,
        press_data: PressSensorData,
        current_weld_params: Optional[Dict[str, float]] = None
    ) -> MLQualityPrediction:
        """
        프레스 데이터로부터 용접 품질 예측

        Args:
            press_data: 프레스 센서 데이터
            current_weld_params: 현재 용접 파라미터

        Returns:
            ML 품질 예측 결과
        """
        # 기본 파라미터
        if current_weld_params is None:
            current_weld_params = {
                "welding_temp1": 300.0,
                "welding_temp2": 290.0,
                "welding_pressure": 20.0,
                "welding_temp3": 270.0,
                "welding_control1": 310.0,
                "welding_control2": 290.0,
            }

        # 모델이 로드되지 않았으면 fallback 로직 사용
        if not self.model_loaded:
            logger.warning("모델 미로드 - Fallback 예측 사용")
            return self._fallback_prediction(press_data)

        # Feature 준비
        features = self._prepare_features(press_data, current_weld_params)

        # 예측
        predicted_strength = self.model.predict(features)[0]

        # 품질 점수 계산 (LSL~USL 범위 기준 정규화)
        # LSL~Target: 0~90점, Target~USL: 90~100점
        if predicted_strength >= self.target:
            # Target 이상: 90~100점
            predicted_quality_score = 0.9 + 0.1 * (predicted_strength - self.target) / (self.usl - self.target)
        else:
            # Target 미만: 0~90점 (LSL 미만도 점수 부여)
            predicted_quality_score = 0.9 * (predicted_strength - self.lsl) / (self.target - self.lsl)
        
        predicted_quality_score = float(np.clip(predicted_quality_score, 0.0, 1.0))

        # 강도 저하율 (target 기준)
        strength_degradation = max(0, (self.target - predicted_strength) / self.target * 100)

        # 신뢰도 (모델 R² 기반, 메트릭에서 가져올 수 있음)
        confidence = 0.92  # 임시값, 실제로는 metrics.json에서

        # 위험 수준 판단
        risk_level = self._determine_risk_level(predicted_quality_score)

        # 권장 사항
        recommendation = self._get_recommendation(risk_level, strength_degradation)

        logger.info(
            f"ML 예측 완료 - 예상 강도: {predicted_strength:.2f}, "
            f"품질: {predicted_quality_score:.2%}, 위험: {risk_level}"
        )

        return MLQualityPrediction(
            predicted_strength=predicted_strength,
            predicted_quality_score=predicted_quality_score,
            strength_degradation_pct=strength_degradation,
            confidence=confidence,
            risk_level=risk_level,
            baseline_strength=self.baseline_strength,
            model_used="XGBoost",
            recommendation=recommendation
        )

    def _fallback_prediction(self, press_data: PressSensorData) -> MLQualityPrediction:
        """모델이 없을 때 사용하는 간단한 예측"""
        # 두께 편차 기반 간단한 계산
        thickness_deviation = abs(press_data.thickness - 2.0)
        impact = thickness_deviation * 100

        predicted_strength = max(0, self.baseline_strength - impact)
        predicted_quality = predicted_strength / self.baseline_strength

        risk_level = self._determine_risk_level(predicted_quality)
        degradation = (self.baseline_strength - predicted_strength) / self.baseline_strength * 100

        return MLQualityPrediction(
            predicted_strength=predicted_strength,
            predicted_quality_score=predicted_quality,
            strength_degradation_pct=degradation,
            confidence=0.7,
            risk_level=risk_level,
            baseline_strength=self.baseline_strength,
            model_used="Fallback (Rule-based)",
            recommendation=self._get_recommendation(risk_level, degradation)
        )

    def _determine_risk_level(self, quality_score: float) -> str:
        """위험 수준 판단"""
        threshold = settings.quality_threshold

        if quality_score >= threshold:
            return "low"
        elif quality_score >= threshold - 0.05:
            return "medium"
        elif quality_score >= threshold - 0.15:
            return "high"
        else:
            return "critical"

    def _get_recommendation(self, risk_level: str, degradation_pct: float) -> str:
        """권장 사항 생성"""
        if risk_level == "low":
            return "현재 파라미터 유지 가능"
        elif risk_level == "medium":
            return "용접 파라미터 미세 조정 검토 권장"
        elif risk_level == "high":
            return "즉시 용접 파라미터 조정 필요 - 협상 에이전트 활성화"
        else:  # critical
            return "긴급 조치 필요 - 생산 중단 고려"

    def predict_with_adjustment(
        self,
        press_data: PressSensorData,
        proposed_adjustments: Dict[str, float]
    ) -> MLQualityPrediction:
        """
        조정안 적용 시 예측

        Args:
            press_data: 프레스 데이터
            proposed_adjustments: 제안된 조정값 (백분율)

        Returns:
            조정 후 예측 품질
        """
        # 기본 파라미터에 조정값 적용
        base_params = {
            "welding_temp1": 300.0,
            "welding_temp2": 290.0,
            "welding_pressure": 20.0,
            "welding_temp3": 270.0,
        }

        adjusted_params = {}
        for key, base_value in base_params.items():
            # adjustments의 키를 매핑 (current → temp1, speed → temp3, pressure → pressure)
            adj_key = key
            if "temp1" in key:
                adjustment = proposed_adjustments.get("current", 0.0)
            elif "temp3" in key:
                adjustment = proposed_adjustments.get("welding_speed", 0.0)
            elif "pressure" in key:
                adjustment = proposed_adjustments.get("pressure", 0.0)
            else:
                adjustment = 0.0

            adjusted_params[key] = base_value * (1 + adjustment)

        return self.predict_from_press_data(press_data, adjusted_params)


# 모듈 테스트
if __name__ == "__main__":
    from src.data.sensor_simulator import SensorSimulator

    logger.info("ML Quality Predictor 테스트 시작")

    simulator = SensorSimulator(seed=42)
    predictor = MLQualityCascadePredictor()

    if predictor.model_loaded:
        print("\n✅ 모델 로드 성공!")

        # 정상 데이터 예측
        press_normal = simulator.generate_press_data(force_anomaly=False)
        prediction_normal = predictor.predict_from_press_data(press_normal)

        print(f"\n[정상 데이터 예측]")
        print(f"프레스 두께: {press_normal.thickness:.4f}mm")
        print(f"예측 강도: {prediction_normal.predicted_strength:.2f}")
        print(f"예측 품질: {prediction_normal.predicted_quality_score:.2%}")
        print(f"위험 수준: {prediction_normal.risk_level}")
        print(f"모델: {prediction_normal.model_used}")

        # 이상 데이터 예측
        press_anomaly = simulator.generate_press_data(force_anomaly=True, anomaly_magnitude=2.5)
        prediction_anomaly = predictor.predict_from_press_data(press_anomaly)

        print(f"\n[이상 데이터 예측]")
        print(f"프레스 두께: {press_anomaly.thickness:.4f}mm")
        print(f"예측 강도: {prediction_anomaly.predicted_strength:.2f}")
        print(f"예측 품질: {prediction_anomaly.predicted_quality_score:.2%}")
        print(f"강도 저하: {prediction_anomaly.strength_degradation_pct:.2f}%")
        print(f"위험 수준: {prediction_anomaly.risk_level}")
        print(f"권장 사항: {prediction_anomaly.recommendation}")
    else:
        print("\n⚠️  모델이 로드되지 않았습니다.")
        print("다음 명령어로 모델을 먼저 학습하세요:")
        print("  python scripts/train_model.py")

    logger.info("테스트 완료")
