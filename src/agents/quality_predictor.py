"""
Quality Cascade Predictor

ML/DL 모델을 활용해 현재 공정의 변동이 후속 공정에 미칠 품질 영향을 예측하는 에이전트
"""
from typing import Dict, Optional, List
from dataclasses import dataclass
import numpy as np
from loguru import logger

from src.data.sensor_simulator import PressSensorData, WeldingSensorData
from config import settings


@dataclass
class QualityPrediction:
    """품질 예측 결과"""
    predicted_quality_score: float  # 0~1
    predicted_strength: float  # MPa
    strength_degradation_pct: float  # %
    confidence: float  # 0~1
    risk_level: str  # "low", "medium", "high", "critical"
    contributing_factors: Dict[str, float]  # 기여 요인들
    recommendation: str


class QualityCascadePredictor:
    """품질 연쇄 예측 에이전트"""

    def __init__(self):
        """
        MVP에서는 간단한 룰 기반 모델 사용
        실제 시스템에서는 Multi-scale CNN이나 MEPN 등의 딥러닝 모델 사용
        """
        # 기준값 (config 기반)
        self.baseline_thickness = 2.0  # mm
        self.baseline_strength = settings.welding_strength_target  # config 기반 target
        self.baseline_quality = 1.0
        self.lsl = settings.welding_strength_lsl
        self.usl = settings.welding_strength_usl

        # 영향 계수 (데이터 기반으로 학습된 값으로 가정)
        self.impact_coefficients = {
            "thickness_deviation": -100.0,  # mm당 MPa 감소
            "pressure": 0.3,  # MPa당 강도 증가
            "current": 0.5,  # A당 강도 증가
            "welding_speed": -20.0,  # mm/s당 강도 감소
        }

        logger.info("QualityCascadePredictor 초기화 완료 (룰 기반 모델)")

    def _calculate_thickness_impact(
        self,
        thickness: float
    ) -> float:
        """두께 편차가 강도에 미치는 영향 계산"""
        deviation = abs(thickness - self.baseline_thickness)
        impact = self.impact_coefficients["thickness_deviation"] * deviation
        return impact

    def _calculate_confidence(
        self,
        thickness_deviation: float
    ) -> float:
        """
        예측 신뢰도 계산

        두께 편차가 작을수록 신뢰도 높음
        """
        # 편차가 0이면 신뢰도 1.0, 편차가 클수록 감소
        max_deviation = 0.05  # 5mm
        confidence = 1.0 - min(thickness_deviation / max_deviation, 1.0) * 0.3
        return max(0.7, confidence)  # 최소 70%

    def _determine_risk_level(
        self,
        quality_score: float
    ) -> str:
        """품질 점수 기반 위험 수준 판단"""
        threshold = settings.quality_threshold

        if quality_score >= threshold:
            return "low"
        elif quality_score >= threshold - 0.05:
            return "medium"
        elif quality_score >= threshold - 0.15:
            return "high"
        else:
            return "critical"

    def predict_from_press_data(
        self,
        press_data: PressSensorData,
        current_weld_params: Optional[Dict[str, float]] = None
    ) -> QualityPrediction:
        """
        프레스 공정 데이터로부터 용접 품질 예측

        Args:
            press_data: 프레스 센서 데이터
            current_weld_params: 현재 용접 파라미터 (없으면 기본값 사용)

        Returns:
            품질 예측 결과
        """
        # 기본 용접 파라미터
        if current_weld_params is None:
            current_weld_params = {
                "current": 200.0,  # A
                "welding_speed": 5.0,  # mm/s
                "pressure": 150.0,  # MPa
            }

        # 두께 영향
        thickness_impact = self._calculate_thickness_impact(press_data.thickness)

        # 압력 영향
        pressure_impact = (
            (press_data.pressure - 150.0) *
            self.impact_coefficients["pressure"]
        )

        # 전류 영향
        current_impact = (
            (current_weld_params["current"] - 200.0) *
            self.impact_coefficients["current"]
        )

        # 속도 영향
        speed_impact = (
            (current_weld_params["welding_speed"] - 5.0) *
            self.impact_coefficients["welding_speed"]
        )

        # 총 강도 예측
        predicted_strength = (
            self.baseline_strength +
            thickness_impact +
            pressure_impact +
            current_impact +
            speed_impact +
            np.random.normal(0, 3)  # 약간의 불확실성
        )
        predicted_strength = max(0, predicted_strength)

        # 품질 점수 계산 (config 기반 LSL/Target/USL)
        if predicted_strength >= self.baseline_strength:
            # Target 이상: 90~100점
            predicted_quality_score = 0.9 + 0.1 * (predicted_strength - self.baseline_strength) / (self.usl - self.baseline_strength)
        else:
            # Target 미만: 0~90점
            predicted_quality_score = 0.9 * (predicted_strength - self.lsl) / (self.baseline_strength - self.lsl)
        
        predicted_quality_score = float(np.clip(predicted_quality_score, 0.0, 1.0))

        # 강도 저하율
        strength_degradation = (
            (self.baseline_strength - predicted_strength) /
            self.baseline_strength * 100
        )

        # 기여 요인들
        contributing_factors = {
            "thickness_deviation": thickness_impact,
            "press_pressure": pressure_impact,
            "weld_current": current_impact,
            "weld_speed": speed_impact,
        }

        # 신뢰도 계산
        thickness_deviation = abs(press_data.thickness - self.baseline_thickness)
        confidence = self._calculate_confidence(thickness_deviation)

        # 위험 수준
        risk_level = self._determine_risk_level(predicted_quality_score)

        # 권장 사항
        if risk_level == "low":
            recommendation = "현재 파라미터 유지 가능"
        elif risk_level == "medium":
            recommendation = "용접 파라미터 미세 조정 검토 권장"
        elif risk_level == "high":
            recommendation = "즉시 용접 파라미터 조정 필요 - 협상 에이전트 활성화"
        else:  # critical
            recommendation = "긴급 조치 필요 - 생산 중단 고려"

        logger.info(
            f"품질 예측 완료 - 예상 품질: {predicted_quality_score:.2%}, "
            f"위험도: {risk_level}"
        )

        return QualityPrediction(
            predicted_quality_score=predicted_quality_score,
            predicted_strength=predicted_strength,
            strength_degradation_pct=strength_degradation,
            confidence=confidence,
            risk_level=risk_level,
            contributing_factors=contributing_factors,
            recommendation=recommendation
        )

    def predict_with_adjustment(
        self,
        press_data: PressSensorData,
        proposed_adjustments: Dict[str, float]
    ) -> QualityPrediction:
        """
        조정안을 적용했을 때의 품질 예측

        Args:
            press_data: 프레스 센서 데이터
            proposed_adjustments: 제안된 조정값 (예: {"current": 0.03, "welding_speed": -0.05})

        Returns:
            조정 후 예측 품질
        """
        # 기본 파라미터에 조정값 적용
        base_params = {
            "current": 200.0,
            "welding_speed": 5.0,
            "pressure": 150.0,
        }

        adjusted_params = {}
        for key, base_value in base_params.items():
            adjustment = proposed_adjustments.get(key, 0.0)
            adjusted_params[key] = base_value * (1 + adjustment)

        return self.predict_from_press_data(press_data, adjusted_params)

    def compare_scenarios(
        self,
        press_data: PressSensorData,
        scenarios: List[Dict[str, float]]
    ) -> List[tuple[Dict[str, float], QualityPrediction]]:
        """
        여러 조정 시나리오 비교

        Args:
            press_data: 프레스 센서 데이터
            scenarios: 조정 시나리오 리스트

        Returns:
            (시나리오, 예측 결과) 튜플 리스트 (품질 점수 내림차순)
        """
        results = []

        for scenario in scenarios:
            prediction = self.predict_with_adjustment(press_data, scenario)
            results.append((scenario, prediction))

        # 품질 점수 기준 정렬
        results.sort(key=lambda x: x[1].predicted_quality_score, reverse=True)

        return results

    def get_agent_state(self) -> Dict:
        """에이전트 상태 정보 반환"""
        return {
            "agent_type": "QualityCascadePredictor",
            "status": "active",
            "model_type": "rule_based",  # MVP에서는 룰 기반
            "baseline_values": {
                "thickness": self.baseline_thickness,
                "strength": self.baseline_strength,
                "quality": self.baseline_quality,
            },
            "impact_coefficients": self.impact_coefficients,
        }


# 모듈 테스트용
if __name__ == "__main__":
    from src.data.sensor_simulator import SensorSimulator

    logger.info("QualityCascadePredictor 테스트 시작")

    simulator = SensorSimulator(seed=42)
    predictor = QualityCascadePredictor()

    print("\n" + "=" * 60)
    print("시나리오 1: 정상 데이터")
    print("=" * 60)

    press_normal = simulator.generate_press_data(force_anomaly=False)
    print(f"프레스 두께: {press_normal.thickness:.4f}mm")

    prediction_normal = predictor.predict_from_press_data(press_normal)
    print(f"예측 품질 점수: {prediction_normal.predicted_quality_score:.2%}")
    print(f"예측 강도: {prediction_normal.predicted_strength:.2f}MPa")
    print(f"강도 저하: {prediction_normal.strength_degradation_pct:.2f}%")
    print(f"신뢰도: {prediction_normal.confidence:.2%}")
    print(f"위험 수준: {prediction_normal.risk_level}")
    print(f"권장 사항: {prediction_normal.recommendation}")

    print("\n" + "=" * 60)
    print("시나리오 2: 이상 데이터 (두께 편차)")
    print("=" * 60)

    press_anomaly = simulator.generate_press_data(force_anomaly=True, anomaly_magnitude=2.5)
    print(f"프레스 두께: {press_anomaly.thickness:.4f}mm")
    print(f"두께 편차: {abs(press_anomaly.thickness - 2.0):.4f}mm")

    prediction_anomaly = predictor.predict_from_press_data(press_anomaly)
    print(f"예측 품질 점수: {prediction_anomaly.predicted_quality_score:.2%}")
    print(f"예측 강도: {prediction_anomaly.predicted_strength:.2f}MPa")
    print(f"강도 저하: {prediction_anomaly.strength_degradation_pct:.2f}%")
    print(f"위험 수준: {prediction_anomaly.risk_level}")
    print(f"권장 사항: {prediction_anomaly.recommendation}")

    print("\n" + "=" * 60)
    print("시나리오 3: 조정안 적용 시뮬레이션")
    print("=" * 60)

    # 여러 조정 시나리오
    scenarios = [
        {"current": 0.0, "welding_speed": 0.0, "pressure": 0.0},  # 조정 없음
        {"current": 0.03, "welding_speed": -0.05, "pressure": 0.02},  # 권장 조정
        {"current": 0.05, "welding_speed": -0.07, "pressure": 0.03},  # 강력 조정
        {"current": 0.10, "welding_speed": 0.0, "pressure": 0.0},  # 전류만 증가
    ]

    results = predictor.compare_scenarios(press_anomaly, scenarios)

    for i, (scenario, pred) in enumerate(results, 1):
        print(f"\n[시나리오 {i}]")
        print(f"조정값: {scenario}")
        print(f"예측 품질: {pred.predicted_quality_score:.2%}")
        print(f"예측 강도: {pred.predicted_strength:.2f}MPa")
        print(f"위험 수준: {pred.risk_level}")

    logger.info("테스트 완료")
