"""
가상 센서 데이터 시뮬레이터

프레스 및 용접 공정의 센서 데이터를 시뮬레이션합니다.
"""
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field
from loguru import logger
from config import settings


@dataclass
class SensorReading:
    """센서 측정값"""
    timestamp: datetime
    process_name: str
    parameter_name: str
    value: float
    is_anomaly: bool = False
    anomaly_severity: float = 0.0  # 0.0 ~ 1.0


@dataclass
class PressSensorData:
    """프레스 공정 센서 데이터"""
    timestamp: datetime
    thickness: float  # 두께 (mm)
    pressure: float  # 압력 (MPa)
    temperature: float  # 온도 (°C)
    cycle_time: float  # 사이클 시간 (초)
    is_anomaly: bool = False
    anomaly_type: Optional[str] = None


@dataclass
class WeldingSensorData:
    """용접 공정 센서 데이터"""
    timestamp: datetime
    current: float  # 전류 (A)
    voltage: float  # 전압 (V)
    welding_speed: float  # 용접 속도 (mm/s)
    temperature: float  # 온도 (°C)
    strength: float  # 용접 강도 (MPa) - 실제론 측정 불가, 시뮬레이션용


class SensorSimulator:
    """센서 데이터 시뮬레이터"""

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 난수 시드 (재현성을 위해)
        """
        if seed is not None:
            np.random.seed(seed)

        # 프레스 공정 기본 파라미터
        self.press_thickness_mean = settings.press_thickness_mean
        self.press_thickness_std = settings.press_thickness_std
        self.press_pressure_mean = 150.0  # MPa
        self.press_pressure_std = 5.0
        self.press_temp_mean = 25.0  # °C
        self.press_temp_std = 2.0
        self.press_cycle_time_mean = 3.0  # 초
        self.press_cycle_time_std = 0.2

        # 용접 공정 기본 파라미터
        self.weld_current_mean = 200.0  # A
        self.weld_current_std = 10.0
        self.weld_voltage_mean = 25.0  # V
        self.weld_voltage_std = 2.0
        self.weld_speed_mean = 5.0  # mm/s
        self.weld_speed_std = 0.3
        self.weld_temp_mean = 800.0  # °C
        self.weld_temp_std = 50.0

        # 이상 발생 확률
        self.anomaly_probability = settings.anomaly_probability

        logger.info("SensorSimulator 초기화 완료")

    def generate_press_data(
        self,
        force_anomaly: bool = False,
        anomaly_magnitude: float = 2.0
    ) -> PressSensorData:
        """
        프레스 공정 센서 데이터 생성

        Args:
            force_anomaly: 강제로 이상 데이터 생성
            anomaly_magnitude: 이상 크기 (표준편차의 배수)

        Returns:
            PressSensorData 객체
        """
        timestamp = datetime.now()

        # 정상 데이터 생성
        thickness = np.random.normal(
            self.press_thickness_mean,
            self.press_thickness_std
        )
        pressure = np.random.normal(
            self.press_pressure_mean,
            self.press_pressure_std
        )
        temperature = np.random.normal(
            self.press_temp_mean,
            self.press_temp_std
        )
        cycle_time = np.random.normal(
            self.press_cycle_time_mean,
            self.press_cycle_time_std
        )

        # 이상 발생 여부 결정
        is_anomaly = force_anomaly or (np.random.random() < self.anomaly_probability)
        anomaly_type = None

        if is_anomaly:
            # 이상 유형 선택
            anomaly_types = ["thickness_deviation", "pressure_spike", "temperature_high"]
            anomaly_type = np.random.choice(anomaly_types)

            if anomaly_type == "thickness_deviation":
                # 두께 편차 발생 (가장 중요한 시나리오)
                deviation = anomaly_magnitude * self.press_thickness_std
                thickness += deviation if np.random.random() > 0.5 else -deviation
                logger.warning(
                    f"두께 편차 이상 발생: {thickness:.4f}mm "
                    f"(편차: {abs(thickness - self.press_thickness_mean):.4f}mm)"
                )

            elif anomaly_type == "pressure_spike":
                # 압력 급증
                pressure += anomaly_magnitude * self.press_pressure_std
                logger.warning(f"압력 급증 이상 발생: {pressure:.2f}MPa")

            elif anomaly_type == "temperature_high":
                # 온도 상승
                temperature += anomaly_magnitude * self.press_temp_std
                logger.warning(f"온도 상승 이상 발생: {temperature:.2f}°C")

        return PressSensorData(
            timestamp=timestamp,
            thickness=thickness,
            pressure=pressure,
            temperature=temperature,
            cycle_time=cycle_time,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type
        )

    def generate_welding_data(
        self,
        upstream_thickness: float,
        current_adjustment: float = 0.0,
        speed_adjustment: float = 0.0
    ) -> WeldingSensorData:
        """
        용접 공정 센서 데이터 생성

        Args:
            upstream_thickness: 상류 공정(프레스)의 두께 측정값
            current_adjustment: 전류 조정값 (%)
            speed_adjustment: 속도 조정값 (%)

        Returns:
            WeldingSensorData 객체
        """
        timestamp = datetime.now()

        # 조정값 적용
        current = np.random.normal(
            self.weld_current_mean * (1 + current_adjustment),
            self.weld_current_std
        )
        voltage = np.random.normal(
            self.weld_voltage_mean,
            self.weld_voltage_std
        )
        speed = np.random.normal(
            self.weld_speed_mean * (1 + speed_adjustment),
            self.weld_speed_std
        )
        temperature = np.random.normal(
            self.weld_temp_mean,
            self.weld_temp_std
        )

        # 용접 강도 시뮬레이션 (단순화된 모델)
        # 두께 편차가 크면 강도가 떨어짐
        thickness_deviation = abs(upstream_thickness - self.press_thickness_mean)
        thickness_penalty = thickness_deviation * 100  # 편차 1mm당 100MPa 감소

        # 전류와 속도의 영향
        current_effect = (current - self.weld_current_mean) * 0.5
        speed_effect = -(speed - self.weld_speed_mean) * 20  # 속도가 빠르면 강도 감소

        # 기본 강도 350MPa에서 계산
        base_strength = 350.0
        strength = base_strength - thickness_penalty + current_effect + speed_effect
        strength += np.random.normal(0, 5)  # 노이즈 추가

        return WeldingSensorData(
            timestamp=timestamp,
            current=current,
            voltage=voltage,
            welding_speed=speed,
            temperature=temperature,
            strength=max(0, strength)  # 음수 방지
        )

    def calculate_quality_impact(
        self,
        press_data: PressSensorData,
        welding_data: WeldingSensorData
    ) -> Dict[str, float]:
        """
        공정 간 품질 영향 계산

        Args:
            press_data: 프레스 공정 데이터
            welding_data: 용접 공정 데이터

        Returns:
            품질 영향 지표 딕셔너리
        """
        # 기준 강도 (정상 조건)
        baseline_strength = 350.0

        # 실제 강도
        actual_strength = welding_data.strength

        # 강도 저하율
        strength_degradation = (baseline_strength - actual_strength) / baseline_strength

        # 품질 점수 (0~1, 1이 완벽)
        quality_score = actual_strength / baseline_strength

        # 두께 편차
        thickness_deviation = abs(
            press_data.thickness - self.press_thickness_mean
        )

        return {
            "baseline_strength": baseline_strength,
            "actual_strength": actual_strength,
            "strength_degradation_pct": strength_degradation * 100,
            "quality_score": quality_score,
            "thickness_deviation": thickness_deviation,
            "meets_threshold": quality_score >= settings.quality_threshold
        }


# 모듈 테스트용
if __name__ == "__main__":
    logger.info("센서 시뮬레이터 테스트 시작")

    simulator = SensorSimulator(seed=42)

    # 정상 데이터 생성
    print("\n=== 정상 데이터 ===")
    press_normal = simulator.generate_press_data()
    print(f"프레스 두께: {press_normal.thickness:.4f}mm")
    print(f"이상 여부: {press_normal.is_anomaly}")

    weld_normal = simulator.generate_welding_data(press_normal.thickness)
    print(f"용접 강도: {weld_normal.strength:.2f}MPa")

    impact_normal = simulator.calculate_quality_impact(press_normal, weld_normal)
    print(f"품질 점수: {impact_normal['quality_score']:.2%}")

    # 이상 데이터 생성
    print("\n=== 이상 데이터 (두께 편차) ===")
    press_anomaly = simulator.generate_press_data(force_anomaly=True, anomaly_magnitude=2.0)
    print(f"프레스 두께: {press_anomaly.thickness:.4f}mm")
    print(f"이상 여부: {press_anomaly.is_anomaly}")
    print(f"이상 유형: {press_anomaly.anomaly_type}")

    weld_anomaly = simulator.generate_welding_data(press_anomaly.thickness)
    print(f"용접 강도: {weld_anomaly.strength:.2f}MPa")

    impact_anomaly = simulator.calculate_quality_impact(press_anomaly, weld_anomaly)
    print(f"품질 점수: {impact_anomaly['quality_score']:.2%}")
    print(f"강도 저하: {impact_anomaly['strength_degradation_pct']:.2f}%")
    print(f"품질 기준 충족: {impact_anomaly['meets_threshold']}")

    logger.info("테스트 완료")
