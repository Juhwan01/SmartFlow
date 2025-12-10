"""
Process Monitor Agent

각 공정의 센서 데이터를 실시간으로 수집하고 이상 징후를 포착하는 에이전트
"""
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger

from src.data.sensor_simulator import (
    SensorSimulator,
    PressSensorData,
    WeldingSensorData
)
from config import settings


@dataclass
class AlertMessage:
    """이상 알림 메시지"""
    timestamp: datetime
    alert_id: str
    severity: str  # "low", "medium", "high", "critical"
    process_stage: str
    issue_description: str
    current_values: Dict[str, float]
    threshold_violated: Optional[str] = None
    recommended_action: Optional[str] = None


class ProcessMonitorAgent:
    """공정 감시 에이전트"""

    def __init__(self, simulator: Optional[SensorSimulator] = None):
        """
        Args:
            simulator: 센서 시뮬레이터 (None이면 새로 생성)
        """
        self.simulator = simulator or SensorSimulator()
        self.alert_history: List[AlertMessage] = []
        self.alert_counter = 0

        # 이상 감지 임계값
        self.thresholds = {
            "thickness_deviation": 0.015,  # ±0.015mm
            "pressure_deviation": 10.0,  # ±10 MPa
            "temperature_high": 30.0,  # 기준 온도 + 30°C
            "quality_score": settings.quality_threshold,  # 0.90
        }

        logger.info("ProcessMonitorAgent 초기화 완료")

    def _generate_alert_id(self) -> str:
        """알림 ID 생성"""
        self.alert_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"ALERT-{timestamp}-{self.alert_counter:04d}"

    def _determine_severity(
        self,
        deviation: float,
        threshold: float
    ) -> str:
        """
        이상 심각도 판단

        Args:
            deviation: 편차 크기
            threshold: 임계값

        Returns:
            심각도 레벨
        """
        ratio = abs(deviation) / threshold

        if ratio < 1.0:
            return "low"
        elif ratio < 1.5:
            return "medium"
        elif ratio < 2.5:
            return "high"
        else:
            return "critical"

    def monitor_press_process(
        self,
        force_anomaly: bool = False
    ) -> tuple[PressSensorData, Optional[AlertMessage]]:
        """
        프레스 공정 모니터링

        Args:
            force_anomaly: 강제로 이상 발생시키기

        Returns:
            (센서 데이터, 알림 메시지) 튜플
        """
        # 센서 데이터 수집
        sensor_data = self.simulator.generate_press_data(force_anomaly=force_anomaly)

        # 이상 감지
        alert = None
        if sensor_data.is_anomaly:
            # 두께 편차 계산
            thickness_deviation = abs(
                sensor_data.thickness - self.simulator.press_thickness_mean
            )

            # 심각도 판단
            severity = self._determine_severity(
                thickness_deviation,
                self.thresholds["thickness_deviation"]
            )

            # 권장 조치 결정
            if severity in ["low"]:
                recommended_action = "모니터링 계속, 추가 조치 대기"
            elif severity in ["medium"]:
                recommended_action = "용접 파라미터 조정 협상 시작 권장"
            else:  # high, critical
                recommended_action = "즉시 용접 파라미터 조정 필요"

            # 알림 생성
            alert = AlertMessage(
                timestamp=sensor_data.timestamp,
                alert_id=self._generate_alert_id(),
                severity=severity,
                process_stage="press",
                issue_description=sensor_data.anomaly_type or "unknown_anomaly",
                current_values={
                    "thickness": sensor_data.thickness,
                    "thickness_deviation": thickness_deviation,
                    "pressure": sensor_data.pressure,
                    "temperature": sensor_data.temperature,
                },
                threshold_violated="thickness_deviation",
                recommended_action=recommended_action
            )

            self.alert_history.append(alert)

            logger.warning(
                f"[{alert.alert_id}] 프레스 공정 이상 감지 - "
                f"심각도: {severity}, 두께: {sensor_data.thickness:.4f}mm"
            )

        return sensor_data, alert

    def monitor_welding_process(
        self,
        welding_data: WeldingSensorData,
        quality_score: float
    ) -> Optional[AlertMessage]:
        """
        용접 공정 모니터링

        Args:
            welding_data: 용접 센서 데이터
            quality_score: 품질 점수 (0~1)

        Returns:
            알림 메시지 (이상 있을 경우)
        """
        # 품질 점수 확인
        if quality_score < self.thresholds["quality_score"]:
            quality_deficit = self.thresholds["quality_score"] - quality_score

            # 심각도 판단
            if quality_deficit < 0.05:
                severity = "low"
                recommended_action = "현재 상태 유지, 모니터링 강화"
            elif quality_deficit < 0.10:
                severity = "medium"
                recommended_action = "파라미터 미세 조정 검토"
            elif quality_deficit < 0.20:
                severity = "high"
                recommended_action = "즉시 파라미터 조정 필요"
            else:
                severity = "critical"
                recommended_action = "라인 중지 및 점검 필요"

            alert = AlertMessage(
                timestamp=welding_data.timestamp,
                alert_id=self._generate_alert_id(),
                severity=severity,
                process_stage="welding",
                issue_description="quality_below_threshold",
                current_values={
                    "quality_score": quality_score,
                    "strength": welding_data.strength,
                    "current": welding_data.current,
                    "welding_speed": welding_data.welding_speed,
                    "temperature": welding_data.temperature,
                },
                threshold_violated="quality_score",
                recommended_action=recommended_action
            )

            self.alert_history.append(alert)

            logger.warning(
                f"[{alert.alert_id}] 용접 품질 저하 감지 - "
                f"심각도: {severity}, 품질: {quality_score:.2%}"
            )

            return alert

        return None

    def is_anomaly_detected(self, predicted_strength: float, predicted_quality_score: float = None) -> bool:
        """
        이상 감지 판단 (MVP 설계 + 비용 최적화)

        비용 고려사항:
        - 실제 불량률 1.8% → 이상 탐지율 목표 3-5%
        - 불필요한 조정 최소화 → 생산성 유지, LLM API 비용 절감

        감지 기준 (보수적):
        1. LSL/USL 범위 벗어남 (명백한 불량)
        2. 품질 점수 < 0.85 (심각한 품질 저하)
        3. LSL/USL 근접 (불량 직전)

        Args:
            predicted_strength: ML 모델이 예측한 강도
            predicted_quality_score: 예측 품질 점수 (0~1, optional)

        Returns:
            이상 여부 (True: 조정 필요, False: 정상)
        """
        lsl = settings.welding_strength_lsl  # 11.50 (불량 기준)
        usl = settings.welding_strength_usl  # 13.20 (불량 기준)

        # ========================================
        # 1단계: 명백한 불량 (LSL/USL 범위 벗어남)
        # ========================================
        if predicted_strength < lsl or predicted_strength > usl:
            return True  # 이미 불량 → 즉시 조정 필요

        # ========================================
        # 2단계: 심각한 품질 저하 (보수적 기준)
        # ========================================
        if predicted_quality_score is not None:
            warning_threshold = settings.anomaly_warning_quality  # 0.85
            if predicted_quality_score < warning_threshold:
                return True  # 심각한 품질 저하 → 사전 조정

        # ========================================
        # 3단계: LSL/USL 근접 (불량 직전 예방)
        # ========================================
        lsl_buffer = settings.lsl_safety_buffer  # 0.20
        usl_buffer = settings.usl_safety_buffer  # 0.20

        # LSL에 너무 가까움
        if predicted_strength < (lsl + lsl_buffer):  # 11.70 미만
            return True  # 불량 직전 → 사전 조정

        # USL에 너무 가까움
        if predicted_strength > (usl - usl_buffer):  # 13.00 초과
            return True  # 불량 직전 → 사전 조정

        # 모든 기준 통과: 정상
        return False

    def get_alert_summary(self) -> Dict:
        """알림 요약 정보 반환"""
        if not self.alert_history:
            return {
                "total_alerts": 0,
                "by_severity": {},
                "by_process": {},
                "recent_alerts": []
            }

        # 심각도별 집계
        by_severity = {}
        for alert in self.alert_history:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1

        # 공정별 집계
        by_process = {}
        for alert in self.alert_history:
            by_process[alert.process_stage] = \
                by_process.get(alert.process_stage, 0) + 1

        # 최근 5개 알림
        recent_alerts = [
            {
                "alert_id": a.alert_id,
                "severity": a.severity,
                "process": a.process_stage,
                "issue": a.issue_description,
                "timestamp": a.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            for a in self.alert_history[-5:]
        ]

        return {
            "total_alerts": len(self.alert_history),
            "by_severity": by_severity,
            "by_process": by_process,
            "recent_alerts": recent_alerts
        }

    def get_agent_state(self) -> Dict:
        """에이전트 상태 정보 반환"""
        return {
            "agent_type": "ProcessMonitorAgent",
            "status": "active",
            "total_alerts_issued": len(self.alert_history),
            "thresholds": self.thresholds,
            "alert_summary": self.get_alert_summary()
        }


# 모듈 테스트용
if __name__ == "__main__":
    logger.info("ProcessMonitorAgent 테스트 시작")

    agent = ProcessMonitorAgent()

    print("\n" + "=" * 60)
    print("프레스 공정 모니터링 테스트")
    print("=" * 60)

    # 정상 데이터
    print("\n[1] 정상 데이터:")
    press_data, alert = agent.monitor_press_process(force_anomaly=False)
    print(f"두께: {press_data.thickness:.4f}mm")
    print(f"알림: {alert}")

    # 이상 데이터
    print("\n[2] 이상 데이터 (자동 발생):")
    for i in range(3):
        press_data, alert = agent.monitor_press_process(force_anomaly=True)
        if alert:
            print(f"\n알림 ID: {alert.alert_id}")
            print(f"심각도: {alert.severity}")
            print(f"문제: {alert.issue_description}")
            print(f"두께 편차: {alert.current_values['thickness_deviation']:.4f}mm")
            print(f"권장 조치: {alert.recommended_action}")

    # 용접 공정 모니터링
    print("\n" + "=" * 60)
    print("용접 공정 모니터링 테스트")
    print("=" * 60)

    weld_data = agent.simulator.generate_welding_data(
        upstream_thickness=2.03  # 편차가 큰 두께
    )
    quality_impact = agent.simulator.calculate_quality_impact(
        press_data,
        weld_data
    )

    print(f"\n품질 점수: {quality_impact['quality_score']:.2%}")
    alert = agent.monitor_welding_process(
        weld_data,
        quality_impact['quality_score']
    )

    if alert:
        print(f"알림 ID: {alert.alert_id}")
        print(f"심각도: {alert.severity}")
        print(f"권장 조치: {alert.recommended_action}")

    # 요약 정보
    print("\n" + "=" * 60)
    print("알림 요약")
    print("=" * 60)
    summary = agent.get_alert_summary()
    print(f"총 알림 수: {summary['total_alerts']}")
    print(f"심각도별: {summary['by_severity']}")
    print(f"공정별: {summary['by_process']}")

    logger.info("테스트 완료")
