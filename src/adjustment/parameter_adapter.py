"""
Parameter Adapter 모듈

제어 변수 ↔ 측정 변수 변환 및 조정값 적용
"""
from typing import Dict, Optional, List
from loguru import logger

from config.data_schema import DataSchema
from src.features.feature_engineer import FeatureEngineer


class ParameterAdapter:
    """
    파라미터 조정 어댑터

    제어 변수 조정 → 측정 변수 업데이트 → 파생 변수 재계산
    """

    def __init__(
        self,
        schema: DataSchema,
        feature_engineer: Optional[FeatureEngineer] = None
    ):
        """
        Args:
            schema: 데이터셋 스키마
            feature_engineer: 피처 엔지니어 (파생 변수 재계산용)
        """
        self.schema = schema
        self.feature_engineer = feature_engineer

        # 제어 변수 → 측정 변수 매핑
        self.control_to_measurement = schema.control_to_measurement_mapping

        logger.info(
            f"ParameterAdapter 초기화: {len(self.control_to_measurement)} 개 매핑"
        )

    def apply_control_adjustments(
        self,
        data: Dict[str, float],
        control_adjustments: Dict[str, float],
        recalculate_features: bool = True
    ) -> Dict[str, float]:
        """
        제어 변수 조정을 데이터에 적용

        Args:
            data: 원본 데이터 딕셔너리
            control_adjustments: 제어 변수 조정값 (예: {"current": 0.03, "speed": -0.05})
            recalculate_features: 파생 변수 재계산 여부

        Returns:
            조정이 적용된 데이터 딕셔너리
        """
        adjusted = data.copy()

        # ========================================
        # 1. 제어 변수 조정 → 측정 변수 업데이트
        # ========================================
        for control_var, adjustment_pct in control_adjustments.items():
            # 대응하는 측정 변수 찾기
            measurement_var = self.control_to_measurement.get(control_var)

            if measurement_var is None:
                logger.warning(
                    f"제어 변수 '{control_var}'에 대응하는 측정 변수 없음 (스키마 확인 필요)"
                )
                continue

            if measurement_var not in adjusted:
                logger.warning(
                    f"측정 변수 '{measurement_var}'가 데이터에 없음"
                )
                continue

            # 조정 적용 (백분율)
            original_value = adjusted[measurement_var]
            adjusted[measurement_var] = original_value * (1 + adjustment_pct)

            logger.debug(
                f"조정 적용: {control_var} ({measurement_var}) "
                f"{original_value:.4f} → {adjusted[measurement_var]:.4f} "
                f"({adjustment_pct:+.1%})"
            )

        # ========================================
        # 2. 파생 변수 재계산
        # ========================================
        if recalculate_features and self.feature_engineer is not None:
            adjusted = self.feature_engineer.recalculate_features(
                adjusted,
                feature_names=self.schema.recalculable_features
            )

        return adjusted

    def get_measurement_var(self, control_var: str) -> Optional[str]:
        """
        제어 변수에 대응하는 측정 변수 반환

        Args:
            control_var: 제어 변수 이름

        Returns:
            측정 변수 이름 (없으면 None)
        """
        return self.control_to_measurement.get(control_var)

    def get_all_control_vars(self) -> List[str]:
        """조정 가능한 모든 제어 변수 반환"""
        return list(self.control_to_measurement.keys())

    def get_adjustment_summary(
        self,
        original_data: Dict[str, float],
        adjusted_data: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        조정 요약 정보 생성

        Args:
            original_data: 원본 데이터
            adjusted_data: 조정된 데이터

        Returns:
            조정 요약 딕셔너리
        """
        summary = {}

        for control_var, measurement_var in self.control_to_measurement.items():
            if measurement_var not in original_data or measurement_var not in adjusted_data:
                continue

            original = original_data[measurement_var]
            adjusted = adjusted_data[measurement_var]
            change_pct = ((adjusted - original) / original) if original != 0 else 0

            summary[control_var] = {
                "measurement_var": measurement_var,
                "original": original,
                "adjusted": adjusted,
                "change": adjusted - original,
                "change_pct": change_pct
            }

        return summary
