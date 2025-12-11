"""
데이터셋 구조 정의 모듈

데이터셋 변경 시 이 파일만 수정하면 전체 시스템에 적용됨
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from pathlib import Path


@dataclass
class ProcessStageSchema:
    """공정 단계 스키마"""
    name: str
    measurement_variables: List[str]  # 측정 변수
    control_variables: Optional[List[str]] = None  # 제어 변수 (있는 경우)


@dataclass
class DataSchema:
    """
    데이터셋 구조 정의

    데이터셋마다 이 클래스를 상속받아 구체적인 스키마 정의
    """
    # 데이터셋 이름
    dataset_name: str

    # 공정 단계들
    stage1: ProcessStageSchema  # 1차 공정 (예: 프레스)
    stage2: ProcessStageSchema  # 2차 공정 (예: 용접)

    # 타겟 변수
    target_variable: str

    # 품질 스펙
    lsl: float  # Lower Specification Limit
    usl: float  # Upper Specification Limit
    target: float  # Target value

    # 제어 변수 → 측정 변수 매핑
    # 예: {"current": "welding_temp1", "welding_speed": "welding_temp3"}
    control_to_measurement_mapping: Dict[str, str] = field(default_factory=dict)

    # Feature Engineering 레시피 (나중에 추가)
    feature_recipes: List[tuple] = field(default_factory=list)

    # 파생 변수 재계산 가능 여부
    recalculable_features: List[str] = field(default_factory=list)

    # 도메인 지식 (프롬프트 생성용)
    process_description: str = ""  # 공정 설명
    parameter_guidance: Dict[str, str] = field(default_factory=dict)  # 파라미터별 영향 설명

    def get_all_measurement_vars(self) -> List[str]:
        """모든 측정 변수 반환"""
        return (
            self.stage1.measurement_variables +
            self.stage2.measurement_variables
        )

    def get_all_control_vars(self) -> List[str]:
        """모든 제어 변수 반환"""
        control_vars = []
        if self.stage1.control_variables:
            control_vars.extend(self.stage1.control_variables)
        if self.stage2.control_variables:
            control_vars.extend(self.stage2.control_variables)
        return control_vars

    def get_measurement_var_for_control(self, control_var: str) -> Optional[str]:
        """제어 변수에 대응하는 측정 변수 반환"""
        return self.control_to_measurement_mapping.get(control_var)


# ============================================================
# Continuous Factory 데이터셋 스키마 (현재 사용 중)
# ============================================================

CONTINUOUS_FACTORY_SCHEMA = DataSchema(
    dataset_name="continuous_factory_process",

    # 1차 공정: 프레스
    stage1=ProcessStageSchema(
        name="press",
        measurement_variables=[
            "press_thickness",
            "press_measurement1",
            "press_measurement2",
        ],
        control_variables=None  # 제어 변수 없음 (측정만)
    ),

    # 2차 공정: 용접
    stage2=ProcessStageSchema(
        name="welding",
        measurement_variables=[
            "welding_temp1",
            "welding_temp2",
            "welding_temp3",
            "welding_temp4",
            "welding_temp5",
            "welding_pressure",
            "welding_control1",
            "welding_control2",
        ],
        control_variables=None  # 실제로는 없지만 개념적으로 매핑
    ),

    # 타겟 변수
    target_variable="welding_strength",

    # 품질 스펙 (LSL~USL)
    lsl=11.50,
    usl=13.20,
    target=12.35,

    # 제어 변수 → 측정 변수 매핑 (개념적 매핑)
    # 실제 제어 변수는 없지만, 시스템은 이 매핑을 사용
    control_to_measurement_mapping={
        "current": "welding_temp1",      # 전류(개념) → 온도1 센서
        "welding_speed": "welding_temp3",  # 속도(개념) → 온도3 센서
        "pressure": "welding_pressure",    # 압력(개념) → 압력 센서
    },

    # 조정 후 재계산 가능한 파생 변수들
    recalculable_features=[
        "heat_input_proxy",
        "welding_temp_mean",
        "welding_temp_span",
        "pressure_x_temp2",
        "total_control",
    ],

    # 도메인 지식
    process_description="자동차 부품 용접 공정 (프레스 → 용접)",
    parameter_guidance={
        "current": "전류 증가 → 열량 증가 → 용접 강도 증가 (과도 시 변형 위험)",
        "welding_speed": "속도 감소 → 입열 시간 증가 → 강도 증가 (생산성 저하)",
        "pressure": "압력 증가 → 밀착도 향상 → 강도 증가 (과도 시 재료 변형)"
    }
)


# ============================================================
# 향후 다른 데이터셋 추가 예시
# ============================================================

# 실제 제어 변수가 있는 데이터셋 예시
EXAMPLE_REAL_CONTROL_SCHEMA = DataSchema(
    dataset_name="welding_with_control_vars",

    stage1=ProcessStageSchema(
        name="press",
        measurement_variables=["press_thickness"],
        control_variables=["press_force_setpoint"]
    ),

    stage2=ProcessStageSchema(
        name="welding",
        measurement_variables=["welding_temp", "welding_current_actual"],
        control_variables=["welding_current_setpoint", "welding_speed_setpoint"]
    ),

    target_variable="weld_strength",
    lsl=100.0,
    usl=150.0,
    target=125.0,

    # 실제 제어 변수 → 측정 변수 매핑
    control_to_measurement_mapping={
        "welding_current_setpoint": "welding_current_actual",
        "welding_speed_setpoint": "welding_temp",  # 속도 → 온도에 영향
    },

    recalculable_features=[]
)


def get_schema(dataset_name: str = "continuous_factory_process") -> DataSchema:
    """
    데이터셋 이름으로 스키마 반환

    Args:
        dataset_name: 데이터셋 이름

    Returns:
        DataSchema 객체
    """
    schemas = {
        "continuous_factory_process": CONTINUOUS_FACTORY_SCHEMA,
        "welding_with_control_vars": EXAMPLE_REAL_CONTROL_SCHEMA,
    }

    if dataset_name not in schemas:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(schemas.keys())}"
        )

    return schemas[dataset_name]
