"""
과거 작업 사례 데이터

RAG 시스템에서 사용할 과거 성공/실패 사례를 정의합니다.
"""
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class HistoricalCase:
    """과거 작업 사례"""
    case_id: str
    date: str
    process_stage: str  # "press" or "welding"
    issue_detected: str
    issue_severity: str  # "low", "medium", "high"
    action_taken: str
    parameters_adjusted: Dict[str, float]
    outcome: str  # "success" or "failure"
    quality_before: float  # 조치 전 품질 점수
    quality_after: float  # 조치 후 품질 점수
    notes: str
    lessons_learned: str


def get_historical_cases() -> List[HistoricalCase]:
    """과거 사례 목록 반환"""

    cases = [
        # 성공 사례들
        HistoricalCase(
            case_id="CASE-2024-001",
            date="2024-11-15",
            process_stage="press",
            issue_detected="두께 편차 +0.025mm 발생",
            issue_severity="medium",
            action_taken="용접 속도 5% 감소, 압력 2% 증가",
            parameters_adjusted={"welding_speed": -0.05, "pressure": 0.02},
            outcome="success",
            quality_before=0.88,
            quality_after=0.95,
            notes="프레스 공정에서 두께 편차가 발생했으나, 용접 파라미터 조정으로 품질 회복",
            lessons_learned="두께 편차 발생 시 속도를 줄이고 압력을 미세 조정하면 효과적"
        ),
        HistoricalCase(
            case_id="CASE-2024-002",
            date="2024-11-18",
            process_stage="press",
            issue_detected="두께 편차 +0.018mm, 압력 미세 상승",
            issue_severity="low",
            action_taken="용접 속도 3% 감소",
            parameters_adjusted={"welding_speed": -0.03},
            outcome="success",
            quality_before=0.91,
            quality_after=0.96,
            notes="작은 편차는 속도 조정만으로 해결 가능",
            lessons_learned="저강도 이상은 단일 파라미터 조정으로 충분"
        ),
        HistoricalCase(
            case_id="CASE-2024-003",
            date="2024-11-20",
            process_stage="press",
            issue_detected="두께 편차 +0.030mm",
            issue_severity="high",
            action_taken="용접 속도 7% 감소, 전류 3% 증가, 압력 3% 증가",
            parameters_adjusted={
                "welding_speed": -0.07,
                "current": 0.03,
                "pressure": 0.03
            },
            outcome="success",
            quality_before=0.85,
            quality_after=0.93,
            notes="높은 편차에는 다중 파라미터 조정 필요",
            lessons_learned="고강도 이상은 복합 조정이 효과적"
        ),

        # 실패 사례들 (교훈을 주는 사례)
        HistoricalCase(
            case_id="CASE-2024-004",
            date="2024-11-22",
            process_stage="press",
            issue_detected="두께 편차 +0.022mm",
            issue_severity="medium",
            action_taken="전류만 10% 증가 (과도한 조정)",
            parameters_adjusted={"current": 0.10},
            outcome="failure",
            quality_before=0.89,
            quality_after=0.75,
            notes="전류를 과도하게 높였더니 비드 균열 발생",
            lessons_learned="전류 단독 증가는 위험. 균형 잡힌 조정 필요"
        ),
        HistoricalCase(
            case_id="CASE-2024-005",
            date="2024-11-25",
            process_stage="press",
            issue_detected="두께 편차 +0.020mm",
            issue_severity="medium",
            action_taken="조치 없이 진행 (무대응)",
            parameters_adjusted={},
            outcome="failure",
            quality_before=0.90,
            quality_after=0.82,
            notes="중간 정도 편차를 무시했더니 후속 공정에서 품질 저하",
            lessons_learned="중간 강도 이상도 반드시 조치 필요"
        ),
        HistoricalCase(
            case_id="CASE-2024-006",
            date="2024-11-28",
            process_stage="press",
            issue_detected="두께 편차 +0.035mm (고강도)",
            issue_severity="high",
            action_taken="속도만 3% 감소 (불충분한 조정)",
            parameters_adjusted={"welding_speed": -0.03},
            outcome="failure",
            quality_before=0.83,
            quality_after=0.86,
            notes="고강도 이상에 대해 단일 파라미터 조정은 불충분",
            lessons_learned="심각한 이상은 복합 조정 전략 필요"
        ),

        # 추가 성공 사례
        HistoricalCase(
            case_id="CASE-2024-007",
            date="2024-12-01",
            process_stage="press",
            issue_detected="두께 편차 -0.020mm (음수 편차)",
            issue_severity="medium",
            action_taken="용접 속도 4% 증가, 압력 2% 감소",
            parameters_adjusted={"welding_speed": 0.04, "pressure": -0.02},
            outcome="success",
            quality_before=0.87,
            quality_after=0.94,
            notes="음수 편차는 양수 편차와 반대 방향으로 조정",
            lessons_learned="편차 방향에 따라 조정 방향도 반대로"
        ),
        HistoricalCase(
            case_id="CASE-2024-008",
            date="2024-12-03",
            process_stage="welding",
            issue_detected="용접 온도 과도 상승 (850°C)",
            issue_severity="medium",
            action_taken="전류 5% 감소, 속도 3% 증가",
            parameters_adjusted={"current": -0.05, "welding_speed": 0.03},
            outcome="success",
            quality_before=0.86,
            quality_after=0.92,
            notes="온도 상승 시 전류 감소와 속도 증가로 냉각 효과",
            lessons_learned="온도 이상은 전류-속도 동시 조정이 효과적"
        ),
        HistoricalCase(
            case_id="CASE-2024-009",
            date="2024-12-05",
            process_stage="press",
            issue_detected="압력 급증 (+15MPa)",
            issue_severity="high",
            action_taken="프레스 압력 재보정, 용접 파라미터 유지",
            parameters_adjusted={},
            outcome="success",
            quality_before=0.84,
            quality_after=0.95,
            notes="프레스 공정 자체 문제는 해당 공정에서 해결",
            lessons_learned="근본 원인 공정에서 직접 조치하는 것이 최선"
        ),
        HistoricalCase(
            case_id="CASE-2024-010",
            date="2024-12-07",
            process_stage="press",
            issue_detected="두께 편차 +0.015mm (경미)",
            issue_severity="low",
            action_taken="모니터링만 수행, 파라미터 조정 없음",
            parameters_adjusted={},
            outcome="success",
            quality_before=0.92,
            quality_after=0.93,
            notes="경미한 편차는 자연 회복 가능",
            lessons_learned="모든 이상에 과잉 반응할 필요 없음, 임계값 관리 중요"
        ),
    ]

    return cases


def save_cases_to_json(filepath: str = "data/historical_cases/cases.json"):
    """과거 사례를 JSON 파일로 저장"""
    cases = get_historical_cases()

    cases_dict = [
        {
            "case_id": case.case_id,
            "date": case.date,
            "process_stage": case.process_stage,
            "issue_detected": case.issue_detected,
            "issue_severity": case.issue_severity,
            "action_taken": case.action_taken,
            "parameters_adjusted": case.parameters_adjusted,
            "outcome": case.outcome,
            "quality_before": case.quality_before,
            "quality_after": case.quality_after,
            "notes": case.notes,
            "lessons_learned": case.lessons_learned,
        }
        for case in cases
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cases_dict, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(cases)}개 사례를 {filepath}에 저장했습니다.")


def get_cases_as_text() -> List[str]:
    """RAG 임베딩을 위한 텍스트 형식 변환"""
    cases = get_historical_cases()
    texts = []

    for case in cases:
        text = f"""
사례 ID: {case.case_id}
날짜: {case.date}
공정: {case.process_stage}
발견된 문제: {case.issue_detected}
문제 심각도: {case.issue_severity}
취한 조치: {case.action_taken}
조정된 파라미터: {case.parameters_adjusted}
결과: {case.outcome}
조치 전 품질: {case.quality_before:.2%}
조치 후 품질: {case.quality_after:.2%}
비고: {case.notes}
교훈: {case.lessons_learned}
        """.strip()
        texts.append(text)

    return texts


# 모듈 테스트용
if __name__ == "__main__":
    print("=== 과거 사례 데이터 생성 ===\n")

    cases = get_historical_cases()
    print(f"총 {len(cases)}개의 사례 생성")

    # 성공/실패 통계
    success_count = sum(1 for c in cases if c.outcome == "success")
    failure_count = sum(1 for c in cases if c.outcome == "failure")
    print(f"성공 사례: {success_count}개")
    print(f"실패 사례: {failure_count}개")

    # 심각도 통계
    severity_counts = {}
    for case in cases:
        severity_counts[case.issue_severity] = \
            severity_counts.get(case.issue_severity, 0) + 1
    print(f"\n심각도별 분포: {severity_counts}")

    # 샘플 출력
    print(f"\n=== 샘플 사례 (성공) ===")
    print(get_cases_as_text()[0])

    print(f"\n=== 샘플 사례 (실패) ===")
    print(get_cases_as_text()[3])

    # JSON 저장
    save_cases_to_json()
