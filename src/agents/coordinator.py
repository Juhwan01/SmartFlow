"""
Coordinator Agent

전체 생산 목표(품질, 비용, 납기)를 고려하여 에이전트 간 협상 결과를 최종 승인/반려하는 조정자
"""
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from src.agents.negotiation_agent import AdjustmentProposal
from config import settings


@dataclass
class ProductionGoals:
    """생산 목표"""
    target_quality: float = settings.quality_threshold  # 목표 품질 점수
    max_cycle_time_increase: float = 0.10  # 최대 사이클 타임 증가율 (10%)
    max_cost_increase: float = 0.05  # 최대 비용 증가율 (5%)
    min_throughput: int = 100  # 최소 시간당 생산량


@dataclass
class ApprovalDecision:
    """승인 결정"""
    decision_id: str
    proposal_id: str
    status: str  # "approved", "rejected", "conditional_approved"
    rationale: str
    conditions: Optional[List[str]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CoordinatorAgent:
    """조정자 에이전트"""

    def __init__(
        self,
        agent_id: str = "coordinator",
        production_goals: Optional[ProductionGoals] = None
    ):
        """
        Args:
            agent_id: 에이전트 ID
            production_goals: 생산 목표 (None이면 기본값 사용)
        """
        self.agent_id = agent_id
        self.production_goals = production_goals or ProductionGoals()

        self.decision_history: List[ApprovalDecision] = []
        self.decision_counter = 0

        logger.info(f"CoordinatorAgent '{agent_id}' 초기화 완료")
        logger.info(f"생산 목표 - 품질: {self.production_goals.target_quality:.0%}, "
                   f"최대 사이클 타임 증가: {self.production_goals.max_cycle_time_increase:.0%}")

    def _generate_decision_id(self) -> str:
        """결정 ID 생성"""
        self.decision_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"DEC-{timestamp}-{self.decision_counter:04d}"

    def _estimate_cycle_time_impact(
        self,
        adjustments: Dict[str, float]
    ) -> float:
        """
        조정안이 사이클 타임에 미치는 영향 추정

        Args:
            adjustments: 파라미터 조정값

        Returns:
            사이클 타임 증가율 (음수면 감소)
        """
        # 용접 속도 감소는 사이클 타임 증가
        speed_adjustment = adjustments.get("welding_speed", 0)

        # 속도가 5% 감소하면 사이클 타임이 약 5% 증가한다고 가정
        cycle_time_impact = -speed_adjustment

        return cycle_time_impact

    def _estimate_cost_impact(
        self,
        adjustments: Dict[str, float]
    ) -> float:
        """
        조정안이 비용에 미치는 영향 추정

        Args:
            adjustments: 파라미터 조정값

        Returns:
            비용 증가율
        """
        # 전류 증가는 에너지 비용 증가
        current_adjustment = adjustments.get("current", 0)

        # 압력 증가는 유지보수 비용 증가
        pressure_adjustment = adjustments.get("pressure", 0)

        # 단순화된 비용 모델
        cost_impact = (current_adjustment * 0.5) + (pressure_adjustment * 0.3)

        return cost_impact

    def evaluate_proposal(
        self,
        proposal: AdjustmentProposal,
        current_quality_score: float
    ) -> ApprovalDecision:
        """
        제안 평가 및 승인/반려 결정

        Args:
            proposal: 조정 제안
            current_quality_score: 현재 품질 점수

        Returns:
            승인 결정
        """
        logger.info(f"제안 평가 시작: {proposal.proposal_id}")

        decision_id = self._generate_decision_id()
        rationale_parts = []
        conditions = []

        # 1. 품질 목표 확인
        quality_check = proposal.expected_quality >= self.production_goals.target_quality

        if quality_check:
            rationale_parts.append(
                f"✅ 품질 목표 충족 (예상: {proposal.expected_quality:.2%}, "
                f"목표: {self.production_goals.target_quality:.2%})"
            )
        else:
            rationale_parts.append(
                f"⚠️  품질 목표 미달 (예상: {proposal.expected_quality:.2%}, "
                f"목표: {self.production_goals.target_quality:.2%})"
            )

        # 2. 사이클 타임 영향 확인
        cycle_time_impact = self._estimate_cycle_time_impact(proposal.adjustments)
        cycle_time_check = cycle_time_impact <= self.production_goals.max_cycle_time_increase

        if cycle_time_check:
            rationale_parts.append(
                f"✅ 사이클 타임 영향 허용 범위 내 "
                f"(증가율: {cycle_time_impact:.1%}, "
                f"한계: {self.production_goals.max_cycle_time_increase:.1%})"
            )
        else:
            rationale_parts.append(
                f"⚠️  사이클 타임 증가 과다 "
                f"(증가율: {cycle_time_impact:.1%}, "
                f"한계: {self.production_goals.max_cycle_time_increase:.1%})"
            )
            conditions.append("사이클 타임 영향 재검토 필요")

        # 3. 비용 영향 확인
        cost_impact = self._estimate_cost_impact(proposal.adjustments)
        cost_check = cost_impact <= self.production_goals.max_cost_increase

        if cost_check:
            rationale_parts.append(
                f"✅ 비용 영향 허용 범위 내 "
                f"(증가율: {cost_impact:.1%}, "
                f"한계: {self.production_goals.max_cost_increase:.1%})"
            )
        else:
            rationale_parts.append(
                f"⚠️  비용 증가 과다 "
                f"(증가율: {cost_impact:.1%}, "
                f"한계: {self.production_goals.max_cost_increase:.1%})"
            )
            conditions.append("비용 영향 재검토 필요")

        # 4. 위험 수준 확인
        if proposal.risk_assessment in ["critical", "high"]:
            rationale_parts.append(
                f"⚠️  위험 수준 높음 ({proposal.risk_assessment})"
            )
            conditions.append("위험 모니터링 강화")

        # 최종 결정
        if quality_check and cycle_time_check and cost_check:
            status = "approved"
            decision_message = "✅ 제안 승인"
        elif quality_check:
            status = "conditional_approved"
            decision_message = "⚠️  조건부 승인 (조건 충족 시 실행)"
        else:
            status = "rejected"
            decision_message = "❌ 제안 반려"

        rationale = f"{decision_message}\n\n" + "\n".join(rationale_parts)

        decision = ApprovalDecision(
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            status=status,
            rationale=rationale,
            conditions=conditions if conditions else None
        )

        self.decision_history.append(decision)

        logger.info(
            f"결정 완료 - {decision_id}: {status} "
            f"(품질: {quality_check}, 시간: {cycle_time_check}, 비용: {cost_check})"
        )

        return decision

    def get_approval_statistics(self) -> Dict:
        """승인 통계"""
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "approved": 0,
                "rejected": 0,
                "conditional_approved": 0,
            }

        stats = {
            "total_decisions": len(self.decision_history),
            "approved": 0,
            "rejected": 0,
            "conditional_approved": 0,
        }

        for decision in self.decision_history:
            stats[decision.status] += 1

        stats["approval_rate"] = (
            stats["approved"] / stats["total_decisions"]
            if stats["total_decisions"] > 0 else 0
        )

        return stats

    def get_agent_state(self) -> Dict:
        """에이전트 상태"""
        return {
            "agent_type": "CoordinatorAgent",
            "agent_id": self.agent_id,
            "status": "active",
            "production_goals": {
                "target_quality": self.production_goals.target_quality,
                "max_cycle_time_increase": self.production_goals.max_cycle_time_increase,
                "max_cost_increase": self.production_goals.max_cost_increase,
                "min_throughput": self.production_goals.min_throughput,
            },
            "statistics": self.get_approval_statistics()
        }


# 모듈 테스트용
if __name__ == "__main__":
    from src.agents.negotiation_agent import AdjustmentProposal

    logger.info("CoordinatorAgent 테스트 시작")

    coordinator = CoordinatorAgent()

    print("\n" + "=" * 70)
    print("테스트 1: 양호한 제안 (승인 예상)")
    print("=" * 70)

    good_proposal = AdjustmentProposal(
        proposal_id="PROP-TEST-001",
        adjustments={"welding_speed": -0.05, "current": 0.03, "pressure": 0.02},
        expected_quality=0.95,
        rationale="과거 성공 사례 기반 조정",
        risk_assessment="medium"
    )

    decision1 = coordinator.evaluate_proposal(good_proposal, current_quality_score=0.88)
    print(f"결정 ID: {decision1.decision_id}")
    print(f"상태: {decision1.status}")
    print(f"\n근거:\n{decision1.rationale}")
    if decision1.conditions:
        print(f"\n조건: {decision1.conditions}")

    print("\n" + "=" * 70)
    print("테스트 2: 과도한 조정 제안 (조건부 승인 또는 반려 예상)")
    print("=" * 70)

    excessive_proposal = AdjustmentProposal(
        proposal_id="PROP-TEST-002",
        adjustments={"welding_speed": -0.15, "current": 0.10, "pressure": 0.08},
        expected_quality=0.96,
        rationale="공격적 조정안",
        risk_assessment="high"
    )

    decision2 = coordinator.evaluate_proposal(excessive_proposal, current_quality_score=0.85)
    print(f"결정 ID: {decision2.decision_id}")
    print(f"상태: {decision2.status}")
    print(f"\n근거:\n{decision2.rationale}")
    if decision2.conditions:
        print(f"\n조건: {decision2.conditions}")

    print("\n" + "=" * 70)
    print("테스트 3: 불충분한 품질 개선 제안 (반려 예상)")
    print("=" * 70)

    insufficient_proposal = AdjustmentProposal(
        proposal_id="PROP-TEST-003",
        adjustments={"welding_speed": -0.02},
        expected_quality=0.88,  # 목표 0.90 미달
        rationale="최소한의 조정",
        risk_assessment="low"
    )

    decision3 = coordinator.evaluate_proposal(insufficient_proposal, current_quality_score=0.86)
    print(f"결정 ID: {decision3.decision_id}")
    print(f"상태: {decision3.status}")
    print(f"\n근거:\n{decision3.rationale}")

    # 통계
    print("\n" + "=" * 70)
    print("승인 통계")
    print("=" * 70)
    stats = coordinator.get_approval_statistics()
    print(f"총 결정 수: {stats['total_decisions']}")
    print(f"승인: {stats['approved']}")
    print(f"조건부 승인: {stats['conditional_approved']}")
    print(f"반려: {stats['rejected']}")
    print(f"승인율: {stats['approval_rate']:.1%}")

    logger.info("테스트 완료")
