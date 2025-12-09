"""
RAG-enabled Negotiation Agent

LLM과 RAG를 활용하여 품질 예측 결과를 해석하고 파라미터 조정안을 제안/협상하는 에이전트
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

from src.rag.retriever import RAGRetriever
from src.agents.quality_predictor import QualityPrediction
from config import settings


@dataclass
class AdjustmentProposal:
    """파라미터 조정 제안"""
    proposal_id: str
    adjustments: Dict[str, float]  # 예: {"welding_speed": -0.05, "current": 0.03}
    expected_quality: float
    rationale: str  # LLM이 생성한 근거
    risk_assessment: str
    alternative_options: Optional[List[Dict]] = None


@dataclass
class NegotiationMessage:
    """협상 메시지"""
    from_agent: str
    to_agent: str
    message_type: str  # "proposal", "counter_proposal", "accept", "reject"
    content: str
    proposal: Optional[AdjustmentProposal] = None


class NegotiationAgent:
    """RAG 기반 협상 에이전트"""

    def __init__(
        self,
        agent_id: str = "negotiation_agent",
        rag_retriever: Optional[RAGRetriever] = None
    ):
        """
        Args:
            agent_id: 에이전트 ID
            rag_retriever: RAG 검색기 (None이면 새로 생성)
        """
        self.agent_id = agent_id
        self.rag_retriever = rag_retriever or RAGRetriever()

        # RAG 초기화
        if not self.rag_retriever.initialized:
            self.rag_retriever.initialize()

        # LLM 초기화
        self.llm = self._initialize_llm()

        self.negotiation_history: List[NegotiationMessage] = []
        self.proposal_counter = 0

        logger.info(f"NegotiationAgent '{agent_id}' 초기화 완료")

    def _initialize_llm(self):
        """LLM 초기화"""
        if settings.llm_provider == "openai":
            return ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                api_key=settings.openai_api_key
            )
        elif settings.llm_provider == "anthropic":
            return ChatAnthropic(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                api_key=settings.anthropic_api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    def _generate_proposal_id(self) -> str:
        """제안 ID 생성"""
        self.proposal_counter += 1
        return f"PROP-{self.agent_id}-{self.proposal_counter:04d}"

    def analyze_situation_and_propose(
        self,
        current_issue: str,
        prediction: QualityPrediction,
        process_data: Dict
    ) -> AdjustmentProposal:
        """
        상황 분석 및 조정안 제안

        Args:
            current_issue: 현재 발생한 문제 설명
            prediction: 품질 예측 결과
            process_data: 공정 데이터

        Returns:
            조정 제안
        """
        logger.info("LLM을 통한 상황 분석 시작...")

        # RAG로 유사 사례 검색
        rag_context = self.rag_retriever.build_context_for_llm(
            current_situation=current_issue,
            n_success=2,
            n_failure=1
        )

        # LLM 프롬프트 구성
        system_prompt = """당신은 스마트 제조 시스템의 품질 관리 전문가입니다.
과거 사례 데이터베이스(RAG)를 참고하여, 현재 발생한 품질 문제에 대한 최적의 파라미터 조정안을 제안해야 합니다.

당신의 역할:
1. ML 모델이 예측한 품질 저하를 언어적으로 해석
2. 과거 성공/실패 사례를 분석하여 교훈 도출
3. 구체적인 파라미터 조정값 제안
4. 제안의 근거와 위험 요소 설명

중요:
- 실패 사례에서 나타난 위험 요소는 피해야 합니다.
- 성공 사례의 패턴을 따르되, 현재 상황에 맞게 조정하세요.
- 파라미터 조정값은 백분율(%)로 제시하세요.
"""

        user_prompt = f"""## 현재 상황
{current_issue}

## ML 예측 결과
- 예상 품질 점수: {prediction.predicted_quality_score:.2%}
- 예상 강도: {prediction.predicted_strength:.2f}MPa
- 강도 저하: {prediction.strength_degradation_pct:.2f}%
- 위험 수준: {prediction.risk_level}

## 공정 데이터
{process_data}

{rag_context}

## 요청사항
위 정보를 바탕으로, 다음 형식으로 조정안을 제안하세요:

1. **상황 해석**: 현재 문제의 핵심 원인과 영향 설명
2. **과거 사례 분석**: 유사 사례에서 얻은 교훈
3. **조정안 제안**:
   - welding_speed (용접 속도): X% 조정
   - current (전류): Y% 조정
   - pressure (압력): Z% 조정
4. **근거**: 왜 이 조정안이 효과적인지 설명
5. **위험 평가**: 이 조정안의 잠재적 위험과 완화 방안
6. **대안**: 다른 접근 방법이 있다면 간단히 제시

응답은 명확하고 간결하게 작성하세요.
"""

        # LLM 호출
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        llm_analysis = response.content

        logger.info("LLM 분석 완료")

        # 조정안 파싱 (간단한 휴리스틱 사용, 실제로는 더 정교한 파싱 필요)
        adjustments = self._parse_adjustments_from_llm(llm_analysis, prediction)

        # 제안 생성
        proposal = AdjustmentProposal(
            proposal_id=self._generate_proposal_id(),
            adjustments=adjustments,
            expected_quality=prediction.predicted_quality_score + 0.05,  # 예상 개선
            rationale=llm_analysis,
            risk_assessment=prediction.risk_level
        )

        logger.info(
            f"조정안 생성 완료 - {proposal.proposal_id}: {adjustments}"
        )

        return proposal

    def _parse_adjustments_from_llm(
        self,
        llm_response: str,
        prediction: QualityPrediction
    ) -> Dict[str, float]:
        """
        LLM 응답에서 조정값 파싱

        실제로는 구조화된 출력(JSON)을 사용하는 것이 좋지만,
        MVP에서는 간단한 휴리스틱 사용
        """
        # 예측 기반 기본 조정안
        adjustments = {}

        # 위험도에 따른 기본 조정
        if prediction.risk_level == "low":
            adjustments = {"welding_speed": -0.02, "pressure": 0.01}
        elif prediction.risk_level == "medium":
            adjustments = {"welding_speed": -0.05, "current": 0.03, "pressure": 0.02}
        elif prediction.risk_level == "high":
            adjustments = {"welding_speed": -0.07, "current": 0.05, "pressure": 0.03}
        else:  # critical
            adjustments = {"welding_speed": -0.10, "current": 0.07, "pressure": 0.05}

        # LLM 응답에서 수정 (실제로는 정규표현식이나 JSON 파싱 사용)
        # MVP에서는 기본값 사용

        return adjustments

    def negotiate_with_process_agent(
        self,
        proposal: AdjustmentProposal,
        process_agent_constraints: Dict
    ) -> NegotiationMessage:
        """
        공정 에이전트와 협상

        Args:
            proposal: 초기 제안
            process_agent_constraints: 공정 에이전트의 제약 조건

        Returns:
            협상 메시지
        """
        logger.info(f"협상 시작: {proposal.proposal_id}")

        # 간단한 협상 로직 (MVP)
        # 실제로는 더 복잡한 multi-turn 협상 가능

        # 제약 조건 확인
        speed_adjustment = proposal.adjustments.get("welding_speed", 0)
        max_speed_reduction = process_agent_constraints.get("max_speed_reduction", -0.10)

        if speed_adjustment < max_speed_reduction:
            # 제약 조건 위반 - 재협상
            message = NegotiationMessage(
                from_agent=self.agent_id,
                to_agent="welding_process_agent",
                message_type="counter_proposal",
                content=f"초기 제안이 공정 제약을 초과합니다. 속도 감소를 {max_speed_reduction:.1%}로 조정하고, 대신 압력을 더 높이겠습니다.",
                proposal=AdjustmentProposal(
                    proposal_id=self._generate_proposal_id(),
                    adjustments={
                        "welding_speed": max_speed_reduction,
                        "current": proposal.adjustments.get("current", 0) + 0.01,
                        "pressure": proposal.adjustments.get("pressure", 0) + 0.01,
                    },
                    expected_quality=proposal.expected_quality - 0.01,
                    rationale="공정 제약 준수를 위한 대안",
                    risk_assessment=proposal.risk_assessment
                )
            )
        else:
            # 제약 조건 만족 - 제안 수용
            message = NegotiationMessage(
                from_agent=self.agent_id,
                to_agent="welding_process_agent",
                message_type="proposal",
                content=f"다음 조정안을 제안합니다: {proposal.adjustments}",
                proposal=proposal
            )

        self.negotiation_history.append(message)
        return message

    def get_negotiation_summary(self) -> Dict:
        """협상 이력 요약"""
        return {
            "total_messages": len(self.negotiation_history),
            "total_proposals": self.proposal_counter,
            "recent_messages": [
                {
                    "from": msg.from_agent,
                    "to": msg.to_agent,
                    "type": msg.message_type,
                    "content": msg.content[:100] + "..."
                }
                for msg in self.negotiation_history[-5:]
            ]
        }


# 모듈 테스트용
if __name__ == "__main__":
    from src.agents.quality_predictor import QualityCascadePredictor
    from src.data.sensor_simulator import SensorSimulator

    logger.info("NegotiationAgent 테스트 시작")

    # 시뮬레이터 및 예측기 초기화
    simulator = SensorSimulator(seed=42)
    predictor = QualityCascadePredictor()

    # 협상 에이전트 초기화
    agent = NegotiationAgent(agent_id="negotiator-01")

    print("\n" + "=" * 70)
    print("테스트: 이상 상황 발생 및 조정안 제안")
    print("=" * 70)

    # 이상 데이터 생성
    press_data = simulator.generate_press_data(force_anomaly=True, anomaly_magnitude=2.5)
    print(f"프레스 두께: {press_data.thickness:.4f}mm")
    print(f"두께 편차: {abs(press_data.thickness - 2.0):.4f}mm")

    # 품질 예측
    prediction = predictor.predict_from_press_data(press_data)
    print(f"\n예측 품질: {prediction.predicted_quality_score:.2%}")
    print(f"위험 수준: {prediction.risk_level}")

    # 상황 설명
    current_issue = f"""
프레스 공정에서 두께 편차 {abs(press_data.thickness - 2.0):.4f}mm가 감지되었습니다.
이로 인해 용접 품질이 {prediction.predicted_quality_score:.2%}로 저하될 것으로 예측됩니다.
목표 품질 {settings.quality_threshold:.0%}에 미달하는 상황입니다.
    """.strip()

    process_data = {
        "thickness": f"{press_data.thickness:.4f}mm",
        "pressure": f"{press_data.pressure:.2f}MPa",
        "temperature": f"{press_data.temperature:.2f}°C",
    }

    # 조정안 제안
    try:
        print("\nLLM을 통한 조정안 생성 중...")
        proposal = agent.analyze_situation_and_propose(
            current_issue=current_issue,
            prediction=prediction,
            process_data=process_data
        )

        print(f"\n[제안 ID] {proposal.proposal_id}")
        print(f"[조정안] {proposal.adjustments}")
        print(f"[예상 품질] {proposal.expected_quality:.2%}")
        print(f"\n[LLM 분석]\n{proposal.rationale[:500]}...")

        # 협상 시뮬레이션
        print("\n" + "=" * 70)
        print("협상 시뮬레이션")
        print("=" * 70)

        constraints = {
            "max_speed_reduction": -0.08,
            "max_current_increase": 0.10
        }

        negotiation_msg = agent.negotiate_with_process_agent(proposal, constraints)
        print(f"[{negotiation_msg.message_type}]")
        print(f"From: {negotiation_msg.from_agent}")
        print(f"To: {negotiation_msg.to_agent}")
        print(f"Content: {negotiation_msg.content}")

    except Exception as e:
        logger.error(f"LLM 호출 실패 (API 키 확인 필요): {e}")
        print(f"\n⚠️  LLM 호출 실패: {e}")
        print("API 키를 .env 파일에 설정해주세요.")

    logger.info("테스트 완료")
