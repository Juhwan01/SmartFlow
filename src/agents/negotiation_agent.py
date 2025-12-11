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
from config.data_schema import get_schema
from src.prompts import PromptGenerator


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

        # 스키마 및 프롬프트 생성기 초기화
        self.schema = get_schema()
        self.prompt_generator = PromptGenerator(self.schema)

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

        # LLM 프롬프트 생성 (PromptGenerator 사용)
        system_prompt = self.prompt_generator.generate_negotiation_system_prompt()

        # 예측 결과를 딕셔너리로 변환
        prediction_dict = {
            'predicted_value': prediction.predicted_strength,
            'quality_score': prediction.predicted_quality_score
        }

        user_prompt = self.prompt_generator.generate_user_prompt_for_negotiation(
            current_issue=current_issue,
            current_data=process_data,
            prediction=prediction_dict,
            rag_context=rag_context
        )

        # LLM 호출
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        llm_analysis = response.content

        logger.info("LLM 분석 완료")
        logger.debug(f"LLM 응답 (처음 500자):\n{llm_analysis[:500]}")

        # 조정안 파싱
        adjustments = self._parse_adjustments_from_llm(llm_analysis, prediction)

        # 조정 크기에 따른 예상 품질 개선 계산
        # 조정이 클수록 더 큰 개선 기대 (물리적 근사)
        adjustment_magnitude = abs(adjustments.get("welding_speed", 0)) + \
                              abs(adjustments.get("current", 0)) + \
                              abs(adjustments.get("pressure", 0))
        
        # 조정 크기에 비례한 품질 개선 (최소 3%, 최대 15%)
        expected_improvement = min(0.15, max(0.03, adjustment_magnitude * 0.5))
        expected_quality = min(1.0, prediction.predicted_quality_score + expected_improvement)

        # 제안 생성
        proposal = AdjustmentProposal(
            proposal_id=self._generate_proposal_id(),
            adjustments=adjustments,
            expected_quality=expected_quality,
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

        JSON 형식 응답을 파싱하거나, 실패 시 위험도 기반 기본값 사용
        """
        import json
        import re
        
        adjustments = {}
        
        # 1. JSON 파싱 시도
        try:
            # JSON 블록 추출 (```json ... ``` 또는 { ... } 형태)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*?"adjustments".*?\})', llm_response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                
                # adjustments 추출 및 변환 (백분율 → 소수)
                if "adjustments" in data:
                    for param, value in data["adjustments"].items():
                        adjustments[param] = float(value) / 100.0
                    
                    logger.info(f"✓ JSON 파싱 성공: {adjustments}")
                    return adjustments
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        except Exception as e:
            logger.warning(f"JSON 처리 오류: {e}")
        
        # 2. 정규표현식 폴백
        patterns = {
            "welding_speed": [
                r'"welding_speed":\s*([+-]?\d+\.?\d*)',
                r"welding[_\s]?speed[:\s]+([+-]?\d+\.?\d*)\s*%",
            ],
            "current": [
                r'"current":\s*([+-]?\d+\.?\d*)',
                r"current[:\s]+([+-]?\d+\.?\d*)\s*%",
            ],
            "pressure": [
                r'"pressure":\s*([+-]?\d+\.?\d*)',
                r"pressure[:\s]+([+-]?\d+\.?\d*)\s*%",
            ]
        }
        
        for param, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, llm_response, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    # 값이 -1~1 범위면 이미 소수, 아니면 백분율로 간주
                    if abs(value) > 1:
                        value = value / 100.0
                    adjustments[param] = value
                    logger.info(f"✓ 정규식 파싱: {param} = {value:.3f}")
                    break
        
        # 3. 파싱 실패 시 위험도 기반 기본 조정값 사용
        if not adjustments:
            logger.warning(f"LLM 응답 파싱 실패. 기본값 사용.")
            logger.debug(f"LLM 응답:\n{llm_response[:500]}")
            if prediction.risk_level == "low":
                adjustments = {"welding_speed": -0.02, "pressure": 0.01}
            elif prediction.risk_level == "medium":
                adjustments = {"welding_speed": -0.05, "current": 0.03, "pressure": 0.02}
            elif prediction.risk_level == "high":
                adjustments = {"welding_speed": -0.08, "current": 0.05, "pressure": 0.04}
            else:  # critical
                adjustments = {"welding_speed": -0.12, "current": 0.08, "pressure": 0.06}

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
