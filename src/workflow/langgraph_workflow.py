"""
LangGraph 기반 Multi-Agent 워크플로우

전체 시스템을 조율하는 워크플로우
"""
from typing import TypedDict, Annotated, Dict, Any
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END
from loguru import logger

from src.data.sensor_simulator import SensorSimulator
from src.data.case_logger import CaseLogger
from src.agents.process_monitor import ProcessMonitorAgent
from src.agents.quality_predictor import QualityCascadePredictor
from src.agents.negotiation_agent import NegotiationAgent
from src.agents.coordinator import CoordinatorAgent, ProductionGoals
from src.rag.retriever import RAGRetriever


# 전역 상태 정의
class WorkflowState(TypedDict):
    """워크플로우 상태"""
    # 센서 데이터
    press_data: Dict[str, Any]
    welding_data: Dict[str, Any]

    # 모니터링 결과
    alert: Dict[str, Any]

    # 예측 결과
    prediction: Dict[str, Any]

    # 협상 결과
    proposal: Dict[str, Any]
    negotiation_log: list

    # 최종 결정
    decision: Dict[str, Any]

    # 실행 결과
    execution_result: Dict[str, Any]

    # 메타 정보
    workflow_status: str
    error_message: str


class SmartFlowWorkflow:
    """SmartFlow Multi-Agent 워크플로우"""

    def __init__(self):
        """워크플로우 초기화"""
        logger.info("SmartFlow Workflow 초기화 시작...")

        # 컴포넌트 초기화
        self.simulator = SensorSimulator()
        self.monitor = ProcessMonitorAgent(simulator=self.simulator)
        self.predictor = QualityCascadePredictor()
        self.rag_retriever = RAGRetriever()
        self.negotiator = NegotiationAgent(rag_retriever=self.rag_retriever)
        self.coordinator = CoordinatorAgent()
        self.case_logger = CaseLogger()

        # RAG 초기화
        self.rag_retriever.initialize()

        # 워크플로우 그래프 구성
        self.workflow = self._build_workflow()

        logger.info("SmartFlow Workflow 초기화 완료")

    def _should_continue_to_predict(self, state: WorkflowState) -> str:
        """
        프레스 이상 여부에 따라 다음 단계 결정 (MVP 2단계 cascade detection)

        프레스 정상이면 조정 불필요 → 즉시 종료
        프레스 이상이면 품질 예측 진행 → cascade effect 분석
        """
        press_data = state.get("press_data", {})
        is_anomaly = press_data.get("is_anomaly", False)

        if is_anomaly:
            logger.info("프레스 이상 감지 → 품질 예측 단계로 진행")
            return "predict"
        else:
            logger.info("프레스 정상 → 조정 불필요, 워크플로우 종료")
            state["workflow_status"] = "normal_operation_no_adjustment"
            return END

    def _build_workflow(self) -> StateGraph:
        """워크플로우 그래프 구성 (MVP 2단계 cascade detection)"""
        workflow = StateGraph(WorkflowState)

        # 노드 추가
        workflow.add_node("monitor", self._monitor_node)
        workflow.add_node("predict", self._predict_node)
        workflow.add_node("negotiate", self._negotiate_node)
        workflow.add_node("coordinate", self._coordinate_node)
        workflow.add_node("execute", self._execute_node)

        # 엣지 추가 (MVP 시나리오: 프레스 정상이면 즉시 종료)
        workflow.set_entry_point("monitor")

        # 조건부 라우팅: 프레스 이상 여부에 따라 분기
        workflow.add_conditional_edges(
            "monitor",
            self._should_continue_to_predict,
            {
                "predict": "predict",
                END: END
            }
        )

        workflow.add_edge("predict", "negotiate")
        workflow.add_edge("negotiate", "coordinate")
        workflow.add_edge("coordinate", "execute")
        workflow.add_edge("execute", END)

        return workflow.compile()

    def _monitor_node(self, state: WorkflowState) -> WorkflowState:
        """1단계: 공정 모니터링"""
        logger.info("[1/5] 공정 모니터링 시작")

        # 프레스 공정 모니터링
        press_data, alert = self.monitor.monitor_press_process(force_anomaly=True)

        state["press_data"] = {
            "thickness": press_data.thickness,
            "pressure": press_data.pressure,
            "temperature": press_data.temperature,
            "is_anomaly": press_data.is_anomaly,
            "anomaly_type": press_data.anomaly_type,
        }

        if alert:
            state["alert"] = {
                "alert_id": alert.alert_id,
                "severity": alert.severity,
                "issue_description": alert.issue_description,
                "current_values": alert.current_values,
                "recommended_action": alert.recommended_action,
            }
            logger.warning(f"이상 감지: {alert.alert_id} - {alert.severity}")
        else:
            state["alert"] = None
            logger.info("정상 운영 중")

        state["workflow_status"] = "monitoring_complete"
        return state

    def _predict_node(self, state: WorkflowState) -> WorkflowState:
        """2단계: 품질 예측"""
        logger.info("[2/5] 품질 예측 시작")

        # 더미 PressSensorData 생성
        from src.data.sensor_simulator import PressSensorData
        from datetime import datetime

        press_data_obj = PressSensorData(
            timestamp=datetime.now(),
            thickness=state["press_data"]["thickness"],
            pressure=state["press_data"]["pressure"],
            temperature=state["press_data"]["temperature"],
            cycle_time=3.0,
            is_anomaly=state["press_data"]["is_anomaly"],
            anomaly_type=state["press_data"].get("anomaly_type"),
        )

        # 품질 예측
        prediction = self.predictor.predict_from_press_data(press_data_obj)

        state["prediction"] = {
            "predicted_quality_score": prediction.predicted_quality_score,
            "predicted_strength": prediction.predicted_strength,
            "strength_degradation_pct": prediction.strength_degradation_pct,
            "confidence": prediction.confidence,
            "risk_level": prediction.risk_level,
            "recommendation": prediction.recommendation,
        }

        logger.info(
            f"품질 예측 완료 - 예상 품질: {prediction.predicted_quality_score:.2%}, "
            f"위험도: {prediction.risk_level}"
        )

        state["workflow_status"] = "prediction_complete"
        return state

    def _negotiate_node(self, state: WorkflowState) -> WorkflowState:
        """3단계: 협상 및 조정안 제안"""
        logger.info("[3/5] 협상 및 조정안 제안 시작")

        # LLM을 사용하지 않는 경우 스킵
        try:
            # 현재 상황 설명
            press_data = state["press_data"]
            prediction = state["prediction"]

            current_issue = f"""
프레스 공정에서 두께 편차 {abs(press_data['thickness'] - 2.0):.4f}mm가 감지되었습니다.
이로 인해 용접 품질이 {prediction['predicted_quality_score']:.2%}로 저하될 것으로 예측됩니다.
            """.strip()

            # Quality Prediction 객체 재구성
            from src.agents.quality_predictor import QualityPrediction

            prediction_obj = QualityPrediction(
                predicted_quality_score=prediction["predicted_quality_score"],
                predicted_strength=prediction["predicted_strength"],
                strength_degradation_pct=prediction["strength_degradation_pct"],
                confidence=prediction["confidence"],
                risk_level=prediction["risk_level"],
                contributing_factors={},
                recommendation=prediction["recommendation"],
            )

            # 조정안 제안 (LLM 사용)
            proposal = self.negotiator.analyze_situation_and_propose(
                current_issue=current_issue,
                prediction=prediction_obj,
                process_data=press_data
            )

            state["proposal"] = {
                "proposal_id": proposal.proposal_id,
                "adjustments": proposal.adjustments,
                "expected_quality": proposal.expected_quality,
                "rationale": proposal.rationale,
                "risk_assessment": proposal.risk_assessment,
            }

            state["negotiation_log"] = [
                f"조정안 생성: {proposal.proposal_id}",
                f"조정값: {proposal.adjustments}",
            ]

            logger.info(f"조정안 제안 완료: {proposal.adjustments}")

        except Exception as e:
            logger.warning(f"LLM 협상 스킵 (오류: {e}). 기본 조정안 사용.")

            # 기본 조정안 (LLM 없이)
            prediction = state["prediction"]
            risk_level = prediction["risk_level"]

            if risk_level == "low":
                adjustments = {"welding_speed": -0.02, "pressure": 0.01}
            elif risk_level == "medium":
                adjustments = {"welding_speed": -0.05, "current": 0.03, "pressure": 0.02}
            elif risk_level == "high":
                adjustments = {"welding_speed": -0.07, "current": 0.05, "pressure": 0.03}
            else:  # critical
                adjustments = {"welding_speed": -0.10, "current": 0.07, "pressure": 0.05}

            state["proposal"] = {
                "proposal_id": "PROP-FALLBACK-001",
                "adjustments": adjustments,
                "expected_quality": prediction["predicted_quality_score"] + 0.05,
                "rationale": f"기본 조정안 ({risk_level} 위험도 기반)",
                "risk_assessment": risk_level,
            }

            state["negotiation_log"] = [
                "기본 조정안 사용 (LLM 미사용)",
                f"조정값: {adjustments}",
            ]

        state["workflow_status"] = "negotiation_complete"
        return state

    def _coordinate_node(self, state: WorkflowState) -> WorkflowState:
        """4단계: 최종 승인/반려"""
        logger.info("[4/5] 최종 승인/반려 결정 시작")

        # AdjustmentProposal 객체 재구성
        from src.agents.negotiation_agent import AdjustmentProposal

        proposal_data = state["proposal"]
        proposal_obj = AdjustmentProposal(
            proposal_id=proposal_data["proposal_id"],
            adjustments=proposal_data["adjustments"],
            expected_quality=proposal_data["expected_quality"],
            rationale=proposal_data["rationale"],
            risk_assessment=proposal_data["risk_assessment"],
        )

        # 승인 결정
        decision = self.coordinator.evaluate_proposal(
            proposal_obj,
            current_quality_score=state["prediction"]["predicted_quality_score"]
        )

        state["decision"] = {
            "decision_id": decision.decision_id,
            "status": decision.status,
            "rationale": decision.rationale,
            "conditions": decision.conditions,
        }

        logger.info(f"결정 완료: {decision.status}")

        state["workflow_status"] = "coordination_complete"
        return state

    @staticmethod
    def _format_adjustment_action(adjustments: Dict[str, float]) -> str:
        if not adjustments:
            return "조정 없음"
        parts = []
        for key, value in adjustments.items():
            parts.append(f"{key} {value:+.1%}")
        return ", ".join(parts)

    def _log_success_case(self, state: WorkflowState) -> None:
        execution = state.get("execution_result") or {}
        if not execution.get("executed"):
            return
        if not execution.get("meets_threshold"):
            return

        prediction = state.get("prediction") or {}
        proposal = state.get("proposal") or {}
        decision = state.get("decision") or {}
        alert = state.get("alert")
        press_data = state.get("press_data") or {}

        adjustments = execution.get("adjustments_applied") or proposal.get("adjustments") or {}
        quality_before = float(round(prediction.get("predicted_quality_score", 0.0), 4))
        quality_after = float(round(execution.get("final_quality_score", 0.0), 4))
        quality_gain = float(round(quality_after - quality_before, 4))

        severity = (alert or {}).get("severity") or prediction.get("risk_level") or "unknown"
        issue_desc = (alert or {}).get("issue_description") or (
            f"predicted_quality_drop ({prediction.get('risk_level', 'unknown')})"
        )

        entry = {
            "process_stage": "composite_line",
            "issue_detected": issue_desc,
            "issue_severity": severity,
            "action_taken": self._format_adjustment_action(adjustments),
            "parameters_adjusted": {k: float(v) for k, v in adjustments.items()},
            "outcome": "success",
            "quality_before": quality_before,
            "quality_after": quality_after,
            "notes": f"decision={decision.get('status')}, alert={alert['alert_id'] if alert else 'none'}",
            "lessons_learned": (
                f"실시간 워크플로우에서 {severity} 이상을 {self._format_adjustment_action(adjustments)}로 대응해 "
                f"품질을 {quality_before:.3f}→{quality_after:.3f}(Δ={quality_gain:+.3f})로 회복"
            ),
            "source_event_id": decision.get("decision_id"),
            "decision_status": decision.get("status"),
            "context_features": {
                "press_thickness": press_data.get("thickness"),
                "press_pressure": press_data.get("pressure"),
                "press_temperature": press_data.get("temperature"),
                "press_anomaly": press_data.get("is_anomaly"),
                "alert_severity": severity,
                "predicted_risk_level": prediction.get("risk_level"),
                "predicted_strength": prediction.get("predicted_strength"),
                "final_strength": execution.get("final_strength"),
            }
        }

        self.case_logger.record_case(entry, source_id=decision.get("decision_id"))

    def _execute_node(self, state: WorkflowState) -> WorkflowState:
        """5단계: 파라미터 조정 실행 (시뮬레이션)"""
        logger.info("[5/5] 파라미터 조정 실행 시작")

        decision = state["decision"]

        if decision["status"] in ["approved", "conditional_approved"]:
            # 조정 실행 (시뮬레이션)
            adjustments = state["proposal"]["adjustments"]

            # 조정 후 용접 데이터 생성
            welding_data = self.simulator.generate_welding_data(
                upstream_thickness=state["press_data"]["thickness"],
                current_adjustment=adjustments.get("current", 0),
                speed_adjustment=adjustments.get("welding_speed", 0),
            )

            # 재측정된 프레스 데이터
            from src.data.sensor_simulator import PressSensorData
            from datetime import datetime

            press_data_obj = PressSensorData(
                timestamp=datetime.now(),
                thickness=state["press_data"]["thickness"],
                pressure=state["press_data"]["pressure"] * (1 + adjustments.get("pressure", 0)),
                temperature=state["press_data"]["temperature"],
                cycle_time=3.0,
                is_anomaly=state["press_data"]["is_anomaly"],
            )

            # 최종 품질 계산
            quality_result = self.simulator.calculate_quality_impact(
                press_data_obj,
                welding_data
            )

            state["execution_result"] = {
                "executed": True,
                "final_quality_score": quality_result["quality_score"],
                "final_strength": quality_result["actual_strength"],
                "meets_threshold": quality_result["meets_threshold"],
                "adjustments_applied": adjustments,
            }

            self._log_success_case(state)

            logger.info(
                f"조정 실행 완료 - 최종 품질: {quality_result['quality_score']:.2%}"
            )
        else:
            state["execution_result"] = {
                "executed": False,
                "reason": "Proposal rejected by coordinator",
            }

            logger.warning("조정 미실행 (제안 반려)")

        state["workflow_status"] = "execution_complete"
        return state

    def run(self) -> Dict:
        """워크플로우 실행"""
        logger.info("=" * 70)
        logger.info("SmartFlow Workflow 실행 시작")
        logger.info("=" * 70)

        # 초기 상태
        initial_state = {
            "press_data": {},
            "welding_data": {},
            "alert": None,
            "prediction": {},
            "proposal": {},
            "negotiation_log": [],
            "decision": {},
            "execution_result": {},
            "workflow_status": "initialized",
            "error_message": "",
        }

        # 워크플로우 실행
        final_state = self.workflow.invoke(initial_state)

        logger.info("=" * 70)
        logger.info("SmartFlow Workflow 실행 완료")
        logger.info("=" * 70)

        return final_state


# 모듈 테스트용
if __name__ == "__main__":
    logger.info("SmartFlow Workflow 테스트 시작")

    # 워크플로우 실행
    workflow = SmartFlowWorkflow()
    result = workflow.run()

    # 결과 출력
    print("\n" + "=" * 70)
    print("최종 결과")
    print("=" * 70)

    print(f"\n[프레스 데이터]")
    print(f"두께: {result['press_data']['thickness']:.4f}mm")
    print(f"이상 여부: {result['press_data']['is_anomaly']}")

    if result['alert']:
        print(f"\n[알림]")
        print(f"ID: {result['alert']['alert_id']}")
        print(f"심각도: {result['alert']['severity']}")

    print(f"\n[품질 예측]")
    print(f"예상 품질: {result['prediction']['predicted_quality_score']:.2%}")
    print(f"위험 수준: {result['prediction']['risk_level']}")

    print(f"\n[조정안]")
    print(f"ID: {result['proposal']['proposal_id']}")
    print(f"조정값: {result['proposal']['adjustments']}")

    print(f"\n[최종 결정]")
    print(f"ID: {result['decision']['decision_id']}")
    print(f"상태: {result['decision']['status']}")
    print(f"근거:\n{result['decision']['rationale']}")

    print(f"\n[실행 결과]")
    if result['execution_result']['executed']:
        print(f"최종 품질: {result['execution_result']['final_quality_score']:.2%}")
        print(f"최종 강도: {result['execution_result']['final_strength']:.2f}MPa")
        print(f"품질 기준 충족: {result['execution_result']['meets_threshold']}")
    else:
        print(f"미실행: {result['execution_result']['reason']}")

    logger.info("테스트 완료")
