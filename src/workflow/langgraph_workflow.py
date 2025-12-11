"""
LangGraph 기반 Multi-Agent 워크플로우

전체 시스템을 조율하는 워크플로우
"""
from typing import TypedDict, Annotated, Dict, Any, Optional
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END
from loguru import logger
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from src.data.sensor_simulator import SensorSimulator, PressSensorData
from src.data.case_logger import CaseLogger
from src.agents.process_monitor import ProcessMonitorAgent
from src.agents.quality_predictor import QualityCascadePredictor
from src.agents.negotiation_agent import NegotiationAgent
from src.agents.coordinator import CoordinatorAgent
from src.rag.retriever import RAGRetriever
from config import settings
from config.data_schema import get_schema
from src.features import FeatureEngineer
from src.adjustment.parameter_adapter import ParameterAdapter

project_root = Path(__file__).resolve().parents[2]


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
    ml_row: Optional[Dict[str, Any]]
    ml_row_adjusted: Optional[Dict[str, Any]]


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
        self.schema = get_schema()
        self.feature_engineer = FeatureEngineer(self.schema)
        self.parameter_adapter = ParameterAdapter(self.schema, self.feature_engineer)

        self.ml_dataset = None
        self.ml_feature_cols: list[str] = []
        self.ml_target_col = "welding_strength"
        self.ml_scaler = None
        self.ml_model = None
        self.ml_context_available = False
        self._load_ml_context()

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

    def _load_ml_context(self) -> None:
        """대시보드와 동일한 ML 모델/데이터 컨텍스트 로드"""
        dataset_candidates = [
            project_root / "data" / "uploaded_data.csv",
            project_root / "data" / "test_set.csv"
        ]

        dataset_path = None
        for candidate in dataset_candidates:
            if candidate.exists():
                dataset_path = candidate
                break

        if dataset_path is None:
            logger.warning("ML 데이터셋을 찾을 수 없어 시뮬레이터 기반으로 동작합니다.")
            return

        try:
            self.ml_dataset = pd.read_csv(dataset_path)
            logger.info(f"ML 워크로드 데이터셋 로드: {dataset_path} ({len(self.ml_dataset)} rows)")
        except Exception as exc:
            logger.warning(f"데이터셋 로드 실패: {exc}")
            self.ml_dataset = None
            return

        if "welding_strength" in self.ml_dataset.columns:
            self.ml_target_col = "welding_strength"
        else:
            self.ml_target_col = None

        if self.ml_target_col:
            self.ml_feature_cols = [
                col for col in self.ml_dataset.columns if col != self.ml_target_col
            ]
        else:
            self.ml_feature_cols = list(self.ml_dataset.columns)

        scaler_path = project_root / "models" / "scaler.pkl"
        model_path = project_root / "models" / "quality_predictor.pkl"

        try:
            with open(scaler_path, 'rb') as f:
                self.ml_scaler = pickle.load(f)
            with open(model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            self.ml_context_available = True
            logger.info("ML 모델/스케일러 로드 완료 (대시보드와 동일한 컨텍스트 사용)")
        except FileNotFoundError as exc:
            logger.warning(f"ML 모델 파일을 찾을 수 없습니다: {exc}")
            self.ml_context_available = False
        except Exception as exc:
            logger.warning(f"ML 모델 로드 실패: {exc}")
            self.ml_context_available = False

    def _add_negotiation_log(
        self,
        state: WorkflowState,
        role: str,
        label: str,
        message: str,
        status: str = "info",
        meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """협상 단계별 진행 로그를 축적"""
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "role": role,
            "label": label,
            "message": message.strip(),
            "status": status,
            "meta": meta or {}
        }
        state.setdefault("negotiation_log", []).append(entry)

    def _sample_ml_row(self) -> Optional[Dict[str, Any]]:
        if self.ml_dataset is None or self.ml_dataset.empty:
            return None
        sample = self.ml_dataset.sample(1).iloc[0]
        return {col: (float(val) if isinstance(val, (np.floating, np.integer)) else val)
                for col, val in sample.to_dict().items()}

    def _row_to_press_sensor_data(self, row: Dict[str, Any]) -> PressSensorData:
        thickness = float(row.get("press_thickness") or row.get("Stage1.Output.Measurement0.U.Actual") or 2.0)
        pressure = float(row.get("welding_pressure") or row.get("Machine4.Pressure.C.Actual") or 150.0)
        temperature = float(row.get("combiner_temp1") or row.get("Machine4.Temperature1.C.Actual") or 25.0)
        cycle_time = float(row.get("cycle_time", 3.0))
        anomaly_type = row.get("anomaly_type")

        return PressSensorData(
            timestamp=datetime.now(),
            thickness=thickness,
            pressure=pressure,
            temperature=temperature,
            cycle_time=cycle_time,
            is_anomaly=False,
            anomaly_type=anomaly_type
        )

    def _predict_quality_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ml_context_available or self.ml_model is None or self.ml_scaler is None:
            raise RuntimeError("ML 컨텍스트가 준비되지 않았습니다.")

        ordered = [row.get(col, 0.0) for col in self.ml_feature_cols]
        features = np.array([ordered])
        features_scaled = self.ml_scaler.transform(features)
        predicted_strength = float(self.ml_model.predict(features_scaled)[0])

        lsl = settings.welding_strength_lsl
        target = settings.welding_strength_target
        usl = settings.welding_strength_usl

        if predicted_strength >= target:
            predicted_quality_score = 0.9 + 0.1 * (predicted_strength - target) / (usl - target)
        else:
            predicted_quality_score = 0.9 * (predicted_strength - lsl) / (target - lsl)
        predicted_quality_score = float(np.clip(predicted_quality_score, 0.0, 1.0))

        strength_degradation = max(0.0, (target - predicted_strength) / target * 100)

        threshold = settings.quality_threshold
        if predicted_quality_score >= threshold:
            risk_level = "low"
        elif predicted_quality_score >= threshold - 0.05:
            risk_level = "medium"
        elif predicted_quality_score >= threshold - 0.15:
            risk_level = "high"
        else:
            risk_level = "critical"

        recommendations = {
            "low": "현재 파라미터 유지 가능",
            "medium": "용접 파라미터 미세 조정 검토",
            "high": "즉시 파라미터 조정 필요",
            "critical": "긴급 조치 필요 - 생산 중단 고려"
        }

        return {
            "predicted_quality_score": predicted_quality_score,
            "predicted_strength": predicted_strength,
            "strength_degradation_pct": strength_degradation,
            "confidence": 0.92,
            "risk_level": risk_level,
            "recommendation": recommendations.get(risk_level, "검토 필요")
        }

    def _apply_adjustments_to_row(self, row: Dict[str, Any], adjustments: Dict[str, float]) -> Dict[str, Any]:
        if not adjustments:
            return row
        adjusted = self.parameter_adapter.apply_control_adjustments(
            data=row,
            control_adjustments=adjustments,
            recalculate_features=True
        )
        return adjusted

    def _monitor_node(self, state: WorkflowState) -> WorkflowState:
        """1단계: 공정 모니터링"""
        logger.info("[1/5] 공정 모니터링 시작")

        ml_row = self._sample_ml_row() if self.ml_context_available else None
        state["ml_row"] = ml_row

        if ml_row is not None:
            press_sensor = self._row_to_press_sensor_data(ml_row)
            press_data, alert = self.monitor.monitor_press_process(
                force_anomaly=False,
                press_data_override=press_sensor
            )
        else:
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
        ml_row = state.get("ml_row")

        if ml_row is not None and self.ml_context_available:
            ml_prediction = self._predict_quality_from_row(ml_row)
            state["prediction"] = ml_prediction
        else:
            press_data_obj = PressSensorData(
                timestamp=datetime.now(),
                thickness=state["press_data"]["thickness"],
                pressure=state["press_data"]["pressure"],
                temperature=state["press_data"]["temperature"],
                cycle_time=3.0,
                is_anomaly=state["press_data"]["is_anomaly"],
                anomaly_type=state["press_data"].get("anomaly_type"),
            )

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

            self._add_negotiation_log(
                state,
                role="quality_predictor",
                label="품질 예측",
                message=(
                    f"예상 품질 {prediction['predicted_quality_score']:.1%}, "
                    f"위험 {prediction['risk_level'].upper()}"
                ),
                status="alert",
                meta={
                    "thickness": f"{press_data['thickness']:.4f}mm",
                    "pressure": f"{press_data['pressure']:.1f}MPa"
                }
            )

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

            self._add_negotiation_log(
                state,
                role="negotiator",
                label="조정안 제안",
                message=self._format_adjustment_action(proposal.adjustments),
                status="proposal",
                meta={
                    "proposal_id": proposal.proposal_id,
                    "expected_quality": f"{proposal.expected_quality:.1%}"
                }
            )

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

            self._add_negotiation_log(
                state,
                role="negotiator",
                label="기본 조정안",
                message=self._format_adjustment_action(adjustments),
                status="fallback",
                meta={"reason": "LLM unavailable"}
            )

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

        self._add_negotiation_log(
            state,
            role="coordinator",
            label="승인 판단",
            message=f"{decision.status.upper()} - {decision.rationale.splitlines()[0] if decision.rationale else ''}",
            status="decision",
            meta={"decision_id": decision.decision_id}
        )

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
            adjustments = state["proposal"]["adjustments"]

            if self.ml_context_available and state.get("ml_row") is not None:
                base_row = state["ml_row"]
                adjusted_row = self._apply_adjustments_to_row(base_row, adjustments)
                final_prediction = self._predict_quality_from_row(adjusted_row)

                state["execution_result"] = {
                    "executed": True,
                    "final_quality_score": final_prediction["predicted_quality_score"],
                    "final_strength": final_prediction["predicted_strength"],
                    "meets_threshold": final_prediction["predicted_quality_score"] >= settings.quality_threshold,
                    "adjustments_applied": adjustments,
                    "risk_level": final_prediction["risk_level"],
                }

                state.setdefault("prediction", {})["post_adjustment_quality_score"] = final_prediction["predicted_quality_score"]
                state["prediction"]["post_adjustment_risk_level"] = final_prediction["risk_level"]
                state["ml_row_adjusted"] = adjusted_row

                self._log_success_case(state)

                self._add_negotiation_log(
                    state,
                    role="execution",
                    label="조정 실행",
                    message=(
                        f"최종 품질 {final_prediction['predicted_quality_score']:.1%}, "
                        f"위험 {final_prediction['risk_level'].upper()}"
                    ),
                    status="result",
                    meta={
                        "meets_threshold": state["execution_result"]["meets_threshold"],
                        "final_strength": f"{final_prediction['predicted_strength']:.2f}MPa"
                    }
                )

                logger.info(
                    f"조정 실행 완료 - 최종 품질: {final_prediction['predicted_quality_score']:.2%}"
                )
            else:
                welding_data = self.simulator.generate_welding_data(
                    upstream_thickness=state["press_data"]["thickness"],
                    current_adjustment=adjustments.get("current", 0),
                    speed_adjustment=adjustments.get("welding_speed", 0),
                )

                press_data_obj = PressSensorData(
                    timestamp=datetime.now(),
                    thickness=state["press_data"]["thickness"],
                    pressure=state["press_data"]["pressure"] * (1 + adjustments.get("pressure", 0)),
                    temperature=state["press_data"]["temperature"],
                    cycle_time=3.0,
                    is_anomaly=state["press_data"]["is_anomaly"],
                )

                quality_result = self.simulator.calculate_quality_impact(
                    press_data_obj,
                    welding_data
                )

                final_risk = self.predictor.determine_risk_from_quality(
                    quality_result["quality_score"]
                )

                state["execution_result"] = {
                    "executed": True,
                    "final_quality_score": quality_result["quality_score"],
                    "final_strength": quality_result["actual_strength"],
                    "meets_threshold": quality_result["meets_threshold"],
                    "adjustments_applied": adjustments,
                    "risk_level": final_risk,
                }

                state.setdefault("prediction", {})["post_adjustment_quality_score"] = quality_result["quality_score"]
                state["prediction"]["post_adjustment_risk_level"] = final_risk

                self._log_success_case(state)

                self._add_negotiation_log(
                    state,
                    role="execution",
                    label="조정 실행",
                    message=(
                        f"최종 품질 {quality_result['quality_score']:.1%}, "
                        f"위험 {final_risk.upper()}"
                    ),
                    status="result",
                    meta={
                        "meets_threshold": quality_result["meets_threshold"],
                        "final_strength": f"{quality_result['actual_strength']:.2f}MPa"
                    }
                )

                logger.info(
                    f"조정 실행 완료 - 최종 품질: {quality_result['quality_score']:.2%}"
                )
        else:
            state["execution_result"] = {
                "executed": False,
                "reason": "Proposal rejected by coordinator",
            }
            logger.warning("조정 미실행 (제안 반려)")

            self._add_negotiation_log(
                state,
                role="execution",
                label="조정 중단",
                message="제안이 반려되어 실행되지 않았습니다.",
                status="warning"
            )

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
            "ml_row": None,
            "ml_row_adjusted": None,
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
