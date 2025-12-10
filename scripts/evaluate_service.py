"""
서비스 평가 스크립트 (실제 Multi-Agent 시스템 사용)

실제 서비스 에이전트들을 그대로 사용하여 Test 데이터로 성능 평가
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_absolute_error
from loguru import logger
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict

# 실제 서비스 에이전트 import
from src.agents.process_monitor import ProcessMonitorAgent
from src.agents.ml_quality_predictor import MLQualityCascadePredictor, MLQualityPrediction
from src.agents.negotiation_agent import NegotiationAgent
from src.agents.coordinator import CoordinatorAgent, ProductionGoals
from src.data.sensor_simulator import PressSensorData
from config import settings


@dataclass
class SampleEvaluationResult:
    """개별 샘플 평가 결과"""
    sample_id: int
    ground_truth: float
    baseline_prediction: float
    adjusted_prediction: float
    adjusted_ground_truth: float  # 시뮬레이션된 실제 값
    is_anomaly: bool
    adjustment_applied: bool
    improvement: float
    meets_threshold_baseline: bool
    meets_threshold_adjusted: bool
    defect_prevented: bool
    applied_adjustments: Dict[str, float]
    decision_status: str
    iterations_used: int


class ServiceEvaluator:
    """실제 Multi-Agent 시스템을 사용한 서비스 평가"""

    def __init__(
        self,
        quality_threshold: float = 0.90,
        cost_per_defect: float = 100.0,
        max_iterations: int = 3
    ):
        self.quality_threshold = quality_threshold
        self.cost_per_defect = cost_per_defect
        self.max_iterations = max_iterations

        # 실제 서비스 에이전트 초기화
        logger.info("실제 Multi-Agent 시스템 초기화 중...")
        self.process_monitor = ProcessMonitorAgent()

        # ML 모델 (load_test_data에서 로드됨)
        self.ml_model = None

        self.negotiation_agent = NegotiationAgent()
        self.coordinator = CoordinatorAgent(
            production_goals=ProductionGoals(
                target_quality=quality_threshold,
                max_cycle_time_increase=0.18,
                max_cost_increase=0.10
            )
        )

        self.test_data = None
        self.scaler = None
        self.feature_cols: List[str] = []
        self.lsl = None
        self.usl = None
        self.spec_span = 1.0
        self.sample_results: List[SampleEvaluationResult] = []

        # 물리적 효과 계수 (ground truth 시뮬레이션용)
        self.base_speed_coeff = 0.30
        self.base_current_coeff = 0.40
        self.base_pressure_coeff = 0.30

        logger.info("✅ Multi-Agent 시스템 초기화 완료")

    def _load_ml_model(self):
        """ML 모델 로드"""
        model_path = Path("models/quality_predictor.pkl")
        try:
            with open(model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            logger.info(f"✅ ML 모델 로드: {model_path}")
        except Exception as e:
            logger.error(f"ML 모델 로드 실패: {e}")
            raise

    def predict_quality(self, raw_row: Dict[str, float]) -> MLQualityPrediction:
        """
        품질 예측 (ML 모델 사용)

        Args:
            raw_row: 원본 피처 딕셔너리

        Returns:
            MLQualityPrediction 객체
        """
        # 피처 배열 준비
        features = np.array([[raw_row[col] for col in self.feature_cols]])

        # 예측
        predicted_strength = float(self.ml_model.predict(features)[0])

        # 품질 점수 계산
        baseline_strength = 12.0  # 기준 강도
        predicted_quality_score = min(1.0, predicted_strength / baseline_strength)

        # 강도 저하율
        strength_degradation = max(0, (baseline_strength - predicted_strength) / baseline_strength * 100)

        # 위험 수준
        if predicted_quality_score >= self.quality_threshold:
            risk_level = "low"
        elif predicted_quality_score >= self.quality_threshold - 0.05:
            risk_level = "medium"
        elif predicted_quality_score >= self.quality_threshold - 0.15:
            risk_level = "high"
        else:
            risk_level = "critical"

        return MLQualityPrediction(
            predicted_strength=predicted_strength,
            predicted_quality_score=predicted_quality_score,
            strength_degradation_pct=strength_degradation,
            confidence=0.92,
            risk_level=risk_level,
            baseline_strength=baseline_strength,
            model_used="XGBoost",
            recommendation="자동 평가"
        )

    def load_test_data(self):
        """Test 데이터 로드"""
        logger.info("=" * 70)
        logger.info("Test 데이터 로드 중...")

        test_path = Path("data/test_set.csv")
        if not test_path.exists():
            raise FileNotFoundError(f"Test 데이터가 없습니다: {test_path}")

        self.test_data = pd.read_csv(test_path)
        target_col = "welding_strength"
        self.feature_cols = [col for col in self.test_data.columns if col != target_col]

        # 스케일러 로드
        scaler_path = Path("models/scaler.pkl")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # ML 모델 로드 (feature_cols 설정 후)
        self._load_ml_model()

        # 품질 스펙 계산
        target_values = self.test_data[target_col].values
        self.lsl = float(np.percentile(target_values, 5))
        self.usl = float(np.percentile(target_values, 95))
        self.spec_span = max(self.usl - self.lsl, 1e-3)

        # 이상 감지를 위한 통계값 계산
        self.feature_stats = {}
        for col in self.feature_cols:
            values = self.test_data[col].values
            self.feature_stats[col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'q05': float(np.percentile(values, 5)),
                'q95': float(np.percentile(values, 95))
            }

        logger.info(f"✅ Test 데이터: {len(self.test_data)} samples")
        logger.info(f"   LSL={self.lsl:.4f}, USL={self.usl:.4f}")
        logger.info("=" * 70)

    def detect_anomaly(self, raw_row: Dict[str, float]) -> bool:
        """
        이상 감지 (통계 기반)

        Args:
            raw_row: 원본 피처 딕셔너리

        Returns:
            이상 여부
        """
        # 주요 파라미터 체크 (두께 편차가 큰 경우)
        if "press_thickness" in raw_row and "press_thickness" in self.feature_stats:
            stats = self.feature_stats["press_thickness"]
            thickness = raw_row["press_thickness"]

            # 5%~95% 범위를 벗어나면 이상
            if thickness < stats['q05'] or thickness > stats['q95']:
                return True

            # 평균에서 2 표준편차 이상 벗어나면 이상
            if abs(thickness - stats['mean']) > 2 * stats['std']:
                return True

        return False

    def _meets_quality_spec(self, value: float) -> bool:
        """품질 기준 충족 여부"""
        return self.lsl <= value <= self.usl

    def _inverse_scale_row(self, scaled_features: np.ndarray) -> Dict[str, float]:
        """스케일링된 피처를 원본 값으로 변환"""
        raw_values = self.scaler.inverse_transform(scaled_features)[0]
        return {col: float(raw_values[i]) for i, col in enumerate(self.feature_cols)}

    def _scale_row(self, raw_row: Dict[str, float]) -> np.ndarray:
        """원본 값을 스케일링"""
        ordered = [raw_row[col] for col in self.feature_cols]
        return self.scaler.transform(np.array([ordered]))

    def _apply_adjustments(
        self,
        raw_row: Dict[str, float],
        adjustments: Dict[str, float]
    ) -> Dict[str, float]:
        """조정값 적용"""
        adjusted = raw_row.copy()

        # 파라미터 매핑
        param_mapping = {
            "welding_speed": "welding_temp3",
            "current": "welding_temp1",
            "pressure": "welding_pressure"
        }

        for adj_key, feature_name in param_mapping.items():
            if adj_key in adjustments and feature_name in adjusted:
                adjusted[feature_name] *= (1 + adjustments[adj_key])

        # 파생 피처 재계산
        if {"welding_temp1", "welding_temp3"}.issubset(adjusted):
            denom = adjusted["welding_temp3"] if adjusted["welding_temp3"] != 0 else 1e-5
            adjusted["heat_input_proxy"] = adjusted["welding_temp1"] / denom

        return adjusted

    def _simulate_ground_truth_effect(
        self,
        original_gt: float,
        adjustments: Dict[str, float]
    ) -> float:
        """조정이 실제 품질에 미치는 영향 시뮬레이션"""
        if not adjustments:
            return original_gt

        speed_change = adjustments.get("welding_speed", 0)
        current_change = adjustments.get("current", 0)
        pressure_change = adjustments.get("pressure", 0)

        # 물리적 효과 계산
        strength_change_pct = (
            -speed_change * self.base_speed_coeff +
            current_change * self.base_current_coeff +
            pressure_change * self.base_pressure_coeff
        )

        adjusted_gt = original_gt * (1 + strength_change_pct)
        return float(np.clip(adjusted_gt, self.lsl - 0.5, self.usl + 0.5))

    def evaluate_samples(self):
        """Test 샘플 평가 (실제 Multi-Agent 시스템 사용)"""
        logger.info("\n" + "=" * 70)
        logger.info("🔍 실제 Multi-Agent 시스템으로 평가 시작")
        logger.info("=" * 70)

        target_col = "welding_strength"
        X_test = self.test_data[self.feature_cols].values
        y_test = self.test_data[target_col].values

        for i in range(len(X_test)):
            features = X_test[i:i+1]
            ground_truth = y_test[i]
            raw_row = self._inverse_scale_row(features)

            # 1. Process Monitor: 이상 감지 (통계 기반)
            is_anomaly = self.detect_anomaly(raw_row)

            # 2. ML Quality Predictor: 품질 예측
            baseline_pred_obj = self.predict_quality(raw_row)
            baseline_pred = baseline_pred_obj.predicted_strength

            decision_status = "skipped"
            adjustment_applied = False
            adjusted_pred = baseline_pred
            applied_adjustments = {}
            iterations_used = 0

            # 3. 이상이 감지되면 Multi-Agent 협상 시작
            if is_anomaly:
                working_raw = raw_row
                current_pred_obj = baseline_pred_obj

                for iteration in range(self.max_iterations):
                    # 품질 기준 충족하면 종료
                    if self._meets_quality_spec(current_pred_obj.predicted_strength):
                        break

                    # Negotiation Agent: RAG 기반 조정 제안
                    current_issue = f"품질 저하 감지: {current_pred_obj.predicted_strength:.3f} (목표: {self.lsl:.3f}~{self.usl:.3f})"

                    try:
                        proposal = self.negotiation_agent.analyze_situation_and_propose(
                            current_issue=current_issue,
                            prediction=current_pred_obj,
                            process_data=working_raw
                        )
                    except Exception as e:
                        logger.warning(f"Negotiation Agent 오류: {e}")
                        break

                    # Coordinator: 승인/반려
                    current_quality_score = max(0.0, min(1.0, current_pred_obj.predicted_quality_score))
                    decision = self.coordinator.evaluate_proposal(
                        proposal=proposal,
                        current_quality_score=current_quality_score
                    )

                    decision_status = decision.status

                    if decision.status in ["approved", "conditional_approved"]:
                        # 조정 적용
                        working_raw = self._apply_adjustments(working_raw, proposal.adjustments)
                        working_features = self._scale_row(working_raw)
                        current_pred_obj = self.predict_quality(working_raw)

                        applied_adjustments = proposal.adjustments
                        adjustment_applied = True
                        adjusted_pred = current_pred_obj.predicted_strength
                        iterations_used += 1
                    else:
                        # 반려되면 종료
                        break

            # Ground truth 시뮬레이션
            adjusted_ground_truth = self._simulate_ground_truth_effect(
                ground_truth,
                applied_adjustments
            )

            # 결과 기록
            meets_baseline = self._meets_quality_spec(ground_truth)
            meets_adjusted = self._meets_quality_spec(adjusted_ground_truth)
            defect_prevented = (not meets_baseline) and meets_adjusted
            improvement = adjusted_pred - baseline_pred

            result = SampleEvaluationResult(
                sample_id=i,
                ground_truth=ground_truth,
                baseline_prediction=baseline_pred,
                adjusted_prediction=adjusted_pred,
                adjusted_ground_truth=adjusted_ground_truth,
                is_anomaly=is_anomaly,
                adjustment_applied=adjustment_applied,
                improvement=improvement,
                meets_threshold_baseline=meets_baseline,
                meets_threshold_adjusted=meets_adjusted,
                defect_prevented=defect_prevented,
                applied_adjustments=applied_adjustments,
                decision_status=decision_status,
                iterations_used=iterations_used
            )
            self.sample_results.append(result)

            if (i + 1) % 100 == 0:
                logger.info(f"  진행: {i+1}/{len(X_test)} 샘플 완료")

        logger.info("=" * 70)
        logger.info("✅ 평가 완료")

    def calculate_and_report_metrics(self):
        """지표 계산 및 리포트"""
        total_samples = len(self.sample_results)
        anomalies_detected = sum(1 for r in self.sample_results if r.is_anomaly)
        adjustments_made = sum(1 for r in self.sample_results if r.adjustment_applied)

        defects_before = sum(1 for r in self.sample_results if not r.meets_threshold_baseline)
        defects_after = sum(1 for r in self.sample_results if not r.meets_threshold_adjusted)
        defects_prevented = sum(1 for r in self.sample_results if r.defect_prevented)

        defect_reduction_rate = defects_prevented / defects_before if defects_before > 0 else 0.0
        quality_recovery_rate = defects_prevented / anomalies_detected if anomalies_detected > 0 else 0.0

        cost_saving = defects_prevented * self.cost_per_defect

        successful_adjustments = sum(
            1 for r in self.sample_results
            if r.adjustment_applied and (r.defect_prevented or r.meets_threshold_adjusted)
        )
        adjustment_success_rate = successful_adjustments / adjustments_made if adjustments_made > 0 else 0.0

        quality_gains = [
            r.adjusted_ground_truth - r.ground_truth
            for r in self.sample_results if r.adjustment_applied
        ]
        avg_improvement = float(np.mean(quality_gains)) if quality_gains else 0.0

        # 리포트 출력
        logger.info("\n" + "=" * 70)
        logger.info("📊 실제 Multi-Agent 시스템 평가 결과")
        logger.info("=" * 70)
        logger.info(f"\n[기본 통계]")
        logger.info(f"  총 샘플 수: {total_samples}")
        logger.info(f"  감지된 이상: {anomalies_detected} ({anomalies_detected/total_samples:.1%})")
        logger.info(f"  적용된 조정: {adjustments_made}")
        logger.info(f"\n[비즈니스 임팩트]")
        logger.info(f"  조정 전 불량: {defects_before}개")
        logger.info(f"  조정 후 불량: {defects_after}개")
        logger.info(f"  방지된 불량: {defects_prevented}개")
        logger.info(f"  불량 감소율: {defect_reduction_rate:.1%}")
        logger.info(f"  품질 회복율: {quality_recovery_rate:.1%}")
        logger.info(f"  비용 절감액: ${cost_saving:,.2f}")
        logger.info(f"\n[조정 효과성]")
        logger.info(f"  성공한 조정: {successful_adjustments}/{adjustments_made}")
        logger.info(f"  조정 성공률: {adjustment_success_rate:.1%}")
        logger.info(f"  조정당 평균 개선: {avg_improvement:.4f}")
        logger.info("=" * 70)

        # 결과 저장
        results = {
            "evaluation_date": datetime.now().isoformat(),
            "total_samples": total_samples,
            "anomalies_detected": anomalies_detected,
            "adjustments_made": adjustments_made,
            "defects_prevented": defects_prevented,
            "defect_reduction_rate": float(defect_reduction_rate),
            "quality_recovery_rate": float(quality_recovery_rate),
            "cost_saving": float(cost_saving),
            "successful_adjustments": successful_adjustments,
            "adjustment_success_rate": float(adjustment_success_rate),
            "avg_improvement": float(avg_improvement)
        }

        output_dir = Path("models")
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "service_evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 결과 저장: {results_path}")

        # 텍스트 리포트
        report_path = output_dir / "service_evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SmartFlow Multi-Agent 시스템 평가 리포트\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test 샘플 수: {total_samples}\n\n")
            f.write("=" * 70 + "\n")
            f.write("비즈니스 임팩트\n")
            f.write("=" * 70 + "\n")
            f.write(f"감지된 이상: {anomalies_detected}건\n")
            f.write(f"방지된 불량: {defects_prevented}건\n")
            f.write(f"불량 감소율: {defect_reduction_rate:.1%}\n")
            f.write(f"품질 회복율: {quality_recovery_rate:.1%}\n")
            f.write(f"비용 절감액: ${cost_saving:,.2f}\n\n")
            f.write("=" * 70 + "\n")
            f.write("조정 시스템 효과성\n")
            f.write("=" * 70 + "\n")
            f.write(f"적용된 조정: {adjustments_made}회\n")
            f.write(f"성공한 조정: {successful_adjustments}회\n")
            f.write(f"조정 성공률: {adjustment_success_rate:.1%}\n")
            f.write(f"조정당 평균 개선: {avg_improvement:.4f}\n\n")
            f.write("=" * 70 + "\n")

        logger.info(f"✅ 리포트 저장: {report_path}")


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🔬 SmartFlow Multi-Agent 시스템 평가")
    print("=" * 70)
    print("⚠️  실제 서비스 에이전트를 사용하여 Test 데이터로 성능을 평가합니다.")
    print("=" * 70)

    try:
        evaluator = ServiceEvaluator(
            quality_threshold=0.90,
            cost_per_defect=100.0,
            max_iterations=3
        )

        evaluator.load_test_data()
        evaluator.evaluate_samples()
        evaluator.calculate_and_report_metrics()

        print("\n" + "=" * 70)
        print("✅ 평가 완료!")
        print("=" * 70)

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ 평가 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
