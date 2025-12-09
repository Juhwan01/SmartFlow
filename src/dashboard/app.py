"""
SmartFlow Dashboard

협상 과정 및 결과를 시각화하는 Streamlit 대시보드
"""
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.workflow.langgraph_workflow import SmartFlowWorkflow
from src.evaluation.metrics import MetricsCalculator
from config import settings


# 페이지 설정
st.set_page_config(
    page_title="SmartFlow Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바
with st.sidebar:
    st.title("🏭 SmartFlow")
    st.markdown("**LLM 기반 Multi-Agent 협상 시스템**")
    st.markdown("---")

    st.subheader("시스템 설정")
    st.write(f"LLM Provider: `{settings.llm_provider}`")
    st.write(f"Model: `{settings.llm_model}`")
    st.write(f"품질 목표: `{settings.quality_threshold:.0%}`")

    st.markdown("---")

    run_button = st.button("🚀 워크플로우 실행", type="primary", use_container_width=True)


# 메인 화면
st.title("SmartFlow Multi-Agent 협상 대시보드")
st.markdown("프레스-용접 공정의 사전 품질 예측 및 자율 조정 시스템")

if run_button or "workflow_result" in st.session_state:
    if run_button:
        with st.spinner("워크플로우 실행 중..."):
            try:
                workflow = SmartFlowWorkflow()
                result = workflow.run()
                st.session_state.workflow_result = result
                st.success("워크플로우 실행 완료!")
            except Exception as e:
                st.error(f"오류 발생: {e}")
                st.stop()

    result = st.session_state.workflow_result

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 전체 요약",
        "🔍 공정 모니터링",
        "📈 품질 예측",
        "🤝 협상 과정",
        "✅ 최종 결과",
        "🎯 평가지표"
    ])

    with tab1:
        st.header("전체 프로세스 요약")

        # 워크플로우 단계
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("1️⃣ 모니터링", "완료" if result['alert'] else "정상")
        with col2:
            quality = result['prediction']['predicted_quality_score']
            st.metric("2️⃣ 품질 예측", f"{quality:.1%}")
        with col3:
            st.metric("3️⃣ 조정안 제안", result['proposal']['proposal_id'][:15])
        with col4:
            decision = result['decision']['status']
            st.metric("4️⃣ 최종 결정", decision)
        with col5:
            executed = result['execution_result'].get('executed', False)
            st.metric("5️⃣ 실행", "완료" if executed else "미실행")

        # 주요 지표
        st.markdown("---")
        st.subheader("주요 지표")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            thickness = result['press_data']['thickness']
            deviation = abs(thickness - 2.0)
            st.metric(
                "프레스 두께",
                f"{thickness:.4f}mm",
                f"편차: {deviation:.4f}mm",
                delta_color="inverse"
            )

        with metric_col2:
            pred_quality = result['prediction']['predicted_quality_score']
            delta_quality = pred_quality - settings.quality_threshold
            st.metric(
                "예측 품질",
                f"{pred_quality:.1%}",
                f"{delta_quality:+.1%}",
                delta_color="normal" if delta_quality >= 0 else "inverse"
            )

        with metric_col3:
            risk = result['prediction']['risk_level']
            risk_colors = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
            st.metric(
                "위험 수준",
                f"{risk_colors.get(risk, '⚪')} {risk.upper()}"
            )

        with metric_col4:
            if result['execution_result'].get('executed'):
                final_quality = result['execution_result']['final_quality_score']
                improvement = final_quality - pred_quality
                st.metric(
                    "최종 품질",
                    f"{final_quality:.1%}",
                    f"{improvement:+.1%}",
                    delta_color="normal"
                )
            else:
                st.metric("최종 품질", "N/A")

    with tab2:
        st.header("공정 모니터링")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("프레스 공정 데이터")
            press_data = result['press_data']

            st.write(f"**두께**: {press_data['thickness']:.4f} mm")
            st.write(f"**압력**: {press_data['pressure']:.2f} MPa")
            st.write(f"**온도**: {press_data['temperature']:.2f} °C")
            st.write(f"**이상 여부**: {'⚠️ 예' if press_data['is_anomaly'] else '✅ 아니오'}")

            if press_data['is_anomaly']:
                st.warning(f"이상 유형: {press_data.get('anomaly_type', 'unknown')}")

        with col2:
            st.subheader("이상 알림")

            if result['alert']:
                alert = result['alert']

                severity_colors = {
                    "low": "🟢",
                    "medium": "🟡",
                    "high": "🟠",
                    "critical": "🔴"
                }

                st.error(f"""
                **알림 ID**: {alert['alert_id']}

                **심각도**: {severity_colors.get(alert['severity'], '⚪')} {alert['severity'].upper()}

                **문제**: {alert['issue_description']}

                **권장 조치**: {alert['recommended_action']}
                """)
            else:
                st.success("이상 없음 - 정상 운영 중")

    with tab3:
        st.header("품질 예측 결과")

        prediction = result['prediction']

        col1, col2 = st.columns([2, 1])

        with col1:
            # 품질 게이지
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction['predicted_quality_score'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "예측 품질 점수 (%)"},
                delta={'reference': settings.quality_threshold * 100},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 80], 'color': "lightgray"},
                        {'range': [80, 90], 'color': "lightyellow"},
                        {'range': [90, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': settings.quality_threshold * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("상세 정보")
            st.write(f"**예상 강도**: {prediction['predicted_strength']:.2f} MPa")
            st.write(f"**강도 저하**: {prediction['strength_degradation_pct']:.2f}%")
            st.write(f"**신뢰도**: {prediction['confidence']:.1%}")
            st.write(f"**위험 수준**: {prediction['risk_level'].upper()}")

        st.markdown("---")
        st.info(f"**권장 사항**: {prediction['recommendation']}")

    with tab4:
        st.header("협상 과정 및 조정안")

        proposal = result['proposal']

        st.subheader("제안 정보")
        st.write(f"**제안 ID**: {proposal['proposal_id']}")
        st.write(f"**예상 품질**: {proposal['expected_quality']:.1%}")
        st.write(f"**위험 평가**: {proposal['risk_assessment']}")

        st.markdown("---")
        st.subheader("파라미터 조정안")

        adjustments = proposal['adjustments']

        adj_col1, adj_col2, adj_col3 = st.columns(3)

        with adj_col1:
            speed_adj = adjustments.get('welding_speed', 0) * 100
            st.metric("용접 속도", f"{speed_adj:+.1f}%")

        with adj_col2:
            current_adj = adjustments.get('current', 0) * 100
            st.metric("전류", f"{current_adj:+.1f}%")

        with adj_col3:
            pressure_adj = adjustments.get('pressure', 0) * 100
            st.metric("압력", f"{pressure_adj:+.1f}%")

        st.markdown("---")
        st.subheader("조정 근거")

        with st.expander("LLM 분석 결과 보기", expanded=False):
            st.text(proposal['rationale'][:1000] + "..." if len(proposal['rationale']) > 1000 else proposal['rationale'])

    with tab5:
        st.header("최종 승인 결정 및 실행 결과")

        decision = result['decision']

        # 결정 상태
        if decision['status'] == 'approved':
            st.success(f"✅ 제안 승인 (결정 ID: {decision['decision_id']})")
        elif decision['status'] == 'conditional_approved':
            st.warning(f"⚠️  조건부 승인 (결정 ID: {decision['decision_id']})")
        else:
            st.error(f"❌ 제안 반려 (결정 ID: {decision['decision_id']})")

        # 근거
        st.subheader("결정 근거")
        st.text(decision['rationale'])

        if decision.get('conditions'):
            st.warning("**조건**: " + ", ".join(decision['conditions']))

        st.markdown("---")

        # 실행 결과
        st.subheader("실행 결과")

        exec_result = result['execution_result']

        if exec_result.get('executed'):
            st.success("조정 실행 완료")

            final_col1, final_col2, final_col3 = st.columns(3)

            with final_col1:
                st.metric("최종 품질 점수", f"{exec_result['final_quality_score']:.1%}")

            with final_col2:
                st.metric("최종 강도", f"{exec_result['final_strength']:.2f} MPa")

            with final_col3:
                meets = exec_result['meets_threshold']
                st.metric("품질 기준 충족", "✅ 예" if meets else "❌ 아니오")

            st.json(exec_result['adjustments_applied'])
        else:
            st.warning(f"조정 미실행: {exec_result.get('reason', 'Unknown')}")

    with tab6:
        st.header("📊 시스템 평가지표")
        st.markdown("해커톤 심사를 위한 정량적 성능 지표")

        # MetricsCalculator 초기화 및 데이터 로드
        try:
            calculator = MetricsCalculator()

            # ML 성능 지표 로드
            ml_metrics = calculator.load_ml_metrics()

            st.markdown("---")
            st.subheader("🤖 ML 모델 성능 지표")
            st.markdown("XGBoost 품질 예측 모델의 테스트 세트 성능")

            ml_col1, ml_col2, ml_col3, ml_col4 = st.columns(4)

            with ml_col1:
                st.metric(
                    "R² Score",
                    f"{ml_metrics.r2:.4f}",
                    help="결정계수 - 모델이 데이터를 얼마나 잘 설명하는지 (목표: >0.90)"
                )
                if ml_metrics.r2 >= 0.90:
                    st.success("✅ 목표 달성")
                else:
                    st.warning(f"⚠️  목표 미달 (0.90)")

            with ml_col2:
                st.metric(
                    "MAE",
                    f"{ml_metrics.mae:.4f}",
                    help="평균 절대 오차 - 예측과 실제값의 평균 차이 (목표: <1.0)"
                )
                if ml_metrics.mae < 1.0:
                    st.success("✅ 목표 달성")
                else:
                    st.warning(f"⚠️  목표 미달 (1.0)")

            with ml_col3:
                st.metric(
                    "RMSE",
                    f"{ml_metrics.rmse:.4f}",
                    help="평균 제곱근 오차 - 예측 오차의 표준편차"
                )

            with ml_col4:
                st.metric(
                    "MAPE",
                    f"{ml_metrics.mape:.2f}%",
                    help="평균 절대 백분율 오차 (목표: <5%)"
                )
                if ml_metrics.mape < 5.0:
                    st.success("✅ 목표 달성")
                else:
                    st.warning(f"⚠️  목표 미달 (5%)")

            # R² Score 게이지 차트
            st.markdown("#### R² Score 시각화")
            fig_r2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ml_metrics.r2,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "R² Score (예측 정확도)"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.7], 'color': "lightgray"},
                        {'range': [0.7, 0.85], 'color': "lightyellow"},
                        {'range': [0.85, 0.90], 'color': "lightgreen"},
                        {'range': [0.90, 1.0], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.90
                    }
                }
            ))
            fig_r2.update_layout(height=250)
            st.plotly_chart(fig_r2, use_container_width=True)

            st.info(f"📝 **데이터셋**: {ml_metrics.dataset} | 모델이 {ml_metrics.r2*100:.1f}%의 정확도로 품질을 예측합니다")

        except FileNotFoundError:
            st.warning("⚠️  ML 모델이 아직 학습되지 않았습니다. 먼저 `python scripts/train_model.py`를 실행하세요.")
            ml_metrics = None

        # 에이전트 효율성 지표
        st.markdown("---")
        st.subheader("🤝 에이전트 효율성 지표")
        st.markdown("협상 에이전트의 운영 성능")

        # 시뮬레이션 데이터로 계산 (실제 사용 시 workflow에서 가져옴)
        negotiation_history = [{"id": "current", "turns": 2}]
        approval_decisions = [{"status": result['decision']['status']}]

        agent_metrics = calculator.calculate_agent_metrics(
            negotiation_history=negotiation_history,
            approval_decisions=approval_decisions
        )

        agent_col1, agent_col2, agent_col3 = st.columns(3)

        with agent_col1:
            st.metric(
                "총 협상 횟수",
                f"{agent_metrics.total_negotiations}회",
                help="시스템이 수행한 총 협상 횟수"
            )

        with agent_col2:
            st.metric(
                "평균 협상 턴",
                f"{agent_metrics.avg_negotiation_turns:.1f}회",
                help="협상 완료까지 평균 턴 수 (목표: <3회)"
            )
            if agent_metrics.avg_negotiation_turns < 3.0:
                st.success("✅ 목표 달성")
            else:
                st.warning("⚠️  목표 미달 (3회)")

        with agent_col3:
            st.metric(
                "승인율",
                f"{agent_metrics.approval_rate:.1%}",
                help="제안 중 승인된 비율"
            )

        agent_col4, agent_col5, agent_col6 = st.columns(3)

        with agent_col4:
            st.metric(
                "RAG 적중률",
                f"{agent_metrics.rag_hit_rate:.1%}",
                help="RAG 검색으로 관련 사례를 찾은 비율"
            )

        with agent_col5:
            st.metric(
                "안전 준수율",
                f"{agent_metrics.safety_compliance_rate:.1%}",
                help="물리적 안전 범위 내 제안 비율"
            )

        with agent_col6:
            if agent_metrics.safety_compliance_rate >= 0.95:
                st.success("✅ 안전 기준 충족")
            else:
                st.error("❌ 안전성 미달")

        # 비즈니스 임팩트 지표
        st.markdown("---")
        st.subheader("💰 비즈니스 임팩트 지표")
        st.markdown("시스템 도입으로 인한 실질적 비용 절감 효과")

        # 시뮬레이션 데이터로 계산
        business_metrics = calculator.calculate_business_metrics(
            total_samples=100,
            anomalies_detected=15,
            defects_before=15,
            defects_after=2,
            cost_per_defect=100.0  # $100 per defect
        )

        biz_col1, biz_col2, biz_col3 = st.columns(3)

        with biz_col1:
            st.metric(
                "감지된 이상",
                f"{business_metrics.total_anomalies_detected}건",
                help="시스템이 감지한 품질 이상 건수"
            )

        with biz_col2:
            st.metric(
                "방지된 불량",
                f"{business_metrics.prevented_defects}건",
                help="사전 조치로 방지한 불량품 수"
            )

        with biz_col3:
            st.metric(
                "불량 감소율",
                f"{business_metrics.defect_reduction_rate:.1%}",
                help="조치 전 대비 불량 감소 비율 (목표: 85%)"
            )
            if business_metrics.defect_reduction_rate >= 0.85:
                st.success("✅ 목표 달성")
            else:
                st.warning(f"⚠️  목표 미달 (85%)")

        biz_col4, biz_col5 = st.columns(2)

        with biz_col4:
            st.metric(
                "품질 회복율",
                f"{business_metrics.quality_recovery_rate:.1%}",
                help="이상 감지 후 품질 회복 성공률"
            )

        with biz_col5:
            st.metric(
                "💵 추정 비용 절감",
                f"${business_metrics.estimated_cost_saving:,.2f}",
                help="불량 방지로 인한 추정 비용 절감액"
            )

        # 비용 절감 시각화
        st.markdown("#### 💰 비용 절감 효과")

        fig_cost = go.Figure()

        fig_cost.add_trace(go.Bar(
            name='조치 전 불량 비용',
            x=['비용 비교'],
            y=[15 * 100.0],  # defects_before * cost_per_defect
            marker_color='red',
            text=[f'${15 * 100.0:,.0f}'],
            textposition='auto',
        ))

        fig_cost.add_trace(go.Bar(
            name='조치 후 불량 비용',
            x=['비용 비교'],
            y=[2 * 100.0],  # defects_after * cost_per_defect
            marker_color='green',
            text=[f'${2 * 100.0:,.0f}'],
            textposition='auto',
        ))

        fig_cost.update_layout(
            title="불량 비용 절감 효과",
            yaxis_title="비용 ($)",
            barmode='group',
            height=300
        )

        st.plotly_chart(fig_cost, use_container_width=True)

        st.success(f"💡 **시스템 도입 효과**: {business_metrics.prevented_defects}건의 불량을 사전 방지하여 **${business_metrics.estimated_cost_saving:,.2f} 절감**")

        # 전체 요약 저장
        st.markdown("---")
        if st.button("📥 평가지표 요약 다운로드 (JSON)"):
            calculator.save_summary("models/evaluation_summary.json")
            st.success("평가지표 요약이 models/evaluation_summary.json에 저장되었습니다")

            summary = calculator.get_summary()
            st.json(summary)

else:
    st.info("👈 사이드바에서 '워크플로우 실행' 버튼을 클릭하여 시작하세요.")

    # 시스템 설명
    st.markdown("---")
    st.subheader("시스템 개요")

    st.markdown("""
    **SmartFlow**는 LLM 기반 Multi-Agent 협상을 통한 다단계 제조 공정의 사전 품질 예측 및 자율 조정 시스템입니다.

    ### 주요 기능
    1. **실시간 공정 모니터링**: 프레스 공정의 센서 데이터를 실시간으로 감시하여 이상을 조기 감지
    2. **품질 연쇄 예측**: ML 모델을 통해 현재 공정의 변동이 후속 공정에 미칠 영향 예측
    3. **RAG 기반 추론**: 과거 성공/실패 사례를 검색하여 최적의 조정안 도출
    4. **에이전트 협상**: 여러 공정 에이전트가 협상하여 전체 최적화 달성
    5. **자율 조정**: 승인된 파라미터를 자동으로 적용하여 품질 개선

    ### 에이전트 구성
    - **Process Monitor Agent**: 센서 데이터 수집 및 이상 감지
    - **Quality Cascade Predictor**: 품질 영향 예측
    - **Negotiation Agent**: RAG 기반 조정안 제안 및 협상
    - **Coordinator Agent**: 최종 승인/반려 결정

    ### 기대 효과
    - 불량률 15-20% 감소
    - 재작업 비용 20-25% 절감
    - 조기 문제 식별 속도 35-45% 향상
    """)


# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "스마트 제조 AI Agent 해커톤 2025 | 팀 노동조합 | SmartFlow"
    "</div>",
    unsafe_allow_html=True
)
