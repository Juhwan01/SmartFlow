"""
SmartFlow End-to-End Pipeline Dashboard

CSV 업로드 → RAG 임베딩 → 모델 학습 → LangGraph 테스트 통합 파이프라인
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
import json
import io
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.workflow.langgraph_workflow import SmartFlowWorkflow
from src.rag.retriever import RAGRetriever
from src.data.case_logger import CaseLogger
from config import settings

# 페이지 설정
st.set_page_config(
    page_title="SmartFlow Pipeline",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'pipeline_step' not in st.session_state:
    st.session_state.pipeline_step = 0
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'work_log_data' not in st.session_state:
    st.session_state.work_log_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
if 'workflow_result' not in st.session_state:
    st.session_state.workflow_result = None


# ============================================================================
# 사이드바 - 파이프라인 진행 상태
# ============================================================================
with st.sidebar:
    st.title("🏭 SmartFlow Pipeline")
    st.markdown("**End-to-End 통합 파이프라인**")
    st.markdown("---")

    st.subheader("📊 파이프라인 단계")

    steps = [
        ("1️⃣ 데이터 업로드", st.session_state.csv_data is not None),
        ("2️⃣ RAG 임베딩", st.session_state.rag_initialized),
        ("3️⃣ 모델 학습", st.session_state.model_trained),
        ("4️⃣ 워크플로우 테스트", st.session_state.workflow_result is not None),
    ]

    for step_name, completed in steps:
        if completed:
            st.success(f"✅ {step_name}")
        else:
            st.info(f"⏸️ {step_name}")

    st.markdown("---")
    st.subheader("시스템 설정")
    st.write(f"LLM: `{settings.llm_provider}/{settings.llm_model}`")
    st.write(f"품질 목표: `{settings.quality_threshold:.0%}`")

    st.markdown("---")
    if st.button("🔄 파이프라인 초기화", type="secondary", use_container_width=True):
        for key in ['csv_data', 'work_log_data', 'model_trained', 'rag_initialized', 'workflow_result']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ============================================================================
# 메인 화면
# ============================================================================
st.title("🏭 SmartFlow End-to-End Pipeline")
st.markdown("**데이터 업로드부터 워크플로우 테스트까지 한 번에**")

# 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 1. 데이터 업로드",
    "🔍 2. RAG 임베딩",
    "🤖 3. 모델 학습",
    "🔄 4. 워크플로우 테스트",
    "📊 5. 결과 분석"
])

# ============================================================================
# TAB 1: 데이터 업로드
# ============================================================================
with tab1:
    st.header("📁 데이터 업로드")
    st.markdown("CSV 공정 데이터와 작업일지를 업로드하세요.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏭 공정 데이터 (CSV)")
        st.markdown("""
        **필요한 데이터:**
        - 프레스 공정 데이터 (press_thickness, press_pressure, etc.)
        - 용접 공정 데이터 (welding_temp1-5, welding_pressure, etc.)
        - 타겟 변수: `welding_strength`
        """)

        csv_file = st.file_uploader(
            "CSV 파일 업로드",
            type=['csv'],
            key="csv_uploader",
            help="공정 데이터가 포함된 CSV 파일을 업로드하세요."
        )

        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
                st.session_state.csv_data = df

                st.success(f"✅ CSV 파일 업로드 완료: {len(df)} rows × {len(df.columns)} columns")

                with st.expander("📊 데이터 미리보기", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)

                    st.markdown("**데이터 통계:**")
                    st.write(f"- 총 행 수: {len(df):,}")
                    st.write(f"- 총 열 수: {len(df.columns)}")

                    if 'welding_strength' in df.columns:
                        st.write(f"- 타겟 변수 평균: {df['welding_strength'].mean():.4f}")
                        st.write(f"- 타겟 변수 범위: {df['welding_strength'].min():.4f} ~ {df['welding_strength'].max():.4f}")

                    # 필수 컬럼 체크
                    required_cols = ['welding_strength']
                    missing_cols = [col for col in required_cols if col not in df.columns]

                    if missing_cols:
                        st.error(f"⚠️ 필수 컬럼 누락: {missing_cols}")
                    else:
                        st.success("✅ 모든 필수 컬럼 확인됨")

            except Exception as e:
                st.error(f"❌ CSV 파일 읽기 오류: {e}")

        # 기존 데이터 사용 옵션
        st.markdown("---")
        if st.button("📂 기존 데이터 사용 (data/continuous_factory_process.csv)", use_container_width=True):
            try:
                from src.data.data_preprocessing import ManufacturingDataProcessor
                processor = ManufacturingDataProcessor()
                df = processor.create_mapped_dataset()
                st.session_state.csv_data = df
                st.success(f"✅ 기존 데이터 로드 완료: {len(df)} rows")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 기존 데이터 로드 실패: {e}")

    with col2:
        st.subheader("📝 작업일지 (RAG용)")
        st.markdown("""
        **작업일지 형식:**
        - JSON 또는 텍스트 파일
        - 과거 문제 해결 사례
        - 파라미터 조정 이력
        """)

        work_log_file = st.file_uploader(
            "작업일지 파일 업로드",
            type=['json', 'txt', 'jsonl'],
            key="work_log_uploader",
            help="과거 작업일지를 업로드하세요."
        )

        if work_log_file is not None:
            try:
                content = work_log_file.read().decode('utf-8')

                # JSON 형식 시도
                try:
                    work_log_data = json.loads(content)
                    st.session_state.work_log_data = work_log_data

                    if isinstance(work_log_data, list):
                        st.success(f"✅ 작업일지 업로드 완료: {len(work_log_data)} 건")
                    else:
                        st.success(f"✅ 작업일지 업로드 완료")

                    with st.expander("📋 작업일지 미리보기", expanded=False):
                        st.json(work_log_data if isinstance(work_log_data, dict) else work_log_data[:3])

                except json.JSONDecodeError:
                    # 텍스트 형식
                    st.session_state.work_log_data = content
                    st.success(f"✅ 작업일지 업로드 완료: {len(content)} 문자")

                    with st.expander("📋 작업일지 미리보기", expanded=False):
                        st.text(content[:500] + "..." if len(content) > 500 else content)

            except Exception as e:
                st.error(f"❌ 작업일지 읽기 오류: {e}")

        # 기존 RAG 데이터 사용 옵션
        st.markdown("---")
        if st.button("📂 기존 RAG 데이터 사용", use_container_width=True):
            st.session_state.work_log_data = "existing"
            st.success("✅ 기존 RAG 데이터를 사용합니다")
            st.info("data/case_history.jsonl에 저장된 케이스를 사용합니다")

    # 공정 단계 설정
    if st.session_state.csv_data is not None:
        st.markdown("---")
        st.subheader("🔧 공정 단계 설정 (2-Stage Cascade Detection)")
        st.markdown("""
        **SmartFlow MVP 시나리오:**
        - **1차 공정 (프레스)**: 두께·압력 이상 감지
        - **2차 공정 (용접)**: 1차 이상이 품질에 미칠 영향 예측
        - **조정**: 용접 파라미터를 조정해 품질 회복
        """)

        from config.data_schema import get_schema
        schema = get_schema()

        col_stage1, col_stage2 = st.columns(2)

        with col_stage1:
            st.markdown("**1️⃣ 프레스 공정 (Stage 1)**")
            stage1_vars = schema.stage1.measurement_variables
            st.info(f"측정 변수: {len(stage1_vars)}개")
            with st.expander("변수 목록 보기", expanded=False):
                for var in stage1_vars:
                    available = "✅" if var in st.session_state.csv_data.columns else "❌"
                    st.write(f"{available} `{var}`")

        with col_stage2:
            st.markdown("**2️⃣ 용접 공정 (Stage 2)**")
            stage2_vars = schema.stage2.measurement_variables
            st.info(f"측정 변수: {len(stage2_vars)}개")
            with st.expander("변수 목록 보기", expanded=False):
                for var in stage2_vars:
                    available = "✅" if var in st.session_state.csv_data.columns else "❌"
                    st.write(f"{available} `{var}`")

        st.markdown("**🎯 제어 변수 → 측정 변수 매핑 (조정 시 사용)**")
        control_mapping = schema.control_to_measurement_mapping
        mapping_rows = []
        for ctrl, measure in control_mapping.items():
            mapping_rows.append({
                "제어 변수 (개념)": ctrl,
                "→ 측정 변수 (실제)": measure,
                "데이터 존재": "✅" if measure in st.session_state.csv_data.columns else "❌"
            })
        mapping_df = pd.DataFrame(mapping_rows)
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)

        st.caption("""
        💡 **Tip**: 워크플로우는 이 매핑을 사용해 제어 변수 조정값(예: current +3%)을
        실제 측정 변수(예: welding_temp1)에 반영하고, 파생 변수를 재계산합니다.
        """)

    # 다음 단계 버튼
    st.markdown("---")
    if st.session_state.csv_data is not None:
        st.success("✅ 데이터 업로드 완료!")
        st.info("👉 이제 상단의 **'2. RAG 임베딩'** 탭을 클릭하여 다음 단계로 진행하세요.")
    else:
        st.warning("⏸️ CSV 파일을 먼저 업로드하세요.")


# ============================================================================
# TAB 2: RAG 임베딩
# ============================================================================
with tab2:
    st.header("🔍 RAG 임베딩")
    st.markdown("작업일지를 벡터 DB에 임베딩하여 유사 사례 검색을 가능하게 합니다.")

    if st.session_state.csv_data is None:
        st.warning("⚠️ 먼저 데이터를 업로드하세요 (Tab 1)")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 RAG 시스템 상태")

            if st.session_state.rag_initialized:
                st.success("✅ RAG 시스템 초기화 완료")

                try:
                    rag = RAGRetriever()
                    if rag.initialized:
                        st.info(f"📚 벡터 DB: {rag.collection.count()} 건의 케이스 저장됨")

                    # 샘플 검색 테스트
                    with st.expander("🔍 검색 테스트", expanded=False):
                        test_query = st.text_input("테스트 쿼리:", "품질 저하 문제")

                        if st.button("검색 실행"):
                            with st.spinner("검색 중..."):
                                results = rag.search(test_query, n_results=3)

                                st.write(f"**검색 결과: {len(results)} 건**")
                                for i, result in enumerate(results, 1):
                                    st.markdown(f"**{i}. 유사도: {result.get('similarity', 0):.3f}**")
                                    st.text(result.get('text', '')[:200] + "...")
                                    st.markdown("---")
                except Exception as e:
                    st.error(f"RAG 시스템 확인 오류: {e}")
            else:
                st.info("⏸️ RAG 시스템이 아직 초기화되지 않았습니다.")

        with col2:
            st.subheader("⚙️ 임베딩 설정")

            use_existing = st.checkbox(
                "기존 RAG 데이터 사용",
                value=st.session_state.work_log_data == "existing",
                help="기존에 저장된 케이스 히스토리를 사용합니다."
            )

            if not use_existing and st.session_state.work_log_data is not None:
                st.info("업로드된 작업일지를 사용합니다.")

        st.markdown("---")

        # RAG 초기화 버튼
        if not st.session_state.rag_initialized:
            if st.button("🚀 RAG 임베딩 시작", type="primary", use_container_width=True):
                with st.spinner("RAG 시스템 초기화 및 임베딩 중..."):
                    try:
                        # RAG 초기화
                        rag = RAGRetriever()
                        rag.initialize()

                        # 작업일지 데이터가 있으면 추가
                        if st.session_state.work_log_data is not None and st.session_state.work_log_data != "existing":
                            case_logger = CaseLogger()

                            # JSON 리스트인 경우
                            if isinstance(st.session_state.work_log_data, list):
                                for case in st.session_state.work_log_data:
                                    case_logger.record_case(case)
                                st.success(f"✅ {len(st.session_state.work_log_data)}건의 작업일지 임베딩 완료")

                            # JSON 객체인 경우
                            elif isinstance(st.session_state.work_log_data, dict):
                                case_logger.record_case(st.session_state.work_log_data)
                                st.success("✅ 작업일지 임베딩 완료")

                            # 재초기화
                            rag.initialize()

                        st.session_state.rag_initialized = True
                        st.success("✅ RAG 시스템 초기화 완료!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ RAG 초기화 실패: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.success("✅ RAG 임베딩 완료!")
            st.info("👉 이제 상단의 **'3. 모델 학습'** 탭을 클릭하여 다음 단계로 진행하세요.")


# ============================================================================
# TAB 3: 모델 학습
# ============================================================================
with tab3:
    st.header("🤖 ML 모델 학습")
    st.markdown("XGBoost 품질 예측 모델을 학습합니다.")

    if st.session_state.csv_data is None:
        st.warning("⚠️ 먼저 데이터를 업로드하세요 (Tab 1)")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 데이터 정보")
            df = st.session_state.csv_data

            st.write(f"- 총 샘플 수: {len(df):,}")
            st.write(f"- 피처 수: {len(df.columns) - 1} (타겟 제외)")

            if 'welding_strength' in df.columns:
                st.write(f"- 타겟 변수 평균: {df['welding_strength'].mean():.4f}")
                st.write(f"- 타겟 변수 표준편차: {df['welding_strength'].std():.4f}")

        with col2:
            st.subheader("⚙️ 학습 설정")

            n_estimators = st.number_input("트리 개수", min_value=100, max_value=5000, value=2000, step=100)
            max_depth = st.number_input("최대 깊이", min_value=3, max_value=15, value=8, step=1)
            learning_rate = st.number_input("학습률", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.3f")

        st.markdown("---")

        # 모델 학습 버튼
        if not st.session_state.model_trained:
            if st.button("🚀 모델 학습 시작", type="primary", use_container_width=True):

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # 데이터를 임시 CSV로 저장
                    status_text.text("1/5 데이터 준비 중...")
                    progress_bar.progress(20)

                    data_dir = Path("data")
                    data_dir.mkdir(parents=True, exist_ok=True)
                    temp_csv_path = data_dir / "uploaded_data.csv"
                    st.session_state.csv_data.to_csv(temp_csv_path, index=False)

                    # ModelTrainer 초기화
                    status_text.text("2/5 모델 트레이너 초기화 중...")
                    progress_bar.progress(40)

                    # ManufacturingDataProcessor가 uploaded_data.csv를 읽도록 수정하거나
                    # 직접 학습 가능하도록 수정 필요
                    # 여기서는 간단히 train_model.py의 로직 사용

                    from scripts.train_model import ModelTrainer

                    trainer = ModelTrainer()

                    # 학습 시작
                    status_text.text("3/5 모델 학습 중... (수 분 소요)")
                    progress_bar.progress(60)

                    model, metrics = trainer.train_xgboost(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate
                    )

                    # 모델 저장
                    status_text.text("4/5 모델 저장 중...")
                    progress_bar.progress(80)

                    trainer.save_model()

                    # 완료
                    status_text.text("5/5 완료!")
                    progress_bar.progress(100)

                    st.session_state.model_trained = True

                    # 결과 표시
                    st.success("✅ 모델 학습 완료!")

                    col_m1, col_m2, col_m3 = st.columns(3)

                    with col_m1:
                        st.metric("Validation MAE", f"{metrics['validation']['mae']:.4f}")

                    with col_m2:
                        st.metric("Validation MAPE", f"{metrics['validation']['mape']:.2f}%")

                    with col_m3:
                        st.metric("Validation RMSE", f"{metrics['validation']['rmse']:.4f}")

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 모델 학습 실패: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.success("✅ 모델 학습 완료! 다음 단계로 진행하세요.")

            # 학습된 모델 정보 표시
            try:
                with open("models/metrics.json", 'r') as f:
                    metrics = json.load(f)

                st.subheader("📊 모델 성능")

                col_m1, col_m2, col_m3 = st.columns(3)

                with col_m1:
                    st.metric("Validation MAE", f"{metrics['validation']['mae']:.4f}",
                             help="평균 절대 오차 (목표: <0.2)")

                with col_m2:
                    st.metric("Validation MAPE", f"{metrics['validation']['mape']:.2f}%",
                             help="평균 절대 백분율 오차 (목표: <2%)")

                with col_m3:
                    st.metric("Validation RMSE", f"{metrics['validation']['rmse']:.4f}",
                             help="평균 제곱근 오차")

            except Exception as e:
                st.info("모델 메트릭 파일을 찾을 수 없습니다.")

            st.markdown("---")
            st.info("👉 이제 상단의 **'4. 워크플로우 테스트'** 탭을 클릭하여 실행하세요.")


# ============================================================================
# TAB 4: 워크플로우 테스트
# ============================================================================
with tab4:
    st.header("🔄 LangGraph 워크플로우 테스트")
    st.markdown("Multi-Agent 협상 워크플로우를 실행합니다.")

    if not st.session_state.model_trained:
        st.warning("⚠️ 먼저 모델을 학습하세요 (Tab 3)")
    elif not st.session_state.rag_initialized:
        st.warning("⚠️ 먼저 RAG를 초기화하세요 (Tab 2)")
    else:
        st.info("✅ 모든 준비 완료! 워크플로우를 실행할 수 있습니다.")

        # 워크플로우 실행 버튼
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button("🚀 워크플로우 실행", type="primary", use_container_width=True):
                with st.spinner("Multi-Agent 워크플로우 실행 중..."):
                    try:
                        workflow = SmartFlowWorkflow()
                        result = workflow.run()
                        st.session_state.workflow_result = result
                        st.success("✅ 워크플로우 실행 완료!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 워크플로우 실행 실패: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        with col2:
            if st.session_state.workflow_result is not None:
                if st.button("🔄 재실행", use_container_width=True):
                    st.session_state.workflow_result = None
                    st.rerun()

        # 워크플로우 결과 표시
        if st.session_state.workflow_result is not None:
            result = st.session_state.workflow_result
            ml_row = result.get("ml_row") or {}
            ml_row_adjusted = result.get("ml_row_adjusted") or {}
            negotiation_log = result.get("negotiation_log") or []

            st.markdown("---")
            st.subheader("📊 실행 결과 요약")

            # 주요 지표
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                press_thickness = result['press_data']['thickness']
                st.metric("프레스 두께", f"{press_thickness:.4f}mm")

            with col2:
                pred_quality = result['prediction']['predicted_quality_score']
                st.metric("예측 품질", f"{pred_quality:.1%}")

            with col3:
                risk = result['prediction']['risk_level']
                risk_colors = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
                st.metric("위험 수준", f"{risk_colors.get(risk, '⚪')} {risk.upper()}")

            with col4:
                if result['execution_result'].get('executed'):
                    final_quality = result['execution_result']['final_quality_score']
                    st.metric("최종 품질", f"{final_quality:.1%}")
                else:
                    st.metric("최종 품질", "N/A")

            # 상세 결과
            st.markdown("---")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.subheader("🔍 이상 감지 (2-Stage Cascade)")
                if result['alert']:
                    alert = result['alert']
                    st.error(f"""
                    **알림 ID**: {alert['alert_id']}
                    **공정 단계**: {alert.get('process_stage', 'press').upper()}
                    **심각도**: {alert['severity'].upper()}
                    **문제**: {alert['issue_description']}
                    """)
                    st.caption("💡 1차(프레스) 이상 → 2차(용접) 품질 저하 예상")
                else:
                    st.success("이상 없음 - 정상 운영")

                st.subheader("🤝 조정안 (2차 공정 파라미터)")
                proposal = result['proposal']
                st.write(f"**제안 ID**: {proposal['proposal_id']}")
                st.write(f"**예상 품질**: {proposal['expected_quality']:.1%}")

                st.markdown("**조정 내역 (제어 변수 → 측정 변수):**")
                adjustments = proposal['adjustments']
                for param, value in adjustments.items():
                    st.write(f"- **{param}**: {value:+.1%}")
                st.caption("💡 제어 변수 조정이 실제 센서 측정값에 반영됩니다.")

            with detail_col2:
                st.subheader("✅ 최종 결정")
                decision = result['decision']

                if decision['status'] == 'approved':
                    st.success(f"✅ 제안 승인")
                elif decision['status'] == 'conditional_approved':
                    st.warning(f"⚠️ 조건부 승인")
                else:
                    st.error(f"❌ 제안 반려")

                st.text(decision['rationale'])

                st.subheader("🎯 실행 결과")
                exec_result = result['execution_result']

                if exec_result.get('executed'):
                    st.success("조정 실행 완료")
                    st.write(f"**최종 품질**: {exec_result['final_quality_score']:.1%}")
                    st.write(f"**최종 강도**: {exec_result['final_strength']:.2f} MPa")
                    st.write(f"**품질 기준 충족**: {'✅ 예' if exec_result['meets_threshold'] else '❌ 아니오'}")
                else:
                    st.warning(f"조정 미실행: {exec_result.get('reason', 'Unknown')}")

            st.markdown("---")
            st.subheader("💬 협상 로그")

            if negotiation_log:
                status_badge = {
                    "alert": "🟥",
                    "info": "🟦",
                    "proposal": "🟩",
                    "decision": "🟨",
                    "result": "🟪",
                    "fallback": "⬜",
                    "warning": "🟧"
                }

                for entry in negotiation_log:
                    # Handle both dict and string entries
                    if isinstance(entry, dict):
                        badge = status_badge.get(entry.get("status", "info"), "🔹")
                        meta = entry.get("meta") or {}
                        meta_text = ", ".join([f"{k}: {v}" for k, v in meta.items()]) if meta else ""

                        st.markdown(
                            f"{badge} **[{entry.get('timestamp','--:--')}] {entry.get('role','unknown')} · {entry.get('label','')}**"
                        )
                        st.write(entry.get("message", ""))
                        if meta_text:
                            st.caption(meta_text)
                    else:
                        # If entry is a string or other type, display it simply
                        st.markdown(f"🔹 {entry}")
                    st.divider()
            else:
                st.info("협상 로그가 비어 있습니다. LLM 협상 없이 기본 조정안이 사용되었을 수 있습니다.")

            st.markdown("---")
            st.subheader("🧮 ML 샘플 변수 비교 (공정별)")

            if ml_row:
                available_fields = sorted(ml_row.keys())
                
                # 공정별 변수 그룹
                from config.data_schema import get_schema
                schema = get_schema()
                stage1_fields = [f for f in schema.stage1.measurement_variables if f in available_fields]
                stage2_fields = [f for f in schema.stage2.measurement_variables if f in available_fields]
                target_field = schema.target_variable if schema.target_variable in available_fields else None
                
                view_mode = st.radio(
                    "변수 선택 모드",
                    options=["공정별 자동 선택", "수동 선택"],
                    horizontal=True,
                    help="공정 단계별로 자동 필터링하거나 직접 선택하세요."
                )
                
                if view_mode == "공정별 자동 선택":
                    show_stage = st.radio(
                        "표시할 공정",
                        options=["1차(프레스)", "2차(용접)", "타겟", "전체"],
                        horizontal=True
                    )
                    if show_stage == "1차(프레스)":
                        selected_fields = stage1_fields
                    elif show_stage == "2차(용접)":
                        selected_fields = stage2_fields
                    elif show_stage == "타겟":
                        selected_fields = [target_field] if target_field else []
                    else:
                        selected_fields = stage1_fields + stage2_fields + ([target_field] if target_field else [])
                else:
                    default_fields = stage1_fields[:3] + stage2_fields[:3] + ([target_field] if target_field else [])
                    selected_fields = st.multiselect(
                        "표시할 변수 선택",
                        options=available_fields,
                        default=default_fields,
                        help="ML 샘플에서 확인하고 싶은 센서·제어 변수를 선택하세요."
                    )

                if selected_fields:
                    comparison_rows = []
                    for field in selected_fields:
                        base_val = ml_row.get(field)
                        adj_val = ml_row_adjusted.get(field) if ml_row_adjusted else None

                        if isinstance(base_val, (int, float)) and isinstance(adj_val, (int, float)) and base_val not in [0, None]:
                            change_pct = (adj_val - base_val) / base_val * 100
                        else:
                            change_pct = None

                        comparison_rows.append({
                            "변수": field,
                            "원본": base_val,
                            "조정후": adj_val if ml_row_adjusted else "-",
                            "변화율(%)": f"{change_pct:+.2f}%" if change_pct is not None else "-"
                        })

                    comparison_df = pd.DataFrame(comparison_rows).set_index("변수")
                    st.dataframe(comparison_df, use_container_width=True)
                else:
                    st.info("표시할 변수를 선택하세요.")
            else:
                st.info("ML 데이터셋이 없어 시뮬레이터 입력을 사용했습니다. 업로드된 CSV로 모델을 학습하면 변수 비교가 가능합니다.")


# ============================================================================
# TAB 5: 결과 분석
# ============================================================================
with tab5:
    st.header("📊 전체 결과 분석")

    if st.session_state.workflow_result is None:
        st.info("⏸️ 먼저 워크플로우를 실행하세요 (Tab 4)")
    else:
        result = st.session_state.workflow_result

        # 품질 게이지
        st.subheader("📈 품질 점수 변화")

        pred_quality = result['prediction']['predicted_quality_score']
        final_quality = result['execution_result'].get('final_quality_score', pred_quality)

        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=final_quality * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "최종 품질 점수 (%)"},
            delta={'reference': pred_quality * 100, 'increasing': {'color': "green"}},
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

        # 상세 메트릭
        st.markdown("---")
        st.subheader("📋 상세 메트릭")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**프레스 공정**")
            st.write(f"두께: {result['press_data']['thickness']:.4f}mm")
            st.write(f"압력: {result['press_data']['pressure']:.2f}MPa")
            st.write(f"온도: {result['press_data']['temperature']:.2f}°C")
            st.write(f"이상: {'예' if result['press_data']['is_anomaly'] else '아니오'}")

        with col2:
            st.markdown("**품질 예측**")
            pred = result['prediction']
            st.write(f"예상 품질: {pred['predicted_quality_score']:.1%}")
            st.write(f"예상 강도: {pred['predicted_strength']:.2f}MPa")
            st.write(f"강도 저하: {pred['strength_degradation_pct']:.2f}%")
            st.write(f"위험 수준: {pred['risk_level'].upper()}")

        with col3:
            st.markdown("**조정 결과**")
            if result['execution_result'].get('executed'):
                exec_res = result['execution_result']
                st.write(f"최종 품질: {exec_res['final_quality_score']:.1%}")
                st.write(f"최종 강도: {exec_res['final_strength']:.2f}MPa")
                improvement = exec_res['final_quality_score'] - pred['predicted_quality_score']
                st.write(f"개선량: {improvement:+.1%}")
                st.write(f"기준 충족: {'예' if exec_res['meets_threshold'] else '아니오'}")
            else:
                st.write("조정 미실행")

        # JSON 다운로드
        st.markdown("---")
        if st.button("📥 전체 결과 다운로드 (JSON)"):
            result_json = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                label="💾 JSON 파일 다운로드",
                data=result_json,
                file_name=f"smartflow_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


# ============================================================================
# 푸터
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "SmartFlow End-to-End Pipeline | Powered by LangGraph & RAG"
    "</div>",
    unsafe_allow_html=True
)
