# SmartFlow

**LLM 기반 Multi-Agent 협상을 통한 다단계 제조 공정의 사전 품질 예측 및 자율 조정 시스템**

## 프로젝트 개요

스마트 제조 환경에서 공정 간 품질 연쇄 효과를 사전에 예측하고, LLM Agent들이 과거 작업 이력(RAG)을 참조하여 추론 및 협상을 통해 공정 파라미터를 자율 조정하는 시스템입니다.

## MVP 범위

### 대상 공정
- 프레스 공정 → 용접 공정 (2단계 연결)

### 핵심 시나리오
두께 편차 발생 시 용접 파라미터 자동 재조정

### 구현 기능
1. **Data Ingestion**: 가상 센서 데이터 생성 및 이상 감지
2. **Reasoning**: RAG를 통한 과거 실패 사례 검색
3. **Negotiation**: 에이전트 간 채팅 로그(협상 과정) 시각화

## 시스템 아키텍처

### 에이전트 구성
1. **Process Monitor Agent** (감시자)
   - 센서 데이터 실시간 수집
   - 이상 징후 1차 포착

2. **Quality Cascade Predictor** (예측자)
   - ML/DL 모델을 통한 품질 영향 예측
   - 후속 공정 품질 저하 수치 계산

3. **RAG-enabled Negotiation Agent** (협상자)
   - ML 예측값 언어적 해석
   - 과거 사례 검색 및 최적 조정안 추론
   - 타 공정 에이전트와 협상

4. **Coordinator Agent** (조정자)
   - 전체 생산 목표 고려
   - 협상 결과 최종 승인/반려

## 기술 스택

- **LLM**: GPT-4, Claude 3.5 Sonnet
- **Multi-Agent Framework**: LangGraph
- **RAG**: LangChain, ChromaDB
- **ML**: XGBoost, scikit-learn
- **Dashboard**: Streamlit
- **Data**: Kaggle Multi-Stage Continuous-Flow Manufacturing Process (14,089 samples)

## 설치 및 실행

### 1. 환경 설정

```bash
# Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일에 API 키 설정
# OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 입력
```

### 3. ML 모델 학습

시스템 실행 전 반드시 ML 모델을 학습해야 합니다.

```bash
# XGBoost 품질 예측 모델 학습
python scripts/train_model.py
```

학습 완료 후 생성되는 파일:
- `models/quality_predictor.pkl` - 학습된 XGBoost 모델
- `models/scaler.pkl` - 데이터 정규화 Scaler
- `models/metrics.json` - 성능 지표 (R², MAE, MAPE)
- `models/variable_mapping.json` - 변수 매핑 정보

목표 성능:
- **R² Score**: >0.90 (92% 예측 정확도)
- **MAE**: <1.0
- **MAPE**: <5%

자세한 학습 가이드는 [SETUP_AND_TRAIN.md](SETUP_AND_TRAIN.md)를 참조하세요.

### 4. 시스템 실행

```bash
# 메인 시스템 실행
python main.py

# 대시보드 실행
streamlit run src/dashboard/app.py
```

## 프로젝트 구조

```
SmartFlow/
├── src/
│   ├── agents/              # 에이전트 모듈
│   │   ├── process_monitor.py
│   │   ├── quality_predictor.py      # 규칙 기반 예측기 (Fallback)
│   │   ├── ml_quality_predictor.py   # XGBoost 기반 예측기 (Production)
│   │   ├── negotiation_agent.py
│   │   └── coordinator.py
│   ├── data/                # 데이터 생성 및 처리
│   │   ├── continuous_factory_process.csv  # Kaggle 실제 데이터 (14,089 samples)
│   │   ├── data_preprocessing.py           # 데이터 전처리 및 변수 매핑
│   │   ├── sensor_simulator.py             # 시뮬레이션 데이터 생성
│   │   └── sample_cases.py                 # RAG용 샘플 케이스
│   ├── evaluation/          # 평가지표 모듈
│   │   └── metrics.py       # ML, Agent, Business 지표 계산
│   ├── rag/                 # RAG 시스템
│   │   ├── vectorstore.py
│   │   └── retriever.py
│   ├── workflow/            # LangGraph 워크플로우
│   │   └── langgraph_workflow.py
│   └── dashboard/           # 시각화 대시보드
│       └── app.py           # 평가지표 탭 포함
├── scripts/                 # 유틸리티 스크립트
│   └── train_model.py       # ML 모델 학습 스크립트
├── models/                  # 학습된 모델 및 메트릭 (학습 후 생성)
│   ├── quality_predictor.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   └── variable_mapping.json
├── data/                    # 데이터 저장
│   └── historical_cases/
├── config/                  # 설정 파일
│   └── settings.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── SETUP_AND_TRAIN.md       # 설치 및 학습 가이드
└── main.py
```

## 데이터 및 ML 모델

### 사용 데이터
**Kaggle: Multi-Stage Continuous-Flow Manufacturing Process**
- 14,089개 샘플
- 116개 컬럼 (Stage1, Machine4-15, Stage2 등)
- 실제 제조 공정 데이터

### 변수 매핑 전략
Kaggle 익명 데이터를 SmartFlow 시나리오에 매핑:

| SmartFlow 변수 | Kaggle 원본 컬럼 | 설명 |
|---------------|-----------------|------|
| press_thickness | Stage1.Output.Measurement0.U.Actual | 프레스 두께 (주요 입력) |
| press_measurement1/2 | Stage1.Output.Measurement1/2.U.Actual | 프레스 측정값 |
| welding_temp1 | Machine4.Temperature1.C.Actual | 용접 온도 1 (전류 관련) |
| welding_pressure | Machine4.Pressure.C.Actual | 용접 압력 |
| welding_temp3 | Machine4.Temperature3.C.Actual | 용접 온도 3 (속도 관련) |
| welding_control1/2 | Machine5.Temperature1/2.C.Actual | 용접 제어 변수 |
| **welding_strength** | **Stage2.Output.Measurement0.U.Actual** | **용접 강도 (예측 타겟)** |

총 9개 Feature를 사용하여 XGBoost 회귀 모델 학습 (Stage2 Output 변수는 Data Leakage 방지를 위해 제외)

### ML 모델 아키텍처
- **알고리즘**: XGBoost Regressor
- **학습**: 80% train, 20% test split
- **정규화**: MinMaxScaler (0-1 범위)
- **하이퍼파라미터**: n_estimators=150, max_depth=8, learning_rate=0.05
- **출력**: 용접 강도 예측값 (연속형)

## 대표 시나리오

1. **감지**: 프레스 공정의 Monitor Agent가 두께 편차(+0.02mm) 발생 감지
2. **예측**: ML Predictor가 XGBoost 모델로 용접 강도 4.8% 저하 예측
3. **검색 및 추론**: Negotiation Agent가 RAG DB에서 과거 유사 사례 검색
4. **협상**: 용접 공정 Agent와 협상하여 합의안 도출 (속도 5% 감소, 압력 2% 증가)
5. **조치**: Coordinator 승인 후 파라미터 자동 조정, 결과를 RAG DB에 저장

## 기대 효과

- 최종 불량률 15-20% 감소
- 재작업 비용 20-25% 절감
- 조기 문제 식별 속도 35-45% 향상
- 예측 정확도 >90%
- 사전 조정 성공률 >85%

## 평가지표

SmartFlow는 3가지 카테고리의 정량적 지표로 성능을 측정합니다:

### 1. ML 모델 성능 지표
- **R² Score**: >0.90 (품질 예측 정확도)
- **MAE** (Mean Absolute Error): <1.0
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error): <5%

### 2. 에이전트 효율성 지표
- **평균 협상 턴 수**: <3회 (빠른 합의 도달)
- **RAG 적중률**: 과거 사례 검색 성공률
- **승인율**: 제안 중 승인된 비율
- **안전 준수율**: 물리적 안전 범위 내 제안 비율

### 3. 비즈니스 임팩트 지표
- **불량 감소율**: 85% 이상
- **품질 회복율**: 이상 감지 후 품질 회복 성공률
- **비용 절감액**: 불량 방지로 인한 추정 절감액 ($)

대시보드의 "평가지표" 탭에서 실시간으로 확인할 수 있습니다.

## 팀 정보

- **팀명**: 노동조합
- **팀원**: 정주환, 박준수
- **대회**: 스마트 제조 AI Agent 해커톤 2025

## 라이센스

MIT License
