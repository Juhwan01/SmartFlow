# SmartFlow 모듈화 구조

## 개요

SmartFlow 시스템을 **데이터셋에 독립적인 모듈화 구조**로 리팩토링했습니다.
이제 데이터셋을 교체해도 **핵심 로직은 그대로 유지**되며, **스키마 설정만 변경**하면 됩니다.

## 모듈 구조

```
SmartFlow/
├── config/
│   └── data_schema.py          # 📋 데이터셋 스키마 정의
├── src/
│   ├── features/
│   │   └── feature_engineer.py # 🔧 Feature Engineering
│   ├── adjustment/
│   │   └── parameter_adapter.py # ⚙️ 파라미터 조정 어댑터
│   ├── prompts/
│   │   └── prompt_generator.py # 💬 LLM 프롬프트 생성기
│   └── agents/
│       ├── process_monitor.py
│       ├── negotiation_agent.py
│       └── coordinator.py
└── scripts/
    ├── train_model.py
    └── evaluate_service.py
```

## 핵심 모듈

### 1️⃣ **Data Schema** (`config/data_schema.py`)

데이터셋 구조와 도메인 지식을 정의합니다.

```python
from config.data_schema import get_schema, CONTINUOUS_FACTORY_SCHEMA

# 현재 데이터셋 스키마 로드
schema = get_schema("continuous_factory_process")

# 스키마 정보
print(schema.stage1.name)  # "press"
print(schema.stage2.name)  # "welding"
print(schema.target_variable)  # "welding_strength"
print(schema.control_to_measurement_mapping)
# {"current": "welding_temp1", ...}
```

**새 데이터셋 추가 방법:**
```python
NEW_DATASET_SCHEMA = DataSchema(
    dataset_name="injection_molding",
    stage1=ProcessStageSchema(...),
    stage2=ProcessStageSchema(...),
    # ...
)
```

### 2️⃣ **Feature Engineer** (`src/features/feature_engineer.py`)

데이터셋에 독립적인 피처 생성 엔진입니다.

```python
from src.features import FeatureEngineer

# 스키마 기반 초기화
fe = FeatureEngineer(schema)

# DataFrame에 피처 추가
df_with_features = fe.apply(df)

# 조정 후 재계산 (dict 형태)
adjusted_data = fe.recalculate_features(
    row_dict,
    feature_names=schema.recalculable_features
)
```

**새 피처 추가 방법:**
```python
from src.features import FeatureRecipe

fe.add_recipe(FeatureRecipe(
    name="my_custom_feature",
    func=lambda df: df['col1'] * df['col2'],
    description="Custom interaction feature",
    dependencies=["col1", "col2"]
))
```

### 3️⃣ **Parameter Adapter** (`src/adjustment/parameter_adapter.py`)

제어 변수 조정을 측정 변수에 적용합니다.

```python
from src.adjustment import ParameterAdapter

# 초기화
adapter = ParameterAdapter(schema, feature_engineer=fe)

# 조정 적용
adjusted_data = adapter.apply_control_adjustments(
    data=original_data,
    control_adjustments={
        "current": 0.03,      # 3% 증가
        "welding_speed": -0.05  # 5% 감소
    },
    recalculate_features=True  # 파생 변수 재계산
)

# 조정 요약
summary = adapter.get_adjustment_summary(original_data, adjusted_data)
```

### 4️⃣ **Prompt Generator** (`src/prompts/prompt_generator.py`)

스키마 기반으로 LLM 프롬프트를 동적 생성합니다.

```python
from src.prompts import PromptGenerator

pg = PromptGenerator(schema)

# Negotiation Agent 프롬프트
system_prompt = pg.generate_negotiation_system_prompt()
# → "당신은 welding 공정 최적화 전문가입니다..."

# 도메인 지식 포함
guidance = pg.get_parameter_descriptions()
# → {"current": "전류 증가 → 열량 증가...", ...}
```

## 사용 예시

### 📊 **데이터셋 교체하기**

**AS-IS (하드코딩)**:
```python
# 모든 파일에서 수동 수정 필요
param_mapping = {"current": "welding_temp1", ...}
system_prompt = "당신은 용접 전문가..."
```

**TO-BE (모듈화)**:
```python
# 1. 새 스키마만 정의
NEW_SCHEMA = DataSchema(
    dataset_name="new_process",
    # ... 스키마 정의
)

# 2. 시스템 전체에 자동 적용
schema = get_schema("new_process")
fe = FeatureEngineer(schema)
adapter = ParameterAdapter(schema, fe)
pg = PromptGenerator(schema)

# → 프롬프트, 조정 로직 모두 자동 업데이트!
```

### 🔄 **기존 코드 마이그레이션**

**evaluate_service.py 예시:**
```python
# Before (하드코딩)
param_mapping = {
    "welding_speed": "welding_temp3",
    "current": "welding_temp1",
}
for adj_key, feature_name in param_mapping.items():
    adjusted[feature_name] *= (1 + adjustments[adj_key])

# After (모듈화)
from config.data_schema import get_schema
from src.adjustment import ParameterAdapter
from src.features import FeatureEngineer

schema = get_schema()
fe = FeatureEngineer(schema)
adapter = ParameterAdapter(schema, fe)

adjusted = adapter.apply_control_adjustments(
    data=raw_row,
    control_adjustments=adjustments,
    recalculate_features=True
)
```

## 장점

✅ **데이터셋 교체 용이**: 스키마만 변경하면 전체 시스템 적용
✅ **재사용성**: 다른 제조 공정에도 동일 구조 사용 가능
✅ **유지보수성**: 로직이 한 곳에 집중, 변경 시 영향 범위 최소화
✅ **테스트 가능**: 각 모듈 독립적으로 테스트
✅ **확장성**: 새 피처, 새 파라미터 동적 추가
✅ **LLM 프롬프트 자동화**: 데이터 구조 변경 시 프롬프트도 자동 업데이트

## 다음 단계

1. 기존 코드 리팩토링 (`evaluate_service.py`, `train_model.py`)
2. Negotiation Agent 프롬프트를 PromptGenerator 사용하도록 수정
3. 테스트 작성
4. Streamlit UI도 스키마 기반으로 업데이트
