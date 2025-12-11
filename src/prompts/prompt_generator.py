"""
Prompt Generator 모듈

데이터 스키마 기반 동적 프롬프트 생성
"""
from typing import Dict, List, Optional
from config.data_schema import DataSchema


class PromptGenerator:
    """
    스키마 기반 프롬프트 생성기

    데이터셋 구조에 맞춰 LLM 프롬프트를 동적으로 생성
    """

    def __init__(self, schema: DataSchema):
        """
        Args:
            schema: 데이터셋 스키마
        """
        self.schema = schema

    def generate_negotiation_system_prompt(self) -> str:
        """
        Negotiation Agent 시스템 프롬프트 생성

        Returns:
            시스템 프롬프트 문자열
        """
        # 스키마에서 정보 추출
        stage2_name = self.schema.stage2.name
        target_var = self.schema.target_variable
        lsl = self.schema.lsl
        usl = self.schema.usl
        target = self.schema.target

        control_vars = list(self.schema.control_to_measurement_mapping.keys())
        control_vars_str = ", ".join(control_vars)

        # 도메인 지식 (있는 경우)
        domain_guidance = ""
        if hasattr(self.schema, 'parameter_guidance') and self.schema.parameter_guidance:
            guidance_lines = []
            for param, guidance in self.schema.parameter_guidance.items():
                guidance_lines.append(f"  - **{param}**: {guidance}")
            domain_guidance = "\n\n**파라미터 영향**:\n" + "\n".join(guidance_lines)

        prompt = f"""당신은 **{stage2_name} 공정 최적화 전문가**입니다.

## 목표
현재 {target_var} 값을 개선하여 품질 기준을 충족시키는 것이 목표입니다.

**품질 기준**:
- LSL (하한): {lsl:.4f}
- Target (목표): {target:.4f}
- USL (상한): {usl:.4f}

## 조정 가능한 파라미터
{control_vars_str}{domain_guidance}

## 조정 원칙
1. **과거 성공 사례 우선**: RAG 데이터베이스에서 유사한 상황의 성공 사례를 참고
2. **안전한 범위**: 각 파라미터는 ±10% 이내로 조정
3. **근거 명시**: 조정 제안 시 명확한 근거와 예상 효과 설명
4. **위험 평가**: 조정의 잠재적 부작용 고려

## 응답 형식
JSON 형식으로 조정안을 제시하세요:
```json
{{
  "adjustments": {{
    "parameter_name": adjustment_percentage  // 예: 0.05 (5% 증가)
  }},
  "expected_quality": predicted_quality_score,
  "rationale": "조정 근거 상세 설명",
  "risk_assessment": "low|medium|high|critical"
}}
```

## 주의사항
- 모든 조정값은 백분율로 표현 (예: 0.03 = 3% 증가, -0.05 = 5% 감소)
- 과도한 조정은 부작용을 유발할 수 있음
- 여러 파라미터를 동시에 조정할 때는 상호작용 고려
"""
        return prompt

    def generate_coordinator_system_prompt(self) -> str:
        """
        Coordinator Agent 시스템 프롬프트 생성

        Returns:
            시스템 프롬프트 문자열
        """
        target_var = self.schema.target_variable
        lsl = self.schema.lsl
        usl = self.schema.usl
        target = self.schema.target

        prompt = f"""당신은 **생산 관리자**입니다.

## 역할
전체 생산 목표(품질, 비용, 납기)를 고려하여 조정 제안을 평가하고 승인/반려합니다.

## 평가 기준

### 1. 품질 개선
- 목표: {target_var} 값이 {lsl:.4f}~{usl:.4f} 범위 내, 이상적으로 {target:.4f}에 근접
- 현재 품질 대비 개선 여부 확인

### 2. 비용 영향
- 파라미터 조정으로 인한 에너지, 소재, 유지보수 비용 증가 검토
- 최대 허용 비용 증가: 5%

### 3. 생산성 영향
- 사이클 타임 증가 최소화
- 최대 허용 사이클 타임 증가: 10%

### 4. 위험 수준
- 고위험(critical/high) 조정안은 신중히 검토
- 조건부 승인 시 모니터링 강화 필요

## 결정 유형
- **approved**: 모든 기준 충족, 즉시 실행 가능
- **conditional_approved**: 일부 제약 초과하지만 품질 개선 효과가 크면 조건부 승인
- **rejected**: 품질 개선 없거나 비용/시간 제약 크게 초과

## 응답 형식
승인/반려 결정과 명확한 근거를 제시하세요.
"""
        return prompt

    def generate_user_prompt_for_negotiation(
        self,
        current_issue: str,
        current_data: Dict[str, float],
        prediction: Optional[Dict] = None,
        rag_context: Optional[str] = None
    ) -> str:
        """
        Negotiation Agent용 유저 프롬프트 생성

        Args:
            current_issue: 현재 이슈 설명
            current_data: 현재 공정 데이터
            prediction: ML 예측 결과 (있는 경우)
            rag_context: RAG 검색 결과 (있는 경우)

        Returns:
            유저 프롬프트 문자열
        """
        # 주요 측정 변수들
        stage2_vars = self.schema.stage2.measurement_variables[:5]  # 처음 5개만
        current_values = {var: current_data.get(var, 'N/A') for var in stage2_vars}

        values_str = "\n".join([f"  - {k}: {v}" for k, v in current_values.items()])

        prompt = f"""## 현재 상황
{current_issue}

## 현재 공정 데이터
{values_str}
"""

        if prediction:
            prompt += f"""
## ML 예측 결과
- 예상 {self.schema.target_variable}: {prediction.get('predicted_value', 'N/A')}
- 예상 품질 점수: {prediction.get('quality_score', 'N/A')}
"""

        if rag_context:
            prompt += f"""
## 과거 유사 사례
{rag_context}
"""

        prompt += """
위 정보를 바탕으로 최적의 파라미터 조정안을 제시해주세요.
"""
        return prompt

    def get_parameter_descriptions(self) -> Dict[str, str]:
        """
        파라미터 설명 반환

        Returns:
            파라미터명 → 설명 딕셔너리
        """
        if hasattr(self.schema, 'parameter_guidance'):
            return self.schema.parameter_guidance
        return {}

    def get_quality_spec_description(self) -> str:
        """
        품질 스펙 설명 반환

        Returns:
            품질 스펙 설명 문자열
        """
        return f"""품질 기준:
- 타겟 변수: {self.schema.target_variable}
- LSL (하한): {self.schema.lsl:.4f}
- Target (목표): {self.schema.target:.4f}
- USL (상한): {self.schema.usl:.4f}
"""
