"""
SmartFlow 시스템 설정
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """시스템 전역 설정"""

    # LLM Configuration
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    llm_provider: Literal["openai", "anthropic"] = Field(default="openai", env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.5, env="LLM_TEMPERATURE")

    # RAG Configuration
    chroma_db_path: str = Field(default="./chroma_db", env="CHROMA_DB_PATH")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )

    # Agent Configuration
    negotiation_max_rounds: int = Field(default=5, env="NEGOTIATION_MAX_ROUNDS")
    quality_threshold: float = Field(default=0.90, env="QUALITY_THRESHOLD")  # 목표 품질 (90%)

    # Quality Specification (업계 표준 고정값)
    welding_strength_lsl: float = Field(default=11.50, env="WELDING_STRENGTH_LSL")  # 하한 규격 (불량 기준)
    welding_strength_usl: float = Field(default=13.20, env="WELDING_STRENGTH_USL")  # 상한 규격 (불량 기준)
    welding_strength_target: float = Field(default=12.60, env="WELDING_STRENGTH_TARGET")  # 목표값 (최적)

    # Anomaly Detection Configuration (MVP 설계 기준)
    # 비용 최적화: 심각한 품질 저하만 감지하여 불필요한 조정 최소화
    # 실제 불량률 1.8% → 이상 탐지율 목표 3-5%
    anomaly_warning_quality: float = Field(default=0.85, env="ANOMALY_WARNING_QUALITY")  # 85% 미만만 이상 감지
    # LSL 근접 경고 (불량 직전 사전 예방)
    lsl_safety_buffer: float = Field(default=0.20, env="LSL_SAFETY_BUFFER")  # LSL + 0.20 미만이면 경고
    usl_safety_buffer: float = Field(default=0.20, env="USL_SAFETY_BUFFER")  # USL - 0.20 초과이면 경고

    # Sensor Simulation Configuration
    press_thickness_mean: float = Field(default=2.0, env="PRESS_THICKNESS_MEAN")
    press_thickness_std: float = Field(default=0.01, env="PRESS_THICKNESS_STD")
    anomaly_probability: float = Field(default=0.15, env="ANOMALY_PROBABILITY")

    # Dashboard Configuration
    dashboard_port: int = Field(default=8501, env="DASHBOARD_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 싱글톤 인스턴스
settings = Settings()
