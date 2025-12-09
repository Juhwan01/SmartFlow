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
    llm_model: str = Field(default="gpt-4o", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")

    # RAG Configuration
    chroma_db_path: str = Field(default="./chroma_db", env="CHROMA_DB_PATH")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )

    # Agent Configuration
    negotiation_max_rounds: int = Field(default=5, env="NEGOTIATION_MAX_ROUNDS")
    quality_threshold: float = Field(default=0.90, env="QUALITY_THRESHOLD")

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
