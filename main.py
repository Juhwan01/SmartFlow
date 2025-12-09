"""
SmartFlow 메인 실행 파일

스마트 제조 AI Agent 해커톤 2025
팀: 노동조합
프로젝트: SmartFlow - LLM 기반 Multi-Agent 협상 시스템
"""
import asyncio
from loguru import logger
from config import settings


def setup_logging():
    """로깅 설정"""
    logger.add(
        "logs/smartflow_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )


async def main():
    """메인 실행 함수"""
    setup_logging()

    logger.info("=" * 60)
    logger.info("SmartFlow 시스템 시작")
    logger.info("=" * 60)
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Quality Threshold: {settings.quality_threshold}")
    logger.info("=" * 60)

    # 워크플로우 실행
    from src.workflow.langgraph_workflow import SmartFlowWorkflow

    workflow = SmartFlowWorkflow()
    result = workflow.run()

    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("실행 결과 요약")
    logger.info("=" * 60)
    logger.info(f"프레스 두께: {result['press_data']['thickness']:.4f}mm")
    logger.info(f"예측 품질: {result['prediction']['predicted_quality_score']:.2%}")
    logger.info(f"최종 결정: {result['decision']['status']}")
    if result['execution_result'].get('executed'):
        logger.info(f"최종 품질: {result['execution_result']['final_quality_score']:.2%}")
        logger.info(f"품질 기준 충족: {result['execution_result']['meets_threshold']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("사용자에 의해 시스템 종료")
    except Exception as e:
        logger.exception(f"시스템 오류 발생: {e}")
