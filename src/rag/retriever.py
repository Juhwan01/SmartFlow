"""
RAG Retriever

과거 사례를 검색하고 LLM 컨텍스트를 구성하는 모듈
"""
from typing import List, Dict, Optional
import json
from loguru import logger

from src.rag.vectorstore import VectorStoreManager
from src.data.sample_cases import get_historical_cases, get_cases_as_text


class RAGRetriever:
    """RAG 검색 시스템"""

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        """
        Args:
            vector_store: 벡터 저장소 (None이면 새로 생성)
        """
        self.vector_store = vector_store or VectorStoreManager()
        self.initialized = False

        logger.info("RAGRetriever 초기화")

    def initialize(self, reset: bool = False) -> None:
        """
        RAG 시스템 초기화 (과거 사례 데이터 로드)

        Args:
            reset: True면 기존 데이터 삭제 후 재구축
        """
        logger.info("RAG 시스템 초기화 시작...")

        # 컬렉션 생성
        self.vector_store.create_collection(reset=reset)

        # 이미 데이터가 있고 reset하지 않으면 스킵
        if not reset and self.vector_store.collection.count() > 0:
            logger.info(
                f"기존 데이터 사용 (문서 수: {self.vector_store.collection.count()})"
            )
            self.initialized = True
            return

        # 과거 사례 로드
        cases = get_historical_cases()
        case_texts = get_cases_as_text()

        # 메타데이터 생성
        metadatas = []
        for case in cases:
            metadatas.append({
                "case_id": case.case_id,
                "date": case.date,
                "process_stage": case.process_stage,
                "issue_severity": case.issue_severity,
                "outcome": case.outcome,
                "quality_before": case.quality_before,
                "quality_after": case.quality_after,
            })

        # ID 생성
        ids = [case.case_id for case in cases]

        # 벡터 DB에 추가
        self.vector_store.add_documents(
            documents=case_texts,
            metadatas=metadatas,
            ids=ids
        )

        self.initialized = True
        logger.info(f"RAG 초기화 완료 - {len(cases)}개 사례 로드")

    def retrieve_similar_cases(
        self,
        query: str,
        n_results: int = 3,
        outcome_filter: Optional[str] = None,
        severity_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        유사 사례 검색

        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            outcome_filter: "success" 또는 "failure"로 필터링
            severity_filter: "low", "medium", "high"로 필터링

        Returns:
            검색된 사례 리스트
        """
        if not self.initialized:
            self.initialize()

        # 필터 조건 구성
        where = {}
        if outcome_filter:
            where["outcome"] = outcome_filter
        if severity_filter:
            where["issue_severity"] = severity_filter

        # 검색
        results = self.vector_store.query(
            query_text=query,
            n_results=n_results,
            where=where if where else None
        )

        # 결과 포맷팅
        formatted_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            formatted_results.append({
                "content": doc,
                "metadata": metadata,
                "similarity_score": 1 - distance,  # 거리를 유사도로 변환
            })

        return formatted_results

    def retrieve_success_cases(
        self,
        query: str,
        n_results: int = 3
    ) -> List[Dict]:
        """성공 사례만 검색"""
        return self.retrieve_similar_cases(
            query=query,
            n_results=n_results,
            outcome_filter="success"
        )

    def retrieve_failure_cases(
        self,
        query: str,
        n_results: int = 3
    ) -> List[Dict]:
        """실패 사례만 검색"""
        return self.retrieve_similar_cases(
            query=query,
            n_results=n_results,
            outcome_filter="failure"
        )

    def build_context_for_llm(
        self,
        current_situation: str,
        n_success: int = 2,
        n_failure: int = 1
    ) -> str:
        """
        LLM에 제공할 컨텍스트 구성

        Args:
            current_situation: 현재 상황 설명
            n_success: 포함할 성공 사례 수
            n_failure: 포함할 실패 사례 수

        Returns:
            포맷된 컨텍스트 문자열
        """
        # 성공 사례 검색
        success_cases = self.retrieve_success_cases(
            current_situation,
            n_results=n_success
        )

        # 실패 사례 검색
        failure_cases = self.retrieve_failure_cases(
            current_situation,
            n_results=n_failure
        )

        # 컨텍스트 구성
        context = f"""## 현재 상황
{current_situation}

## 과거 유사 사례 분석

### 성공 사례 (참고할 만한 조치)
"""

        for i, case in enumerate(success_cases, 1):
            context += f"\n#### 성공 사례 {i} (유사도: {case['similarity_score']:.2%})\n"
            context += f"{case['content']}\n"

        context += "\n### 실패 사례 (피해야 할 조치)\n"

        for i, case in enumerate(failure_cases, 1):
            context += f"\n#### 실패 사례 {i} (유사도: {case['similarity_score']:.2%})\n"
            context += f"{case['content']}\n"

        context += "\n## 권장 사항\n"
        context += "위 사례들을 참고하여, 성공 사례의 접근법을 따르되 실패 사례에서 드러난 위험 요소는 피하십시오.\n"

        return context

    def get_stats(self) -> Dict:
        """RAG 시스템 통계"""
        if not self.initialized:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "vector_store_stats": self.vector_store.get_collection_stats()
        }


# 모듈 테스트용
if __name__ == "__main__":
    logger.info("RAGRetriever 테스트 시작")

    retriever = RAGRetriever()
    retriever.initialize(reset=True)

    print("\n" + "=" * 70)
    print("테스트 1: 두께 편차 관련 성공 사례 검색")
    print("=" * 70)

    success_cases = retriever.retrieve_success_cases(
        "프레스 공정에서 두께 편차 0.025mm가 발생했습니다.",
        n_results=3
    )

    for i, case in enumerate(success_cases, 1):
        print(f"\n[성공 사례 {i}] 유사도: {case['similarity_score']:.2%}")
        print(f"사례 ID: {case['metadata']['case_id']}")
        print(f"조치 전 품질: {case['metadata']['quality_before']:.2%}")
        print(f"조치 후 품질: {case['metadata']['quality_after']:.2%}")
        print(f"내용:\n{case['content'][:200]}...")

    print("\n" + "=" * 70)
    print("테스트 2: 실패 사례 검색 (피해야 할 조치)")
    print("=" * 70)

    failure_cases = retriever.retrieve_failure_cases(
        "두께 편차가 발생했을 때 조치 방법",
        n_results=2
    )

    for i, case in enumerate(failure_cases, 1):
        print(f"\n[실패 사례 {i}] 유사도: {case['similarity_score']:.2%}")
        print(f"사례 ID: {case['metadata']['case_id']}")
        print(f"내용:\n{case['content'][:200]}...")

    print("\n" + "=" * 70)
    print("테스트 3: LLM 컨텍스트 구성")
    print("=" * 70)

    current_situation = """
프레스 공정에서 두께 편차 +0.028mm가 감지되었습니다.
이는 기준값 2.0mm에서 벗어난 값으로, 용접 공정에서 품질 저하가 예상됩니다.
현재 예측 품질 점수는 0.87로, 목표 0.90에 미달합니다.
    """.strip()

    context = retriever.build_context_for_llm(
        current_situation,
        n_success=2,
        n_failure=1
    )

    print(context)

    print("\n" + "=" * 70)
    print("RAG 시스템 통계")
    print("=" * 70)
    stats = retriever.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    logger.info("테스트 완료")
