"""
Vector Store Manager

ChromaDB를 사용한 벡터 저장소 관리
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from loguru import logger
from pathlib import Path

from config import settings


class VectorStoreManager:
    """벡터 저장소 관리자"""

    def __init__(
        self,
        collection_name: str = "manufacturing_cases",
        persist_directory: Optional[str] = None
    ):
        """
        Args:
            collection_name: 컬렉션 이름
            persist_directory: 저장 디렉토리
        """
        if persist_directory is None:
            persist_directory = settings.chroma_db_path

        # 디렉토리 생성
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        self.collection_name = collection_name
        self.collection = None

        logger.info(f"VectorStoreManager 초기화 - 저장 경로: {persist_directory}")

    def create_collection(self, reset: bool = False) -> None:
        """
        컬렉션 생성 또는 로드

        Args:
            reset: True면 기존 컬렉션 삭제 후 재생성
        """
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"기존 컬렉션 '{self.collection_name}' 삭제")
            except Exception:
                pass

        # 컬렉션 생성 또는 로드
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Manufacturing process historical cases"}
        )

        logger.info(
            f"컬렉션 '{self.collection_name}' 준비 완료 "
            f"(문서 수: {self.collection.count()})"
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ) -> None:
        """
        문서 추가

        Args:
            documents: 문서 텍스트 리스트
            metadatas: 메타데이터 리스트
            ids: 문서 ID 리스트
        """
        if self.collection is None:
            self.create_collection()

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"{len(documents)}개 문서를 벡터 DB에 추가")

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        유사 문서 검색

        Args:
            query_text: 쿼리 텍스트
            n_results: 반환할 결과 수
            where: 필터 조건 (예: {"outcome": "success"})

        Returns:
            검색 결과
        """
        if self.collection is None:
            self.create_collection()

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )

        logger.info(
            f"쿼리 '{query_text[:50]}...'에 대해 {len(results['documents'][0])}개 결과 반환"
        )

        return results

    def get_collection_stats(self) -> Dict:
        """컬렉션 통계 정보"""
        if self.collection is None:
            return {"error": "Collection not initialized"}

        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "metadata": self.collection.metadata
        }

    def reset_collection(self) -> None:
        """컬렉션 초기화"""
        self.create_collection(reset=True)


# 모듈 테스트용
if __name__ == "__main__":
    logger.info("VectorStoreManager 테스트 시작")

    # 벡터 저장소 초기화
    vs_manager = VectorStoreManager()
    vs_manager.create_collection(reset=True)

    # 테스트 문서 추가
    test_docs = [
        "프레스 공정에서 두께 편차 +0.025mm 발생. 용접 속도 5% 감소, 압력 2% 증가로 조치. 품질 회복 성공.",
        "두께 편차 +0.022mm 발생. 전류만 10% 증가했더니 비드 균열 발생. 실패 사례.",
        "두께 편차 +0.030mm (고강도). 다중 파라미터 조정으로 품질 회복. 성공 사례.",
    ]

    test_metadata = [
        {"case_id": "TEST-001", "outcome": "success", "severity": "medium"},
        {"case_id": "TEST-002", "outcome": "failure", "severity": "medium"},
        {"case_id": "TEST-003", "outcome": "success", "severity": "high"},
    ]

    test_ids = ["test-001", "test-002", "test-003"]

    vs_manager.add_documents(test_docs, test_metadata, test_ids)

    # 검색 테스트
    print("\n" + "=" * 60)
    print("검색 테스트 1: 두께 편차 관련")
    print("=" * 60)

    results = vs_manager.query("두께 편차가 발생했을 때 어떻게 조치해야 하나요?", n_results=2)

    for i, (doc, meta, distance) in enumerate(
        zip(results['documents'][0], results['metadatas'][0], results['distances'][0]),
        1
    ):
        print(f"\n[결과 {i}] (유사도: {1-distance:.3f})")
        print(f"사례 ID: {meta['case_id']}")
        print(f"결과: {meta['outcome']}")
        print(f"내용: {doc}")

    # 성공 사례만 검색
    print("\n" + "=" * 60)
    print("검색 테스트 2: 성공 사례만 필터링")
    print("=" * 60)

    results_success = vs_manager.query(
        "두께 편차 조치 방법",
        n_results=5,
        where={"outcome": "success"}
    )

    for i, (doc, meta) in enumerate(
        zip(results_success['documents'][0], results_success['metadatas'][0]),
        1
    ):
        print(f"\n[성공 사례 {i}]")
        print(f"사례 ID: {meta['case_id']}")
        print(f"내용: {doc[:100]}...")

    # 통계 정보
    print("\n" + "=" * 60)
    print("컬렉션 통계")
    print("=" * 60)
    stats = vs_manager.get_collection_stats()
    print(f"컬렉션 이름: {stats['collection_name']}")
    print(f"총 문서 수: {stats['total_documents']}")

    logger.info("테스트 완료")
