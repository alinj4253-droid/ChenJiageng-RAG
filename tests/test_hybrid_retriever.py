"""
HybridRetriever / Weighted RRF 融合测试
使用 mock 检索结果，不加载真实 FAISS / BM25 索引。
验证 VECTOR_WEIGHT、BM25_WEIGHT、RRF_K 确实参与打分。
"""
import pytest
from unittest.mock import patch, MagicMock

from src.hybrid_retriever import HybridRetriever
from src.config import VECTOR_WEIGHT, BM25_WEIGHT, RRF_K


def _make_hybrid_with_mocks(vector_results, bm25_results):
    """构造一个内部检索器被 mock 的 HybridRetriever"""
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.vector_retriever = MagicMock()
    retriever.bm25_retriever = MagicMock()
    retriever.vector_retriever.search.return_value = vector_results
    retriever.bm25_retriever.search.return_value = bm25_results
    return retriever


class TestWeightedRRF:
    """验证 Weighted RRF 打分公式"""

    def test_rrf_basic_fusion(self, dense_results, bm25_results):
        """
        Dense: chunk_001 rank1, chunk_002 rank2
        BM25:  chunk_002 rank1, chunk_003 rank2
        融合后 chunk_002 同时命中两路，应排第一
        """
        retriever = _make_hybrid_with_mocks(dense_results, bm25_results)
        results = retriever.search("测试查询", top_k=5)

        ids = [r["chunk_id"] for r in results]
        # chunk_002 在两路都出现，融合分最高
        assert ids[0] == "chunk_002"
        # 所有结果都有 rrf_score
        assert all("rrf_score" in r for r in results)

    def test_weights_participate_in_scoring(self, dense_results, bm25_results):
        """显式验证权重参与计算：chunk_002 的分数 = 两路加权 RRF 之和"""
        retriever = _make_hybrid_with_mocks(dense_results, bm25_results)
        results = retriever.search("q", top_k=5)

        chunk_002 = next(r for r in results if r["chunk_id"] == "chunk_002")
        # dense rank2 (index=1)，bm25 rank1 (index=0)
        expected = (
            VECTOR_WEIGHT / (RRF_K + 2)
            + BM25_WEIGHT / (RRF_K + 1)
        )
        assert chunk_002["rrf_score"] == pytest.approx(expected, rel=1e-6)

    def test_vector_only_chunk_score(self, dense_results, bm25_results):
        """只在向量结果出现的 chunk，分数 = VECTOR_WEIGHT / (RRF_K + rank)"""
        retriever = _make_hybrid_with_mocks(dense_results, bm25_results)
        results = retriever.search("q", top_k=5)

        chunk_001 = next(r for r in results if r["chunk_id"] == "chunk_001")
        # chunk_001 dense rank1
        expected = VECTOR_WEIGHT / (RRF_K + 1)
        assert chunk_001["rrf_score"] == pytest.approx(expected, rel=1e-6)

    def test_bm25_only_chunk_score(self, dense_results, bm25_results):
        """只在 BM25 结果出现的 chunk，分数 = BM25_WEIGHT / (RRF_K + rank)"""
        retriever = _make_hybrid_with_mocks(dense_results, bm25_results)
        results = retriever.search("q", top_k=5)

        chunk_004 = next(r for r in results if r["chunk_id"] == "chunk_004")
        # chunk_004 bm25 rank3 (index=2)
        expected = BM25_WEIGHT / (RRF_K + 3)
        assert chunk_004["rrf_score"] == pytest.approx(expected, rel=1e-6)

    def test_results_sorted_descending(self, dense_results, bm25_results):
        """结果按 rrf_score 降序"""
        retriever = _make_hybrid_with_mocks(dense_results, bm25_results)
        results = retriever.search("q", top_k=5)

        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self):
        """top_k 限制返回数量"""
        dense = [
            {"chunk_id": f"d{i}", "book": "b", "chapter": "c",
             "text": f"t{i}", "score": 0.9 - i * 0.1, "char_count": 10}
            for i in range(10)
        ]
        bm25 = [
            {"chunk_id": f"b{i}", "book": "b", "chapter": "c",
             "text": f"t{i}", "score": 10 - i, "char_count": 10}
            for i in range(10)
        ]
        retriever = _make_hybrid_with_mocks(dense, bm25)
        results = retriever.search("q", top_k=4)
        assert len(results) == 4

    def test_empty_retrieval(self):
        """两路都为空 → 返回空列表"""
        retriever = _make_hybrid_with_mocks([], [])
        results = retriever.search("q", top_k=5)
        assert results == []

    def test_identical_results_merge(self):
        """两路返回完全相同的排名 → 融合分翻倍且去重"""
        shared = [
            {"chunk_id": "x1", "book": "b", "chapter": "c",
             "text": "t", "score": 0.9, "char_count": 10},
        ]
        retriever = _make_hybrid_with_mocks(shared,
                                            [dict(shared[0])])
        results = retriever.search("q", top_k=5)
        assert len(results) == 1
        expected = (VECTOR_WEIGHT + BM25_WEIGHT) / (RRF_K + 1)
        assert results[0]["rrf_score"] == pytest.approx(expected, rel=1e-6)
