"""
RAG Pipeline Smoke Test
全部检索器 / Reranker / LLM 均 mock，只验证：输入 question → 返回 answer
不依赖真实模型、FAISS 索引或网络。
"""
import pytest
from unittest.mock import MagicMock

from src.rag_pipeline import RAGPipeline


def _make_pipeline(use_graph=True, use_rerank=True, mock_chunks=None):
    """构造一个组件全部被 mock 的 RAGPipeline"""
    if mock_chunks is None:
        mock_chunks = []

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.query_cache = {}

    # query analyzer
    pipeline.query_analyzer = MagicMock()
    pipeline.query_analyzer.analyze.return_value = {
        "original_query": "测试问题",
        "low_level_keywords": ["陈嘉庚"],
        "high_level_keywords": [],
        "query_mode": "hybrid",
        "reason": "mock",
    }

    # hybrid retriever
    pipeline.hybrid_retriever = MagicMock()
    pipeline.hybrid_retriever.search.return_value = mock_chunks

    # graph
    pipeline.use_graph = use_graph
    if use_graph:
        pipeline.graph_retriever = MagicMock()
        pipeline.graph_retriever.search.return_value = {
            "entities": [], "expanded_entities": [], "chunks": []
        }

    # reranker
    pipeline.use_rerank = use_rerank
    if use_rerank:
        pipeline.reranker = MagicMock()
        pipeline.reranker.model = object()  # 非 None 表示模型可用
        pipeline.reranker.rerank.side_effect = lambda q, chunks, top_k=5: chunks[:top_k]

    # generator
    pipeline.generator = MagicMock()
    pipeline.generator.generate.return_value = {
        "answer": "这是mock的最终答案。",
        "references": [{"chunk_id": c["chunk_id"]} for c in mock_chunks[:3]],
    }

    return pipeline


def _chunks():
    return [
        {"chunk_id": "c1", "book": "书", "chapter": "章",
         "text": "陈嘉庚1921年创办厦门大学。", "score": 0.9, "char_count": 20,
         "source": "vector"},
        {"chunk_id": "c2", "book": "书", "chapter": "章",
         "text": "他主张教育救国。", "score": 0.8, "char_count": 12,
         "source": "bm25"},
    ]


class TestPipelineSmoke:
    """端到端冒烟测试"""

    def test_query_returns_answer(self):
        """输入问题 → 返回包含 answer 字段的结果"""
        pipeline = _make_pipeline(mock_chunks=_chunks())
        result = pipeline.query("陈嘉庚创办了什么大学？")

        assert "answer" in result
        assert result["answer"] == "这是mock的最终答案。"
        assert result["question"] == "陈嘉庚创办了什么大学？"

    def test_query_returns_latency(self):
        """结果包含各阶段耗时"""
        pipeline = _make_pipeline(mock_chunks=_chunks())
        result = pipeline.query("问题")

        assert "latency" in result
        assert "total" in result["latency"]

    def test_query_returns_references(self):
        """结果包含引用"""
        pipeline = _make_pipeline(mock_chunks=_chunks())
        result = pipeline.query("问题")
        assert "references" in result

    def test_query_without_graph(self):
        """关闭图谱检索也能正常返回"""
        pipeline = _make_pipeline(use_graph=False, mock_chunks=_chunks())
        result = pipeline.query("问题")
        assert result["answer"] == "这是mock的最终答案。"
        pipeline.hybrid_retriever.search.assert_called_once()

    def test_query_without_rerank(self):
        """关闭 rerank 也能正常返回"""
        pipeline = _make_pipeline(use_rerank=False, mock_chunks=_chunks())
        result = pipeline.query("问题")
        assert result["answer"] == "这是mock的最终答案。"

    def test_query_cache_hit(self):
        """相同问题第二次命中缓存"""
        pipeline = _make_pipeline(mock_chunks=_chunks())
        first = pipeline.query("相同问题")
        second = pipeline.query("相同问题")

        assert second.get("from_cache") is True
        assert first["answer"] == second["answer"]

    def test_fuse_results_no_graph(self):
        """无图谱结果时，融合结果直接返回 hybrid 结果"""
        pipeline = _make_pipeline(mock_chunks=_chunks())
        hybrid = _chunks()
        fused = pipeline._fuse_results(hybrid, [])
        assert len(fused) == 2

    def test_fuse_results_with_graph(self):
        """图谱结果与 hybrid 结果融合（含重叠 chunk 去重）"""
        pipeline = _make_pipeline()
        hybrid = _chunks()
        graph = [
            {"chunk_id": "c2", "book": "书", "chapter": "章",
             "text": "他主张教育救国。", "score": 1.0, "char_count": 12,
             "source": "graph"},
            {"chunk_id": "c3", "book": "书", "chapter": "章",
             "text": "另一个图谱chunk。", "score": 0.5, "char_count": 12,
             "source": "graph"},
        ]
        fused = pipeline._fuse_results(hybrid, graph)
        ids = [r["chunk_id"] for r in fused]
        # c2 同时命中两路，c3 仅图谱，共 3 个去重结果
        assert set(ids) == {"c1", "c2", "c3"}
