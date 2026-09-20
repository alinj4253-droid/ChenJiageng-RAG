"""
RAG Pipeline Smoke Test + 路由集成测试
检索器 / Reranker / LLM 全部 mock，不依赖真实模型、FAISS 索引或网络。
"""
import pytest
from unittest.mock import MagicMock

from src.rag_pipeline import RAGPipeline
from src.retrieval_router import build_plan
from src.evidence_judge import EvidenceJudgement


def _chunk(cid, text=None):
    return {
        "chunk_id": cid, "book": "书", "chapter": "章",
        "text": text or f"内容{cid}", "score": 0.9,
        "char_count": 10, "source": "vector",
    }


def _make_pipeline(use_graph=True, use_rerank=True, enable_routing=True,
                   dense=None, bm25=None, graph_chunks=None, graph_entities=None):
    """构造组件全部被 mock 的 RAGPipeline"""
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.query_cache = {}
    pipeline.use_graph = use_graph
    pipeline.use_rerank = use_rerank
    pipeline.enable_routing = enable_routing

    pipeline.query_analyzer = MagicMock()
    pipeline.query_analyzer.analyze.return_value = {
        "original_query": "测试问题", "low_level_keywords": ["陈嘉庚"],
        "high_level_keywords": ["教育思想"], "query_mode": "hybrid", "reason": "mock",
    }

    pipeline.hybrid_retriever = MagicMock()
    pipeline.vector_retriever = MagicMock()
    pipeline.bm25_retriever = MagicMock()
    pipeline.vector_retriever.search.return_value = dense if dense is not None else [_chunk("d1"), _chunk("d2")]
    pipeline.bm25_retriever.search.return_value = bm25 if bm25 is not None else [_chunk("b1"), _chunk("b2")]

    if use_graph:
        pipeline.graph_retriever = MagicMock()
        pipeline.graph_retriever.search.return_value = {
            "entities": graph_entities or [],
            "expanded_entities": [],
            "chunks": graph_chunks if graph_chunks is not None else [_chunk("g1")],
        }

    if use_rerank:
        pipeline.reranker = MagicMock()
        pipeline.reranker.model = object()
        pipeline.reranker.rerank.side_effect = lambda q, chunks, top_k=5: chunks[:top_k]

    pipeline.generator = MagicMock()
    pipeline.generator.generate.return_value = {
        "answer": "这是mock的最终答案。",
        "references": [{"chunk_id": "d1"}],
    }

    pipeline.enable_evidence_judge = True
    pipeline.evidence_judge = MagicMock()
    pipeline.evidence_judge.judge.return_value = EvidenceJudgement(
        sufficient=True, missing=[], reason="mock 充分"
    )
    return pipeline


class TestPipelineSmoke:
    """端到端冒烟测试（用足够长的问题触发 LLM 分析）"""

    def test_query_returns_answer(self):
        p = _make_pipeline()
        result = p.query("陈嘉庚创办厦门大学的具体经过是怎样的？")
        assert result["answer"] == "这是mock的最终答案。"
        assert "latency" in result and "total" in result["latency"]
        assert "references" in result
        assert result["query_mode"] == "hybrid"
        assert "retrieval_plan" in result

    def test_query_without_graph(self):
        p = _make_pipeline(use_graph=False)
        result = p.query("陈嘉庚创办厦门大学的具体经过是怎样的？")
        assert result["answer"] == "这是mock的最终答案。"
        p.graph_retriever = None  # 明确无图谱检索器
        p.vector_retriever.search.assert_called()

    def test_query_cache_hit(self):
        p = _make_pipeline()
        q = "陈嘉庚创办厦门大学的具体经过是怎样的？"
        first = p.query(q)
        second = p.query(q)
        assert second.get("from_cache") is True
        assert first["answer"] == second["answer"]


class TestFuseRanked:
    """多路 Weighted RRF 融合"""

    def test_fuse_multiple_routes_dedup(self):
        p = _make_pipeline(
            dense=[_chunk("c1"), _chunk("c2")],
            bm25=[_chunk("c2"), _chunk("c3")],
            graph_chunks=[_chunk("c2"), _chunk("c4")],
        )
        plan = build_plan({"query_mode": "hybrid"})
        ranked, _ = p._retrieve_by_plan("q", {"low_level_keywords": [], "high_level_keywords": []}, plan)
        fused = p._fuse_ranked(ranked, top_k=10)

        ids = [r["chunk_id"] for r in fused]
        assert set(ids) == {"c1", "c2", "c3", "c4"}
        # c2 命中三路，融合分最高
        assert ids[0] == "c2"
        # sources 记录命中的检索路
        c2 = next(r for r in fused if r["chunk_id"] == "c2")
        assert set(c2["sources"]) == {"dense", "bm25", "graph"}

    def test_fuse_empty(self):
        p = _make_pipeline(dense=[], bm25=[], graph_chunks=[])
        plan = build_plan({"query_mode": "hybrid"})
        ranked, _ = p._retrieve_by_plan("q", {}, plan)
        assert p._fuse_ranked(ranked) == []


class TestRoutingIntegration:
    """验证不同 query_mode 真正走不同检索路径"""

    def _analysis(self, mode):
        return {"low_level_keywords": [], "high_level_keywords": [], "query_mode": mode}

    def test_naive_skips_graph(self):
        p = _make_pipeline()
        plan = build_plan({"query_mode": "naive"})
        ranked, entities = p._retrieve_by_plan("短问题", self._analysis("naive"), plan)
        names = [name for _, _, name in ranked]
        assert "dense" in names and "bm25" in names
        assert "graph" not in names
        p.graph_retriever.search.assert_not_called()

    def test_local_calls_graph_not_bm25(self):
        p = _make_pipeline()
        plan = build_plan({"query_mode": "local"})
        ranked, _ = p._retrieve_by_plan("q", self._analysis("local"), plan)
        names = [name for _, _, name in ranked]
        assert "dense" in names and "graph" in names
        assert "bm25" not in names
        p.graph_retriever.search.assert_called_once()
        p.bm25_retriever.search.assert_not_called()

    def test_global_enables_graph(self):
        p = _make_pipeline()
        plan = build_plan({"query_mode": "global"})
        ranked, _ = p._retrieve_by_plan("q", self._analysis("global"), plan)
        names = [name for _, _, name in ranked]
        assert "dense" in names and "graph" in names
        assert "bm25" not in names

    def test_hybrid_enables_all(self):
        p = _make_pipeline()
        plan = build_plan({"query_mode": "hybrid"})
        ranked, _ = p._retrieve_by_plan("q", self._analysis("hybrid"), plan)
        names = [name for _, _, name in ranked]
        assert "dense" in names and "bm25" in names and "graph" in names

    def test_routing_disabled_uses_full_hybrid(self):
        """关闭 Agent 路由 → 无论分析结果如何都走全量检索"""
        p = _make_pipeline(enable_routing=False)
        # 即使分析成 naive，关闭路由后仍走 hybrid 全量
        plan = p._resolve_plan({"query_mode": "naive"})
        assert plan.use_dense and plan.use_bm25 and plan.use_graph

    def test_naive_plan_does_not_rerank(self):
        p = _make_pipeline()
        naive = build_plan({"query_mode": "naive"})
        hybrid = build_plan({"query_mode": "hybrid"})
        fused = [_chunk(f"c{i}") for i in range(8)]

        _, applied_naive = p._rerank_by_plan("q", fused, naive)
        assert applied_naive is False
        p.reranker.rerank.assert_not_called()

        _, applied_hybrid = p._rerank_by_plan("q", fused, hybrid)
        assert applied_hybrid is True
        p.reranker.rerank.assert_called_once()

    def test_short_question_defaults_naive(self):
        """短问题跳过 LLM 分析，默认 naive 模式"""
        p = _make_pipeline()
        analysis = p._analyze("陈嘉庚是谁")  # 5 字
        assert analysis["query_mode"] == "naive"
        p.query_analyzer.analyze.assert_not_called()

    def test_long_question_calls_analyzer(self):
        p = _make_pipeline()
        p._analyze("陈嘉庚创办厦门大学的具体经过和历史背景是什么？")
        p.query_analyzer.analyze.assert_called_once()


class TestEvidenceJudgeIntegration:
    """证据裁判与 pipeline 的集成"""

    def test_complex_question_invokes_judge(self):
        """长问题（hybrid）会调用证据裁判，结果写入返回"""
        p = _make_pipeline()
        result = p.query("陈嘉庚创办厦门大学的具体经过和历史背景是什么？")
        p.evidence_judge.judge.assert_called_once()
        assert result["evidence_sufficient"] is True
        assert result["evidence_judgement"]["sufficient"] is True

    def test_naive_question_skips_judge(self):
        """短问题（naive）不调用证据裁判"""
        p = _make_pipeline()
        result = p.query("陈嘉庚是谁")
        p.evidence_judge.judge.assert_not_called()
        assert result["evidence_judgement"] is None
        assert result["evidence_sufficient"] is None

    def test_judge_disabled_globally(self):
        """全局关闭证据裁判 → 不调用"""
        p = _make_pipeline()
        p.enable_evidence_judge = False
        p.evidence_judge = None
        result = p.query("陈嘉庚创办厦门大学的具体经过和历史背景是什么？")
        assert result["evidence_judgement"] is None

    def test_insufficient_judgement_recorded(self):
        """证据不足时，结果记录 insufficient 与 missing"""
        p = _make_pipeline()
        p.evidence_judge.judge.return_value = EvidenceJudgement(
            sufficient=False, missing=["缺少创办经费来源"], reason="证据不足"
        )
        result = p.query("陈嘉庚创办厦门大学的具体经过和历史背景是什么？")
        assert result["evidence_sufficient"] is False
        assert "缺少创办经费来源" in result["evidence_judgement"]["missing"]
