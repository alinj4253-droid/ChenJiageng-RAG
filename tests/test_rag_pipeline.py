"""
RAG Pipeline Smoke Test + 路由集成测试
检索器 / Reranker / LLM 全部 mock，不依赖真实模型、FAISS 索引或网络。
"""
import pytest
from unittest.mock import MagicMock

from src.rag_pipeline import RAGPipeline
from src.retrieval_router import build_plan
from src.evidence_judge import EvidenceJudgement
from src.query_rewriter import RewrittenQuery


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
    pipeline.fixed_plan = None
    # trace 只组装不落盘，避免测试产生日志文件
    from src.trace_logger import StructuredTraceLogger
    pipeline.tracer = StructuredTraceLogger(enabled=False)

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
        # 关系检索默认无结果，避免 MagicMock 被当作可迭代结果
        pipeline.graph_retriever.relation_chunks.return_value = []

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

    # 默认改写器返回 None（不因 mock 的 insufficient 意外触发重试）
    pipeline.max_retry = 1
    pipeline.query_rewriter = MagicMock()
    pipeline.query_rewriter.rewrite.return_value = None

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
        # trace 已组装并随结果返回
        trace = result["trace"]
        assert trace["query_mode"] == "hybrid"
        assert "retrieval_plan" in trace and "latency" in trace
        assert trace["reranked_docs"] == len(result["retrieved_chunks"])
        assert "retrieved_docs" in trace and "retry_count" in trace

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

    def test_query_writes_structured_trace(self, tmp_path):
        import json as _json
        from src.trace_logger import StructuredTraceLogger
        p = _make_pipeline()
        log_file = tmp_path / "rag_trace.jsonl"
        p.tracer = StructuredTraceLogger(log_file=log_file, enabled=True)
        p.query("陈嘉庚创办厦门大学的具体经过是怎样的？")

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        trace = _json.loads(lines[0])
        assert trace["query"].startswith("陈嘉庚")
        assert trace["query_mode"] == "hybrid"
        assert "retrieval_plan" in trace and "latency" in trace
        assert trace["retry_count"] == 0


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

    def test_global_calls_relation_retrieval_with_high_level_keywords(self):
        p = _make_pipeline()
        plan = build_plan({"query_mode": "global"})
        p.graph_retriever.relation_chunks.return_value = [_chunk("r1")]
        analysis = {"low_level_keywords": [],
                    "high_level_keywords": ["教育", "救国"],
                    "query_mode": "global"}
        ranked, _ = p._retrieve_by_plan("陈嘉庚教育救国思想", analysis, plan)
        names = [name for _, _, name in ranked]
        assert "graph_relation" in names
        # 关系检索查询由 high-level 关键词拼接而成
        rel_query = p.graph_retriever.relation_chunks.call_args[0][0]
        assert "教育" in rel_query and "救国" in rel_query

    def test_local_does_not_call_relation_retrieval(self):
        p = _make_pipeline()
        plan = build_plan({"query_mode": "local"})
        p._retrieve_by_plan("q", self._analysis("local"), plan)
        p.graph_retriever.relation_chunks.assert_not_called()

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

    def test_short_question_still_calls_analyzer(self):
        """短问题不再按长度 shortcut，统一交给 QueryAnalyzer"""
        p = _make_pipeline()
        # 5 字短问题：长度不再等价于 naive，必须真正调用 LLM 分析
        analysis = p._analyze("陈嘉庚出生哪年？")
        p.query_analyzer.analyze.assert_called_once()
        # mock analyzer 默认返回 hybrid（短问题长度不再决定 query_mode）
        assert analysis["query_mode"] == "hybrid"

    @pytest.mark.parametrize("short_query", [
        "陈嘉庚出生哪年？",          # 短事实问题
        "陈嘉庚和李光前什么关系？",    # 短但属实体关系查询
        "教育救国思想是什么？",       # 短但属 global/semantic
    ])
    def test_short_queries_all_go_through_analyzer(self, short_query):
        """三类短 query 都必须经过 QueryAnalyzer，不被长度截断"""
        p = _make_pipeline()
        p._analyze(short_query)
        p.query_analyzer.analyze.assert_called_once()

    def test_empty_question_rejected(self):
        """空 / 全空格输入属于非法输入校验，不进入检索"""
        p = _make_pipeline()
        with pytest.raises(ValueError):
            p._analyze("   ")
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
        """naive 模式（由 QueryAnalyzer 判定，而非问题长度）不调用证据裁判"""
        p = _make_pipeline()
        # 显式让 analyzer 判定为 naive：路由后 use_evidence_judge=False
        p.query_analyzer.analyze.return_value = {
            "original_query": "陈嘉庚是谁", "low_level_keywords": ["陈嘉庚"],
            "high_level_keywords": [], "query_mode": "naive", "reason": "mock naive",
        }
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


class TestRetrievalCallCounting:
    """Task3：区分 retrieval_rounds（轮次）与 retrieval_calls（真实 Retriever 调用）"""

    def _counters(self):
        return {"dense": 0, "bm25": 0, "entity_graph": 0, "relation_graph": 0}

    def test_naive_plan_calls_two_retrievers(self):
        """Case1：naive = dense + bm25 → 1 轮 / 2 次调用"""
        p = _make_pipeline()
        counters = self._counters()
        plan = build_plan({"query_mode": "naive"})
        p._retrieve_by_plan("q", {"high_level_keywords": []}, plan, counters)
        assert counters == {"dense": 1, "bm25": 1, "entity_graph": 0,
                            "relation_graph": 0}

    def test_hybrid_plan_calls_four_retrievers(self):
        """Case2：hybrid = dense+bm25+entity_graph+relation → 1 轮 / 4 次调用"""
        p = _make_pipeline()
        counters = self._counters()
        plan = build_plan({"query_mode": "hybrid"})
        p._retrieve_by_plan("q", {"high_level_keywords": ["教育"]}, plan, counters)
        assert sum(counters.values()) == 4
        assert counters["entity_graph"] == 1
        assert counters["relation_graph"] == 1

    def test_relation_call_counted_even_when_empty(self):
        """relation_chunks 即使召回为空也算一次真实调用"""
        p = _make_pipeline()
        p.graph_retriever.relation_chunks.return_value = []
        counters = self._counters()
        plan = build_plan({"query_mode": "hybrid"})
        p._retrieve_by_plan("q", {"high_level_keywords": []}, plan, counters)
        assert counters["relation_graph"] == 1

    def test_retry_accumulates_calls_across_rounds(self):
        """Case3：hybrid 每轮 4 次，重试 1 次 → 2 轮 / 8 次调用"""
        p = _make_pipeline()
        p.max_retry = 1
        p.evidence_judge.judge.side_effect = [
            EvidenceJudgement(sufficient=False, missing=["缺经费"]),
            EvidenceJudgement(sufficient=True, missing=[]),
        ]
        p.query_rewriter.rewrite.return_value = RewrittenQuery(query="改写后")
        # hybrid plan → 每轮 dense+bm25+entity_graph+relation = 4
        p.fixed_plan = build_plan({"query_mode": "hybrid"})

        result = p.query("陈嘉庚创办厦门大学的经过和背景？")
        assert result["retrieval_rounds"] == 2
        assert result["retrieval_calls"] == 8
        assert result["retrieval_call_detail"]["dense"] == 2

    def test_router_reduces_calls_naive_vs_hybrid(self):
        """Case4：router 让 naive 走 2 次调用，hybrid 走 4 次调用"""
        p = _make_pipeline()
        na = self._counters()
        hy = self._counters()
        p._retrieve_by_plan("q", {"high_level_keywords": []},
                            build_plan({"query_mode": "naive"}), na)
        p._retrieve_by_plan("q", {"high_level_keywords": []},
                            build_plan({"query_mode": "hybrid"}), hy)
        assert sum(na.values()) == 2
        assert sum(hy.values()) == 4

    def test_trace_contains_calls_and_detail(self):
        """trace 同时记录 rounds / calls / call_detail"""
        p = _make_pipeline()
        result = p.query("陈嘉庚创办厦门大学的经过？")
        trace = result["trace"]
        assert trace["retrieval_rounds"] == result["retrieval_rounds"]
        assert trace["retrieval_calls"] == result["retrieval_calls"]
        assert "dense" in trace["retrieval_call_detail"]
