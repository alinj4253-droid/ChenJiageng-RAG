"""
Retrieval Router 测试
只验证：给定明确 query_mode 时，RetrievalPlan 的调用路径是否正确。
不验证 LLM 是否一定判断对模式（那是 QueryAnalyzer 的职责）。
"""
import pytest

from src.retrieval_router import build_plan, RetrievalPlan


class TestRoutingPolicy:

    def test_naive_disables_graph_and_rerank(self):
        """naive：稠密+BM25，不调图谱、不精排"""
        plan = build_plan({"query_mode": "naive"})
        assert plan.use_dense is True
        assert plan.use_bm25 is True
        assert plan.use_graph is False
        assert plan.use_relation is False
        assert plan.use_rerank is False
        assert plan.enabled_retrievers() == ["dense", "bm25"]

    def test_local_enables_entity_graph(self):
        """local：稠密 + 实体图谱 + 精排，不调 BM25 / 关系检索"""
        plan = build_plan({"query_mode": "local"})
        assert plan.use_dense is True
        assert plan.use_bm25 is False
        assert plan.use_graph is True
        assert plan.use_relation is False
        assert plan.use_rerank is True
        assert "graph" in plan.enabled_retrievers()

    def test_global_enables_relation_retrieval(self):
        """global：启用关系检索 + 图谱 + 精排"""
        plan = build_plan({"query_mode": "global"})
        assert plan.use_dense is True
        assert plan.use_graph is True
        assert plan.use_relation is True
        assert plan.use_rerank is True
        assert "relation" in plan.enabled_retrievers()

    def test_hybrid_enables_all_retrievers(self):
        """hybrid：四路检索全开 + 精排"""
        plan = build_plan({"query_mode": "hybrid"})
        assert plan.use_dense is True
        assert plan.use_bm25 is True
        assert plan.use_graph is True
        assert plan.use_relation is True
        assert plan.use_rerank is True
        assert set(plan.enabled_retrievers()) == {"dense", "bm25", "graph", "relation"}

    def test_unknown_mode_falls_back_to_hybrid(self):
        """未知 / 缺失 query_mode → 兜底全量检索（hybrid）"""
        plan = build_plan({"query_mode": "something_new"})
        assert plan == build_plan({"query_mode": "hybrid"})

        plan_empty = build_plan({})
        assert plan_empty.use_dense is True
        assert plan_empty.use_graph is True

        plan_none = build_plan(None)
        assert plan_none.use_dense is True

    def test_plan_is_immutable(self):
        """RetrievalPlan 冻结，防止流程中被意外修改"""
        plan = build_plan({"query_mode": "naive"})
        with pytest.raises(Exception):
            plan.use_graph = True

    def test_to_dict_serializable(self):
        """plan 可序列化为 dict（用于结构化日志）"""
        plan = build_plan({"query_mode": "local"})
        d = plan.to_dict()
        assert d["use_graph"] is True
        assert d["use_rerank"] is True
