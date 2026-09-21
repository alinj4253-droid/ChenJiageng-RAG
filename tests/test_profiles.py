"""
消融配置（evaluation/profiles.py）单变量递进测试

不跑真实 benchmark，只验证：每个 profile 相对前一组恰好只新增一个主要能力，
保证消融实验可区分各模块的独立贡献。
"""
from evaluation.profiles import (
    PROFILES,
    DENSE_ONLY,
    DENSE_BM25,
    DENSE_BM25_ENTITY_GRAPH,
    DENSE_BM25_ENTITY_RELATION,
    FULL_FIXED_RAG,
)


def _kwargs(key):
    return next(p["kwargs"] for p in PROFILES if p["key"] == key)


class TestSevenProfiles:

    def test_exactly_seven_profiles(self):
        assert len(PROFILES) == 7
        keys = [p["key"] for p in PROFILES]
        assert keys == [
            "dense", "dense_bm25", "entity_graph",
            "entity_relation", "full_fixed_rag",
            "agent_router", "agent_retry",
        ]

    def test_letters_a_to_g(self):
        assert [p["letter"] for p in PROFILES] == list("ABCDEFG")


class TestSingleVariableProgression:
    """固定工作流 A~E：相邻两组只允许一个布尔开关变化"""

    def test_a_to_b_only_bm25_added(self):
        assert DENSE_ONLY.use_dense is True
        assert DENSE_ONLY.use_bm25 is False
        assert DENSE_BM25.use_bm25 is True
        # 其余能力保持关闭
        assert DENSE_BM25.use_graph is False
        assert DENSE_BM25.use_relation is False
        assert DENSE_BM25.use_rerank is False

    def test_b_to_c_only_entity_graph_added(self):
        assert DENSE_BM25.use_graph is False
        assert DENSE_BM25_ENTITY_GRAPH.use_graph is True
        assert DENSE_BM25_ENTITY_GRAPH.use_relation is False
        assert DENSE_BM25_ENTITY_GRAPH.use_rerank is False

    def test_c_to_d_only_relation_added(self):
        assert DENSE_BM25_ENTITY_GRAPH.use_relation is False
        assert DENSE_BM25_ENTITY_RELATION.use_relation is True
        # 关键：relation 与 rerank 拆成两步，D 不引入 rerank
        assert DENSE_BM25_ENTITY_RELATION.use_rerank is False

    def test_d_to_e_only_rerank_added(self):
        assert DENSE_BM25_ENTITY_RELATION.use_rerank is False
        assert FULL_FIXED_RAG.use_rerank is True
        # Relation 在 E 保持开启，不回退
        assert FULL_FIXED_RAG.use_relation is True

    def test_relation_and_rerank_are_isolated(self):
        """Relation 与 Rerank 必须分属不同 profile，不能同时首次引入"""
        assert DENSE_BM25_ENTITY_RELATION.use_relation is True
        assert DENSE_BM25_ENTITY_RELATION.use_rerank is False
        assert FULL_FIXED_RAG.use_rerank is True


class TestAgentProfiles:
    """F / G：路由都开，区别只在证据裁判 + 重试"""

    def test_agent_router_routing_on_retry_off(self):
        kw = _kwargs("agent_router")
        assert kw["enable_routing"] is True
        assert kw["enable_judge"] is False
        assert kw["max_retry"] == 0

    def test_agent_retry_routing_on_retry_on(self):
        kw = _kwargs("agent_retry")
        assert kw["enable_routing"] is True
        assert kw["enable_judge"] is True
        assert kw["max_retry"] == 1

    def test_agent_profiles_do_not_pin_fixed_plan(self):
        """Agent 组走 Router，不能用 fixed_plan 绕过动态路由"""
        for key in ("agent_router", "agent_retry"):
            assert "fixed_plan" not in _kwargs(key)
