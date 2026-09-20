"""
六组消融配置（Ablation Profiles）

前四组为固定工作流（fixed_plan 绕过 Router，控制变量）：
  A. Dense                     仅向量检索
  B. Dense + BM25              向量 + 关键词
  C. + KG                      再加知识图谱实体检索
  D. + Rerank                  再加关系扩展 + Cross-Encoder 精排（最强固定工作流）

后两组为 Agentic：
  E. Agent Router              Router 按 query_mode 动态选路（关闭裁判/重试）
  F. Agent Router + Retry      路由 + 证据裁判 + 有限查询改写重试（完整 Agentic）

所有组都初始化全套检索组件（初始化耗时不计入 query latency），
实际调用哪些组件完全由 plan / 开关决定，保证对比公平。
"""
from src.retrieval_router import RetrievalPlan

# ---- 固定工作流 Plan（显式指定每一路，避免依赖默认值）----
DENSE_PLAN = RetrievalPlan(
    use_dense=True, use_bm25=False, use_graph=False,
    use_relation=False, use_rerank=False, use_evidence_judge=False,
)
DENSE_BM25_PLAN = RetrievalPlan(
    use_dense=True, use_bm25=True, use_graph=False,
    use_relation=False, use_rerank=False, use_evidence_judge=False,
)
PLUS_KG_PLAN = RetrievalPlan(
    use_dense=True, use_bm25=True, use_graph=True,
    use_relation=False, use_rerank=False, use_evidence_judge=False,
)
PLUS_RERANK_PLAN = RetrievalPlan(
    use_dense=True, use_bm25=True, use_graph=True,
    use_relation=True, use_rerank=True, use_evidence_judge=False,
)

# (key, 展示名, 构造 RAGPipeline 的额外 kwargs)
PROFILES = [
    {
        "key": "dense",
        "name": "Dense",
        "kwargs": {"fixed_plan": DENSE_PLAN},
    },
    {
        "key": "dense_bm25",
        "name": "Dense+BM25",
        "kwargs": {"fixed_plan": DENSE_BM25_PLAN},
    },
    {
        "key": "plus_kg",
        "name": "+KG",
        "kwargs": {"fixed_plan": PLUS_KG_PLAN},
    },
    {
        "key": "plus_rerank",
        "name": "+Rerank",
        "kwargs": {"fixed_plan": PLUS_RERANK_PLAN},
    },
    {
        "key": "agent_router",
        "name": "Agent Router",
        "kwargs": {"enable_routing": True, "enable_judge": False, "max_retry": 0},
    },
    {
        "key": "agent_retry",
        "name": "Agent Router + Retry",
        "kwargs": {"enable_routing": True, "enable_judge": True, "max_retry": 1},
    },
]
