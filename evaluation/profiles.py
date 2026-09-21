"""
七组消融配置（Ablation Profiles）—— 严格单变量递进

固定工作流 A~E（fixed_plan 绕过 Router，每次只新增一个主要检索/后处理能力）：

  A. DENSE_ONLY               仅 Dense（最基础 RAG baseline）
  B. DENSE_BM25               + BM25（验证 Sparse 关键词召回增量）
  C. DENSE_BM25_ENTITY_GRAPH  + 实体图谱检索（验证领域实体关系扩展）
  D. DENSE_BM25_ENTITY_RELATION + 关系检索（验证 high-level relation 召回增益）
  E. FULL_FIXED_RAG           + Cross-Encoder Rerank（验证二阶段精排对 Top-K 证据质量）

Agentic F~G（开启 QueryAnalyzer → RetrievalRouter 的动态路由，一次只开一个 Agent 能力）：

  F. AGENT_ROUTER             路由（关闭证据裁判 / 重试），验证动态选路能否在保住效果的
                              同时减少不必要 Retriever 调用、降低延迟
  G. AGENT_RETRY              路由 + 证据裁判 + 有限 Query Rewrite / Retry，
                              验证证据不足时补检对回答质量的提升

关键设计：C→D 只新增 Relation，D→E 只新增 Rerank——Relation Retrieval 与 Rerank
不再在同一步同时引入，以便区分二者各自的独立贡献（严格单变量消融）。

所有组都初始化全套检索组件（初始化耗时不计入 query latency），实际调用哪些组件
完全由 plan / 开关决定，保证对比公平。
"""
from src.retrieval_router import RetrievalPlan

# ---- 固定工作流 Plan（显式指定每一路，避免依赖默认值）----
# A. 最基础：仅 Dense
DENSE_ONLY = RetrievalPlan(
    use_dense=True, use_bm25=False, use_graph=False,
    use_relation=False, use_rerank=False, use_evidence_judge=False,
)
# B. + BM25
DENSE_BM25 = RetrievalPlan(
    use_dense=True, use_bm25=True, use_graph=False,
    use_relation=False, use_rerank=False, use_evidence_judge=False,
)
# C. + 实体图谱检索
DENSE_BM25_ENTITY_GRAPH = RetrievalPlan(
    use_dense=True, use_bm25=True, use_graph=True,
    use_relation=False, use_rerank=False, use_evidence_judge=False,
)
# D. + 关系检索（只新增 relation，不精排）
DENSE_BM25_ENTITY_RELATION = RetrievalPlan(
    use_dense=True, use_bm25=True, use_graph=True,
    use_relation=True, use_rerank=False, use_evidence_judge=False,
)
# E. + Rerank（只在 D 基础上新增精排，最强固定工作流）
FULL_FIXED_RAG = RetrievalPlan(
    use_dense=True, use_bm25=True, use_graph=True,
    use_relation=True, use_rerank=True, use_evidence_judge=False,
)

# (key, 展示名, 构造 RAGPipeline 的额外 kwargs)
PROFILES = [
    {
        "key": "dense",
        "name": "Dense Only",
        "letter": "A",
        "kwargs": {"fixed_plan": DENSE_ONLY},
    },
    {
        "key": "dense_bm25",
        "name": "Dense + BM25",
        "letter": "B",
        "kwargs": {"fixed_plan": DENSE_BM25},
    },
    {
        "key": "entity_graph",
        "name": "+ Entity Graph",
        "letter": "C",
        "kwargs": {"fixed_plan": DENSE_BM25_ENTITY_GRAPH},
    },
    {
        "key": "entity_relation",
        "name": "+ Relation Retrieval",
        "letter": "D",
        "kwargs": {"fixed_plan": DENSE_BM25_ENTITY_RELATION},
    },
    {
        "key": "full_fixed_rag",
        "name": "+ Rerank (Full Fixed RAG)",
        "letter": "E",
        "kwargs": {"fixed_plan": FULL_FIXED_RAG},
    },
    {
        "key": "agent_router",
        "name": "Agent Router",
        "letter": "F",
        # 路由开启；关闭证据裁判即同时关闭重试
        "kwargs": {"enable_routing": True, "enable_judge": False, "max_retry": 0},
    },
    {
        "key": "agent_retry",
        "name": "Agent Router + Retry",
        "letter": "G",
        "kwargs": {"enable_routing": True, "enable_judge": True, "max_retry": 1},
    },
]
