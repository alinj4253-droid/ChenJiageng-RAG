"""
RAG主Pipeline

查询理解(QueryAnalyzer)
  → 检索路由(RetrievalRouter, query_mode → RetrievalPlan)
  → 按 Plan 执行 Dense / BM25 / Graph 多路检索
  → Weighted RRF 融合
  → Cross-Encoder Rerank（按 Plan）
  → 生成答案

Agentic 决策只出现在 query_mode（LLM 判断）这一处，
检索路径本身是确定性规则映射，保证可控、可测、便于消融。
"""
import time
from typing import Dict, List, Optional, Tuple

from src.query_analyzer import QueryAnalyzer
from src.hybrid_retriever import HybridRetriever
from src.graph_retriever import GraphRetriever
from src.reranker import Reranker
from src.rag_generator import RAGGenerator
from src.retrieval_router import build_plan, RetrievalPlan
from src.evidence_judge import EvidenceJudge, EvidenceJudgement
from src.config import (
    FINAL_TOP_K,
    RERANK_TOP_N,
    RRF_K,
    VECTOR_TOP_K,
    BM25_TOP_K,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    KG_WEIGHT,
    ENABLE_AGENT_ROUTING,
    ENABLE_EVIDENCE_JUDGE,
)

# 融合候选数：送入 rerank 前保留的候选规模
FUSION_CANDIDATES = RERANK_TOP_N


class RAGPipeline:
    """完整的RAG问答流水线"""

    def __init__(self, use_graph: bool = True, use_rerank: bool = True,
                 enable_routing: Optional[bool] = None):
        print('初始化RAG Pipeline...')
        self.query_analyzer = QueryAnalyzer()
        self.hybrid_retriever = HybridRetriever()
        # 复用 hybrid 内部的基础检索器，使 Router 能按 Plan 单独调用 dense / bm25
        self.vector_retriever = self.hybrid_retriever.vector_retriever
        self.bm25_retriever = self.hybrid_retriever.bm25_retriever

        self.use_graph = use_graph
        if use_graph:
            self.graph_retriever = GraphRetriever()

        self.use_rerank = use_rerank
        if use_rerank:
            self.reranker = Reranker()

        self.generator = RAGGenerator()

        # 证据裁判（Agentic 决策点之二）；可全局关闭用于消融
        self.enable_evidence_judge = ENABLE_EVIDENCE_JUDGE
        self.evidence_judge = EvidenceJudge() if ENABLE_EVIDENCE_JUDGE else None

        # Agent 路由开关：显式参数优先，否则读全局配置（便于消融 Agent On/Off）
        self.enable_routing = (
            ENABLE_AGENT_ROUTING if enable_routing is None else enable_routing
        )

        # 简单查询缓存：相同问题直接返回结果
        self.query_cache = {}
        print('RAG Pipeline 初始化完成')

    # ------------------------------------------------------------------
    # 查询分析
    # ------------------------------------------------------------------
    def _analyze(self, question: str) -> Dict:
        """查询分析；很短的事实问题跳过 LLM，直接判为 naive 以降低延迟"""
        if len(question) < 15:
            return {
                'original_query': question,
                'low_level_keywords': [],
                'high_level_keywords': [],
                'query_mode': 'naive',
                'reason': '短问题跳过LLM分析，按简单事实查询处理',
            }
        return self.query_analyzer.analyze(question)

    def _resolve_plan(self, analysis: Dict) -> RetrievalPlan:
        """根据分析结果决定检索计划；关闭路由时统一走全量 hybrid"""
        if self.enable_routing:
            return build_plan(analysis)
        return build_plan({'query_mode': 'hybrid'})

    # ------------------------------------------------------------------
    # 按 Plan 检索 + 融合
    # ------------------------------------------------------------------
    def _retrieve_by_plan(
        self, query: str, analysis: Dict, plan: RetrievalPlan
    ) -> Tuple[List[Tuple[List[Dict], float, str]], List[Dict]]:
        """
        按 RetrievalPlan 执行多路检索。

        Returns:
            ranked_lists: [(results, weight, name), ...] 供 Weighted RRF 融合
            graph_entities: 图谱命中的实体（用于状态展示/可观测）
        """
        ranked_lists: List[Tuple[List[Dict], float, str]] = []
        graph_entities: List[Dict] = []

        if plan.use_dense:
            dense = self.vector_retriever.search(query, top_k=VECTOR_TOP_K)
            ranked_lists.append((dense, VECTOR_WEIGHT, 'dense'))

        if plan.use_bm25:
            bm25 = self.bm25_retriever.search(query, top_k=BM25_TOP_K)
            ranked_lists.append((bm25, BM25_WEIGHT, 'bm25'))

        if plan.use_graph and self.use_graph:
            # 实体图谱检索；关系级检索（use_relation）在 Phase 6 接入，
            # 当前图谱子图扩展本身已沿关系边遍历，可覆盖部分关系需求。
            graph_result = self.graph_retriever.search(query)
            graph_chunks = graph_result.get('chunks', [])
            graph_entities = graph_result.get('entities', [])
            ranked_lists.append((graph_chunks, KG_WEIGHT, 'graph'))

        return ranked_lists, graph_entities

    @staticmethod
    def _fuse_ranked(
        ranked_lists: List[Tuple[List[Dict], float, str]],
        top_k: int = FUSION_CANDIDATES,
    ) -> List[Dict]:
        """
        Weighted RRF 融合任意多路检索结果。
        每路贡献 weight / (RRF_K + rank + 1)，相同 chunk_id 累加并去重。
        """
        rrf_scores: Dict[str, list] = {}

        for results, weight, name in ranked_lists:
            for rank, r in enumerate(results):
                chunk_id = r['chunk_id']
                score = weight / (RRF_K + rank + 1)
                if chunk_id in rrf_scores:
                    rrf_scores[chunk_id][0] += score
                    rrf_scores[chunk_id][1].setdefault('sources', []).append(name)
                else:
                    merged = dict(r)
                    merged['sources'] = [name]
                    merged[f'{name}_score'] = r.get('score', 0)
                    rrf_scores[chunk_id] = [score, merged]

        ordered = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        fused = []
        for score, r in ordered[:top_k]:
            r['fused_score'] = score
            fused.append(r)
        return fused

    def _rerank_by_plan(self, question: str, fused: List[Dict],
                        plan: RetrievalPlan) -> Tuple[List[Dict], bool]:
        """按 Plan 决定是否精排，返回 (chunks, rerank_applied)"""
        if plan.use_rerank and self.use_rerank and self.reranker.model is not None:
            return self.reranker.rerank(question, fused, top_k=FINAL_TOP_K), True
        return fused[:FINAL_TOP_K], False

    def _retrieve_fuse_rerank(self, query: str, analysis: Dict,
                              plan: RetrievalPlan,
                              latency: Optional[Dict] = None
                              ) -> Tuple[List[Dict], List[Dict]]:
        """对给定 query 执行：按 Plan 检索 → Weighted RRF 融合 → 精排。
        Retry 时可对改写后的 query 重复调用。"""
        latency = latency if latency is not None else {}

        t2 = time.time()
        ranked_lists, graph_entities = self._retrieve_by_plan(query, analysis, plan)
        latency['retrieval'] = time.time() - t2

        t3 = time.time()
        fused = self._fuse_ranked(ranked_lists, FUSION_CANDIDATES)
        latency['fusion'] = time.time() - t3

        t4 = time.time()
        reranked, rerank_applied = self._rerank_by_plan(query, fused, plan)
        latency['rerank'] = time.time() - t4 if rerank_applied else 0.0

        return reranked, graph_entities

    def _judge_evidence(self, question: str, contexts: List[Dict],
                        plan: RetrievalPlan) -> Optional[EvidenceJudgement]:
        """按 Plan / 全局开关决定是否做证据裁判；naive 模式不裁判"""
        if self.evidence_judge is None or not plan.use_evidence_judge:
            return None
        return self.evidence_judge.judge(question, contexts)

    def _prepare_context(self, question: str) -> Tuple[List[Dict], Dict, RetrievalPlan, List[Dict], Dict]:
        """查询分析 → 路由 → 检索 → 融合 → 精排（非流式路径共用）"""
        latency: Dict[str, float] = {}

        t1 = time.time()
        analysis = self._analyze(question)
        plan = self._resolve_plan(analysis)
        latency['query_analysis'] = time.time() - t1

        reranked, graph_entities = self._retrieve_fuse_rerank(
            question, analysis, plan, latency
        )

        return reranked, analysis, plan, graph_entities, latency

    # ------------------------------------------------------------------
    # 非流式查询
    # ------------------------------------------------------------------
    def query(self, question: str, verbose: bool = False) -> Dict:
        """
        执行完整的RAG问答。

        返回: {
            "question", "query_analysis", "query_mode", "retrieval_plan",
            "retrieved_chunks", "answer", "references", "latency"
        }
        """
        cache_key = question.strip()
        if cache_key in self.query_cache:
            cached = self.query_cache[cache_key].copy()
            cached["from_cache"] = True
            return cached

        t0 = time.time()
        reranked, analysis, plan, graph_entities, latency = self._prepare_context(question)

        if verbose:
            print(f'[查询分析] 模式: {analysis.get("query_mode")}')
            print(f'[路由] 启用检索器: {plan.enabled_retrievers()} | 精排: {plan.use_rerank}')
            print(f'[图谱检索] 匹配实体: {[e["name"] for e in graph_entities]}')
            print(f'[融合+精排] 最终上下文 {len(reranked)} 个chunk')

        # 证据裁判（Phase 4 将扩展为有限 Query Rewrite + Retry 循环）
        t5 = time.time()
        judgement = self._judge_evidence(question, reranked, plan)
        latency['evidence_judge'] = time.time() - t5 if judgement else 0.0
        if verbose and judgement:
            print(f'[证据裁判] sufficient={judgement.sufficient} missing={judgement.missing}')

        t6 = time.time()
        gen_result = self.generator.generate(question, reranked, analysis)
        latency['generation'] = time.time() - t6
        latency['total'] = time.time() - t0

        result = {
            'question': question,
            'query_analysis': analysis,
            'query_mode': analysis.get('query_mode'),
            'retrieval_plan': plan.to_dict(),
            'retrieved_chunks': reranked,
            'evidence_judgement': judgement.to_dict() if judgement else None,
            'evidence_sufficient': judgement.sufficient if judgement else None,
            'retry_count': 0,
            'answer': gen_result['answer'],
            'references': gen_result['references'],
            'latency': latency,
        }
        self.query_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # 流式查询
    # ------------------------------------------------------------------
    def query_stream(self, question: str, history: list = None):
        """
        流式查询，yield事件字典：
        - {"type": "status", "message": "..."}  思考过程状态
        - {"type": "token", "content": "..."}    生成的token
        - {"type": "done", "references": [...], "latency": {...}, ...}
        """
        t0 = time.time()
        latency: Dict[str, float] = {}

        # 1. 查询分析 + 路由
        yield {"type": "status", "message": "正在理解问题..."}
        t1 = time.time()
        analysis = self._analyze(question)
        plan = self._resolve_plan(analysis)
        latency['query_analysis'] = time.time() - t1

        # 2. 按 Plan 多路检索
        yield {"type": "status", "message": "正在检索相关资料..."}
        t2 = time.time()
        ranked_lists, graph_entities = self._retrieve_by_plan(question, analysis, plan)
        latency['retrieval'] = time.time() - t2
        if graph_entities:
            yield {"type": "status",
                   "message": f"图谱匹配到 {len(graph_entities)} 个实体"}

        # 3. 融合
        t3 = time.time()
        fused = self._fuse_ranked(ranked_lists, FUSION_CANDIDATES)
        latency['fusion'] = time.time() - t3

        # 4. 精排（按 Plan）
        if plan.use_rerank and self.use_rerank and self.reranker.model is not None:
            yield {"type": "status", "message": "正在精排结果..."}
            t4 = time.time()
            reranked = self.reranker.rerank(question, fused, top_k=FINAL_TOP_K)
            latency['rerank'] = time.time() - t4
        else:
            reranked = fused[:FINAL_TOP_K]
            latency['rerank'] = 0.0

        # 5. 流式生成
        yield {"type": "status", "message": "正在生成答案..."}
        t6 = time.time()
        references = []
        for delta, refs in self.generator.generate_stream(
            question, reranked, analysis, history=history
        ):
            if refs is not None:
                references = refs
            else:
                yield {"type": "token", "content": delta}
        latency['generation'] = time.time() - t6
        latency['total'] = time.time() - t0

        yield {
            "type": "done",
            "references": references,
            "latency": latency,
            "query_mode": analysis.get("query_mode"),
            "retrieval_plan": plan.to_dict(),
        }


if __name__ == '__main__':
    pipeline = RAGPipeline()
    result = pipeline.query('陈嘉庚创办了哪些学校？', verbose=True)
    print(f'\n答案: {result["answer"]}')
    print(f'\n引用: {len(result["references"])} 个')
    print(f'\n耗时: {result["latency"]}')
