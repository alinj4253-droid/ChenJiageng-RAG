"""
RAG主Pipeline（Workflow-first Agentic RAG）

固定主流程：
查询理解(QueryAnalyzer)
  → 检索路由(RetrievalRouter, query_mode → RetrievalPlan)
  → 按 Plan 执行 Dense / BM25 / Entity Graph / Relation 多路检索
  → Weighted RRF 融合
  → Cross-Encoder Rerank（按 Plan）
  → Evidence Judge
  → （证据不足且未超上限）Query Rewrite + Bounded Retry
  → 生成答案

Agentic 决策点共三处（其余步骤为确定性代码，保证可控、可测、便于消融）：
  1. Query Classification / Routing：LLM 在 QueryAnalyzer 判定 query_mode，
     再由确定性 RetrievalRouter 映射为 RetrievalPlan；
  2. Evidence Sufficiency Judgement：LLM 判断证据是否足以回答原问题；
  3. Query Rewrite / Bounded Retry：证据不足时按缺失点改写检索 query 并补检，
     有严格上界、重复即停，杜绝死循环。
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
from src.query_rewriter import QueryRewriter, RewrittenQuery
from src.trace_logger import StructuredTraceLogger
from src.config import (
    FINAL_TOP_K,
    RERANK_TOP_N,
    RRF_K,
    VECTOR_TOP_K,
    BM25_TOP_K,
    GRAPH_TOP_K,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    ENTITY_GRAPH_WEIGHT,
    RELATION_GRAPH_WEIGHT,
    ENABLE_AGENT_ROUTING,
    ENABLE_EVIDENCE_JUDGE,
    MAX_RETRIEVAL_RETRY,
)

# 融合候选数：送入 rerank 前保留的候选规模
FUSION_CANDIDATES = RERANK_TOP_N


class RAGPipeline:
    """完整的RAG问答流水线"""

    def __init__(self, use_graph: bool = True, use_rerank: bool = True,
                 enable_routing: Optional[bool] = None,
                 enable_judge: Optional[bool] = None,
                 max_retry: Optional[int] = None,
                 fixed_plan: Optional[RetrievalPlan] = None):
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

        # 证据裁判（Agentic 决策点之二）；可全局/显式关闭用于消融
        self.enable_evidence_judge = (
            ENABLE_EVIDENCE_JUDGE if enable_judge is None else enable_judge
        )
        self.evidence_judge = EvidenceJudge() if self.enable_evidence_judge else None

        # 查询改写器 + 有限重试上限（Agentic 决策点之三）
        self.query_rewriter = QueryRewriter() if self.evidence_judge else None
        self.max_retry = MAX_RETRIEVAL_RETRY if max_retry is None else max_retry

        # Agent 路由开关：显式参数优先，否则读全局配置（便于消融 Agent On/Off）
        self.enable_routing = (
            ENABLE_AGENT_ROUTING if enable_routing is None else enable_routing
        )

        # 消融用：强制使用固定检索 Plan（绕过 Router），用于 Dense/+BM25/+KG/+Rerank 组
        self.fixed_plan = fixed_plan

        # 简单查询缓存：相同问题直接返回结果
        self.query_cache = {}

        # 结构化执行 trace 日志
        self.tracer = StructuredTraceLogger()
        print('RAG Pipeline 初始化完成')

    # ------------------------------------------------------------------
    # 查询分析
    # ------------------------------------------------------------------
    def _analyze(self, question: str) -> Dict:
        """
        查询分析。

        所有非空用户 Query 统一交给 QueryAnalyzer 由 LLM 判定 query_mode，
        不再按“问题长度”做 naive 启发式 shortcut——query 长度与难度没有可靠
        对应关系（如“陈嘉庚和李光前什么关系？”很短却属实体关系查询）。
        仅保留空输入 / 全空格这类非法输入校验。
        """
        if not question or not question.strip():
            raise ValueError("用户问题不能为空")
        return self.query_analyzer.analyze(question.strip())

    def _resolve_plan(self, analysis: Dict) -> RetrievalPlan:
        """消融固定 Plan 优先；否则按路由开关决定 Router 还是固定 hybrid"""
        if self.fixed_plan is not None:
            return self.fixed_plan
        if self.enable_routing:
            return build_plan(analysis)
        return build_plan({'query_mode': 'hybrid'})

    # ------------------------------------------------------------------
    # 按 Plan 检索 + 融合
    # ------------------------------------------------------------------
    def _retrieve_by_plan(
        self, query: str, analysis: Dict, plan: RetrievalPlan,
        counters: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[Tuple[List[Dict], float, str]], List[Dict]]:
        """
        按 RetrievalPlan 执行多路检索。

        Args:
            counters: 可选的可变计数器字典；每真正调用一次 Retriever，对应
                      key（dense/bm25/entity_graph/relation_graph）+1。用于
                      区分 retrieval_rounds（检索轮次）与 retrieval_calls
                      （实际 Retriever 调用次数）——一轮可能调用多个 Retriever。

        Returns:
            ranked_lists: [(results, weight, name), ...] 供 Weighted RRF 融合
            graph_entities: 图谱命中的实体（用于状态展示/可观测）
        """
        ranked_lists: List[Tuple[List[Dict], float, str]] = []
        graph_entities: List[Dict] = []
        if counters is None:
            counters = {}

        if plan.use_dense:
            counters["dense"] = counters.get("dense", 0) + 1
            dense = self.vector_retriever.search(query, top_k=VECTOR_TOP_K)
            ranked_lists.append((dense, VECTOR_WEIGHT, 'dense'))

        if plan.use_bm25:
            counters["bm25"] = counters.get("bm25", 0) + 1
            bm25 = self.bm25_retriever.search(query, top_k=BM25_TOP_K)
            ranked_lists.append((bm25, BM25_WEIGHT, 'bm25'))

        if plan.use_graph and self.use_graph:
            # 实体图谱检索：实体匹配 + 双向子图扩展 + 关联 chunk 召回
            counters["entity_graph"] = counters.get("entity_graph", 0) + 1
            graph_result = self.graph_retriever.search(query)
            graph_chunks = graph_result.get('chunks', [])
            graph_entities = graph_result.get('entities', [])
            ranked_lists.append((graph_chunks, ENTITY_GRAPH_WEIGHT, 'graph'))

        if plan.use_relation and self.use_graph:
            # 关系检索：high-level 抽象关键词走关系向量索引；
            # 无 high-level 关键词时回退用原始 query。
            # 无论是否召回结果，relation_chunks 都被真正调用一次，计入 calls。
            rel_query = " ".join(analysis.get('high_level_keywords', [])) or query
            rel_chunks = self.graph_retriever.relation_chunks(
                rel_query, top_k=GRAPH_TOP_K
            )
            counters["relation_graph"] = counters.get("relation_graph", 0) + 1
            if rel_chunks:
                ranked_lists.append((rel_chunks, RELATION_GRAPH_WEIGHT, 'graph_relation'))

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
                              latency: Optional[Dict] = None,
                              counters: Optional[Dict[str, int]] = None
                              ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """对给定 query 执行：按 Plan 检索 → Weighted RRF 融合 → 精排。
        Retry 时可对改写后的 query 重复调用，耗时在 latency 中累加，
        Retriever 调用次数在 counters 中累加（跨轮次）。
        返回 (reranked, graph_entities, fused)：
          - reranked：精排后最终上下文（FINAL_TOP_K）
          - fused：精排前融合候选（FUSION_CANDIDATES），供检索 Recall@K 评估
        """
        latency = latency if latency is not None else {}

        t2 = time.time()
        ranked_lists, graph_entities = self._retrieve_by_plan(
            query, analysis, plan, counters=counters
        )
        latency['retrieval'] = latency.get('retrieval', 0.0) + (time.time() - t2)

        t3 = time.time()
        fused = self._fuse_ranked(ranked_lists, FUSION_CANDIDATES)
        latency['fusion'] = latency.get('fusion', 0.0) + (time.time() - t3)

        t4 = time.time()
        reranked, rerank_applied = self._rerank_by_plan(query, fused, plan)
        if rerank_applied:
            latency['rerank'] = latency.get('rerank', 0.0) + (time.time() - t4)
        else:
            latency.setdefault('rerank', 0.0)

        return reranked, graph_entities, fused

    def _judge_evidence(self, question: str, contexts: List[Dict],
                        plan: RetrievalPlan) -> Optional[EvidenceJudgement]:
        """按 Plan / 全局开关决定是否做证据裁判；naive 模式不裁判"""
        if self.evidence_judge is None or not plan.use_evidence_judge:
            return None
        return self.evidence_judge.judge(question, contexts)

    def _retrieval_loop(self, question: str, analysis: Dict,
                        plan: RetrievalPlan, latency: Dict):
        """
        检索 → 证据裁判 → （不足则）有限 Query Rewrite + Retry 的核心循环。

        生成器，依次 yield：
            ("status", str)   可供流式接口转发的进度状态
            ("result", dict)  循环结束后的最终结果（仅一次，放在最后）

        - 裁判始终针对用户原始问题；
        - 重试保留原 analysis / plan，只替换检索 query；
        - 最多重试 max_retry 次；
        - 改写结果与历史查询重复时立即停止，杜绝死循环。
        """
        current_query = question
        previous_queries = {question.strip()}
        judgement: Optional[EvidenceJudgement] = None
        retry_count = 0
        candidates: List[Dict] = []

        # 跨轮次累加真实 Retriever 调用次数（区别于检索轮次 rounds）
        retrieval_counters: Dict[str, int] = {
            "dense": 0, "bm25": 0, "entity_graph": 0, "relation_graph": 0,
        }

        yield ("status", "正在检索相关资料...")
        contexts, graph_entities, candidates = self._retrieve_fuse_rerank(
            current_query, analysis, plan, latency, counters=retrieval_counters
        )

        while self.evidence_judge is not None and plan.use_evidence_judge:
            yield ("status", "正在评估证据是否充分...")
            t = time.time()
            judgement = self.evidence_judge.judge(question, contexts)
            latency['evidence_judge'] = latency.get('evidence_judge', 0.0) + (time.time() - t)

            if judgement.sufficient:
                break

            # 证据不足：判断是否还能重试
            if retry_count >= self.max_retry or self.query_rewriter is None:
                break

            yield ("status", "证据不足，正在改写查询并补充检索...")
            t = time.time()
            rewritten = self.query_rewriter.rewrite(
                original_query=question,
                current_query=current_query,
                missing=judgement.missing,
                contexts=contexts,
            )
            latency['query_rewrite'] = latency.get('query_rewrite', 0.0) + (time.time() - t)

            # 改写失败 / 空查询 → 停止
            if rewritten is None or not rewritten.query.strip():
                break

            new_query = rewritten.query.strip()
            # 与当前或历史查询重复 → 直接停止，防止死循环
            if new_query == current_query.strip() or new_query in previous_queries:
                break

            previous_queries.add(new_query)
            current_query = new_query
            retry_count += 1

            yield ("status", "正在用改写后的查询补充检索...")
            contexts, graph_entities, candidates = self._retrieve_fuse_rerank(
                current_query, analysis, plan, latency, counters=retrieval_counters
            )

        yield ("result", {
            "contexts": contexts,
            "graph_entities": graph_entities,
            "judgement": judgement,
            "retry_count": retry_count,
            "candidates": candidates,
            "candidate_count": len(candidates),
            "retrieval_calls": sum(retrieval_counters.values()),
            "retrieval_call_detail": dict(retrieval_counters),
        })

    def _prepare_context(self, question: str) -> Tuple[List[Dict], Dict, RetrievalPlan, List[Dict], Dict, Optional[EvidenceJudgement], int, List[Dict]]:
        """查询分析 → 路由 → （检索-裁判-有限重试）循环（非流式路径共用）"""
        latency: Dict[str, float] = {}

        t1 = time.time()
        analysis = self._analyze(question)
        plan = self._resolve_plan(analysis)
        latency['query_analysis'] = time.time() - t1

        outcome = {}
        for kind, payload in self._retrieval_loop(question, analysis, plan, latency):
            if kind == "result":
                outcome = payload

        return (outcome["contexts"], analysis, plan, outcome["graph_entities"],
                latency, outcome["judgement"], outcome["retry_count"],
                outcome["candidates"],
                outcome["retrieval_calls"], outcome["retrieval_call_detail"])

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
        (reranked, analysis, plan, graph_entities, latency,
         judgement, retry_count, candidates,
         retrieval_calls, retrieval_call_detail) = self._prepare_context(question)

        if verbose:
            print(f'[查询分析] 模式: {analysis.get("query_mode")}')
            print(f'[路由] 启用检索器: {plan.enabled_retrievers()} | 精排: {plan.use_rerank}')
            print(f'[图谱检索] 匹配实体: {[e["name"] for e in graph_entities]}')
            if judgement:
                print(f'[证据裁判] sufficient={judgement.sufficient} '
                      f'missing={judgement.missing} | 重试 {retry_count} 次')
            print(f'[融合+精排] 最终上下文 {len(reranked)} 个chunk')

        # 证据最终仍不足时，要求生成器谨慎作答（不编造）
        caution = judgement is not None and not judgement.sufficient

        t6 = time.time()
        gen_result = self.generator.generate(
            question, reranked, analysis, caution=caution
        )
        latency['generation'] = time.time() - t6
        latency['total'] = time.time() - t0

        result = {
            'question': question,
            'query_analysis': analysis,
            'query_mode': analysis.get('query_mode'),
            'retrieval_plan': plan.to_dict(),
            'retrieved_chunks': reranked,
            'retrieval_candidates': candidates,
            'evidence_judgement': judgement.to_dict() if judgement else None,
            'evidence_sufficient': judgement.sufficient if judgement else None,
            'retry_count': retry_count,
            'retrieval_rounds': retry_count + 1,
            'retrieval_calls': retrieval_calls,
            'retrieval_call_detail': retrieval_call_detail,
            'answer': gen_result['answer'],
            'references': gen_result['references'],
            'latency': latency,
        }

        # 结构化执行 trace（落盘 JSONL，失败不影响主流程）
        try:
            trace = self.tracer.build_trace(
                question=question, analysis=analysis, plan=plan,
                contexts=reranked, graph_entities=graph_entities,
                judgement=judgement, retry_count=retry_count,
                candidate_count=len(candidates), latency=latency,
                retrieval_calls=retrieval_calls,
                retrieval_call_detail=retrieval_call_detail,
            )
            self.tracer.log(trace)
            result['trace'] = trace
        except Exception as e:
            print(f'trace 记录失败: {e}')

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

        # 2~4. 检索 → 证据裁判 → 有限改写重试（生成器，实时转发状态）
        reranked = []
        candidates = []
        graph_entities = []
        judgement = None
        retry_count = 0
        retrieval_calls = 0
        retrieval_call_detail: Dict[str, int] = {}
        for kind, payload in self._retrieval_loop(question, analysis, plan, latency):
            if kind == "status":
                yield {"type": "status", "message": payload}
            elif kind == "result":
                reranked = payload["contexts"]
                judgement = payload["judgement"]
                retry_count = payload["retry_count"]
                graph_entities = payload["graph_entities"]
                candidates = payload["candidates"]
                retrieval_calls = payload["retrieval_calls"]
                retrieval_call_detail = payload["retrieval_call_detail"]
        if graph_entities:
            yield {"type": "status",
                   "message": f"图谱匹配到 {len(graph_entities)} 个实体"}

        # 5. 流式生成（证据最终不足时谨慎作答）
        caution = judgement is not None and not judgement.sufficient
        yield {"type": "status", "message": "正在生成答案..."}
        t6 = time.time()
        references = []
        for delta, refs in self.generator.generate_stream(
            question, reranked, analysis, history=history, caution=caution
        ):
            if refs is not None:
                references = refs
            else:
                yield {"type": "token", "content": delta}
        latency['generation'] = time.time() - t6
        latency['total'] = time.time() - t0

        # 结构化执行 trace（失败不影响流式主流程）
        trace = None
        try:
            trace = self.tracer.build_trace(
                question=question, analysis=analysis, plan=plan,
                contexts=reranked, graph_entities=graph_entities,
                judgement=judgement, retry_count=retry_count,
                candidate_count=len(candidates), latency=latency,
                retrieval_calls=retrieval_calls,
                retrieval_call_detail=retrieval_call_detail,
            )
            self.tracer.log(trace)
        except Exception as e:
            print(f'trace 记录失败: {e}')

        yield {
            "type": "done",
            "references": references,
            "latency": latency,
            "query_mode": analysis.get("query_mode"),
            "retrieval_plan": plan.to_dict(),
            "evidence_sufficient": judgement.sufficient if judgement else None,
            "retry_count": retry_count,
            "retrieval_rounds": retry_count + 1,
            "retrieval_calls": retrieval_calls,
            "retrieval_call_detail": retrieval_call_detail,
            "trace": trace,
        }


if __name__ == '__main__':
    pipeline = RAGPipeline()
    result = pipeline.query('陈嘉庚创办了哪些学校？', verbose=True)
    print(f'\n答案: {result["answer"]}')
    print(f'\n引用: {len(result["references"])} 个')
    print(f'\n耗时: {result["latency"]}')
