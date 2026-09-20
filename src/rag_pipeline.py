"""
RAG主Pipeline
查询理解 -> 混合检索 + 图谱检索 -> 结果融合 -> Rerank -> 生成答案
"""
import time
from typing import Dict, List, Optional

from src.query_analyzer import QueryAnalyzer
from src.hybrid_retriever import HybridRetriever
from src.graph_retriever import GraphRetriever
from src.reranker import Reranker
from src.rag_generator import RAGGenerator
from src.config import FINAL_TOP_K, KG_WEIGHT


class RAGPipeline:
    """完整的RAG问答流水线"""

    def __init__(self, use_graph: bool = True, use_rerank: bool = True):
        print('初始化RAG Pipeline...')
        self.query_analyzer = QueryAnalyzer()
        self.hybrid_retriever = HybridRetriever()
        self.use_graph = use_graph
        if use_graph:
            self.graph_retriever = GraphRetriever()
        self.use_rerank = use_rerank
        if use_rerank:
            self.reranker = Reranker()
        self.generator = RAGGenerator()
        # 简单查询缓存：相同问题直接返回结果
        self.query_cache = {}
        print('RAG Pipeline 初始化完成')

    def _fuse_results(self, hybrid_results: List[Dict], graph_results: List[Dict]) -> List[Dict]:
        """
        融合混合检索和图谱检索结果。
        使用加权RRF融合。
        """
        if not graph_results:
            return hybrid_results

        # 图谱结果权重更高（因为是实体关联的）
        rrf_scores = {}  # chunk_id -> [score, result]

        # 混合检索结果
        for rank, r in enumerate(hybrid_results):
            chunk_id = r['chunk_id']
            score = 1.0 / (60 + rank + 1)  # RRF，k=60
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id][0] += score
                rrf_scores[chunk_id][1]['hybrid_score'] = r.get('rrf_score', r.get('score', 0))
            else:
                r['hybrid_score'] = r.get('rrf_score', r.get('score', 0))
                rrf_scores[chunk_id] = [score, r]

        # 图谱检索结果（乘上 KG_WEIGHT，权重可配）
        for rank, r in enumerate(graph_results):
            chunk_id = r['chunk_id']
            score = KG_WEIGHT / (60 + rank + 1)
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id][0] += score
                rrf_scores[chunk_id][1]['graph_score'] = r.get('score', 0)
            else:
                r['graph_score'] = r.get('score', 0)
                rrf_scores[chunk_id] = [score, r]

        # 排序
        merged = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        results = []
        for score, r in merged:
            r['fused_score'] = score
            results.append(r)

        return results

    def query(self, question: str, verbose: bool = False) -> Dict:
        """
        执行完整的RAG问答。

        Args:
            question: 用户问题
            verbose: 是否打印中间过程

        返回: {
            "question": 原始问题,
            "query_analysis": 查询分析结果,
            "retrieved_chunks": 检索到的chunk,
            "answer": 生成的答案,
            "references": 引用列表,
            "latency": 各阶段耗时,
        }
        """
        # 查询缓存：相同问题直接返回
        cache_key = question.strip()
        if cache_key in self.query_cache:
            cached = self.query_cache[cache_key].copy()
            cached["from_cache"] = True
            return cached

        t0 = time.time()
        latency = {}

        # 1. 查询理解
        t1 = time.time()
        query_analysis = self.query_analyzer.analyze(question)
        latency['query_analysis'] = time.time() - t1
        if verbose:
            print(f'[查询分析] 模式: {query_analysis.get("query_mode")}')
            print(f'[查询分析] 低层关键词: {query_analysis.get("low_level_keywords")}')
            print(f'[查询分析] 高层关键词: {query_analysis.get("high_level_keywords")}')

        # 2. 混合检索（直接用原问题，查询分析仅做意图识别）
        t2 = time.time()
        effective_query = question
        hybrid_results = self.hybrid_retriever.search(effective_query, top_k=FINAL_TOP_K * 2)
        latency['hybrid_retrieval'] = time.time() - t2
        if verbose:
            print(f'[混合检索] 召回 {len(hybrid_results)} 个chunk')

        # 3. 图谱检索
        graph_results = []
        if self.use_graph:
            t3 = time.time()
            graph_result = self.graph_retriever.search(effective_query)
            graph_results = graph_result.get('chunks', [])
            latency['graph_retrieval'] = time.time() - t3
            if verbose:
                print(f'[图谱检索] 匹配实体: {[e["name"] for e in graph_result.get("entities", [])]}')
                print(f'[图谱检索] 召回 {len(graph_results)} 个chunk')

        # 4. 结果融合
        t4 = time.time()
        fused = self._fuse_results(hybrid_results, graph_results)
        latency['fusion'] = time.time() - t4
        if verbose:
            print(f'[结果融合] 共 {len(fused)} 个chunk')

        # 5. Rerank
        if self.use_rerank and self.reranker.model is not None:
            t5 = time.time()
            reranked = self.reranker.rerank(effective_query, fused, top_k=FINAL_TOP_K)
            latency['rerank'] = time.time() - t5
            if verbose:
                print(f'[Rerank] 保留前 {len(reranked)} 个')
        else:
            reranked = fused[:FINAL_TOP_K]
            latency['rerank'] = 0

        # 6. 生成答案
        t6 = time.time()
        gen_result = self.generator.generate(question, reranked, query_analysis)
        latency['generation'] = time.time() - t6

        latency['total'] = time.time() - t0

        result = {
            'question': question,
            'query_analysis': query_analysis,
            'retrieved_chunks': reranked,
            'answer': gen_result['answer'],
            'references': gen_result['references'],
            'latency': latency,
        }
        # 存入查询缓存
        self.query_cache[cache_key] = result
        return result

    def query_stream(self, question: str, history: list = None):
        """
        流式查询，yield事件字典：
        - {"type": "status", "message": "..."}  思考过程状态
        - {"type": "token", "content": "..."}    生成的token
        - {"type": "done", "references": [...], "latency": {...}}  完成
        history: 历史对话messages列表，[{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
        """
        t0 = time.time()
        latency = {}

        # 1. 查询理解（简单问题跳过LLM改写，直接用原问题，提速）
        yield {"type": "status", "message": "正在检索相关资料..."}
        t1 = time.time()
        # 简单的短问题直接跳过改写，省掉一次LLM调用
        if len(question) < 15:
            query_analysis = {
                "original_query": question,
                "low_level_keywords": [],
                "high_level_keywords": [],
                "query_mode": "hybrid",
                "reason": "短问题跳过LLM分析",
            }
        else:
            query_analysis = self.query_analyzer.analyze(question)
        latency['query_analysis'] = time.time() - t1

        # 2. 混合检索（直接用原问题）
        yield {"type": "status", "message": "正在检索相关资料..."}
        t2 = time.time()
        effective_query = question
        hybrid_results = self.hybrid_retriever.search(effective_query, top_k=FINAL_TOP_K * 2)
        latency['hybrid_retrieval'] = time.time() - t2

        # 3. 图谱检索
        graph_results = []
        if self.use_graph:
            t3 = time.time()
            graph_result = self.graph_retriever.search(effective_query)
            graph_results = graph_result.get('chunks', [])
            latency['graph_retrieval'] = time.time() - t3
            yield {"type": "status", "message": f"图谱匹配到 {len(graph_result.get('entities', []))} 个实体"}

        # 4. 结果融合
        fused = self._fuse_results(hybrid_results, graph_results)

        # 5. Rerank
        if self.use_rerank and self.reranker.model is not None:
            yield {"type": "status", "message": "正在精排结果..."}
            t5 = time.time()
            reranked = self.reranker.rerank(effective_query, fused, top_k=FINAL_TOP_K)
            latency['rerank'] = time.time() - t5
        else:
            reranked = fused[:FINAL_TOP_K]
            latency['rerank'] = 0

        # 6. 流式生成
        yield {"type": "status", "message": "正在生成答案..."}
        t6 = time.time()
        references = []
        for delta, refs in self.generator.generate_stream(question, reranked, query_analysis, history=history):
            if refs is not None:
                references = refs
            else:
                yield {"type": "token", "content": delta}
        latency['generation'] = time.time() - t6
        latency['total'] = time.time() - t0

        yield {"type": "done", "references": references, "latency": latency}


if __name__ == '__main__':
    # 测试
    pipeline = RAGPipeline()
    result = pipeline.query('陈嘉庚创办了哪些学校？', verbose=True)
    print(f'\n答案: {result["answer"]}')
    print(f'\n引用: {len(result["references"])} 个')
    print(f'\n耗时: {result["latency"]}')
