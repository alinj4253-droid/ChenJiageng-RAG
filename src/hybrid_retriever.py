"""
混合检索器
向量检索 + BM25检索 + RRF融合
"""
from typing import List, Dict
from src.config import HYBRID_TOP_K, RRF_K
from src.vector_store import VectorRetriever
from src.bm25_store import BM25Retriever


class HybridRetriever:
    """混合检索器：向量+BM25，RRF融合"""

    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> List[Dict]:
        """
        混合检索，返回top_k个相关chunk。
        使用RRF（Reciprocal Rank Fusion）融合向量和BM25结果。
        """
        # 两路检索（各取更多候选）
        vector_results = self.vector_retriever.search(query, top_k=top_k * 2)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)

        # RRF融合
        rrf_scores = {}  # chunk_id -> (score, result_dict)

        for rank, r in enumerate(vector_results):
            chunk_id = r['chunk_id']
            rrf_score = 1.0 / (RRF_K + rank + 1)
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id][0] += rrf_score
                rrf_scores[chunk_id][1]['vector_score'] = r['score']
            else:
                r['vector_score'] = r['score']
                rrf_scores[chunk_id] = [rrf_score, r]

        for rank, r in enumerate(bm25_results):
            chunk_id = r['chunk_id']
            rrf_score = 1.0 / (RRF_K + rank + 1)
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id][0] += rrf_score
                rrf_scores[chunk_id][1]['bm25_score'] = r['score']
            else:
                r['bm25_score'] = r['score']
                rrf_scores[chunk_id] = [rrf_score, r]

        # 按融合分数排序
        merged = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)

        results = []
        for rrf_score, r in merged[:top_k]:
            r['rrf_score'] = rrf_score
            r['source'] = 'hybrid'
            results.append(r)

        return results


if __name__ == '__main__':
    # 测试
    retriever = HybridRetriever()
    results = retriever.search('陈嘉庚的教育思想')
    for r in results[:5]:
        print(f"[RRF={r['rrf_score']:.4f}] {r['book']} - {r['chapter']}")
        print(f"  向量分={r.get('vector_score', 0):.4f}, BM25分={r.get('bm25_score', 0):.4f}")
        print(f"  {r['text'][:80]}...")
