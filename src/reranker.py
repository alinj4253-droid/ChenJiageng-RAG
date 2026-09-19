"""
Reranker重排序模块
使用BGE-reranker对检索结果重新排序，提高相关性
"""
from typing import List, Dict
from src.config import RERANK_TOP_K, RERANK_MODEL_PATH


class Reranker:
    """BGE Reranker重排序"""

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载reranker模型"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(RERANK_MODEL_PATH)
            print(f'Reranker模型加载成功: {RERANK_MODEL_PATH}')
        except Exception as e:
            print(f'Reranker模型加载失败: {e}')
            print('将跳过重排序步骤')
            self.model = None

    def rerank(self, query: str, chunks: List[Dict], top_k: int = RERANK_TOP_K) -> List[Dict]:
        """
        对检索结果重排序。

        Args:
            query: 用户查询
            chunks: 检索结果列表
            top_k: 返回前k个

        返回: 重排序后的chunk列表
        """
        if self.model is None or not chunks:
            return chunks[:top_k]

        # 构造(query, text)对
        pairs = [(query, c['text'][:512]) for c in chunks]

        # 计算相关性分数
        scores = self.model.predict(pairs)

        # 附加分数并排序
        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(scores[i])

        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k]


if __name__ == '__main__':
    print('Reranker模块加载成功')
