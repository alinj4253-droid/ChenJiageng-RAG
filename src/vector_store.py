"""
向量存储与检索
BGE嵌入 + FAISS索引
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from src.config import (
    CHUNKS_DIR,
    VECTOR_DIR,
    EMBED_DIM,
    EMBED_MODEL_PATH,
    VECTOR_TOP_K,
)


def load_embedding_model():
    """加载BGE嵌入模型"""
    from sentence_transformers import SentenceTransformer
    from pathlib import Path
    # 尝试多个本地路径
    possible_paths = [
        Path("data/models/models/BAAI--bge-base-zh-v1.5/snapshots/master"),
        Path(EMBED_MODEL_PATH),
    ]
    for p in possible_paths:
        if p.exists() and (p / "config.json").exists():
            return SentenceTransformer(str(p))
    # 都没有则从名称加载
    return SentenceTransformer("BAAI/bge-base-zh-v1.5")


def build_vector_store():
    """从chunks构建FAISS向量索引"""
    chunks_file = CHUNKS_DIR / 'chunks.jsonl'
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f'加载 {len(chunks)} 个chunk')

    model = load_embedding_model()

    # 生成embedding
    texts = [c['text'] for c in chunks]
    print('生成向量...')
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype='float32')

    # 归一化（余弦相似度）
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # 建FAISS索引
    import faiss
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    # 保存
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    import faiss
    faiss.write_index(index, str(VECTOR_DIR / 'index.faiss'))

    # 保存元数据
    with open(VECTOR_DIR / 'metadata.jsonl', 'w', encoding='utf-8') as f:
        for c in chunks:
            f.write(json.dumps({
                'chunk_id': c['chunk_id'],
                'book': c['book'],
                'chapter': c['chapter'],
                'text': c['text'],
                'char_count': c['char_count'],
            }, ensure_ascii=False) + '\n')

    # 保存配置
    config = {'dim': EMBED_DIM, 'model': EMBED_MODEL_PATH, 'count': len(chunks)}
    with open(VECTOR_DIR / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f'向量索引构建完成: {len(chunks)} 个向量')


class VectorRetriever:
    """向量检索器"""

    def __init__(self):
        import faiss
        self.index = faiss.read_index(str(VECTOR_DIR / 'index.faiss'))
        self.metadata = []
        with open(VECTOR_DIR / 'metadata.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line))
        self.model = None  # 延迟加载

    def _get_model(self):
        if self.model is None:
            self.model = load_embedding_model()
        return self.model

    def search(self, query: str, top_k: int = VECTOR_TOP_K) -> List[Dict]:
        """
        向量检索，返回top_k个相关chunk。
        每个结果: {chunk_id, book, chapter, text, score, char_count}
        """
        model = self._get_model()
        query_emb = model.encode([query])
        query_emb = np.array(query_emb, dtype='float32')
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                'chunk_id': meta['chunk_id'],
                'book': meta['book'],
                'chapter': meta['chapter'],
                'text': meta['text'],
                'score': float(score),
                'char_count': meta['char_count'],
                'source': 'vector',
            })
        return results


if __name__ == '__main__':
    build_vector_store()
