"""
BM25关键词检索
基于jieba分词的稀疏检索，与向量检索互补
"""
import json
import math
from pathlib import Path
from collections import Counter
from typing import List, Dict

from src.config import CHUNKS_DIR, BM25_TOP_K, STOPWORDS_FILE, CUSTOM_DICT_FILE


class BM25Retriever:
    """BM25检索器"""

    def __init__(self):
        self.chunks = []
        self.doc_freqs = []  # 每个文档的词频
        self.doc_len = []    # 每个文档长度
        self.avgdl = 0       # 平均文档长度
        self.df = Counter()  # 文档频率
        self.idf = {}        # IDF值
        self.N = 0           # 文档总数
        self.stopwords = set()
        self._load_stopwords()
        self._load()

    def _load_stopwords(self):
        """加载停用词"""
        if STOPWORDS_FILE.exists():
            with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.stopwords.add(word)
        # 额外停用词
        extra = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                 '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
                 '自己', '这', '那', '他', '她', '它', '们', '这个', '那个', '什么', '怎么',
                 '因为', '所以', '但是', '如果', '虽然', '而且', '并且', '或者', '以及'}
        self.stopwords.update(extra)

    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        import jieba
        # 加载自定义词典
        if CUSTOM_DICT_FILE.exists():
            jieba.load_userdict(str(CUSTOM_DICT_FILE))
        words = jieba.cut(text)
        return [w for w in words if w.strip() and w not in self.stopwords and len(w) > 1]

    def _load(self):
        """加载chunks并构建索引"""
        chunks_file = CHUNKS_DIR / 'chunks.jsonl'
        if not chunks_file.exists():
            print(f'警告: {chunks_file} 不存在')
            return

        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.chunks.append(json.loads(line))

        self.N = len(self.chunks)
        print(f'BM25加载 {self.N} 个文档')

        # 计算词频和文档长度
        for chunk in self.chunks:
            words = self._tokenize(chunk['text'])
            self.doc_freqs.append(Counter(words))
            self.doc_len.append(len(words))
            for word in set(words):
                self.df[word] += 1

        self.avgdl = sum(self.doc_len) / max(self.N, 1)

        # 计算IDF（BM25公式）
        for word, freq in self.df.items():
            self.idf[word] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

        print(f'BM25索引构建完成, 词汇量={len(self.df)}, 平均长度={self.avgdl:.1f}')

    def search(self, query: str, top_k: int = BM25_TOP_K) -> List[Dict]:
        """
        BM25检索，返回top_k个相关chunk。
        """
        if self.N == 0:
            return []

        query_words = self._tokenize(query)
        if not query_words:
            return []

        k1 = 1.5  # BM25参数
        b = 0.75

        scores = []
        for i in range(self.N):
            score = 0.0
            for word in query_words:
                if word not in self.idf:
                    continue
                tf = self.doc_freqs[i].get(word, 0)
                if tf == 0:
                    continue
                # BM25打分公式
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * self.doc_len[i] / self.avgdl)
                score += self.idf[word] * numerator / denominator
            if score > 0:
                scores.append((score, i))

        # 排序
        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            chunk = self.chunks[idx]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'book': chunk['book'],
                'chapter': chunk['chapter'],
                'text': chunk['text'],
                'score': float(score),
                'char_count': chunk['char_count'],
                'source': 'bm25',
            })
        return results


if __name__ == '__main__':
    # 测试
    retriever = BM25Retriever()
    results = retriever.search('陈嘉庚创办厦门大学')
    for r in results[:3]:
        print(f"[{r['score']:.3f}] {r['book']} - {r['chapter']}")
        print(f"  {r['text'][:80]}...")
