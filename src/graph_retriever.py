"""
图谱检索器
基于知识图谱的实体匹配、关系扩展、子图召回
与向量检索形成双层检索
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple

from src.config import (
    KG_DIR, CHUNKS_DIR, GRAPH_TOP_K, GRAPH_EXPAND_DEPTH,
    KG_RELATION_TOP_K, GRAPH_ENTITY_SIM_THRESHOLD,
    GRAPH_RELATION_SIM_THRESHOLD,
)


class GraphRetriever:
    """图谱检索器"""

    def __init__(self):
        self.entities = {}       # name -> entity dict
        self.relations = []      # list of relation dicts
        self.adj_list = {}       # entity -> [(relation, target, direction)]
        self.entity_index = None # FAISS 实体索引
        self.entity_meta = []    # 实体索引元数据
        self.relation_index = None  # FAISS 关系索引
        self.relation_meta = []    # 关系索引元数据
        self._embedding_model = None  # lazy 加载，进程内只加载一次
        self.chunk_map = self._load_chunk_map()  # 启动时一次加载，避免每查询遍历文件
        self._load_kg()
        self._load_index()

    def _load_chunk_map(self) -> Dict:
        """启动时一次性加载 chunk_id -> chunk 内容映射（绝对路径，避免 cwd 隐患）"""
        chunks_file = CHUNKS_DIR / 'chunks.jsonl'
        chunk_map: Dict = {}
        if not chunks_file.exists():
            print(f'警告: chunks 文件不存在: {chunks_file}')
            return chunk_map
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                c = json.loads(line)
                chunk_map[c['chunk_id']] = c
        print(f'chunk_map 加载: {len(chunk_map)} 个 chunk')
        return chunk_map

    def _get_embedding_model(self):
        """lazy 加载并缓存 embedding 模型，避免每次查询重复加载"""
        if self._embedding_model is None:
            from src.vector_store import load_embedding_model
            self._embedding_model = load_embedding_model()
        return self._embedding_model

    def _load_kg(self):
        """加载知识图谱"""
        entities_file = KG_DIR / 'entities.jsonl'
        if entities_file.exists():
            with open(entities_file, 'r', encoding='utf-8') as f:
                for line in f:
                    e = json.loads(line)
                    self.entities[e['name']] = e

        relations_file = KG_DIR / 'triples.jsonl'
        if relations_file.exists():
            with open(relations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    self.relations.append(r)
                    head, tail, rel = r['head'], r['tail'], r['relation']
                    # 双向邻接表：正向 head -> tail（outgoing），反向 tail -> head（incoming），
                    # 保留 relation direction 不丢失语义
                    self.adj_list.setdefault(head, []).append(
                        (rel, tail, 'outgoing'))
                    self.adj_list.setdefault(tail, []).append(
                        (rel, head, 'incoming'))

        print(f'图谱加载: {len(self.entities)} 实体, {len(self.relations)} 关系')

    def _load_index(self):
        """加载实体向量索引与关系向量索引"""
        index_file = KG_DIR / 'entity_index.faiss'
        if index_file.exists():
            import faiss
            self.entity_index = faiss.read_index(str(index_file))
            meta_file = KG_DIR / 'entity_meta.jsonl'
            with open(meta_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.entity_meta.append(json.loads(line))
            print(f'实体索引加载: {self.entity_index.ntotal} 个向量')

        rel_index_file = KG_DIR / 'relation_index.faiss'
        if rel_index_file.exists():
            import faiss
            self.relation_index = faiss.read_index(str(rel_index_file))
            rel_meta_file = KG_DIR / 'relation_meta.jsonl'
            if rel_meta_file.exists():
                with open(rel_meta_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.relation_meta.append(json.loads(line))
            print(f'关系索引加载: {self.relation_index.ntotal} 个向量')

    def match_entities(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        从查询中匹配相关实体。
        先尝试精确匹配（查询中包含实体名），再用向量相似度匹配。
        """
        matched = []

        # 1. 精确匹配（查询中包含实体名）
        for name, entity in self.entities.items():
            if name in query:
                matched.append({
                    'name': name,
                    'type': entity['type'],
                    'description': entity.get('description', ''),
                    'match_type': 'exact',
                    'score': 1.0,
                })
            # 别名匹配
            for alias in entity.get('aliases', []):
                if alias in query:
                    matched.append({
                        'name': name,
                        'type': entity['type'],
                        'description': entity.get('description', ''),
                        'match_type': 'alias',
                        'score': 0.9,
                    })

        # 去重
        seen = set()
        unique_matched = []
        for m in matched:
            if m['name'] not in seen:
                seen.add(m['name'])
                unique_matched.append(m)

        # 2. 向量匹配（补充）
        if self.entity_index is not None and len(unique_matched) < top_k:
            try:
                model = self._get_embedding_model()
                query_emb = model.encode([query])
                query_emb = np.array(query_emb, dtype='float32')
                query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
                scores, indices = self.entity_index.search(query_emb, top_k * 2)
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self.entity_meta):
                        continue
                    meta = self.entity_meta[idx]
                    if meta['name'] not in seen and score > 0.5:
                        unique_matched.append({
                            'name': meta['name'],
                            'type': meta['type'],
                            'description': meta.get('description', ''),
                            'match_type': 'vector',
                            'score': float(score),
                        })
                        seen.add(meta['name'])
                        if len(unique_matched) >= top_k:
                            break
            except Exception as e:
                print(f'实体向量匹配失败: {e}')

        return unique_matched[:top_k]

    def expand_subgraph(self, seed_entities: List[str], depth: int = GRAPH_EXPAND_DEPTH) -> Set[str]:
        """
        从种子实体出发，扩展子图（BFS）。
        返回相关实体集合。
        """
        visited = set(seed_entities)
        frontier = list(seed_entities)

        for d in range(depth):
            next_frontier = []
            for entity in frontier:
                if entity in self.adj_list:
                    for _, target, _ in self.adj_list[entity]:
                        if target not in visited:
                            visited.add(target)
                            next_frontier.append(target)
            frontier = next_frontier
            if not frontier:
                break

        return visited

    def get_neighbors(self, entity: str) -> List[Dict]:
        """
        返回实体的全部邻居（双向），每条带关系方向：
            {"neighbor": 对端实体, "relation": 关系, "direction": "outgoing"/"incoming"}
        """
        return [
            {"neighbor": target, "relation": rel, "direction": direction}
            for rel, target, direction in self.adj_list.get(entity, [])
        ]

    def _materialize_chunk(self, chunk_id: str, score: float,
                           source: str = 'graph',
                           matched: int = 0) -> Dict:
        """把 chunk_id 结合缓存的 chunk_map 物化为检索结果 dict；缺失返回 None"""
        c = self.chunk_map.get(chunk_id)
        if c is None:
            return None
        return {
            'chunk_id': chunk_id,
            'book': c['book'],
            'chapter': c['chapter'],
            'text': c['text'],
            'score': float(score),
            'char_count': c['char_count'],
            'source': source,
            'matched_entities': matched,
        }

    def get_related_chunks(self, entities: Dict[str, float]) -> List[Dict]:
        """
        获取与实体集合相关的chunk。
        通过实体的source_chunks字段关联，用实体匹配分数加权。

        Args:
            entities: {entity_name: match_score} 实体匹配分数
        """
        chunk_scores = {}  # chunk_id -> score

        for entity_name, entity_score in entities.items():
            if entity_name in self.entities:
                entity = self.entities[entity_name]
                for chunk_id in entity.get('source_chunks', []):
                    if chunk_id not in chunk_scores:
                        chunk_scores[chunk_id] = 0
                    # 用实体匹配分数加权，而不是简单计数
                    chunk_scores[chunk_id] += entity_score

        # 按加权分数排序
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, score in sorted_chunks[:GRAPH_TOP_K * 2]:
            chunk = self._materialize_chunk(
                chunk_id, score, source='graph', matched=len(entities)
            )
            if chunk is not None:
                results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # 关系检索（relation index）：high-level 抽象问题 → 关系语义匹配
    # ------------------------------------------------------------------
    def search_relations(self, query_text: str,
                         top_k: int = KG_RELATION_TOP_K,
                         score_threshold: float = GRAPH_RELATION_SIM_THRESHOLD
                         ) -> List[Dict]:
        """
        用关系向量索引检索与 query_text 最相关的三元组。
        FAISS relation index 的第 i 个向量与 self.relations[i]（triples.jsonl
        逐行）一一对应，因此可直接取回带 source_chunks 的完整关系记录。
        """
        if self.relation_index is None or not query_text:
            return []
        try:
            model = self._get_embedding_model()
            emb = model.encode([query_text])
            emb = np.array(emb, dtype='float32')
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            scores, indices = self.relation_index.search(emb, top_k)
        except Exception as e:
            print(f'关系向量检索失败: {e}')
            return []

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.relations):
                continue
            if score < score_threshold:
                continue
            r = self.relations[idx]
            results.append({
                'head': r['head'],
                'relation': r['relation'],
                'tail': r['tail'],
                'description': r.get('description', ''),
                'score': float(score),
                'source_chunks': r.get('source_chunks', []),
            })
        return results

    def relation_chunks(self, query_text: str,
                        top_k: int = GRAPH_TOP_K) -> List[Dict]:
        """
        关系检索 → 聚合命中关系的 source_chunks（按关系相似度加权）→ 物化 chunk。
        供 global / hybrid 模式在实体邻居之外补充关系语义证据。
        """
        rels = self.search_relations(query_text)
        if not rels:
            return []

        chunk_scores: Dict[str, float] = {}
        for rel in rels:
            for cid in rel.get('source_chunks', []):
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + rel['score']

        ordered = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, score in ordered[:top_k]:
            chunk = self._materialize_chunk(
                chunk_id, score, source='graph_relation', matched=len(rels)
            )
            if chunk is not None:
                results.append(chunk)
        return results

    def expand_subgraph_weighted(self, seed_entities: List[Dict], depth: int = GRAPH_EXPAND_DEPTH) -> Dict[str, float]:
        """
        从种子实体出发，扩展子图（BFS），带权重衰减。
        返回 {entity_name: score}，种子实体分数最高，扩展层逐级衰减。
        """
        entity_scores = {}  # entity_name -> score
        
        # 种子实体用匹配分数
        for e in seed_entities:
            entity_scores[e['name']] = e['score']
        
        frontier = [e['name'] for e in seed_entities]
        decay_factor = 0.5  # 每扩展一层，分数衰减一半

        for d in range(depth):
            next_frontier = []
            for entity in frontier:
                if entity not in self.adj_list:
                    continue
                current_score = entity_scores.get(entity, 0)
                for _, target, _ in self.adj_list[entity]:
                    if target not in entity_scores:
                        # 新扩展的实体，分数衰减
                        entity_scores[target] = current_score * decay_factor
                        next_frontier.append(target)
            frontier = next_frontier
            if not frontier:
                break

        return entity_scores

    def search(self, query: str, top_k: int = GRAPH_TOP_K) -> Dict:
        """
        图谱检索完整流程：
        1. 实体匹配
        2. 子图扩展（带权重衰减）
        3. 关联chunk召回（用实体匹配分数加权）

        返回: {entities, expanded_entities, chunks}
        """
        # 1. 实体匹配
        matched_entities = self.match_entities(query, top_k=5)

        if not matched_entities:
            return {'entities': [], 'expanded_entities': [], 'chunks': []}

        # 2. 子图扩展（带权重衰减）
        entity_scores = self.expand_subgraph_weighted(matched_entities)

        # 3. 关联chunk召回（用实体分数加权）
        chunks = self.get_related_chunks(entity_scores)

        return {
            'entities': matched_entities,
            'expanded_entities': list(entity_scores.keys())[:20],  # 只返回前20个
            'chunks': chunks[:top_k],
        }


if __name__ == '__main__':
    # 测试
    retriever = GraphRetriever()
    result = retriever.search('陈嘉庚创办了哪些学校？')
    print(f"匹配实体: {[e['name'] for e in result['entities']]}")
    print(f"扩展实体数: {len(result['expanded_entities'])}")
    print(f"召回chunk数: {len(result['chunks'])}")
    for c in result['chunks'][:3]:
        print(f"  [{c['score']}] {c['book']} - {c['chapter']}")
