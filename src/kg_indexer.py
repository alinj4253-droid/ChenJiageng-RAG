"""
知识图谱向量索引器
为实体和关系构建FAISS向量索引，支持双层检索
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.config import KG_DIR, EMBED_DIM, EMBED_MODEL_NAME
from pathlib import Path


def load_embedding_model():
    """加载BGE嵌入模型"""
    try:
        from sentence_transformers import SentenceTransformer
        # 尝试多个本地路径
        possible_paths = [
            Path("data/models/models/BAAI--bge-base-zh-v1.5/snapshots/master"),
            Path("data/models/bge-base-zh-v1.5"),
        ]
        for p in possible_paths:
            if p.exists() and (p / "config.json").exists():
                model = SentenceTransformer(str(p))
                print(f'嵌入模型加载成功(本地): {p}')
                return model
        # 都没有则从名称加载
        model = SentenceTransformer(EMBED_MODEL_NAME)
        print(f'嵌入模型加载成功: {EMBED_MODEL_NAME}')
        return model
    except Exception as e:
        print(f'嵌入模型加载失败: {e}')
        raise


def build_entity_index(model):
    """构建实体向量索引"""
    entities_file = KG_DIR / 'entities.jsonl'
    if not entities_file.exists():
        print(f'错误: {entities_file} 不存在')
        return

    entities = []
    with open(entities_file, 'r', encoding='utf-8') as f:
        for line in f:
            entities.append(json.loads(line))

    print(f'加载 {len(entities)} 个实体')

    # 构造实体文本（名称+类型+描述）
    entity_texts = []
    for e in entities:
        text = f"{e['name']}（{e['type']}）{e.get('description', '')}"
        entity_texts.append(text)

    # 生成embedding
    print('生成实体向量...')
    embeddings = model.encode(entity_texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype='float32')

    # 归一化（余弦相似度）
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # 建FAISS索引
    import faiss
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    # 保存
    import faiss
    faiss.write_index(index, str(KG_DIR / 'entity_index.faiss'))

    # 保存元数据
    meta = [{'name': e['name'], 'type': e['type'], 'description': e.get('description', ''),
             'aliases': e.get('aliases', []), 'chunk_count': e.get('chunk_count', 0)}
            for e in entities]
    with open(KG_DIR / 'entity_meta.jsonl', 'w', encoding='utf-8') as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')

    print(f'实体索引保存: {index.ntotal} 个向量')
    return entities


def build_relation_index(model):
    """构建关系向量索引"""
    relations_file = KG_DIR / 'triples.jsonl'
    if not relations_file.exists():
        print(f'错误: {relations_file} 不存在')
        return

    relations = []
    with open(relations_file, 'r', encoding='utf-8') as f:
        for line in f:
            relations.append(json.loads(line))

    print(f'加载 {len(relations)} 条关系')

    # 构造关系文本（头实体+关系+尾实体+描述）
    relation_texts = []
    for r in relations:
        text = f"{r['head']} {r['relation']} {r['tail']}。{r.get('description', '')}"
        relation_texts.append(text)

    # 生成embedding
    print('生成关系向量...')
    embeddings = model.encode(relation_texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype='float32')

    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # 建FAISS索引
    import faiss
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    # 保存
    faiss.write_index(index, str(KG_DIR / 'relation_index.faiss'))

    # 保存元数据
    meta = [{'head': r['head'], 'relation': r['relation'], 'tail': r['tail'],
             'description': r.get('description', ''), 'chunk_count': r.get('chunk_count', 0)}
            for r in relations]
    with open(KG_DIR / 'relation_meta.jsonl', 'w', encoding='utf-8') as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')

    print(f'关系索引保存: {index.ntotal} 个向量')
    return relations


def build_all_indexes():
    """构建实体和关系向量索引"""
    model = load_embedding_model()
    print()
    build_entity_index(model)
    print()
    build_relation_index(model)
    print()
    print('=== 图谱索引构建完成 ===')


if __name__ == '__main__':
    build_all_indexes()
