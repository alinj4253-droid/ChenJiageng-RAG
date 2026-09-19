"""
Milvus Lite 迁移脚本
将FAISS向量索引迁移到Milvus Lite
"""
import json
import numpy as np
from pathlib import Path
from pymilvus import MilvusClient, DataType

# 配置
MILVUS_DB_PATH = "data/milvus/rag.db"
CHUNK_COLLECTION = "chunks"
ENTITY_COLLECTION = "entities"
RELATION_COLLECTION = "relations"

VECTOR_DIM = 768  # BGE-base-zh-v1.5 维度


def load_faiss_index(index_path: str):
    """读取FAISS索引中的所有向量"""
    import faiss
    index = faiss.read_index(index_path)
    # 重建索引获取所有向量
    if hasattr(index, 'reconstruct_n'):
        vectors = np.array([index.reconstruct_n(i, 1)[0] for i in range(index.ntotal)])
    else:
        # IndexFlatIP等，直接获取
        vectors = np.asarray(index.xb).copy()
    return vectors


def load_metadata(meta_path: str):
    """读取元数据"""
    metas = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            metas.append(json.loads(line))
    return metas


def migrate_chunks(client: MilvusClient):
    """迁移chunk向量"""
    print("迁移chunk向量...")
    
    # 读取FAISS索引
    vectors = load_faiss_index("data/vector_store/index.faiss")
    metas = load_metadata("data/vector_store/metadata.jsonl")
    
    print(f"  读取到 {len(vectors)} 个向量, {len(metas)} 条元数据")
    
    # 创建Collection
    if client.has_collection(CHUNK_COLLECTION):
        client.drop_collection(CHUNK_COLLECTION)
    
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=64)
    schema.add_field("book", DataType.VARCHAR, max_length=128)
    schema.add_field("chapter", DataType.VARCHAR, max_length=256)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("char_count", DataType.INT64)
    
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="IP", metric_type="IP")
    
    client.create_collection(
        collection_name=CHUNK_COLLECTION,
        schema=schema,
        index_params=index_params
    )
    
    # 插入数据
    data = []
    for i, (vec, meta) in enumerate(zip(vectors, metas)):
        data.append({
            "pk": i,
            "vector": vec.tolist(),
            "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
            "book": meta.get("book", ""),
            "chapter": meta.get("chapter", ""),
            "text": meta.get("text", "")[:4096],
            "char_count": meta.get("char_count", 0),
        })
    
    client.insert(collection_name=CHUNK_COLLECTION, data=data)
    print(f"  插入 {len(data)} 条chunk数据")


def migrate_entities(client: MilvusClient):
    """迁移实体向量"""
    print("迁移实体向量...")
    
    vectors = load_faiss_index("data/kg/entity_index.faiss")
    metas = load_metadata("data/kg/entity_meta.jsonl")
    
    print(f"  读取到 {len(vectors)} 个向量, {len(metas)} 条元数据")
    
    if client.has_collection(ENTITY_COLLECTION):
        client.drop_collection(ENTITY_COLLECTION)
    
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("entity_id", DataType.VARCHAR, max_length=64)
    schema.add_field("name", DataType.VARCHAR, max_length=256)
    schema.add_field("type", DataType.VARCHAR, max_length=32)
    
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="IP", metric_type="IP")
    
    client.create_collection(
        collection_name=ENTITY_COLLECTION,
        schema=schema,
        index_params=index_params
    )
    
    data = []
    for i, (vec, meta) in enumerate(zip(vectors, metas)):
        data.append({
            "pk": i,
            "vector": vec.tolist(),
            "entity_id": meta.get("entity_id", f"ent_{i}"),
            "name": meta.get("name", ""),
            "type": meta.get("type", ""),
        })
    
    client.insert(collection_name=ENTITY_COLLECTION, data=data)
    print(f"  插入 {len(data)} 条实体数据")


def migrate_relations(client: MilvusClient):
    """迁移关系向量"""
    print("迁移关系向量...")
    
    vectors = load_faiss_index("data/kg/relation_index.faiss")
    metas = load_metadata("data/kg/relation_meta.jsonl")
    
    print(f"  读取到 {len(vectors)} 个向量, {len(metas)} 条元数据")
    
    if client.has_collection(RELATION_COLLECTION):
        client.drop_collection(RELATION_COLLECTION)
    
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("relation_id", DataType.VARCHAR, max_length=64)
    schema.add_field("head", DataType.VARCHAR, max_length=256)
    schema.add_field("relation", DataType.VARCHAR, max_length=128)
    schema.add_field("tail", DataType.VARCHAR, max_length=256)
    
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="IP", metric_type="IP")
    
    client.create_collection(
        collection_name=RELATION_COLLECTION,
        schema=schema,
        index_params=index_params
    )
    
    data = []
    for i, (vec, meta) in enumerate(zip(vectors, metas)):
        data.append({
            "pk": i,
            "vector": vec.tolist(),
            "relation_id": meta.get("relation_id", f"rel_{i}"),
            "head": meta.get("head", ""),
            "relation": meta.get("relation", ""),
            "tail": meta.get("tail", ""),
        })
    
    client.insert(collection_name=RELATION_COLLECTION, data=data)
    print(f"  插入 {len(data)} 条关系数据")


def main():
    # 创建Milvus Lite数据库
    Path("data/milvus").mkdir(parents=True, exist_ok=True)
    client = MilvusClient(MILVUS_DB_PATH)
    print(f"Milvus Lite 数据库: {MILVUS_DB_PATH}")
    
    # 迁移三个Collection
    migrate_chunks(client)
    migrate_entities(client)
    migrate_relations(client)
    
    # 验证
    print("\n验证迁移结果:")
    for coll in [CHUNK_COLLECTION, ENTITY_COLLECTION, RELATION_COLLECTION]:
        stats = client.get_collection_stats(coll)
        print(f"  {coll}: {stats['row_count']} 条")
    
    print("\n✅ 迁移完成!")
    client.close()


if __name__ == "__main__":
    main()
