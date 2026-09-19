"""
Neo4j知识图谱导入脚本
把实体和关系导入Neo4j
"""
import json
import os
from neo4j import GraphDatabase

# Neo4j连接配置（从环境变量 / .env 读取，不要硬编码密码）
URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

# 数据文件路径
ENTITIES_FILE = "data/kg/entities.jsonl"
TRIPLES_FILE = "data/kg/triples.jsonl"


def clear_data(tx):
    """清空现有数据"""
    tx.run("MATCH (n) DETACH DELETE n")
    print("已清空现有数据")


def import_entities(tx, entities):
    """批量导入实体"""
    query = """
    UNWIND $entities AS e
    MERGE (n:Entity {name: e.name})
    SET n.type = e.type,
        n.description = e.description,
        n.chunk_count = e.chunk_count
    """
    tx.run(query, entities=entities)


def import_relations(tx, relations):
    """批量导入关系"""
    query = """
    UNWIND $relations AS r
    MERGE (a:Entity {name: r.source_name})
    MERGE (b:Entity {name: r.target_name})
    MERGE (a)-[rel:RELATED {type: r.relation}]->(b)
    SET rel.chunk_count = r.chunk_count
    """
    tx.run(query, relations=relations)


def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    with driver.session() as session:
        # 1. 清空数据
        print("1. 清空现有数据...")
        session.execute_write(clear_data)

        # 2. 导入实体
        print("2. 导入实体...")
        entities = []
        with open(ENTITIES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                e = json.loads(line.strip())
                entities.append({
                    "name": e["name"],
                    "type": e.get("type", "OTHER"),
                    "description": e.get("description", ""),
                    "chunk_count": e.get("chunk_count", 0)
                })
        print(f"   共 {len(entities)} 个实体")

        # 批量导入，每批500个
        batch_size = 500
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            session.execute_write(import_entities, batch)
            print(f"   已导入 {min(i+batch_size, len(entities))}/{len(entities)}")

        # 3. 导入关系
        print("3. 导入关系...")
        relations = []
        with open(TRIPLES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                t = json.loads(line.strip())
                relations.append({
                    "source_name": t["head"],
                    "target_name": t["tail"],
                    "relation": t["relation"],
                    "chunk_count": t.get("chunk_count", 1)
                })
        print(f"   共 {len(relations)} 条关系")

        # 批量导入，每批500条
        for i in range(0, len(relations), batch_size):
            batch = relations[i:i+batch_size]
            session.execute_write(import_relations, batch)
            print(f"   已导入 {min(i+batch_size, len(relations))}/{len(relations)}")

        # 4. 创建索引
        print("4. 创建索引...")
        session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
        session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)")
        print("   索引创建完成")

        # 5. 验证
        print("5. 验证数据...")
        result = session.run("MATCH (n:Entity) RETURN count(n) AS count")
        entity_count = result.single()["count"]
        result = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS count")
        relation_count = result.single()["count"]
        print(f"   实体数: {entity_count}")
        print(f"   关系数: {relation_count}")

    driver.close()
    print("\n✅ 导入完成！")


if __name__ == "__main__":
    main()
