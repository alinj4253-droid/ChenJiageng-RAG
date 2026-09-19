"""
知识图谱三元组抽取器
用LLM从每个chunk中抽取实体和关系
"""
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional

from src.config import (
    CHUNKS_DIR,
    KG_DIR,
    KG_MAX_ENTITIES_PER_CHUNK,
    KG_MAX_RELATIONS_PER_CHUNK,
)
from src.llm_client import get_client
from src.prompts import KG_EXTRACT_SYSTEM, KG_EXTRACT_USER


def extract_from_chunk(chunk: Dict, client=None) -> Optional[Dict]:
    """
    从单个chunk抽取实体和关系。
    返回: {"entities": [...], "relations": [...], "chunk_id": "..."}
    """
    if client is None:
        client = get_client()  # 用全部6个可用模型

    prompt = KG_EXTRACT_USER.format(
        max_entities=KG_MAX_ENTITIES_PER_CHUNK,
        max_relations=KG_MAX_RELATIONS_PER_CHUNK,
        text=chunk['text'][:800],  # 限制输入长度，加快响应
    )

    result = client.extract_json(prompt, system_prompt=KG_EXTRACT_SYSTEM)

    if not result:
        return None

    # 标准化输出
    entities = result.get('entities', [])
    relations = result.get('relations', [])

    # 过滤无效实体
    valid_entities = []
    for e in entities:
        name = e.get('name', '').strip()
        if name and len(name) <= 30:
            valid_entities.append({
                'name': name,
                'type': e.get('type', 'OTHER').upper(),
                'description': e.get('description', '')[:200],
                'source_chunk': chunk['chunk_id'],
            })

    # 过滤无效关系
    valid_relations = []
    for r in relations:
        head = r.get('head', '').strip()
        tail = r.get('tail', '').strip()
        relation = r.get('relation', '').strip()
        if head and tail and relation and head != tail:
            valid_relations.append({
                'head': head,
                'relation': relation,
                'tail': tail,
                'description': r.get('description', '')[:200],
                'source_chunk': chunk['chunk_id'],
            })

    return {
        'chunk_id': chunk['chunk_id'],
        'book': chunk.get('book', ''),
        'chapter': chunk.get('chapter', ''),
        'entities': valid_entities,
        'relations': valid_relations,
    }


def run_extraction(limit: Optional[int] = None, resume: bool = True, max_workers: int = 5):
    """
    批量抽取所有chunk的三元组（并发处理）。

    Args:
        limit: 只处理前N个chunk（测试用）
        resume: 是否断点续传
        max_workers: 并发线程数
    """
    # 读取chunks
    chunks_file = CHUNKS_DIR / 'chunks.jsonl'
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    if limit:
        chunks = chunks[:limit]

    print(f'共 {len(chunks)} 个chunk待处理, 并发数={max_workers}')

    # 断点续传
    output_file = KG_DIR / 'triples_raw.jsonl'
    processed = set()
    if resume and output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data.get('chunk_id', ''))
                except:
                    pass
        print(f'断点续传：已处理 {len(processed)} 个')

    # 过滤已处理的
    todo_chunks = [c for c in chunks if c['chunk_id'] not in processed]
    print(f'实际待处理: {len(todo_chunks)} 个')

    stats = {'success': 0, 'failed': 0, 'total_entities': 0, 'total_relations': 0}
    lock = threading.Lock()
    write_lock = threading.Lock()
    start_time = time.time()

    def process_chunk(chunk):
        """处理单个chunk（线程函数）"""
        client = get_client()  # 每个线程独立客户端，全部模型轮询
        try:
            result = extract_from_chunk(chunk, client)
            if result:
                # 增量写入（加锁）
                with write_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                with lock:
                    stats['success'] += 1
                    stats['total_entities'] += len(result['entities'])
                    stats['total_relations'] += len(result['relations'])
                    done = stats['success'] + stats['failed']
                    if done % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = done / elapsed * 60
                        eta = (len(todo_chunks) - done) / rate if rate > 0 else 0
                        print(f'进度: {done}/{len(todo_chunks)}, 成功={stats["success"]}, '
                              f'速度={rate:.1f}个/分钟, 预计剩余={eta:.0f}分钟')
                return True
            else:
                with lock:
                    stats['failed'] += 1
                return False
        except Exception as e:
            print(f'  错误 chunk={chunk["chunk_id"]}: {e}')
            with lock:
                stats['failed'] += 1
            return False

    # 并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): chunk for chunk in todo_chunks}
        for future in as_completed(futures):
            future.result()  # 触发异常（如果有）

    elapsed = time.time() - start_time
    print()
    print('=== 抽取完成 ===')
    print(f'成功: {stats["success"]}, 失败: {stats["failed"]}')
    print(f'总实体: {stats["total_entities"]}, 总关系: {stats["total_relations"]}')
    print(f'总耗时: {elapsed/60:.1f}分钟, 平均: {elapsed/max(stats["success"]+stats["failed"],1):.1f}秒/个')
    print(f'结果保存: {output_file}')

    return stats


if __name__ == '__main__':
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_extraction(limit=limit)
