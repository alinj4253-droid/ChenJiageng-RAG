"""
知识图谱三元组抽取器
用LLM从每个chunk中抽取实体和关系
"""
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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
    从单个chunk抽取三元组。

    LLM 原始输出为 {entities, relations}，这里将其转换为落盘格式 triples，
    并给每条三元组的头/尾补上实体类型（head_type / tail_type），
    与 data/kg/triples_raw.jsonl 的结构严格对齐。

    返回: {
        chunk_id, source_file, chapter,
        triples: [{head, head_type, relation, tail, tail_type}],
        model, timestamp
    }；无有效三元组时返回 None。
    """
    if client is None:
        client = get_client()  # 用全部可用模型

    prompt = KG_EXTRACT_USER.format(
        max_entities=KG_MAX_ENTITIES_PER_CHUNK,
        max_relations=KG_MAX_RELATIONS_PER_CHUNK,
        text=chunk['text'][:800],  # 限制输入长度，加快响应
    )

    result = client.extract_json(prompt, system_prompt=KG_EXTRACT_SYSTEM, max_tokens=4096)

    if not isinstance(result, dict) or not isinstance(result.get('entities'), list) or not isinstance(result.get('relations'), list):
        return None

    entities = result['entities']
    relations = result['relations']

    # 实体名 -> 标准化类型（用于给三元组头尾补类型）
    type_map = {}
    for e in entities:
        name = e.get('name', '').strip()
        if name and len(name) <= 30:
            type_map[name] = e.get('type', 'OTHER').upper()

    # 组装三元组：补头/尾类型，去重，过滤无效与自环
    triples = []
    seen = {}
    for r in relations:
        head = r.get('head', '').strip()
        tail = r.get('tail', '').strip()
        relation = r.get('relation', '').strip()
        if not head or not tail or not relation or head == tail:
            continue
        if len(head) > 30 or len(tail) > 30:
            continue
        key = (head, relation, tail)
        description = (r.get('description') or '').strip()
        if key in seen:
            previous = seen[key]
            if len(description) > len(previous['description']):
                previous['description'] = description
            continue
        triples.append({
            'head': head,
            'head_type': type_map.get(head, 'OTHER'),
            'relation': relation,
            'tail': tail,
            'tail_type': type_map.get(tail, 'OTHER'),
            'description': description,
        })

        seen[key] = triples[-1]

    # 保留 LLM 抽取的实体原始信息（name / type / description），
    # 供 kg_normalizer 直接使用，避免从三元组头/尾重建时丢失类型与描述。
    entity_records = []
    for e in entities:
        name = e.get('name', '').strip()
        if not name or len(name) > 30:
            continue
        entity_records.append({
            'name': name,
            'type': e.get('type', 'OTHER').strip().upper(),
            'description': (e.get('description', '') or '').strip(),
        })

    # A valid empty extraction is still a completed chunk (e.g. navigation text).
    # Persist it so resume does not retry such chunks forever.

    return {
        'chunk_id': chunk['chunk_id'],
        'source_file': chunk.get('book', ''),
        'chapter': chunk.get('chapter', ''),
        'entities': entity_records,
        'triples': triples,
        'model': getattr(client, 'last_model', '') or '',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def run_extraction(limit: Optional[int] = None, resume: bool = True, max_workers: int = 5,
                   max_consecutive_failures: int = 20):
    """
    批量抽取所有chunk的三元组（并发处理）。

    Args:
        limit: 只处理前N个chunk（测试用）
        resume: 是否断点续传
        max_workers: 并发线程数
        max_consecutive_failures: 服务持续失败时停止提交实际请求，保留断点
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

    if max_consecutive_failures <= 0:
        raise ValueError('max_consecutive_failures must be positive')
    stats = {'success': 0, 'failed': 0, 'total_triples': 0, 'skipped': 0}
    lock = threading.Lock()
    write_lock = threading.Lock()
    stop_event = threading.Event()
    consecutive_failures = 0
    start_time = time.time()

    def record_failure():
        nonlocal consecutive_failures
        with lock:
            stats['failed'] += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                stop_event.set()

    def process_chunk(chunk):
        """处理单个chunk（线程函数）"""
        nonlocal consecutive_failures
        if stop_event.is_set():
            with lock:
                stats['skipped'] += 1
            return False
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
                    consecutive_failures = 0
                    stats['total_triples'] += len(result['triples'])
                    done = stats['success'] + stats['failed']
                    if done % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = done / elapsed * 60
                        eta = (len(todo_chunks) - done) / rate if rate > 0 else 0
                        print(f'进度: {done}/{len(todo_chunks)}, 成功={stats["success"]}, '
                              f'速度={rate:.1f}个/分钟, 预计剩余={eta:.0f}分钟')
                return True
            else:
                record_failure()
                return False
        except Exception as e:
            print(f'  错误 chunk={chunk["chunk_id"]}: {e}')
            record_failure()
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
    if stats['skipped']:
        print(f'服务连续失败，未请求 {stats["skipped"]} 个chunk；恢复服务后重跑可继续')
    print(f'三元组总数: {stats["total_triples"]}')
    print(f'总耗时: {elapsed/60:.1f}分钟, 平均: {elapsed/max(stats["success"]+stats["failed"],1):.1f}秒/个')
    print(f'结果保存: {output_file}')

    return stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('limit', nargs='?', type=int)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    if args.workers <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error('limit and workers must be positive')
    stats = run_extraction(limit=args.limit, max_workers=args.workers)
    if stats['failed']:
        raise SystemExit(1)  # rerun resumes only failed/unprocessed chunks
