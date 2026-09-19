"""
知识图谱归一化器
实体别名归并、类型校正、噪声过滤
"""
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set

from src.config import KG_DIR


# 实体类型映射（统一为标准类型）
TYPE_MAPPING = {
    '人物': 'PERSON', '人': 'PERSON', 'person': 'PERSON', 'PERSON': 'PERSON',
    '组织': 'ORG', '机构': 'ORG', '团体': 'ORG', 'org': 'ORG', 'ORG': 'ORG',
    '教育机构': 'ORG', '政府': 'ORG', '政府机构': 'ORG', '学校': 'ORG',
    '地点': 'LOC', '地方': 'LOC', '区域': 'LOC', 'loc': 'LOC', 'LOC': 'LOC',
    '国家': 'LOC', '城市': 'LOC',
    '事件': 'EVENT', 'event': 'EVENT', 'EVENT': 'EVENT', '历史事件': 'EVENT',
    '概念': 'CONCEPT', '思想': 'CONCEPT', '主义': 'CONCEPT', 'concept': 'CONCEPT', 'CONCEPT': 'CONCEPT',
    '时间': 'TIME', '年代': 'TIME', 'time': 'TIME', 'TIME': 'TIME',
    '作品': 'WORK', '著作': 'WORK', '文件': 'WORK', 'work': 'WORK', 'WORK': 'WORK',
    '身份': 'OTHER', '其他': 'OTHER', 'other': 'OTHER', 'OTHER': 'OTHER',
}

# 噪声实体黑名单（太通用或无意义）
NOISE_ENTITIES = {
    '余', '我', '他', '她', '他们', '我们', '本人', '先生', '女士',
    '政府', '国家', '人民', '社会', '世界', '中国', '南洋', '华侨',
    '教育', '实业', '经济', '政治', '文化', '历史',
}

# 别名映射（手动指定常见别名）
MANUAL_ALIASES = {
    '陈嘉庚': ['陈嘉庚先生', '嘉庚', '陈校主', '校主'],
    '厦门大学': ['厦大'],
    '集美学校': ['集校', '集美'],
    '南侨总会': ['南洋华侨筹赈祖国难民总会', '南侨筹赈总会'],
    '新加坡': ['星洲', '星加坡'],
    '马来西亚': ['马来亚'],
    '印度尼西亚': ['印尼', '荷印'],
}


def normalize_entity_name(name: str) -> str:
    """标准化实体名称：去空格、去标点、统一全半角"""
    name = name.strip()
    # 去掉常见后缀
    for suffix in ['先生', '女士', '君', '兄', '公']:
        if name.endswith(suffix) and len(name) > len(suffix) + 1:
            name = name[:-len(suffix)]
    return name.strip()


def normalize_entity_type(etype: str) -> str:
    """标准化实体类型"""
    etype = etype.strip().upper()
    return TYPE_MAPPING.get(etype, TYPE_MAPPING.get(etype.lower(), 'OTHER'))


def is_noise_entity(name: str, etype: str) -> bool:
    """判断是否为噪声实体"""
    if len(name) < 2:
        return True
    if name in NOISE_ENTITIES:
        return True
    # 纯数字
    if re.match(r'^\d+$', name):
        return True
    return False


def build_alias_map(entities: List[Dict]) -> Dict[str, str]:
    """
    构建别名映射：相似实体名归并。
    策略：1. 手动别名映射 2. 包含关系（短名是长名的子串且长度差<=2）
    """
    alias_map = {}  # 别名 -> 标准名

    # 1. 手动别名
    for standard, aliases in MANUAL_ALIASES.items():
        for alias in aliases:
            alias_map[alias] = standard

    # 2. 基于包含关系的自动归并
    entity_names = list(set(e['name'] for e in entities))
    entity_names.sort(key=len)  # 短名在前

    for i, short in enumerate(entity_names):
        if short in alias_map or len(short) < 3:
            continue
        for long in entity_names[i+1:]:
            if short in long and len(long) - len(short) <= 3:
                alias_map[long] = short

    return alias_map


def normalize_kg():
    """执行知识图谱归一化"""
    # 读取原始抽取结果
    raw_file = KG_DIR / 'triples_raw.jsonl'
    if not raw_file.exists():
        print(f'错误: {raw_file} 不存在，请先运行抽取')
        return

    raw_results = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                raw_results.append(json.loads(line))
            except:
                pass

    print(f'读取 {len(raw_results)} 个chunk的抽取结果')

    # 收集所有实体和关系（从triples格式提取）
    all_entities = {}  # name -> entity dict
    all_relations = []
    entity_chunk_map = defaultdict(set)  # name -> set of chunk_ids

    for result in raw_results:
        chunk_id = result.get('chunk_id', '')
        # 从triples中提取实体和关系
        for t in result.get('triples', []):
            head = normalize_entity_name(t['head'])
            tail = normalize_entity_name(t['tail'])
            head_type = normalize_entity_type(t.get('head_type', 'OTHER'))
            tail_type = normalize_entity_type(t.get('tail_type', 'OTHER'))
            relation = t.get('relation', '').strip()
            if not relation:
                continue

            # 过滤噪声
            if is_noise_entity(head, '') or is_noise_entity(tail, ''):
                continue
            if head == tail:
                continue
            if len(relation) < 1:
                continue

            # 添加head实体
            if head not in all_entities:
                all_entities[head] = {
                    'name': head,
                    'type': head_type,
                    'description': '',
                    'source_chunks': set(),
                }
            all_entities[head]['source_chunks'].add(chunk_id)
            entity_chunk_map[head].add(chunk_id)

            # 添加tail实体
            if tail not in all_entities:
                all_entities[tail] = {
                    'name': tail,
                    'type': tail_type,
                    'description': '',
                    'source_chunks': set(),
                }
            all_entities[tail]['source_chunks'].add(chunk_id)
            entity_chunk_map[tail].add(chunk_id)

            # 添加关系
            all_relations.append({
                'head': head,
                'relation': relation,
                'tail': tail,
                'description': '',
                'source_chunk': chunk_id,
            })

    print(f'原始实体: {len(all_entities)}, 原始关系: {len(all_relations)}')

    # 别名归并
    alias_map = build_alias_map(list(all_entities.values()))
    print(f'别名映射: {len(alias_map)} 条')

    # 应用别名映射
    def resolve_name(name):
        return alias_map.get(name, name)

    # 归并实体
    merged_entities = {}
    for name, entity in all_entities.items():
        standard = resolve_name(name)
        if standard not in merged_entities:
            merged_entities[standard] = {
                'name': standard,
                'type': entity['type'],
                'description': entity['description'],
                'aliases': set(),
                'source_chunks': set(entity['source_chunks']),
            }
        else:
            # 合并
            if len(entity['description']) > len(merged_entities[standard]['description']):
                merged_entities[standard]['description'] = entity['description']
            merged_entities[standard]['source_chunks'].update(entity['source_chunks'])
            if name != standard:
                merged_entities[standard]['aliases'].add(name)

    # 归并关系（去重）
    merged_relations = {}
    for r in all_relations:
        head = resolve_name(r['head'])
        tail = resolve_name(r['tail'])
        if head == tail:
            continue
        key = (head, r['relation'], tail)
        if key not in merged_relations:
            merged_relations[key] = {
                'head': head,
                'relation': r['relation'],
                'tail': tail,
                'description': r['description'],
                'source_chunks': set(),
            }
        merged_relations[key]['source_chunks'].add(r['source_chunk'])

    # 过滤：只保留至少出现在一个关系中的实体（或者出现次数>=2的实体）
    relation_entities = set()
    for r in merged_relations.values():
        relation_entities.add(r['head'])
        relation_entities.add(r['tail'])

    final_entities = {}
    for name, entity in merged_entities.items():
        if name in relation_entities or len(entity['source_chunks']) >= 2:
            entity['aliases'] = list(entity['aliases'])
            entity['source_chunks'] = list(entity['source_chunks'])
            entity['chunk_count'] = len(entity['source_chunks'])
            final_entities[name] = entity

    final_relations = []
    for r in merged_relations.values():
        if r['head'] in final_entities and r['tail'] in final_entities:
            r['source_chunks'] = list(r['source_chunks'])
            r['chunk_count'] = len(r['source_chunks'])
            final_relations.append(r)

    print(f'归一化后实体: {len(final_entities)}, 关系: {len(final_relations)}')

    # 保存
    entities_file = KG_DIR / 'entities.jsonl'
    with open(entities_file, 'w', encoding='utf-8') as f:
        for entity in final_entities.values():
            f.write(json.dumps(entity, ensure_ascii=False) + '\n')

    relations_file = KG_DIR / 'triples.jsonl'
    with open(relations_file, 'w', encoding='utf-8') as f:
        for r in final_relations:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 统计
    type_counts = defaultdict(int)
    for e in final_entities.values():
        type_counts[e['type']] += 1

    stats = {
        'total_entities': len(final_entities),
        'total_relations': len(final_relations),
        'entity_types': dict(type_counts),
        'alias_count': len(alias_map),
        'raw_chunks': len(raw_results),
    }
    stats_file = KG_DIR / 'kg_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f'\n=== 归一化完成 ===')
    print(f'实体: {len(final_entities)}, 关系: {len(final_relations)}')
    print(f'实体类型分布: {dict(type_counts)}')
    print(f'保存: {entities_file}, {relations_file}')

    return stats


if __name__ == '__main__':
    normalize_kg()
