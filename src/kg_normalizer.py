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

    # ---- 从 LLM 原始 entities 字段收集实体信息（type / description）----
    # 这是实体类型与描述的权威来源；三元组中 head_type/tail_type 仅作回退。
    entity_info: Dict[str, Dict] = {}  # normalized_name -> {type, description}

    def _register_entity_info(name: str, etype: str, desc: str):
        """合并同一实体在不同 chunk 中的信息：优先非 OTHER 类型、保留最长描述"""
        std = normalize_entity_name(name)
        if not std:
            return
        etype = normalize_entity_type(etype) if etype else 'OTHER'
        desc = (desc or '').strip()
        if std not in entity_info:
            entity_info[std] = {'type': etype, 'description': desc}
        else:
            existing = entity_info[std]
            # 类型升级：OTHER -> 任意具体类型
            if existing['type'] == 'OTHER' and etype != 'OTHER':
                existing['type'] = etype
            # 描述：保留更长的
            if len(desc) > len(existing['description']):
                existing['description'] = desc

    for result in raw_results:
        for e in result.get('entities', []):
            _register_entity_info(
                e.get('name', ''),
                e.get('type', 'OTHER'),
                e.get('description', ''),
            )

    # 收集所有实体和关系（从 triples 中提取）
    all_entities = {}  # name -> entity dict
    all_relations = []
    entity_chunk_map = defaultdict(set)  # name -> set of chunk_ids

    for result in raw_results:
        chunk_id = result.get('chunk_id', '')
        # 从 triples 中提取实体和关系
        for t in result.get('triples', []):
            head = normalize_entity_name(t['head'])
            tail = normalize_entity_name(t['tail'])
            # 类型优先从 entity_info（LLM entities 字段）取，回退到三元组中的 head_type
            head_info = entity_info.get(head, {})
            tail_info = entity_info.get(tail, {})
            head_type = head_info.get('type') or normalize_entity_type(t.get('head_type', 'OTHER'))
            tail_type = tail_info.get('type') or normalize_entity_type(t.get('tail_type', 'OTHER'))
            head_desc = head_info.get('description', '')
            tail_desc = tail_info.get('description', '')
            relation = t.get('relation', '').strip()
            rel_description = (t.get('description', '') or '').strip()
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
                    'description': head_desc,
                    'source_chunks': set(),
                }
            all_entities[head]['source_chunks'].add(chunk_id)
            entity_chunk_map[head].add(chunk_id)

            # 添加tail实体
            if tail not in all_entities:
                all_entities[tail] = {
                    'name': tail,
                    'type': tail_type,
                    'description': tail_desc,
                    'source_chunks': set(),
                }
            all_entities[tail]['source_chunks'].add(chunk_id)
            entity_chunk_map[tail].add(chunk_id)

            # 添加关系（保留关系描述）
            all_relations.append({
                'head': head,
                'relation': relation,
                'tail': tail,
                'description': rel_description,
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
            # 合并：类型升级（OTHER → 具体类型）、保留最长描述、聚合 source_chunks
            existing = merged_entities[standard]
            if existing['type'] == 'OTHER' and entity['type'] != 'OTHER':
                existing['type'] = entity['type']
            if len(entity['description']) > len(existing['description']):
                existing['description'] = entity['description']
            existing['source_chunks'].update(entity['source_chunks'])
        # 无论新建还是已存在，只要原名与标准名不同就记录为别名
        # （修复：标准实体首次由别名创建时，别名此前会丢失）
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
        else:
            # 保留最长的关系描述
            if len(r['description']) > len(merged_relations[key]['description']):
                merged_relations[key]['description'] = r['description']
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
    entities_with_desc = 0
    for e in final_entities.values():
        type_counts[e['type']] += 1
        if e.get('description'):
            entities_with_desc += 1

    stats = {
        'total_entities': len(final_entities),
        'total_relations': len(final_relations),
        'entity_types': dict(type_counts),
        'entities_with_description': entities_with_desc,
        'description_coverage': round(entities_with_desc / len(final_entities), 4) if final_entities else 0,
        'other_type_count': type_counts.get('OTHER', 0),
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
