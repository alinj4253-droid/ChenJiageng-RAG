"""
查询理解模块
查询改写、实体识别、查询分类、子查询分解
"""
import json
from typing import Dict, List, Optional

from src.llm_client import get_client
from src.prompts import QUERY_ANALYSIS_SYSTEM, QUERY_ANALYSIS_USER


class QueryAnalyzer:
    """查询分析器"""

    def __init__(self):
        self.client = get_client()

    def analyze(self, query: str) -> Dict:
        """
        分析用户查询，做意图识别与关键词提取。

        返回: {
            "original_query": 原始查询,
            "low_level_keywords": [低层关键词（具体实体/人名/地名）],
            "high_level_keywords": [高层关键词（抽象主题/概念）],
            "query_mode": "naive|local|global|hybrid",
            "reason": "简短说明",
        }
        """
        prompt = QUERY_ANALYSIS_USER.format(query=query)
        result = self.client.extract_json(prompt, system_prompt=QUERY_ANALYSIS_SYSTEM)

        if not result:
            # 降级：直接用原问题做混合检索
            return {
                'original_query': query,
                'low_level_keywords': [],
                'high_level_keywords': [],
                'query_mode': 'hybrid',
                'reason': '',
            }

        result['original_query'] = query
        # 补齐可能缺失的字段
        if 'low_level_keywords' not in result:
            result['low_level_keywords'] = []
        if 'high_level_keywords' not in result:
            result['high_level_keywords'] = []
        if 'query_mode' not in result:
            result['query_mode'] = 'hybrid'

        return result

    def rewrite(self, query: str) -> str:
        """仅做查询改写"""
        result = self.analyze(query)
        return result.get('rewritten_query', query)


if __name__ == '__main__':
    # 测试
    analyzer = QueryAnalyzer()
    test_queries = [
        '陈嘉庚创办了哪些学校？',
        '陈嘉庚的教育思想是什么？',
        '南侨回忆录主要讲了什么内容？',
        '陈嘉庚和李光前是什么关系？',
    ]
    for q in test_queries:
        result = analyzer.analyze(q)
        print(f'查询: {q}')
        print(f'  改写: {result.get("rewritten_query")}')
        print(f'  类型: {result.get("query_type")}')
        print(f'  实体: {result.get("entities")}')
        print(f'  子查询: {result.get("sub_queries")}')
        print()
