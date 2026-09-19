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
        分析用户查询，返回结构化结果。

        返回: {
            "original_query": 原始查询,
            "rewritten_query": 改写后的查询（更适合检索）,
            "query_type": 事实型/解释型/比较型/列表型/开放型,
            "entities": [识别到的实体],
            "sub_queries": [子查询列表]（复杂查询才分解）,
            "time_range": 时间范围（如有）,
        }
        """
        prompt = QUERY_ANALYSIS_USER.format(query=query)
        result = self.client.extract_json(prompt, system_prompt=QUERY_ANALYSIS_SYSTEM)

        if not result:
            # 降级：返回原始查询
            return {
                'original_query': query,
                'rewritten_query': query,
                'query_type': '开放型',
                'entities': [],
                'sub_queries': [query],
                'time_range': None,
            }

        result['original_query'] = query
        # 确保sub_queries存在
        if 'sub_queries' not in result or not result['sub_queries']:
            result['sub_queries'] = [result.get('rewritten_query', query)]
        if 'entities' not in result:
            result['entities'] = []

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
