"""
查询理解模块
意图识别 + 低层/高层关键词提取 + 查询模式分类（query_mode）
"""
from typing import Dict

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
        if 'reason' not in result:
            result['reason'] = ''

        return result


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
        print(f'  模式: {result.get("query_mode")}')
        print(f'  低层关键词: {result.get("low_level_keywords")}')
        print(f'  高层关键词: {result.get("high_level_keywords")}')
        print(f'  理由: {result.get("reason")}')
        print()
