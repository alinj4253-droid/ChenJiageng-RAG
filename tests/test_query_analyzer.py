"""
QueryAnalyzer 测试
覆盖：短问题、事实问题、实体关系问题、复杂混合问题、LLM返回非法JSON、LLM调用失败
重点验证 query_mode / low_level_keywords / high_level_keywords / reason 永远有默认值
"""
import pytest
from unittest.mock import patch, MagicMock

from src.query_analyzer import QueryAnalyzer


def _make_analyzer_with_response(json_response):
    """构造一个 LLM 返回固定 JSON 的 QueryAnalyzer"""
    client = MagicMock()
    client.extract_json.return_value = json_response
    analyzer = QueryAnalyzer.__new__(QueryAnalyzer)
    analyzer.client = client
    return analyzer, client


class TestQueryAnalyzerNormal:
    """正常 LLM 返回"""

    def test_naive_short_question(self):
        """短事实问题 → naive 模式"""
        resp = {
            "low_level_keywords": ["陈嘉庚"],
            "high_level_keywords": [],
            "query_mode": "naive",
            "reason": "简单事实查询",
        }
        analyzer, _ = _make_analyzer_with_response(resp)
        result = analyzer.analyze("陈嘉庚是谁？")

        assert result["query_mode"] == "naive"
        assert "陈嘉庚" in result["low_level_keywords"]
        assert result["high_level_keywords"] == []
        assert result["reason"] == "简单事实查询"
        assert result["original_query"] == "陈嘉庚是谁？"

    def test_local_entity_question(self):
        """特定实体详细查询 → local 模式"""
        resp = {
            "low_level_keywords": ["陈嘉庚", "厦门大学"],
            "high_level_keywords": ["创办"],
            "query_mode": "local",
            "reason": "围绕特定实体展开",
        }
        analyzer, _ = _make_analyzer_with_response(resp)
        result = analyzer.analyze("陈嘉庚创办厦门大学的经过是怎样的？")

        assert result["query_mode"] == "local"
        assert "厦门大学" in result["low_level_keywords"]

    def test_global_relation_question(self):
        """全局性/主题性问题 → global 模式"""
        resp = {
            "low_level_keywords": [],
            "high_level_keywords": ["教育思想", "救国理念"],
            "query_mode": "global",
            "reason": "抽象主题问题",
        }
        analyzer, _ = _make_analyzer_with_response(resp)
        result = analyzer.analyze("陈嘉庚的教育救国思想有什么内涵？")

        assert result["query_mode"] == "global"
        assert len(result["high_level_keywords"]) >= 1

    def test_hybrid_complex_question(self):
        """复杂混合问题 → hybrid 模式"""
        resp = {
            "low_level_keywords": ["陈嘉庚", "李光前"],
            "high_level_keywords": ["实业关系", "传承"],
            "query_mode": "hybrid",
            "reason": "涉及多实体与抽象关系",
        }
        analyzer, _ = _make_analyzer_with_response(resp)
        result = analyzer.analyze("陈嘉庚和李光前在实业上是什么关系，如何传承？")

        assert result["query_mode"] == "hybrid"
        assert len(result["low_level_keywords"]) >= 1
        assert len(result["high_level_keywords"]) >= 1


class TestQueryAnalyzerDegradation:
    """异常降级：保证永远有默认值"""

    def test_llm_returns_empty_dict(self):
        """LLM 返回空 dict（解析失败）→ 降级 hybrid"""
        analyzer, _ = _make_analyzer_with_response({})
        result = analyzer.analyze("任意问题")

        assert result["query_mode"] == "hybrid"
        assert result["low_level_keywords"] == []
        assert result["high_level_keywords"] == []
        assert result["reason"] == ""
        assert result["original_query"] == "任意问题"

    def test_llm_returns_none(self):
        """LLM 调用失败返回 None/空 → 降级 hybrid"""
        client = MagicMock()
        client.extract_json.return_value = {}
        analyzer = QueryAnalyzer.__new__(QueryAnalyzer)
        analyzer.client = client

        result = analyzer.analyze("测试问题")
        assert result["query_mode"] == "hybrid"

    def test_missing_keywords_fields_filled(self):
        """LLM 返回缺少 keywords 字段 → 自动补空列表"""
        resp = {"query_mode": "local", "reason": "部分返回"}
        analyzer, _ = _make_analyzer_with_response(resp)
        result = analyzer.analyze("测试")

        assert result["low_level_keywords"] == []
        assert result["high_level_keywords"] == []
        assert result["query_mode"] == "local"

    def test_missing_query_mode_defaults_hybrid(self):
        """LLM 返回缺少 query_mode → 默认 hybrid"""
        resp = {"low_level_keywords": ["X"], "high_level_keywords": []}
        analyzer, _ = _make_analyzer_with_response(resp)
        result = analyzer.analyze("测试")

        assert result["query_mode"] == "hybrid"

    def test_result_always_has_required_schema(self):
        """任何情况下结果都包含 Schema 要求的全部字段"""
        analyzer, _ = _make_analyzer_with_response({})
        result = analyzer.analyze("q")

        for key in ("original_query", "low_level_keywords",
                    "high_level_keywords", "query_mode", "reason"):
            assert key in result, f"缺少字段 {key}"
