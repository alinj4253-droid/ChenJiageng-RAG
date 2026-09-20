"""
QueryRewriter 测试（LLM 全部 mock）
"""
import pytest
from unittest.mock import MagicMock

from src.query_rewriter import QueryRewriter


def _make_rewriter(json_response):
    client = MagicMock()
    client.extract_json.return_value = json_response
    return QueryRewriter(client=client), client


def _ctx(text="一些证据"):
    return [{"chunk_id": "c1", "book": "书", "chapter": "章", "text": text}]


class TestQueryRewriter:

    def test_valid_rewrite(self):
        """正常返回改写查询与理由"""
        resp = {
            "rewritten_query": "陈嘉庚创办厦门大学的经费来源与办学资金",
            "reason": "补充经费这一缺失维度",
        }
        rewriter, client = _make_rewriter(resp)
        result = rewriter.rewrite(
            original_query="陈嘉庚为什么创办厦门大学？",
            current_query="陈嘉庚为什么创办厦门大学？",
            missing=["缺少创办经费来源"],
            contexts=_ctx(),
        )
        assert result is not None
        assert "经费" in result.query
        assert result.reason == "补充经费这一缺失维度"
        client.extract_json.assert_called_once()

    def test_prompt_contains_missing_and_original(self):
        """改写 prompt 必须包含原始问题、当前查询和缺失信息点"""
        rewriter, client = _make_rewriter({"rewritten_query": "新查询"})
        rewriter.rewrite(
            original_query="原始问题XYZ",
            current_query="当前查询ABC",
            missing=["缺失点一", "缺失点二"],
            contexts=_ctx("证据正文"),
        )
        prompt = client.extract_json.call_args[0][0]
        assert "原始问题XYZ" in prompt
        assert "当前查询ABC" in prompt
        assert "缺失点一" in prompt
        assert "缺失点二" in prompt

    def test_empty_response_returns_none(self):
        rewriter, _ = _make_rewriter({})
        assert rewriter.rewrite("o", "c", ["m"], _ctx()) is None

    def test_blank_rewritten_query_returns_none(self):
        rewriter, _ = _make_rewriter({"rewritten_query": "   "})
        assert rewriter.rewrite("o", "c", ["m"], _ctx()) is None

    def test_missing_rewritten_query_field(self):
        rewriter, _ = _make_rewriter({"reason": "没有query字段"})
        assert rewriter.rewrite("o", "c", None, _ctx()) is None

    def test_empty_missing_rendered_placeholder(self):
        """missing 为空也能正常构造 prompt（不报错）"""
        rewriter, client = _make_rewriter({"rewritten_query": "q"})
        result = rewriter.rewrite("o", "o", [], _ctx())
        assert result.query == "q"
        prompt = client.extract_json.call_args[0][0]
        assert "未明确列出" in prompt
