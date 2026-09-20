"""
EvidenceJudge 测试（LLM 全部 mock）
覆盖：证据充分 / 不足 / 空证据 / 裁判失败 fail-open / 字段解析。
"""
import pytest
from unittest.mock import MagicMock

from src.evidence_judge import EvidenceJudge, EvidenceJudgement


def _make_judge(json_response):
    client = MagicMock()
    client.extract_json.return_value = json_response
    return EvidenceJudge(client=client), client


def _ctx(text, cid="c1", book="陈嘉庚传", chapter="第一章"):
    return {"chunk_id": cid, "book": book, "chapter": chapter,
            "text": text, "score": 0.9}


class TestEvidenceJudge:

    def test_insufficient_when_missing_motivation(self):
        """问题问动机，证据只提到创办厦大、完全没有动机 → insufficient"""
        resp = {
            "sufficient": False,
            "missing": ["缺少陈嘉庚创办厦门大学的动机与背景"],
            "reason": "证据仅陈述创办事实，未解释原因",
        }
        judge, client = _make_judge(resp)
        contexts = [_ctx("1921年陈嘉庚创办了厦门大学。", "c1")]

        result = judge.judge("陈嘉庚为什么创办厦门大学？", contexts)

        assert result.sufficient is False
        assert len(result.missing) >= 1
        assert "动机" in result.missing[0] or "背景" in result.missing[0]
        client.extract_json.assert_called_once()

    def test_sufficient_when_core_info_covered(self):
        """证据覆盖背景、教育理念、社会环境 → sufficient"""
        resp = {
            "sufficient": True,
            "missing": [],
            "reason": "证据涵盖创办背景、教育理念与社会环境",
        }
        judge, _ = _make_judge(resp)
        contexts = [
            _ctx("陈嘉庚痛感国家积弱，认为教育为立国之本。", "c1"),
            _ctx("他在南洋实业有成后回乡兴学，创办厦门大学培养人才。", "c2"),
            _ctx("当时民初社会急需专门学校，他决心独资创办。", "c3"),
        ]
        result = judge.judge("陈嘉庚为什么创办厦门大学？", contexts)
        assert result.sufficient is True
        assert result.missing == []

    def test_empty_contexts_is_insufficient_without_llm(self):
        """空证据直接判不充分，且不调用 LLM"""
        judge, client = _make_judge({"sufficient": True})
        result = judge.judge("任意问题", [])
        assert result.sufficient is False
        assert "未检索到" in result.missing[0]
        client.extract_json.assert_not_called()

    def test_none_contexts_is_insufficient(self):
        judge, _ = _make_judge({})
        result = judge.judge("q", None)
        assert result.sufficient is False

    def test_judge_failure_fails_open(self):
        """Judge LLM 返回空 / 非法 → fail-open，判充分以免阻塞生成"""
        judge, _ = _make_judge({})
        contexts = [_ctx("一些证据内容")]
        result = judge.judge("问题", contexts)
        assert result.sufficient is True

    def test_missing_normalized_to_list(self):
        """missing 返回非列表时归一化为字符串列表"""
        judge, _ = _make_judge({"sufficient": False, "missing": "缺少经费来源"})
        result = judge.judge("q", [_ctx("x")])
        assert result.missing == ["缺少经费来源"]

    def test_missing_empty_entries_filtered(self):
        judge, _ = _make_judge({"sufficient": False, "missing": ["有效缺失点", "", None]})
        result = judge.judge("q", [_ctx("x")])
        assert result.missing == ["有效缺失点"]

    def test_to_dict_serializable(self):
        j = EvidenceJudgement(sufficient=False, missing=["a"], reason="r")
        d = j.to_dict()
        assert d == {"sufficient": False, "missing": ["a"], "reason": "r"}

    def test_evidence_formatting_includes_sources(self):
        """证据文本应包含编号与书名章节"""
        judge, client = _make_judge({"sufficient": True})
        contexts = [_ctx("正文内容", "c9", book="南侨回忆录", chapter="教育篇")]
        judge.judge("问题", contexts)

        prompt = client.extract_json.call_args[0][0]
        assert "[证据1]" in prompt
        assert "南侨回忆录" in prompt
        assert "正文内容" in prompt
