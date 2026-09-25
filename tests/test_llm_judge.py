# -*- coding: utf-8 -*-
"""
llm_judge 测试（LLM 全部 mock）
"""
from unittest.mock import MagicMock

from evaluation.llm_judge import LLMJudge


def _client(json_response):
    client = MagicMock()
    client.extract_json.return_value = json_response
    return client


class TestFaithfulness:

    def test_faithfulness_score_extracted(self):
        judge = LLMJudge(client=_client({"score": 0.8, "reason": "大部分有依据"}))
        contexts = [{"text": "陈嘉庚创办了厦门大学。", "book": "书", "chapter": "章"}]
        result = judge.faithfulness("陈嘉庚创办了厦门大学。", contexts)
        assert result is not None
        assert result["score"] == 0.8
        assert "依据" in result["reason"]

    def test_faithfulness_clamped_to_0_1(self):
        judge = LLMJudge(client=_client({"score": 1.5, "reason": ""}))
        contexts = [{"text": "test", "book": "", "chapter": ""}]
        result = judge.faithfulness("answer", contexts)
        assert result["score"] == 1.0

    def test_faithfulness_negative_clamped(self):
        judge = LLMJudge(client=_client({"score": -0.5, "reason": ""}))
        contexts = [{"text": "test", "book": "", "chapter": ""}]
        result = judge.faithfulness("answer", contexts)
        assert result["score"] == 0.0

    def test_faithfulness_empty_answer_returns_none(self):
        judge = LLMJudge(client=_client({"score": 0.5}))
        assert judge.faithfulness("", [{"text": "ctx"}]) is None

    def test_faithfulness_empty_contexts_returns_none(self):
        judge = LLMJudge(client=_client({"score": 0.5}))
        assert judge.faithfulness("answer", []) is None

    def test_faithfulness_llm_failure_returns_none(self):
        judge = LLMJudge(client=_client({}))
        contexts = [{"text": "test", "book": "", "chapter": ""}]
        assert judge.faithfulness("answer", contexts) is None


class TestAnswerCorrectness:

    def test_correctness_score_extracted(self):
        judge = LLMJudge(client=_client({"score": 0.9, "reason": "答案正确"}))
        result = judge.answer_correctness("问题？", "答案", "标准答案")
        assert result is not None
        assert result["score"] == 0.9

    def test_correctness_empty_ground_truth_returns_none(self):
        judge = LLMJudge(client=_client({"score": 0.5}))
        assert judge.answer_correctness("q", "a", "") is None

    def test_correctness_invalid_score_returns_none(self):
        judge = LLMJudge(client=_client({"score": "not_a_number"}))
        assert judge.answer_correctness("q", "a", "gt") is None

    def test_correctness_missing_score_key_returns_none(self):
        judge = LLMJudge(client=_client({"reason": "no score"}))
        assert judge.answer_correctness("q", "a", "gt") is None
