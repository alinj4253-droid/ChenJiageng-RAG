# -*- coding: utf-8 -*-
"""
rag_generator 测试：句子边界截断、上下文组装
"""
from src.rag_generator import _truncate_at_sentence, RAGGenerator


class TestTruncateAtSentence:

    def test_short_text_not_truncated(self):
        text = "陈嘉庚创办了厦门大学。"
        assert _truncate_at_sentence(text, 100) == text

    def test_truncate_at_period(self):
        text = "第一句话。第二句话比较长，需要被截断。第三句话。"
        result = _truncate_at_sentence(text, 15)
        # 应在第一个句号后截断
        assert result.endswith("。")
        assert "第二句话" not in result

    def test_truncate_at_exclamation(self):
        text = "非常重要！这是第二句。"
        result = _truncate_at_sentence(text, 8)
        assert result.endswith("！")

    def test_truncate_at_question(self):
        text = "你是谁？我是陈嘉庚。"
        result = _truncate_at_sentence(text, 6)
        assert result.endswith("？")

    def test_no_sentence_boundary_fallback_to_comma(self):
        text = "这是一段没有句号只有逗号的文本，后面还有更多内容需要被截断掉"
        result = _truncate_at_sentence(text, 20)
        # 应在逗号处截断
        assert result.endswith("，") or result.endswith("...")

    def test_hard_truncate_when_no_boundary(self):
        text = "这是一段完全没有任何标点符号的超长文本内容需要被硬截断"
        result = _truncate_at_sentence(text, 10)
        # 找不到边界时硬截断加省略号
        assert result.endswith("...")
        assert len(result) <= 13  # 10 + 3

    def test_exact_length_not_truncated(self):
        text = "正好十个字。"
        assert _truncate_at_sentence(text, len(text)) == text

    def test_empty_text(self):
        assert _truncate_at_sentence("", 10) == ""


class TestBuildContext:

    def test_context_respects_max_chunks(self):
        gen = RAGGenerator()
        chunks = [
            {"chunk_id": f"c{i}", "text": f"第{i}块内容。" * 10, "book": "书", "chapter": "章"}
            for i in range(5)
        ]
        context = gen._build_context(chunks, max_chunks=2, max_chars=10000)
        # 只应包含 2 个 chunk
        assert context.count("[1]") == 1
        assert context.count("[2]") == 1
        assert "[3]" not in context

    def test_context_truncates_at_sentence_boundary(self):
        gen = RAGGenerator()
        long_text = "第一句话完整。" + "第二句话也完整。" + "第三句话需要被截断。" * 10
        chunks = [{"chunk_id": "c1", "text": long_text, "book": "书", "chapter": "章"}]
        context = gen._build_context(chunks, max_chunks=1, max_chars=30)
        # 不应在句子中间硬截断（除非完全找不到边界）
        assert "..." in context or context.rstrip().endswith(("。", "！", "？", "；"))
