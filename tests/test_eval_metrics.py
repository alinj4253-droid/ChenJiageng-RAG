"""
evaluation.metrics 纯函数指标测试
"""
import pytest
from evaluation.metrics import retrieval_metrics, answer_accuracy


def _chunk(cid, text):
    return {"chunk_id": cid, "text": text}


class TestChunkLevelRetrieval:

    def test_recall_and_mrr_with_annotation(self):
        retrieved = [_chunk(f"c{i}", f"文本{i}") for i in range(1, 11)]
        item = {"relevant_chunks": ["c2", "c7", "c99"]}
        m = retrieval_metrics(retrieved, item)

        assert m["ground_truth"] == "chunk_id"
        # top5 命中 c2 一个 → recall@5 = 1/3
        assert m["recall@5"] == pytest.approx(1 / 3)
        # top10 命中 c2、c7 两个 → recall@10 = 2/3
        assert m["recall@10"] == pytest.approx(2 / 3)
        # 首个相关 c2 排在第 2 位 → MRR = 1/2
        assert m["mrr"] == pytest.approx(0.5)

    def test_no_hit_returns_zero(self):
        retrieved = [_chunk(f"x{i}", "无关") for i in range(5)]
        m = retrieval_metrics(retrieved, {"relevant_chunks": ["c99"]})
        assert m["recall@5"] == 0.0
        assert m["mrr"] == 0.0


class TestKeywordProxyRetrieval:

    def test_keyword_coverage_and_mrr(self):
        retrieved = [
            _chunk("c1", "这里没有关键词"),
            _chunk("c2", "陈嘉庚 1921 年创办厦门大学"),   # 命中 1921、厦门大学
            _chunk("c3", "另一处提到集美"),
        ]
        item = {"answer_keywords": ["1921", "厦门大学", "集美"]}
        m = retrieval_metrics(retrieved, item)

        assert m["ground_truth"] == "keyword_proxy"
        # top5（全部3条）覆盖 3 个关键词
        assert m["recall@5"] == pytest.approx(1.0)
        # top1 无命中 → recall@1 概念这里测 top5 即可
        # 首个命中在第 2 位 → MRR = 1/2
        assert m["mrr"] == pytest.approx(0.5)

    def test_partial_coverage(self):
        retrieved = [_chunk("c1", "只提到 1921")]
        item = {"answer_keywords": ["1921", "厦门大学"]}
        m = retrieval_metrics(retrieved, item)
        assert m["recall@5"] == pytest.approx(0.5)

    def test_refusal_or_no_keywords_returns_none(self):
        assert retrieval_metrics([_chunk("c1", "x")],
                                 {"type": "refusal", "answer_keywords": []}) is None

    def test_relevant_chunks_take_precedence(self):
        # 同时给 relevant_chunks 和 keywords，应优先 chunk 级口径
        retrieved = [_chunk("c1", "1921 厦门大学")]
        item = {"relevant_chunks": ["c99"], "answer_keywords": ["1921"]}
        m = retrieval_metrics(retrieved, item)
        assert m["ground_truth"] == "chunk_id"
        assert m["recall@5"] == 0.0


class TestAnswerAccuracy:

    def test_keyword_full_and_partial(self):
        item = {"type": "fact", "answer_keywords": ["橡胶", "菠萝", "种植"]}
        full, d1 = answer_accuracy("他经营橡胶和菠萝种植园", item)
        assert full == pytest.approx(1.0)
        assert d1["keyword_hits"] == 3

        partial, d2 = answer_accuracy("只提到橡胶", item)
        assert partial == pytest.approx(1 / 3)
        assert d2["keyword_total"] == 3

    def test_refusal_correct(self):
        item = {"type": "refusal", "answer_keywords": []}
        score, detail = answer_accuracy("抱歉，语料中没有相关记载，无法确定。", item)
        assert score == 1.0
        assert detail["refusal_correct"] is True

    def test_refusal_failure(self):
        item = {"type": "refusal", "answer_keywords": []}
        score, detail = answer_accuracy("他的孙子是一名工程师，住在北京。", item)
        assert score == 0.0
        assert detail["refusal_correct"] is False

    def test_no_signal_returns_none(self):
        score, detail = answer_accuracy("随便答", {"type": "overview",
                                                   "answer_keywords": []})
        assert score is None
