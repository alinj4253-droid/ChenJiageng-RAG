"""
Bounded Retry Loop 测试（检索 / 裁判 / 改写全部 mock）
覆盖：
  - 首轮充分 → 不 retry
  - 不足 → retry 一次后充分
  - 持续不足 → 达到上限停止
  - 改写返回相同 query → 不死循环
  - 改写失败返回 None → 停止
  - naive 模式不裁判、不 retry
"""
import pytest
from unittest.mock import MagicMock

from src.rag_pipeline import RAGPipeline
from src.retrieval_router import build_plan
from src.evidence_judge import EvidenceJudgement
from src.query_rewriter import RewrittenQuery


def _chunk(cid):
    return {"chunk_id": cid, "book": "书", "chapter": "章",
            "text": f"内容{cid}", "score": 0.9}


def _make_loop_pipeline(max_retry=1, judge_responses=None, rewrite_responses=None):
    """
    构造只保留 retry 循环相关组件的 pipeline。
    judge_responses: EvidenceJudgement 列表（按调用次序返回）
    rewrite_responses: RewrittenQuery/None 列表（按调用次序返回）
    """
    p = RAGPipeline.__new__(RAGPipeline)
    p.max_retry = max_retry
    p.query_cache = {}

    # 证据裁判
    p.evidence_judge = MagicMock()
    if judge_responses is not None:
        p.evidence_judge.judge.side_effect = judge_responses
    else:
        p.evidence_judge.judge.return_value = EvidenceJudgement(sufficient=True)

    # 查询改写器
    p.query_rewriter = MagicMock()
    if rewrite_responses is not None:
        p.query_rewriter.rewrite.side_effect = rewrite_responses
    else:
        p.query_rewriter.rewrite.return_value = RewrittenQuery(query="改写后的查询")

    # 检索（含融合/精排）整体 mock，返回 (contexts, graph_entities)
    p._retrieve_fuse_rerank = MagicMock(return_value=([_chunk("c1")], []))

    # latency 累加需要的方法在真实实现里，这里整体 mock 掉检索即可
    return p


def _run(p, question, plan):
    statuses, outcome, latency = [], {}, {}
    for kind, payload in p._retrieval_loop(
        question, {"low_level_keywords": [], "high_level_keywords": []},
        plan, latency
    ):
        if kind == "status":
            statuses.append(payload)
        else:
            outcome = payload
    return outcome, statuses, latency


HYBRID = build_plan({"query_mode": "hybrid"})
NAIVE = build_plan({"query_mode": "naive"})
SUFFICIENT = EvidenceJudgement(sufficient=True)
INSUFFICIENT = EvidenceJudgement(sufficient=False, missing=["缺少经费来源"])


class TestBoundedRetry:

    def test_sufficient_first_round_no_retry(self):
        p = _make_loop_pipeline(judge_responses=[SUFFICIENT])
        outcome, _, _ = _run(p, "问题", HYBRID)

        assert outcome["retry_count"] == 0
        assert outcome["judgement"].sufficient is True
        p._retrieve_fuse_rerank.assert_called_once()
        p.query_rewriter.rewrite.assert_not_called()

    def test_insufficient_then_retry_once_and_succeed(self):
        p = _make_loop_pipeline(
            max_retry=1,
            judge_responses=[INSUFFICIENT, SUFFICIENT],
        )
        outcome, _, _ = _run(p, "原始问题", HYBRID)

        assert outcome["retry_count"] == 1
        assert outcome["judgement"].sufficient is True
        # 首轮 + 一次重试 = 2 次检索
        assert p._retrieve_fuse_rerank.call_count == 2
        p.query_rewriter.rewrite.assert_called_once()
        # 第二次检索使用改写后的 query
        second_query = p._retrieve_fuse_rerank.call_args_list[1][0][0]
        assert second_query == "改写后的查询"

    def test_still_insufficient_stops_at_limit(self):
        p = _make_loop_pipeline(
            max_retry=1,
            judge_responses=[INSUFFICIENT, INSUFFICIENT],
        )
        outcome, _, _ = _run(p, "原始问题", HYBRID)

        assert outcome["retry_count"] == 1
        assert outcome["judgement"].sufficient is False
        # 首轮 + 1 次重试，到达上限后不再检索/改写
        assert p._retrieve_fuse_rerank.call_count == 2
        assert p.query_rewriter.rewrite.call_count == 1

    def test_max_retry_zero_never_retries(self):
        p = _make_loop_pipeline(max_retry=0, judge_responses=[INSUFFICIENT])
        outcome, _, _ = _run(p, "q", HYBRID)

        assert outcome["retry_count"] == 0
        p._retrieve_fuse_rerank.assert_called_once()
        p.query_rewriter.rewrite.assert_not_called()

    def test_identical_rewrite_stops_loop(self):
        """改写结果与当前查询完全相同 → 立即停止，不重试"""
        p = _make_loop_pipeline(
            max_retry=2,
            judge_responses=[INSUFFICIENT],
            rewrite_responses=[RewrittenQuery(query="原始问题")],
        )
        outcome, _, _ = _run(p, "原始问题", HYBRID)

        assert outcome["retry_count"] == 0
        p._retrieve_fuse_rerank.assert_called_once()

    def test_rewrite_failure_stops_loop(self):
        """改写器返回 None → 停止，不重试"""
        p = _make_loop_pipeline(
            max_retry=2,
            judge_responses=[INSUFFICIENT],
            rewrite_responses=[None],
        )
        outcome, _, _ = _run(p, "q", HYBRID)

        assert outcome["retry_count"] == 0
        p._retrieve_fuse_rerank.assert_called_once()

    def test_repeated_rewrite_across_rounds_stops(self):
        """多轮重试中改写查询与历史重复 → 停止，杜绝死循环"""
        p = _make_loop_pipeline(
            max_retry=3,
            judge_responses=[INSUFFICIENT, INSUFFICIENT, INSUFFICIENT],
            rewrite_responses=[
                RewrittenQuery(query="查询A"),
                RewrittenQuery(query="查询A"),  # 与上一轮重复
            ],
        )
        outcome, _, _ = _run(p, "q", HYBRID)

        # 第一次改写 A 触发一次重试，第二次又得到 A（已在历史中）→ 停止
        assert outcome["retry_count"] == 1
        assert p._retrieve_fuse_rerank.call_count == 2

    def test_retry_never_exceeds_max(self):
        """任何情况下 retry_count 不超过 max_retry"""
        p = _make_loop_pipeline(
            max_retry=2,
            judge_responses=[INSUFFICIENT, INSUFFICIENT, INSUFFICIENT, INSUFFICIENT],
            rewrite_responses=[
                RewrittenQuery(query="q1"),
                RewrittenQuery(query="q2"),
                RewrittenQuery(query="q3"),
            ],
        )
        outcome, _, _ = _run(p, "q", HYBRID)
        assert outcome["retry_count"] <= 2
        assert outcome["retry_count"] == 2
        assert p._retrieve_fuse_rerank.call_count == 3

    def test_naive_plan_skips_judge_and_retry(self):
        p = _make_loop_pipeline(judge_responses=[INSUFFICIENT])
        outcome, _, _ = _run(p, "短问题", NAIVE)

        assert outcome["retry_count"] == 0
        assert outcome["judgement"] is None
        p.evidence_judge.judge.assert_not_called()
        p.query_rewriter.rewrite.assert_not_called()

    def test_judge_always_uses_original_question(self):
        """裁判始终针对原始问题，而非改写后的查询"""
        p = _make_loop_pipeline(
            max_retry=1,
            judge_responses=[INSUFFICIENT, SUFFICIENT],
        )
        _run(p, "原始问题XYZ", HYBRID)
        for call in p.evidence_judge.judge.call_args_list:
            assert call[0][0] == "原始问题XYZ"
