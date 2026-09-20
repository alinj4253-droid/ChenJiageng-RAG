"""
StructuredTraceLogger 测试
"""
import json

from src.trace_logger import StructuredTraceLogger
from src.retrieval_router import build_plan
from src.evidence_judge import EvidenceJudgement


def _logger(tmp_path, enabled=True):
    return StructuredTraceLogger(log_file=tmp_path / "trace.jsonl", enabled=enabled)


def _kwargs():
    return dict(
        question="陈嘉庚为什么创办厦门大学？",
        analysis={"query_mode": "global", "high_level_keywords": ["教育救国"]},
        plan=build_plan({"query_mode": "global"}),
        contexts=[{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        graph_entities=[{"name": "陈嘉庚"}],
        judgement=EvidenceJudgement(sufficient=False, missing=["缺少经费来源"]),
        retry_count=1,
        candidate_count=10,
        latency={"query_analysis": 0.4, "retrieval": 0.22, "generation": 1.8},
    )


class TestBuildTrace:

    def test_fields(self, tmp_path):
        trace = _logger(tmp_path).build_trace(**_kwargs())
        assert trace["query"] == "陈嘉庚为什么创办厦门大学？"
        assert trace["query_mode"] == "global"
        assert trace["retrieved_docs"] == 10
        assert trace["reranked_docs"] == 2
        assert trace["matched_entities"] == 1
        assert trace["evidence_sufficient"] is False
        assert trace["retry_count"] == 1
        # plan 被序列化
        assert trace["retrieval_plan"]["use_graph"] is True
        # latency 被 round
        assert trace["latency"]["generation"] == 1.8
        assert "timestamp" in trace

    def test_none_judgement(self, tmp_path):
        kw = _kwargs()
        kw["judgement"] = None
        trace = _logger(tmp_path).build_trace(**kw)
        assert trace["evidence_sufficient"] is None


class TestLogPersistence:

    def test_disabled_does_not_write(self, tmp_path):
        logger = _logger(tmp_path, enabled=False)
        result = logger.log(logger.build_trace(**_kwargs()))
        assert result is None
        assert not (tmp_path / "trace.jsonl").exists()

    def test_enabled_appends_jsonl(self, tmp_path):
        logger = _logger(tmp_path, enabled=True)
        logger.log(logger.build_trace(**_kwargs()))
        logger.log(logger.build_trace(**_kwargs()))

        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)  # 每行必须是合法 JSON
            assert obj["query_mode"] == "global"
            assert obj["retry_count"] == 1
