import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.benchmark_loader import load_crud_rag
from evaluation.prepare_benchmark import prepare
from evaluation.run_benchmark import evaluate, validate_workspace
from evaluation.benchmark_metrics import recall_at_ten, answer_f1
from src.kg_extractor import extract_from_chunk
import src.kg_normalizer as normalizer


def test_extraction_description_dedup():
    client = MagicMock()
    client.last_model = "mock"
    client.extract_json.return_value = {
        "entities": [], "relations": [
            {"head": "甲方", "tail": "乙方", "relation": "合作"},
            {"head": "甲方", "tail": "乙方", "relation": "合作", "description": "共同办学"},
        ]}
    row = extract_from_chunk({"chunk_id": "c1", "text": "甲方与乙方共同办学"}, client)
    assert len(row["triples"]) == 1
    assert row["triples"][0]["description"] == "共同办学"
    assert client.extract_json.call_args.kwargs["max_tokens"] == 4096


def test_normalizer_preserves_entity_only_sources_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(normalizer, "KG_DIR", tmp_path)
    rows = [
        {"chunk_id": "c1", "entities": [
            {"name": "嘉庚", "type": "OTHER", "description": "华侨领袖"},
            {"name": "独立实体", "type": "ORG", "description": "未参与关系"}],
         "triples": []},
        {"chunk_id": "c2", "entities": [], "triples": [
            {"head": "嘉庚", "head_type": "PERSON", "relation": "访问", "tail": "新加坡政府",
             "tail_type": "ORG", "description": "交流教育问题"},
            {"head": "嘉庚", "head_type": "PERSON", "relation": "访问", "tail": "新加坡",
             "tail_type": "LOC"}]}]
    (tmp_path / "triples_raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    normalizer.normalize_kg()
    entities = {r["name"]: r for r in map(json.loads, (tmp_path / "entities.jsonl").read_text(encoding="utf-8").splitlines())}
    assert entities["陈嘉庚"]["type"] == "PERSON"
    assert entities["陈嘉庚"]["source_chunks"] == ["c1", "c2"]
    assert entities["独立实体"]["description"] == "未参与关系"
    assert entities["新加坡"]["type"] == "LOC"
    assert entities["新加坡政府"]["type"] == "ORG"
    assert "交流教育问题" in (tmp_path / "triples.jsonl").read_text(encoding="utf-8")


def crud_file(tmp_path):
    path = tmp_path / "crud.json"
    path.write_text(json.dumps({"questanswer_1doc": [
        {"ID": "1", "news1": "检索语料", "questions": "完整问题？", "answers": "完整答案"},
        {"ID": "2", "news1": "干扰资料", "questions": ["另一问题"], "answers": ["答案"]}]}), encoding="utf-8")
    return path


def test_real_crud_string_shape_and_isolated_mapping(tmp_path):
    path = crud_file(tmp_path)
    rows = load_crud_rag(path)
    assert len(rows) == 2
    assert rows[0]["question"] == "完整问题？"
    assert rows[0]["ground_truth"] == "完整答案"
    root = tmp_path / "isolated"
    manifest = prepare(path, "crud-rag", root)
    assert set(rows[0]["relevant_docs"]) <= set(manifest["chunk_to_doc"].values())
    content = (root / "chunks/chunks.jsonl").read_text(encoding="utf-8")
    assert "完整答案" not in content and "完整问题" not in content
    (root / "vector_store").mkdir()
    (root / "vector_store/metadata.jsonl").write_text(content, encoding="utf-8")
    assert validate_workspace(root, rows)["question_count"] == 2
    (root / "vector_store/metadata.jsonl").write_text('{"chunk_id":"domain"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        validate_workspace(root, rows)
    with pytest.raises(ValueError, match="empty"):
        prepare(path, "crud-rag", root)


def test_missing_multihop_support_is_rejected(tmp_path):
    qa = tmp_path / "qa.json"
    qa.write_text(json.dumps([{"query": "q", "answer": "a", "evidence_list": [{"url": "missing"}]}]))
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([{"title": "article", "url": "present", "body": "source"}]))
    with pytest.raises(ValueError, match="Missing supporting"):
        prepare(qa, "multihop-rag", tmp_path / "out", corpus)


def test_official_evidence_mapping_is_separate_from_index_text(tmp_path):
    qa = tmp_path / "qa.json"
    qa.write_text(json.dumps([{"query": "question", "answer": "SECRET_ANSWER",
                              "evidence_list": [{"url": "https://source/article", "fact": "Alice won."}]}]))
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([{"title": "article", "url": "https://source/article", "body": "Alice won. Bob lost."}]))
    root = tmp_path / "out"
    prepare(qa, "multihop-rag", root, corpus)
    mapping = json.loads((root / "chunk_mapping.json").read_text(encoding="utf-8"))
    q = json.loads((root / "questions.json").read_text(encoding="utf-8"))[0]
    row = next(iter(mapping.values()))
    assert row["source_id"] == "https://source/article"
    assert row["document_id"] == q["supporting_evidence"][0]["document_id"]
    assert row["benchmark_evidence_id"] == [q["supporting_evidence"][0]["evidence_id"]]
    chunks = (root / "chunks/chunks.jsonl").read_text(encoding="utf-8")
    assert "SECRET_ANSWER" not in chunks and "benchmark_evidence_id" not in chunks


def test_recall_matches_facts_not_just_documents():
    item = {"supporting_evidence": [{"fact": "Alice founded School."}, {"fact": "Bob funded School."}]}
    assert recall_at_ten([{"text": "Alice founded School."}], item) == .5
    assert recall_at_ten([{"text": "Other paragraph in same article"}], item) == 0
    assert recall_at_ten([{"text": "Alice  founded\nSchool."}] * 10 + [{"text": "Bob funded School."}], item) == .5
    assert recall_at_ten([], {"category": "null_query", **item}) is None
    assert recall_at_ten([], {}) is None


def test_token_f1_multiset_and_normalization():
    assert answer_f1("The Sam Bankman-Fried!", "Sam Bankman-Fried") == 1
    assert answer_f1("Alice Alice Bob", "Alice Bob") == .8
    assert answer_f1("unrelated", "answer") == 0
    assert answer_f1("", "") == 1
    assert answer_f1("", "answer") == 0


def test_three_core_metrics_and_separate_diagnostics():
    pipeline = MagicMock()
    pipeline.query.return_value = {"answer": "Alice", "retrieved_chunks": [{"chunk_id": "wrong"}],
                                   "retrieval_candidates": [{"chunk_id": "right", "text": "Alice won."}],
                                   "retrieval_calls": 6, "retry_count": 1, "latency": {"total": 999}}
    ticks = iter([10.0, 12.5])
    outcome = evaluate([{"question": "q", "ground_truth": "Alice",
                         "supporting_evidence": [{"fact": "Alice won."}]}], pipeline, lambda: next(ticks))
    summary = outcome["summary"]
    assert summary["recall@10"] == 1
    assert summary["f1"] == 1
    assert summary["avg_latency_seconds"] == 2.5
    assert set(summary) == {"recall@10", "f1", "avg_latency_seconds", "total_questions", "retrieval_scored_questions"}
    assert outcome["diagnostics"] == {"avg_retrieval_calls": 6, "retry_rate": 1}
    pipeline.query_cache.clear.assert_called_once()


def test_missing_evidence_label_is_not_zero():
    pipeline = MagicMock()
    pipeline.query.return_value = {"answer": "a", "retrieved_chunks": []}
    result = evaluate([{"question": "q", "ground_truth": "a", "category": "null_query"}], pipeline)
    assert result["summary"]["recall@10"] is None
    assert result["summary"]["retrieval_scored_questions"] == 0


def test_evaluation_checkpoint_retains_completed_queries():
    pipeline = MagicMock()
    pipeline.query.side_effect = [{"answer": "a", "retrieval_candidates": []}, RuntimeError("network")]
    questions = [{"id": i, "question": str(i), "ground_truth": "a"} for i in range(2)]
    saved = []
    with pytest.raises(RuntimeError, match="network"):
        evaluate(questions, pipeline, checkpoint=lambda rows: saved.extend(rows))
    assert len(saved) == 1
    pipeline.query.side_effect = None
    pipeline.query.return_value = {"answer": "a", "retrieval_candidates": []}
    pipeline.query.reset_mock()
    result = evaluate(questions, pipeline, completed=saved)
    pipeline.query.assert_called_once_with("1", verbose=False)
    assert result["summary"]["total_questions"] == 2
    with pytest.raises(ValueError, match="Checkpoint"):
        evaluate(list(reversed(questions)), pipeline, completed=saved)


def test_downloader_rejects_html_and_invalid_cached_file(tmp_path, monkeypatch):
    from evaluation import download_benchmarks as dl
    destination = tmp_path / "crud_rag.json"
    destination.write_text("<html>error</html>")
    monkeypatch.setattr(dl, "PUBLIC_DIR", tmp_path)
    download = MagicMock(return_value=False)
    monkeypatch.setattr(dl, "download_file", download)
    assert dl.download_benchmark("crud-rag") is False
    download.assert_called_once()


def test_json_budget_forwarded():
    from src.llm_client import LLMClient
    client = LLMClient(use_online=False)
    client.chat = MagicMock(return_value='{"ok": true}')
    assert client.extract_json("prompt", max_tokens=4096) == {"ok": True}
    assert client.chat.call_args.kwargs["max_tokens"] == 4096


def test_prompt_matches_actual_router():
    from src.prompts import QUERY_ANALYSIS_SYSTEM
    assert "社区检索" not in QUERY_ANALYSIS_SYSTEM
    assert "实体图检索与关系检索" in QUERY_ANALYSIS_SYSTEM
