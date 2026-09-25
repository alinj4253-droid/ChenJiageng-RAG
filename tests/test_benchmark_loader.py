# -*- coding: utf-8 -*-
"""
benchmark_loader 测试：CRUD-RAG / MultiHop-RAG / 通用格式加载
"""
import json
import pytest

from evaluation.benchmark_loader import (
    load_crud_rag,
    load_multihop_rag,
    load_generic_json,
    load_benchmark,
)


@pytest.fixture
def crud_rag_file(tmp_path):
    """构造一个模拟的 CRUD-RAG 格式文件"""
    data = {
        "questanswer_1doc": [
            {
                "ID": "doc_001",
                "event": "测试事件",
                "news1": "新闻内容1",
                "questions": ["问题1？", "问题2？"],
                "answers": ["答案1", "答案2"],
            }
        ],
        "questanswer_2docs": [
            {
                "ID": "doc_002",
                "event": "多文档事件",
                "news1": "新闻1",
                "news2": "新闻2",
                "questions": ["多跳问题？"],
                "answers": ["多跳答案"],
            }
        ],
        "continuing_writing": [{"ID": "x", "beginning": "...", "continuing": "..."}],
    }
    p = tmp_path / "crud_rag.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return p


@pytest.fixture
def multihop_rag_file(tmp_path):
    """构造一个模拟的 MultiHop-RAG 格式文件"""
    data = [
        {
            "query": "What is the capital of France?",
            "answer": "Paris",
            "question_type": "inference_query",
            "evidence_list": [
                {"title": "France Overview", "author": "A", "url": "http://x", "content": "..."},
                {"title": "European Capitals", "author": "B", "url": "http://y", "content": "..."},
            ],
        },
        {
            "query": "Who wrote Hamlet?",
            "answer": "Shakespeare",
            "question_type": "comparison",
            "evidence_list": [{"title": "English Literature", "content": "..."}],
        },
    ]
    p = tmp_path / "multihop_rag.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return p


class TestLoadCrudRag:

    def test_flattens_questions(self, crud_rag_file):
        result = load_crud_rag(crud_rag_file)
        # 1doc 有 2 题 + 2docs 有 1 题 = 3 题（continuing_writing 不计入）
        assert len(result) == 3

    def test_unified_format_fields(self, crud_rag_file):
        result = load_crud_rag(crud_rag_file)
        for item in result:
            assert "id" in item
            assert "question" in item
            assert "type" in item
            assert "ground_truth" in item
            assert "relevant_chunks" in item
            assert item["source"] == "CRUD-RAG"

    def test_single_doc_type_is_fact(self, crud_rag_file):
        result = load_crud_rag(crud_rag_file)
        single_doc = [r for r in result if "1doc" in r["category"]]
        assert all(r["type"] == "fact" for r in single_doc)

    def test_multi_doc_type_is_multihop(self, crud_rag_file):
        result = load_crud_rag(crud_rag_file)
        multi_doc = [r for r in result if "2doc" in r["category"]]
        assert all(r["type"] == "multihop" for r in multi_doc)

    def test_relevant_chunks_include_news_ids(self, crud_rag_file):
        result = load_crud_rag(crud_rag_file)
        single = [r for r in result if "1doc" in r["category"]][0]
        assert "doc_001_news1" in single["relevant_chunks"]


class TestLoadMultiHopRag:

    def test_loads_all_questions(self, multihop_rag_file):
        result = load_multihop_rag(multihop_rag_file)
        assert len(result) == 2

    def test_unified_format(self, multihop_rag_file):
        result = load_multihop_rag(multihop_rag_file)
        assert result[0]["question"] == "What is the capital of France?"
        assert result[0]["ground_truth"] == "Paris"
        assert result[0]["type"] == "multihop"
        assert result[0]["source"] == "MultiHop-RAG"

    def test_evidence_list_as_relevant_chunks(self, multihop_rag_file):
        result = load_multihop_rag(multihop_rag_file)
        assert "France Overview" in result[0]["relevant_chunks"]
        assert "European Capitals" in result[0]["relevant_chunks"]

    def test_category_from_question_type(self, multihop_rag_file):
        result = load_multihop_rag(multihop_rag_file)
        assert result[0]["category"] == "inference_query"
        assert result[1]["category"] == "comparison"


class TestLoadBenchmarkAuto:

    def test_auto_detect_crud(self, crud_rag_file):
        result = load_benchmark(str(crud_rag_file), fmt="auto")
        assert len(result) == 3
        assert result[0]["source"] == "CRUD-RAG"

    def test_auto_detect_multihop(self, multihop_rag_file):
        result = load_benchmark(str(multihop_rag_file), fmt="auto")
        assert len(result) == 2
        assert result[0]["source"] == "MultiHop-RAG"

    def test_explicit_format_override(self, multihop_rag_file):
        # 即使文件名不含 multihop，显式指定格式也能正确加载
        import shutil
        custom = multihop_rag_file.parent / "custom_data.json"
        shutil.copy(multihop_rag_file, custom)
        result = load_benchmark(str(custom), fmt="multihop-rag")
        assert len(result) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_benchmark(str(tmp_path / "nonexistent.json"))
