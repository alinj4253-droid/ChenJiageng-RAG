# -*- coding: utf-8 -*-
"""
kg_normalizer 测试（不加载真实大数据，用 tmp_path 构造 triples_raw.jsonl）

重点：normalize_kg 从 triples 字段读取，完成类型标准化、噪声过滤、
别名归并、关系去重，并产出 entities.jsonl / triples.jsonl / kg_stats.json。
"""
import json

import src.kg_normalizer as N


def _triple(head, relation, tail, head_type="OTHER", tail_type="OTHER"):
    return {"head": head, "head_type": head_type, "relation": relation,
            "tail": tail, "tail_type": tail_type}


def _row(chunk_id, triples, model="qwen2.5:32b"):
    return {"chunk_id": chunk_id, "source_file": "书", "chapter": "章",
            "triples": triples, "model": model, "timestamp": "2026-09-18 00:00:00"}


def _write_raw(tmp_path, rows):
    p = tmp_path / "triples_raw.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestNormalizePipeline:

    def test_basic_pipeline_and_outputs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
        _write_raw(tmp_path, [
            _row("c1", [_triple("陈嘉庚", "创办", "厦门大学", "PERSON", "ORG")]),
        ])

        stats = N.normalize_kg()

        assert stats is not None
        assert stats["total_entities"] == 2
        assert stats["total_relations"] == 1
        assert stats["raw_chunks"] == 1
        # 三个产物文件都生成
        assert (tmp_path / "entities.jsonl").exists()
        assert (tmp_path / "triples.jsonl").exists()
        assert (tmp_path / "kg_stats.json").exists()

        entities = _read_jsonl(tmp_path / "entities.jsonl")
        names = {e["name"] for e in entities}
        assert names == {"陈嘉庚", "厦门大学"}
        triples = _read_jsonl(tmp_path / "triples.jsonl")
        assert triples[0]["head"] == "陈嘉庚"
        assert triples[0]["tail"] == "厦门大学"

    def test_type_normalized(self, tmp_path, monkeypatch):
        """中文/小写类型统一为标准枚举"""
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
        _write_raw(tmp_path, [
            _row("c1", [_triple("陈嘉庚", "创办", "厦门大学", "人物", "机构")]),
        ])
        N.normalize_kg()
        entities = {e["name"]: e for e in _read_jsonl(tmp_path / "entities.jsonl")}
        assert entities["陈嘉庚"]["type"] == "PERSON"
        assert entities["厦门大学"]["type"] == "ORG"

    def test_noise_entity_filtered(self, tmp_path, monkeypatch):
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
        _write_raw(tmp_path, [
            _row("c1", [_triple("我", "说", "陈嘉庚", "PERSON", "PERSON")]),
            _row("c2", [_triple("陈嘉庚", "创办", "厦门大学", "PERSON", "ORG")]),
        ])
        N.normalize_kg()
        names = {e["name"] for e in _read_jsonl(tmp_path / "entities.jsonl")}
        assert "我" not in names
        assert names == {"陈嘉庚", "厦门大学"}

    def test_alias_merging_manual(self, tmp_path, monkeypatch):
        """嘉庚 经手动别名表归并到 陈嘉庚，aliases 记录原名"""
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
        _write_raw(tmp_path, [
            _row("c1", [_triple("嘉庚", "创办", "厦门大学", "PERSON", "ORG")]),
            _row("c2", [_triple("陈嘉庚", "主持", "南侨总会", "PERSON", "ORG")]),
        ])
        N.normalize_kg()
        entities = {e["name"]: e for e in _read_jsonl(tmp_path / "entities.jsonl")}
        assert set(entities) == {"陈嘉庚", "厦门大学", "南侨总会"}
        assert "嘉庚" in entities["陈嘉庚"]["aliases"]

        triples = _read_jsonl(tmp_path / "triples.jsonl")
        heads = {(t["head"], t["relation"], t["tail"]) for t in triples}
        assert ("陈嘉庚", "创办", "厦门大学") in heads
        assert ("陈嘉庚", "主持", "南侨总会") in heads

    def test_relation_dedup(self, tmp_path, monkeypatch):
        """同一 (头,关系,尾) 去重，来源 chunk 聚合"""
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
        _write_raw(tmp_path, [
            _row("c1", [_triple("陈嘉庚", "创办", "厦门大学", "PERSON", "ORG")]),
            _row("c2", [_triple("陈嘉庚", "创办", "厦门大学", "PERSON", "ORG")]),
        ])
        N.normalize_kg()
        triples = _read_jsonl(tmp_path / "triples.jsonl")
        assert len(triples) == 1
        assert set(triples[0]["source_chunks"]) == {"c1", "c2"}
        assert triples[0]["chunk_count"] == 2

    def test_self_loop_dropped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
        _write_raw(tmp_path, [
            _row("c1", [_triple("陈嘉庚", "创办", "厦门大学", "PERSON", "ORG")]),
            _row("c2", [_triple("厦门大学", "合并", "厦门大学", "ORG", "ORG")]),
        ])
        N.normalize_kg()
        triples = _read_jsonl(tmp_path / "triples.jsonl")
        assert len(triples) == 1

    def test_missing_raw_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
        assert N.normalize_kg() is None
