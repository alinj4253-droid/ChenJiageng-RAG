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


def _row(chunk_id, triples, entities=None, model="qwen2.5:32b"):
    return {"chunk_id": chunk_id, "source_file": "书", "chapter": "章",
            "entities": entities or [], "triples": triples,
            "model": model, "timestamp": "2026-09-18 00:00:00"}


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


class TestEntityInfoPreservation:
    """验证从 LLM entities 字段保留 type / description（修复归一化丢失问题）"""

    def test_entity_description_preserved(self, tmp_path, monkeypatch):
        """entities 字段中的 description 应保留到最终 entities.jsonl"""
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
        _write_raw(tmp_path, [
            _row("c1",
                 triples=[_triple("陈嘉庚", "创办", "厦门大学", "PERSON", "ORG")],
                 entities=[
                     {"name": "陈嘉庚", "type": "PERSON", "description": "著名华侨领袖，创办厦门大学"},
                     {"name": "厦门大学", "type": "ORG", "description": "1921年由陈嘉庚创办的大学"},
                 ]),
        ])
        N.normalize_kg()
        entities = {e["name"]: e for e in _read_jsonl(tmp_path / "entities.jsonl")}
        assert entities["陈嘉庚"]["description"] == "著名华侨领袖，创办厦门大学"
        assert entities["厦门大学"]["description"] == "1921年由陈嘉庚创办的大学"

    def test_entity_type_from_entities_preferred(self, tmp_path, monkeypatch):
        """entities 字段的 type 优先于 triples 中的 head_type/tail_type"""
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
        _write_raw(tmp_path, [
            _row("c1",
                 # triples 中类型给错为 OTHER
                 triples=[_triple("陈嘉庚", "创办", "厦门大学", "OTHER", "OTHER")],
                 # entities 中给出正确类型
                 entities=[
                     {"name": "陈嘉庚", "type": "PERSON", "description": ""},
                     {"name": "厦门大学", "type": "ORG", "description": ""},
                 ]),
        ])
        N.normalize_kg()
        entities = {e["name"]: e for e in _read_jsonl(tmp_path / "entities.jsonl")}
        assert entities["陈嘉庚"]["type"] == "PERSON"
        assert entities["厦门大学"]["type"] == "ORG"

    def test_type_upgrade_on_alias_merge(self, tmp_path, monkeypatch):
        """别名归并时，OTHER 类型应被具体类型升级"""
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
        _write_raw(tmp_path, [
            _row("c1",
                 triples=[_triple("嘉庚", "创办", "厦门大学", "OTHER", "ORG")],
                 entities=[{"name": "嘉庚", "type": "OTHER", "description": ""}]),
            _row("c2",
                 triples=[_triple("陈嘉庚", "主持", "南侨总会", "PERSON", "ORG")],
                 entities=[{"name": "陈嘉庚", "type": "PERSON", "description": "华侨领袖"}]),
        ])
        N.normalize_kg()
        entities = {e["name"]: e for e in _read_jsonl(tmp_path / "entities.jsonl")}
        # 嘉庚 归并到 陈嘉庚，类型应升级为 PERSON
        assert "陈嘉庚" in entities
        assert entities["陈嘉庚"]["type"] == "PERSON"
        assert entities["陈嘉庚"]["description"] == "华侨领袖"

    def test_description_coverage_in_stats(self, tmp_path, monkeypatch):
        """统计信息应包含 description_coverage 和 other_type_count"""
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
        _write_raw(tmp_path, [
            _row("c1",
                 triples=[_triple("陈嘉庚", "创办", "厦门大学", "PERSON", "ORG")],
                 entities=[
                     {"name": "陈嘉庚", "type": "PERSON", "description": "华侨领袖"},
                     {"name": "厦门大学", "type": "ORG", "description": ""},
                 ]),
        ])
        stats = N.normalize_kg()
        assert "entities_with_description" in stats
        assert "description_coverage" in stats
        assert "other_type_count" in stats
        assert stats["entities_with_description"] == 1
        assert stats["description_coverage"] == 0.5

    def test_extraction_description_dedup(self):
        from unittest.mock import MagicMock
        from src.kg_extractor import extract_from_chunk
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

    def test_normalizer_preserves_entity_only_sources_and_fallback(self, tmp_path, monkeypatch):
        import src.kg_normalizer as N
        monkeypatch.setattr(N, "KG_DIR", tmp_path)
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
        _write_raw(tmp_path, rows)
        N.normalize_kg()
        entities = {r["name"]: r for r in _read_jsonl(tmp_path / "entities.jsonl")}
        assert entities["陈嘉庚"]["type"] == "PERSON"
        assert entities["陈嘉庚"]["source_chunks"] == ["c1", "c2"]
        assert entities["独立实体"]["description"] == "未参与关系"
        assert entities["新加坡"]["type"] == "LOC"
        assert entities["新加坡政府"]["type"] == "ORG"
        assert "交流教育问题" in (tmp_path / "triples.jsonl").read_text(encoding="utf-8")
