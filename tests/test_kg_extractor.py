# -*- coding: utf-8 -*-
"""
kg_extractor 测试（LLM 全部 mock）

重点：extract_from_chunk 把 LLM 原始输出 {entities, relations}
转换为落盘格式 triples，每条含 head_type / tail_type，
并与 data/kg/triples_raw.jsonl 的结构对齐。
"""
from unittest.mock import MagicMock

from src.kg_extractor import extract_from_chunk


def _chunk(cid="book_ch000_000", text="陈嘉庚创办了厦门大学。"):
    return {"chunk_id": cid, "book": "南侨回忆录", "chapter": "前言", "text": text}


def _client(raw):
    client = MagicMock()
    client.extract_json.return_value = raw
    client.last_model = "qwen2.5:32b"
    return client


def _llm_raw(entities, relations):
    return {"entities": entities, "relations": relations}


class TestExtractToTriples:

    def test_outputs_triples_with_head_tail_types(self):
        raw = _llm_raw(
            entities=[
                {"name": "陈嘉庚", "type": "PERSON", "description": "华侨领袖"},
                {"name": "厦门大学", "type": "ORG", "description": "大学"},
            ],
            relations=[
                {"head": "陈嘉庚", "relation": "创办", "tail": "厦门大学",
                 "description": "1921年"},
            ],
        )
        result = extract_from_chunk(_chunk(), _client(raw))

        assert result is not None
        assert "triples" in result
        # entities 字段现在被保留（含 name/type/description），供 kg_normalizer 使用
        assert "entities" in result
        assert len(result["entities"]) == 2
        assert result["entities"][0]["name"] == "陈嘉庚"
        assert result["entities"][0]["type"] == "PERSON"
        assert result["entities"][0]["description"] == "华侨领袖"
        t = result["triples"][0]
        assert set(t.keys()) == {"head", "head_type", "relation", "tail", "tail_type"}
        assert t["head"] == "陈嘉庚" and t["head_type"] == "PERSON"
        assert t["relation"] == "创办"
        assert t["tail"] == "厦门大学" and t["tail_type"] == "ORG"

    def test_unknown_entity_type_defaults_other(self):
        """关系里出现、但 entities 没给类型的实体，类型兜底为 OTHER"""
        raw = _llm_raw(
            entities=[{"name": "陈嘉庚", "type": "PERSON"}],
            relations=[{"head": "陈嘉庚", "relation": "到访", "tail": "某地"}],
        )
        result = extract_from_chunk(_chunk(), _client(raw))
        t = result["triples"][0]
        assert t["head_type"] == "PERSON"
        assert t["tail_type"] == "OTHER"

    def test_duplicate_triples_dedup(self):
        raw = _llm_raw(
            entities=[{"name": "陈嘉庚", "type": "PERSON"},
                      {"name": "厦门大学", "type": "ORG"}],
            relations=[
                {"head": "陈嘉庚", "relation": "创办", "tail": "厦门大学"},
                {"head": "陈嘉庚", "relation": "创办", "tail": "厦门大学"},
            ],
        )
        result = extract_from_chunk(_chunk(), _client(raw))
        assert len(result["triples"]) == 1

    def test_self_loop_filtered(self):
        """自环关系被过滤，但 entities 仍保留（供 normalizer 富集）"""
        raw = _llm_raw(
            entities=[{"name": "陈嘉庚", "type": "PERSON"}],
            relations=[{"head": "陈嘉庚", "relation": "自称", "tail": "陈嘉庚"}],
        )
        result = extract_from_chunk(_chunk(), _client(raw))
        assert result is not None
        assert result["triples"] == []  # 自环被过滤
        assert len(result["entities"]) == 1  # 实体仍保留

    def test_missing_parts_filtered(self):
        raw = _llm_raw(
            entities=[{"name": "陈嘉庚", "type": "PERSON"},
                      {"name": "厦门大学", "type": "ORG"}],
            relations=[
                {"head": "", "relation": "创办", "tail": "厦门大学"},
                {"head": "陈嘉庚", "relation": "", "tail": "厦门大学"},
                {"head": "陈嘉庚", "relation": "创办", "tail": ""},
                {"head": "陈嘉庚", "relation": "创办", "tail": "厦门大学"},
            ],
        )
        result = extract_from_chunk(_chunk(), _client(raw))
        assert len(result["triples"]) == 1

    def test_overlong_entity_name_filtered(self):
        """超长实体名的关系被过滤，但正常 entities 仍保留"""
        long_name = "超长实体名称" * 6  # 36 字，明确 > 30
        raw = _llm_raw(
            entities=[{"name": "陈嘉庚", "type": "PERSON"}],
            relations=[{"head": "陈嘉庚", "relation": "提及",
                        "tail": long_name}],
        )
        result = extract_from_chunk(_chunk(), _client(raw))
        assert result is not None
        assert result["triples"] == []  # 超长实体关系被过滤
        assert len(result["entities"]) == 1  # 正常实体保留

    def test_entities_only_still_returned(self):
        """只有 entities 没有 triples 时，仍返回结果（entities 供 normalizer 富集类型/描述）"""
        raw = _llm_raw(entities=[{"name": "陈嘉庚", "type": "PERSON", "description": "华侨领袖"}], relations=[])
        result = extract_from_chunk(_chunk(), _client(raw))
        assert result is not None
        assert result["triples"] == []
        assert len(result["entities"]) == 1
        assert result["entities"][0]["description"] == "华侨领袖"

    def test_llm_empty_response_returns_none(self):
        assert extract_from_chunk(_chunk(), _client({})) is None


class TestTopLevelFields:

    def test_top_level_shape_matches_raw_format(self):
        raw = _llm_raw(
            entities=[{"name": "陈嘉庚", "type": "PERSON"},
                      {"name": "厦门大学", "type": "ORG"}],
            relations=[{"head": "陈嘉庚", "relation": "创办", "tail": "厦门大学"}],
        )
        result = extract_from_chunk(_chunk("bookX_ch001_002"), _client(raw))

        assert result["chunk_id"] == "bookX_ch001_002"
        assert result["source_file"] == "南侨回忆录"   # 来自 chunk.book
        assert result["chapter"] == "前言"
        assert result["model"] == "qwen2.5:32b"        # 来自 client.last_model
        assert "timestamp" in result and result["timestamp"]

    def test_prompt_contains_text_and_called_with_system(self):
        raw = _llm_raw(
            entities=[{"name": "陈嘉庚", "type": "PERSON"}],
            relations=[{"head": "陈嘉庚", "relation": "做", "tail": "事"}])
        client = _client(raw)
        extract_from_chunk(_chunk(text="独特文本内容XYZ"), client)

        client.extract_json.assert_called_once()
        args = client.extract_json.call_args
        assert "独特文本内容XYZ" in args[0][0]
        assert args[1]["system_prompt"]  # 带 system prompt

    def test_model_missing_attr_does_not_crash(self):
        """client 没有 last_model 属性时也不报错（兜底空串）"""
        client = MagicMock()
        client.extract_json.return_value = _llm_raw(
            entities=[{"name": "陈嘉庚", "type": "PERSON"},
                      {"name": "厦门大学", "type": "ORG"}],
            relations=[{"head": "陈嘉庚", "relation": "创办", "tail": "厦门大学"}],
        )
        del client.last_model  # 让 getattr 走默认
        result = extract_from_chunk(_chunk(), client)
        assert result["model"] == ""
