"""
GraphRetriever 测试（轻量构造，不加载真实 faiss / 大数据）

Phase 6 Commit 1：embedding lazy 缓存 + chunk_map 启动加载
Phase 6 Commit 2：双向邻接表 + relation 索引检索
"""
import json
import numpy as np
from unittest.mock import MagicMock, patch

from src.graph_retriever import GraphRetriever


def _chunk(cid, text="正文"):
    return {"chunk_id": cid, "book": "陈嘉庚传", "chapter": "第一章",
            "text": text, "char_count": len(text)}


def _make_graph():
    g = GraphRetriever.__new__(GraphRetriever)
    g.entities = {
        "陈嘉庚": {"name": "陈嘉庚", "type": "PER",
                   "source_chunks": ["c1", "c2"]},
        "厦门大学": {"name": "厦门大学", "type": "ORG",
                    "source_chunks": ["c2", "c3"]},
    }
    g.relations = []
    g.adj_list = {}
    g.entity_index = None
    g.entity_meta = []
    g.relation_index = None
    g.relation_meta = []
    g._embedding_model = None
    g.chunk_map = {cid: _chunk(cid) for cid in ["c1", "c2", "c3"]}
    return g


class TestChunkMapCache:

    def test_get_related_chunks_uses_cached_chunk_map(self):
        g = _make_graph()
        chunks = g.get_related_chunks({"陈嘉庚": 1.0, "厦门大学": 0.5})
        ids = [c["chunk_id"] for c in chunks]
        # c2 同时被两个实体命中，加权分最高应排最前
        assert ids[0] == "c2"
        assert all(c["source"] == "graph" for c in chunks)
        # 物化字段完整
        assert {"book", "chapter", "text", "score", "char_count"} <= set(chunks[0].keys())

    def test_materialize_missing_chunk_returns_none(self):
        g = _make_graph()
        assert g._materialize_chunk("not-exist", 1.0) is None

    def test_unknown_entity_contributes_nothing(self):
        g = _make_graph()
        chunks = g.get_related_chunks({"不存在的实体": 1.0})
        assert chunks == []


class TestEmbeddingLazyCache:

    def test_embedding_model_loaded_once(self):
        g = _make_graph()
        g._embedding_model = None
        with patch("src.vector_store.load_embedding_model",
                   return_value="MODEL") as mock_load:
            first = g._get_embedding_model()
            second = g._get_embedding_model()
        assert first == "MODEL" and second == "MODEL"
        # 两次调用只真正加载一次
        mock_load.assert_called_once()


class TestBidirectionalGraph:

    def _load_small_kg(self, tmp_path, monkeypatch):
        import src.graph_retriever as gr
        monkeypatch.setattr(gr, "KG_DIR", tmp_path)
        (tmp_path / "entities.jsonl").write_text(
            json.dumps({"name": "陈嘉庚", "type": "PER", "source_chunks": []},
                       ensure_ascii=False) + "\n" +
            json.dumps({"name": "厦门大学", "type": "ORG", "source_chunks": []},
                       ensure_ascii=False) + "\n",
            encoding="utf-8")
        (tmp_path / "triples.jsonl").write_text(
            json.dumps({"head": "陈嘉庚", "relation": "创办", "tail": "厦门大学",
                        "source_chunks": ["c1"]}, ensure_ascii=False) + "\n",
            encoding="utf-8")
        g = GraphRetriever.__new__(GraphRetriever)
        g.entities, g.relations, g.adj_list = {}, [], {}
        g._load_kg()
        return g

    def test_adjlist_is_bidirectional_with_direction(self, tmp_path, monkeypatch):
        g = self._load_small_kg(tmp_path, monkeypatch)

        outgoing = {n["neighbor"]: n for n in g.get_neighbors("陈嘉庚")}
        assert outgoing["厦门大学"]["direction"] == "outgoing"
        assert outgoing["厦门大学"]["relation"] == "创办"

        incoming = {n["neighbor"]: n for n in g.get_neighbors("厦门大学")}
        assert incoming["陈嘉庚"]["direction"] == "incoming"
        assert incoming["陈嘉庚"]["relation"] == "创办"

    def test_expand_can_walk_incoming_edges(self, tmp_path, monkeypatch):
        g = self._load_small_kg(tmp_path, monkeypatch)
        # 从尾实体出发，沿 incoming 边也能扩展回头实体
        visited = g.expand_subgraph(["厦门大学"], depth=1)
        assert "陈嘉庚" in visited


class TestRelationRetrieval:

    def _graph_with_relation_index(self):
        g = _make_graph()
        g.relations = [
            {"head": "陈嘉庚", "relation": "创办", "tail": "厦门大学",
             "source_chunks": ["c1"]},
            {"head": "无关", "relation": "位于", "tail": "某地",
             "source_chunks": ["c2"]},
        ]
        fake_index = MagicMock()
        # 第 0 条相似 0.9（保留），第 1 条 0.3（低于阈值过滤）
        fake_index.search.return_value = (
            np.array([[0.9, 0.3]], dtype="float32"),
            np.array([[0, 1]]),
        )
        g.relation_index = fake_index
        g._embedding_model = MagicMock()
        g._embedding_model.encode.return_value = np.ones((1, 8))
        return g

    def test_search_relations_threshold_and_fields(self):
        g = self._graph_with_relation_index()
        rels = g.search_relations("陈嘉庚的教育救国理念")
        assert len(rels) == 1
        assert rels[0]["head"] == "陈嘉庚"
        assert rels[0]["tail"] == "厦门大学"
        assert rels[0]["source_chunks"] == ["c1"]

    def test_relation_chunks_materialize_source(self):
        g = self._graph_with_relation_index()
        chunks = g.relation_chunks("教育救国")
        assert [c["chunk_id"] for c in chunks] == ["c1"]
        assert chunks[0]["source"] == "graph_relation"

    def test_no_relation_index_returns_empty(self):
        g = _make_graph()
        g.relation_index = None
        assert g.search_relations("q") == []
        assert g.relation_chunks("q") == []

    def test_empty_query_returns_empty(self):
        g = self._graph_with_relation_index()
        assert g.search_relations("") == []
