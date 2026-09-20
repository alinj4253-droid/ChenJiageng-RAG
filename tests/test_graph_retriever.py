"""
GraphRetriever 测试（轻量构造，不加载真实 faiss / 大数据）

Phase 6 Commit 1：embedding lazy 缓存 + chunk_map 启动加载
Phase 6 Commit 2：双向邻接表 + relation 索引检索
"""
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
