"""
/api/chat 与 /api/chat/stream 职责边界测试

- /api/chat：单轮同步 RAG，不传会话历史，不调用 build_chat_history；
- /api/chat/stream：多轮聊天主链路，会先 build_chat_history 并把 history
  传给 query_stream（配合 session 持久化 / 滚动摘要 / SSE）。

pipeline 与 db 全部 mock，不依赖真实模型 / 数据库。
"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

import src.app as app_module


def _fake_stream():
    yield {"type": "status", "message": "正在生成"}
    yield {"type": "token", "content": "接着说的答案"}
    yield {"type": "done", "references": [{"chunk_id": "x"}]}


@pytest.fixture
def wired(monkeypatch):
    fake_db = MagicMock()
    fake_db.create_session.return_value = "sess_1"
    fake_db.build_chat_history.return_value = [
        {"role": "user", "content": "上文问题"},
        {"role": "assistant", "content": "上文答案"},
    ]

    fake_pipeline = MagicMock()
    fake_pipeline.query.return_value = {
        "answer": "单轮答案",
        "references": [{"chunk_id": "x"}],
        "latency": {"total": 1.0},
    }
    fake_pipeline.query_stream.return_value = _fake_stream()

    monkeypatch.setattr(app_module, "get_db", lambda: fake_db)
    monkeypatch.setattr(app_module, "get_pipeline", lambda: fake_pipeline)
    return TestClient(app_module.app), fake_db, fake_pipeline


class TestChatSyncSemantics:

    def test_single_turn_answers_without_history(self, wired):
        client, fake_db, fake_pipeline = wired
        resp = client.post("/api/chat", json={"message": "陈嘉庚是谁？"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "单轮答案"
        # 走单轮 query 路径
        fake_pipeline.query.assert_called_once_with("陈嘉庚是谁？", verbose=False)
        fake_pipeline.query_stream.assert_not_called()

    def test_single_turn_does_not_build_chat_history(self, wired):
        """单轮接口不维护会话记忆，不应拉取历史"""
        client, fake_db, _ = wired
        client.post("/api/chat", json={"message": "陈嘉庚是谁？"})
        fake_db.build_chat_history.assert_not_called()


class TestChatStreamSemantics:

    def test_stream_builds_chat_history(self, wired):
        """流式多轮接口会先拉取会话历史"""
        client, fake_db, fake_pipeline = wired
        with client.stream(
            "POST", "/api/chat/stream", json={"message": "接着说"}
        ) as resp:
            resp.read()

        fake_db.build_chat_history.assert_called_once()
        # history 被传给 query_stream
        _, kwargs = fake_pipeline.query_stream.call_args
        assert "history" in kwargs
        assert kwargs["history"] is not None

    def test_stream_persists_user_message(self, wired):
        client, fake_db, _ = wired
        with client.stream(
            "POST", "/api/chat/stream", json={"message": "接着说"}
        ) as resp:
            resp.read()
        # 会话持久化：创建会话 + 写入用户消息
        fake_db.create_session.assert_called_once()
        fake_db.add_message.assert_any_call("sess_1", "user", "接着说")
