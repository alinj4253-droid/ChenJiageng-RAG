"""
对话存储（DBManager）测试
使用临时 SQLite 数据库，不污染真实 data/chat.db
"""
import pytest
from src.db import DBManager


@pytest.fixture
def db(tmp_path):
    """每个测试使用独立临时数据库"""
    db_path = tmp_path / "test_chat.db"
    manager = DBManager(db_url=f"sqlite:///{db_path}")
    yield manager
    manager.close()


class TestSessionAndMessage:
    """会话与消息基本读写"""

    def test_create_session_returns_id(self, db):
        sid = db.create_session("测试对话")
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_add_and_get_messages(self, db):
        sid = db.create_session("t")
        db.add_message(sid, "user", "问题A")
        db.add_message(sid, "assistant", "回答B")

        messages = db.get_session_messages(sid)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "问题A"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "回答B"

    def test_messages_ordered_chronologically(self, db):
        sid = db.create_session("t")
        for i in range(5):
            db.add_message(sid, "user", f"msg-{i}")

        messages = db.get_session_messages(sid)
        contents = [m["content"] for m in messages]
        assert contents == [f"msg-{i}" for i in range(5)]

    def test_delete_session_removes_messages(self, db):
        sid = db.create_session("t")
        db.add_message(sid, "user", "hello")
        db.delete_session(sid)

        assert db.get_session_messages(sid) == []

    def test_first_user_message_updates_title(self, db):
        sid = db.create_session("默认标题")
        db.add_message(sid, "user", "这是第一个问题的内容" * 3)

        sessions = db.get_sessions()
        target = next(s for s in sessions if s["session_id"] == sid)
        assert "这是第一个问题" in target["title"]


class TestRollingSummary:
    """滚动摘要读写"""

    def test_summary_initially_empty(self, db):
        sid = db.create_session("t")
        assert db.get_session_summary(sid) == ""

    def test_update_and_get_summary(self, db):
        sid = db.create_session("t")
        db.update_session_summary(sid, "这是摘要内容")
        assert db.get_session_summary(sid) == "这是摘要内容"

    def test_summary_independent_between_sessions(self, db):
        s1 = db.create_session("a")
        s2 = db.create_session("b")
        db.update_session_summary(s1, "摘要A")

        assert db.get_session_summary(s1) == "摘要A"
        assert db.get_session_summary(s2) == ""
