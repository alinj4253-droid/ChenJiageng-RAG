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


class TestChatHistory:
    """build_chat_history：保证当前问题不重复进入生成器"""

    def test_history_excludes_current_query_before_insert(self, db):
        """
        历史 user A / assistant B，当前问题 C 尚未入库时构建 history：
        history 只能包含 A、B，不能包含 C。
        """
        sid = db.create_session("t")
        db.add_message(sid, "user", "A")
        db.add_message(sid, "assistant", "B")

        # 关键：在存入当前问题 C 之前构建历史
        history = db.build_chat_history(sid)
        contents = [m["content"] for m in history]

        assert "C" not in contents
        assert "A" in contents
        assert "B" in contents

    def test_current_query_appears_once_in_generator_messages(self, db):
        """
        模拟 app.py 的正确顺序：先取 history，再存当前问题。
        最终送入生成器的 messages = history + [当前 prompt]，
        当前问题 C 在整条 messages 中只能出现一次。
        """
        sid = db.create_session("t")
        db.add_message(sid, "user", "历史问题A")
        db.add_message(sid, "assistant", "历史回答B")

        current_query = "当前问题C"

        # 1. 入库前取历史
        history = db.build_chat_history(sid)
        # 2. 入库当前问题
        db.add_message(sid, "user", current_query)

        # 3. 生成器侧：history 之后再 append 包含当前问题的 prompt
        generator_messages = list(history)
        generator_messages.append({"role": "user", "content": f"上下文...\n问题：{current_query}"})

        # 当前问题在 messages 中只应出现一次（最后的 prompt）
        occurrences = sum(
            1 for m in generator_messages if current_query in m["content"]
        )
        assert occurrences == 1

    def test_history_includes_summary_as_system(self, db):
        """存在摘要时，history 第一条为 system 摘要"""
        sid = db.create_session("t")
        db.update_session_summary(sid, "长期摘要")
        db.add_message(sid, "user", "A")

        history = db.build_chat_history(sid)
        assert history[0]["role"] == "system"
        assert "长期摘要" in history[0]["content"]

    def test_history_respects_recent_window(self, db):
        """recent_window 之外的旧消息不进入短期历史（应由摘要承载）"""
        sid = db.create_session("t")
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            db.add_message(sid, role, f"msg-{i}")

        history = db.build_chat_history(sid, recent_window=6)
        # 不含摘要，所以 history 只有最近 6 条
        contents = [m["content"] for m in history]
        assert "msg-0" not in contents
        assert "msg-9" in contents
        assert len(history) == 6


class TestIncrementalSummaryWindow:
    """get_pending_summary_messages 的增量窗口逻辑"""

    def _fill(self, db, sid, n):
        """在现有消息基础上追加到共 n 条（幂等，不重复插入）"""
        existing = len(db.get_session_messages(sid))
        for i in range(existing + 1, n + 1):
            role = "user" if i % 2 == 1 else "assistant"
            db.add_message(sid, role, f"msg-{i}")

    def test_no_pending_within_recent_window(self, db):
        """消息数未超过 recent window → 无待摘要消息"""
        sid = db.create_session("t")
        self._fill(db, sid, 6)
        pending, last_id = db.get_pending_summary_messages(sid, recent_window=6)
        assert pending == []
        assert last_id is None

    def test_first_summary_covers_messages_slid_out(self, db):
        """
        8 条消息、recent_window=6：
        最近 6 条（3~8）保留原文，前 2 条（1~2）滑出窗口待摘要。
        """
        sid = db.create_session("t")
        self._fill(db, sid, 8)
        pending, last_id = db.get_pending_summary_messages(sid, recent_window=6)

        contents = [m["content"] for m in pending]
        assert contents == ["msg-1", "msg-2"]
        assert last_id == pending[-1]["id"]

    def test_recent_window_not_summarized_early(self, db):
        """recent window 内的消息绝不出现在待摘要列表"""
        sid = db.create_session("t")
        self._fill(db, sid, 7)  # 只有 msg-1 滑出
        pending, _ = db.get_pending_summary_messages(sid, recent_window=6)
        contents = [m["content"] for m in pending]
        assert contents == ["msg-1"]

    def test_no_duplicate_after_cursor_advanced(self, db):
        """
        第一次摘要覆盖 1~2 后推进游标；
        增长到 10 条消息时，第二次只摘要 3~4，不再包含 1~2。
        """
        sid = db.create_session("t")
        self._fill(db, sid, 8)

        # 第一次摘要
        pending1, last1 = db.get_pending_summary_messages(sid, recent_window=6)
        assert [m["content"] for m in pending1] == ["msg-1", "msg-2"]
        db.update_session_summary(sid, "摘要(1-2)", until_message_id=last1)

        # 又新增 2 条 → 共 10 条
        self._fill(db, sid, 10)
        pending2, last2 = db.get_pending_summary_messages(sid, recent_window=6)
        contents2 = [m["content"] for m in pending2]

        # 不得再次包含已摘要的 1~2
        assert "msg-1" not in contents2
        assert "msg-2" not in contents2
        assert contents2 == ["msg-3", "msg-4"]
        assert last2 > last1

    def test_cursor_persisted(self, db):
        """摘要游标随会话持久化"""
        sid = db.create_session("t")
        self._fill(db, sid, 8)
        pending, last_id = db.get_pending_summary_messages(sid, recent_window=6)
        db.update_session_summary(sid, "s", until_message_id=last_id)

        # 重新查询，游标仍在 → 无重复待摘要
        pending_again, _ = db.get_pending_summary_messages(sid, recent_window=6)
        assert pending_again == []
        assert db.get_session_summary(sid) == "s"


