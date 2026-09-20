"""
RollingSummaryManager 增量摘要测试
LLM 全部 mock，验证：只摘要滑出窗口的新消息、游标正确推进、不重复摘要。
"""
import pytest
from unittest.mock import MagicMock

from src.db import DBManager
from src.memory_manager import RollingSummaryManager


@pytest.fixture
def db(tmp_path):
    manager = DBManager(db_url=f"sqlite:///{tmp_path / 'mem.db'}")
    yield manager
    manager.close()


def _fill(db, sid, n):
    """在现有消息基础上追加到共 n 条（幂等，不重复插入）"""
    existing = len(db.get_session_messages(sid))
    for i in range(existing + 1, n + 1):
        role = "user" if i % 2 == 1 else "assistant"
        db.add_message(sid, role, f"msg-{i}")


def _make_manager(db, summary_text="合并后的摘要"):
    client = MagicMock()
    client.chat.return_value = summary_text
    manager = RollingSummaryManager(db, client=client, recent_window=6)
    return manager, client


class TestMaybeSummarize:

    def test_no_summarize_within_window(self, db):
        """消息都在 recent window 内 → 不调用 LLM，返回 False"""
        sid = db.create_session("t")
        _fill(db, sid, 4)
        manager, client = _make_manager(db)

        changed = manager.maybe_summarize(sid)
        assert changed is False
        client.chat.assert_not_called()

    def test_first_summarize_calls_llm_and_advances_cursor(self, db):
        """8 条消息 → 摘要滑出窗口的 1~2，更新摘要与游标"""
        sid = db.create_session("t")
        _fill(db, sid, 8)
        manager, client = _make_manager(db, "新摘要")

        changed = manager.maybe_summarize(sid)
        assert changed is True
        client.chat.assert_called_once()
        assert db.get_session_summary(sid) == "新摘要"

        # 游标推进后，再次检查没有重复待摘要
        pending, _ = db.get_pending_summary_messages(sid, recent_window=6)
        assert pending == []

    def test_second_run_does_not_resummarize_old_messages(self, db):
        """
        第二次摘要的 LLM 输入中不得包含第一次已摘要的 msg-1/msg-2。
        """
        sid = db.create_session("t")
        _fill(db, sid, 8)
        manager, client = _make_manager(db)
        manager.maybe_summarize(sid)

        # 增长到 10 条
        _fill(db, sid, 10)
        manager.maybe_summarize(sid)

        # 第二次调用的 prompt
        second_call_prompt = client.chat.call_args_list[1].kwargs["messages"][1]["content"] \
            if "messages" in client.chat.call_args_list[1].kwargs \
            else client.chat.call_args_list[1][0][0][1]["content"]

        assert "msg-1" not in second_call_prompt
        assert "msg-2" not in second_call_prompt
        assert "msg-3" in second_call_prompt
        assert "msg-4" in second_call_prompt
        # recent window（msg-5~10）不应被摘要
        assert "msg-9" not in second_call_prompt

    def test_existing_summary_passed_to_llm(self, db):
        """第二次摘要时，已有摘要作为基础传入 LLM"""
        sid = db.create_session("t")
        _fill(db, sid, 8)
        manager, client = _make_manager(db)
        manager.maybe_summarize(sid)

        _fill(db, sid, 10)
        manager.maybe_summarize(sid)

        second_prompt = client.chat.call_args_list[1].kwargs["messages"][1]["content"]
        assert "已有摘要" in second_prompt

    def test_empty_llm_output_does_not_update(self, db):
        """LLM 返回空 → 不更新摘要、不推进游标，返回 False"""
        sid = db.create_session("t")
        _fill(db, sid, 8)
        manager, client = _make_manager(db, summary_text="")
        client.chat.return_value = "   "

        changed = manager.maybe_summarize(sid)
        assert changed is False
        assert db.get_session_summary(sid) == ""
        # 游标未推进，消息仍处于待摘要状态
        pending, _ = db.get_pending_summary_messages(sid, recent_window=6)
        assert len(pending) == 2
