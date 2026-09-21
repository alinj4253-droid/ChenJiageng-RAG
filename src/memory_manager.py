"""
滚动摘要管理（长期记忆）

设计：
    长期摘要（summary）
    + 近期未摘要、且已滑出 recent window 的对话（增量并入摘要）
    + recent window 原文（由 DBManager.build_chat_history 提供）

关键约束：
    - 只摘要 summarized_until_message_id 之后、且已滑出 recent window 的消息；
    - recent window 内的消息保留原文，不提前摘要；
    - 已摘要过的消息不会被重复摘要（靠摘要游标保证）。
"""
from typing import Optional

from src.llm_client import get_client
from src.prompts import ROLLING_SUMMARY_SYSTEM, ROLLING_SUMMARY_USER
from src.config import RECENT_WINDOW


class RollingSummaryManager:
    """增量滚动摘要器"""

    def __init__(self, db, client=None, recent_window: int = RECENT_WINDOW):
        """
        Args:
            db: DBManager 实例
            client: 可注入的 LLM 客户端（测试时传入 mock）
            recent_window: 保留原文的最近消息条数
        """
        self.db = db
        self.client = client if client is not None else get_client()
        self.recent_window = recent_window

    def maybe_summarize(self, session_id: str) -> bool:
        """
        检查是否有滑出 recent window 的新消息，有则增量并入摘要。

        Returns:
            True 表示本次更新了摘要；False 表示没有待摘要内容。
        """
        pending, last_id = self.db.get_pending_summary_messages(
            session_id, self.recent_window
        )
        if not pending:
            return False

        old_summary = self.db.get_session_summary(session_id)
        new_summary = self._merge_summary(old_summary, pending)
        if not new_summary:
            return False

        self.db.update_session_summary(
            session_id, new_summary, until_message_id=last_id
        )
        return True

    def _merge_summary(self, old_summary: str, pending_messages) -> str:
        """调用 LLM，把新增对话并入已有摘要"""
        dialogue_lines = [
            f"{m['role']}: {m['content'][:300]}" for m in pending_messages
        ]
        existing_part = (
            f"已有摘要（更早的对话，请勿遗漏其中关键信息）：\n{old_summary}\n\n"
            if old_summary
            else ""
        )
        prompt = ROLLING_SUMMARY_USER.format(
            existing_summary_part=existing_part,
            new_dialogue="\n".join(dialogue_lines),
        )
        result = self.client.chat(
            messages=[
                {"role": "system", "content": ROLLING_SUMMARY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        return (result or "").strip()
