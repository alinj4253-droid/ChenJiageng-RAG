"""
对话历史存储模块
基于SQLAlchemy ORM，支持SQLite/PostgreSQL切换
"""
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


class Session(Base):
    """对话会话表"""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(256), nullable=False)
    summary = Column(Text, nullable=True)  # 对话滚动摘要（长期记忆）
    # 摘要已覆盖到哪一条消息（Message.id）；之后滑出 recent window 的消息才需增量摘要
    summarized_until_message_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    messages = relationship("Message", back_populates="session", order_by="Message.created_at")


class Message(Base):
    """对话消息表"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # 引用来源JSON
    created_at = Column(DateTime, default=datetime.now)
    
    session = relationship("Session", back_populates="messages")


class DBManager:
    """数据库管理器"""
    
    def __init__(self, db_url: str = "sqlite:///data/chat.db"):
        """
        初始化数据库
        :param db_url: 数据库连接字符串
            - SQLite: sqlite:///data/chat.db
            - PostgreSQL: postgresql://user:password@localhost:5432/ragdb
        """
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self._ensure_schema_migrations()
        SessionLocal = sessionmaker(bind=self.engine)
        self.session = SessionLocal()

    def _ensure_schema_migrations(self):
        """轻量迁移：为旧版数据库补充后加的列（SQLite 也支持 ADD COLUMN）"""
        from sqlalchemy import inspect, text
        inspector = inspect(self.engine)
        if "sessions" not in inspector.get_table_names():
            return
        columns = {c["name"] for c in inspector.get_columns("sessions")}
        with self.engine.begin() as conn:
            if "summarized_until_message_id" not in columns:
                conn.execute(text(
                    "ALTER TABLE sessions ADD COLUMN summarized_until_message_id INTEGER"
                ))
    
    def create_session(self, title: str = "新对话") -> str:
        """创建新对话，返回session_id"""
        import uuid
        session_id = str(uuid.uuid4())[:8]
        s = Session(session_id=session_id, title=title)
        self.session.add(s)
        self.session.commit()
        return session_id
    
    def get_sessions(self) -> List[Dict]:
        """获取所有对话列表"""
        sessions = self.session.query(Session).order_by(Session.updated_at.desc()).all()
        return [
            {
                "session_id": s.session_id,
                "title": s.title,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]
    
    def get_session_messages(self, session_id: str) -> List[Dict]:
        """获取某个对话的所有消息"""
        messages = self.session.query(Message)\
            .filter(Message.session_id == session_id)\
            .order_by(Message.created_at.asc())\
            .all()
        return [
            {
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ]
    
    def build_chat_history(self, session_id: str, recent_window: int = 6) -> List[Dict]:
        """
        构建传给生成器的对话历史：长期摘要(system) + 最近 recent_window 条消息。

        关键约束：必须在“存入本轮 user 消息之前”调用，
        否则当前问题会同时出现在 history 和最终 prompt 中，造成重复。

        返回: [{"role": "system"/"user"/"assistant", "content": ...}, ...]
        """
        history = []
        summary = self.get_session_summary(session_id)
        if summary:
            history.append({"role": "system",
                            "content": f"之前对话的摘要：{summary}"})
        messages = self.get_session_messages(session_id)
        for m in messages[-recent_window:]:
            history.append({"role": m["role"], "content": m["content"]})
        return history

    def add_message(self, session_id: str, role: str, content: str, sources: str = None):
        """添加一条消息"""
        # 更新session的updated_at
        s = self.session.query(Session).filter(Session.session_id == session_id).first()

        # 标题与计数判断必须在 add 新消息之前查询，
        # 否则 SQLAlchemy 自动 flush 会把本条消息计入 count。
        is_first_user_message = False
        if s and role == "user":
            count = self.session.query(Message).filter(Message.session_id == session_id).count()
            if count == 0:
                is_first_user_message = True

        msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources
        )
        self.session.add(msg)

        if s:
            s.updated_at = datetime.now()
            # 如果是第一条用户消息，更新标题
            if is_first_user_message:
                s.title = content[:30] + "..."

        self.session.commit()
    
    def delete_session(self, session_id: str):
        """删除对话"""
        # 先删消息
        self.session.query(Message).filter(Message.session_id == session_id).delete()
        # 再删session
        self.session.query(Session).filter(Session.session_id == session_id).delete()
        self.session.commit()
    
    def get_session_summary(self, session_id: str) -> str:
        """获取会话的滚动摘要"""
        s = self.session.query(Session).filter(Session.session_id == session_id).first()
        return s.summary if s and s.summary else ""

    def get_pending_summary_messages(self, session_id: str,
                                     recent_window: int = 6):
        """
        返回需要“增量摘要”的消息：
        位于 summarized_until_message_id 之后、且已经滑出 recent window 的消息。

        recent window（最近 recent_window 条）保留原文，不参与本次摘要。

        返回: (messages, last_message_id)
            messages = [{"id", "role", "content"}, ...]
            无待摘要消息时返回 ([], None)
        """
        s = self.session.query(Session).filter(Session.session_id == session_id).first()
        until_id = s.summarized_until_message_id if (s and s.summarized_until_message_id) else 0

        all_msgs = (
            self.session.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.id.asc())
            .all()
        )

        # 消息数未超过 recent window，没有需要摘要的旧消息
        if len(all_msgs) <= recent_window:
            return [], None

        candidates = all_msgs[:-recent_window]
        pending = [m for m in candidates if m.id > until_id]
        if not pending:
            return [], None

        messages = [{"id": m.id, "role": m.role, "content": m.content} for m in pending]
        return messages, pending[-1].id

    def update_session_summary(self, session_id: str, summary: str,
                               until_message_id: Optional[int] = None):
        """
        更新会话的滚动摘要。
        until_message_id: 本次摘要已覆盖到的最后一条消息 id（摘要游标）。
        """
        s = self.session.query(Session).filter(Session.session_id == session_id).first()
        if s:
            s.summary = summary
            if until_message_id is not None:
                s.summarized_until_message_id = until_message_id
            self.session.commit()
    
    def close(self):
        self.session.close()


# 测试
if __name__ == "__main__":
    db = DBManager()
    print("数据库初始化成功")
    
    # 测试创建会话
    sid = db.create_session("测试对话")
    print(f"创建会话: {sid}")
    
    # 测试添加消息
    db.add_message(sid, "user", "陈嘉庚创办了哪些学校？")
    db.add_message(sid, "assistant", "厦门大学、集美学校等", "《陈嘉庚言论新集》")
    
    # 测试查询
    sessions = db.get_sessions()
    print(f"对话列表: {len(sessions)} 个")
    
    messages = db.get_session_messages(sid)
    print(f"消息数: {len(messages)}")
    for m in messages:
        print(f"  [{m['role']}] {m['content'][:50]}")
    
    db.close()
    print("✅ 测试通过")
