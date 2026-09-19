from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(256), nullable=False)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    messages = relationship("Message", back_populates="session", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    session = relationship("Session", back_populates="messages")


class DBManager:
    def __init__(self, db_url: str = "sqlite:///data/chat.db"):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        SessionLocal = sessionmaker(bind=self.engine)
        self.session = SessionLocal()

    def create_session(self, title: str = "新对话") -> str:
        session_id = str(uuid.uuid4())[:8]
        s = Session(session_id=session_id, title=title)
        self.session.add(s)
        self.session.commit()
        return session_id

    def get_sessions(self):
        sessions = self.session.query(Session).order_by(Session.updated_at.desc()).all()
        return [{"session_id": s.session_id, "title": s.title} for s in sessions]

    def get_session_messages(self, session_id: str):
        messages = self.session.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at.asc()).all()
        return [{"role": m.role, "content": m.content, "sources": m.sources} for m in messages]

    def add_message(self, session_id: str, role: str, content: str, sources: str = None):
        msg = Message(session_id=session_id, role=role, content=content, sources=sources)
        self.session.add(msg)
        self.session.commit()

    def delete_session(self, session_id: str):
        self.session.query(Message).filter(Message.session_id == session_id).delete()
        self.session.query(Session).filter(Session.session_id == session_id).delete()
        self.session.commit()

    def get_session_summary(self, session_id: str) -> str:
        s = self.session.query(Session).filter(Session.session_id == session_id).first()
        return s.summary if s and s.summary else ""

    def update_session_summary(self, session_id: str, summary: str):
        s = self.session.query(Session).filter(Session.session_id == session_id).first()
        if s:
            s.summary = summary
            self.session.commit()