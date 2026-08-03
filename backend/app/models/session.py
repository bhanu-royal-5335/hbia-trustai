from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, func, Text, Float, JSON
from app.core.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), default="New Chat")
    message_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' | 'assistant' | 'system'
    content = Column(Text, nullable=False)
    trust_score = Column(Float, nullable=True)
    hallucination_risk = Column(String(20), nullable=True)  # low | medium | high
    sources = Column(JSON, nullable=True)
    verification_data = Column(JSON, nullable=True)
    correction_iterations = Column(Integer, default=0)
    latency_ms = Column(Integer, nullable=True)
    model_used = Column(String(100), nullable=True)
    token_usage = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
