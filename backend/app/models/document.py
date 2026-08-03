from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, func, BigInteger, JSON
from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)  # pdf | docx | txt
    file_size = Column(BigInteger, nullable=True)
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="processing")  # processing | ready | error
    error_message = Column(String(500), nullable=True)
    collection_name = Column(String(100), nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    session_id = Column(Integer, nullable=True)
    query = Column(String(2000), nullable=False)
    response_preview = Column(String(500), nullable=True)
    trust_score = Column(Integer, nullable=True)
    hallucination_risk = Column(String(20), nullable=True)
    correction_iterations = Column(Integer, default=0)
    latency_ms = Column(Integer, nullable=True)
    layer_traces = Column(JSON, nullable=True)
    model_used = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
