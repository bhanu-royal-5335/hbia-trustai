from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


class SourceCitation(BaseModel):
    document_id: Optional[str] = None
    filename: str
    page: Optional[int] = None
    excerpt: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    chunk_index: Optional[int] = None


class ClaimResult(BaseModel):
    claim_text: str
    is_supported: bool
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_source: Optional[str] = None


class VerificationResult(BaseModel):
    trust_score: float = Field(ge=0.0, le=100.0)
    hallucination_risk: str  # low | medium | high
    claims: List[ClaimResult] = []
    verified_claims: int = 0
    total_claims: int = 0
    has_hallucinations: bool = False
    hallucinated_spans: List[str] = []


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: Optional[int] = None
    stream: bool = False
    use_web_search: bool = False


class ChatResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    response: str
    trust_score: float
    verification: VerificationResult
    sources: List[SourceCitation]
    correction_iterations: int
    latency_ms: int
    session_id: int
    message_id: int
    model_used: str
    token_usage: Optional[dict] = None


class StreamChunk(BaseModel):
    type: str  # token | verification_start | verification_complete | done | error
    content: Optional[str] = None
    data: Optional[Any] = None


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    trust_score: Optional[float]
    hallucination_risk: Optional[str]
    sources: Optional[List[SourceCitation]]
    verification_data: Optional[VerificationResult]
    correction_iterations: int
    latency_ms: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class SessionCreate(BaseModel):
    title: str = "New Chat"


class SessionResponse(BaseModel):
    id: int
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SessionWithMessages(BaseModel):
    id: int
    title: str
    message_count: int
    messages: List[MessageResponse]
    created_at: datetime

    class Config:
        from_attributes = True
