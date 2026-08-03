import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse
from app.layers.l0_orchestrator.orchestrator import orchestrator
from app.services.session_service import session_service
import structlog

logger = structlog.get_logger()
router = APIRouter()


@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session_id = request.session_id
    if session_id:
        # Verify session belongs to user
        session = await session_service.get_session(db, session_id, current_user.id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        # Create new session
        session = await session_service.create_session(
            db, current_user.id, title=request.query[:50]
        )
        session_id = session.id

    chat_history = await session_service.get_recent_messages(db, session_id, limit=6)
    
    # Save user message
    await session_service.add_message(
        db, session_id, "user", request.query
    )

    try:
        if request.stream:
            # Handle streaming in a separate endpoint usually, but left here for completeness
            raise HTTPException(status_code=400, detail="Use /stream endpoint for streaming")
        
        # Process through HBIA
        hbia_response = await orchestrator.process(request.query, chat_history=chat_history)
        
        # Save assistant message
        msg = await session_service.add_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=hbia_response.response,
            trust_score=hbia_response.trust_score,
            hallucination_risk=hbia_response.verification.hallucination_risk,
            sources=[s.model_dump() for s in hbia_response.sources],
            verification_data=hbia_response.verification.model_dump(),
            correction_iterations=hbia_response.correction_iterations,
            latency_ms=hbia_response.latency_ms,
            model_used=hbia_response.model_used,
            token_usage=hbia_response.token_usage,
        )
        
        return ChatResponse(
            response=hbia_response.response,
            trust_score=hbia_response.trust_score,
            verification=hbia_response.verification,
            sources=hbia_response.sources,
            correction_iterations=hbia_response.correction_iterations,
            latency_ms=hbia_response.latency_ms,
            session_id=session_id,
            message_id=msg.id,
            model_used=hbia_response.model_used,
            token_usage=hbia_response.token_usage,
        )
    except Exception as e:
        logger.error("chat_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error during chat processing")

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session_id = request.session_id
    if session_id:
        session = await session_service.get_session(db, session_id, current_user.id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = await session_service.create_session(
            db, current_user.id, title=request.query[:50]
        )
        session_id = session.id

    chat_history = await session_service.get_recent_messages(db, session_id, limit=6)
    
    await session_service.add_message(
        db, session_id, "user", request.query
    )

    return StreamingResponse(
        orchestrator.stream_process(request.query, chat_history),
        media_type="text/event-stream"
    )
