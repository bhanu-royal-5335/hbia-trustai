from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.session import Session
from app.models.message import Message


class SessionService:
    async def get_session(self, db: AsyncSession, session_id: int, user_id: int) -> Optional[Session]:
        result = await db.execute(
            select(Session).where(Session.id == session_id, Session.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def create_session(self, db: AsyncSession, user_id: int, title: str = "New Chat") -> Session:
        session = Session(user_id=user_id, title=title)
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return session

    async def get_recent_messages(self, db: AsyncSession, session_id: int, limit: int = 6) -> List[dict]:
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        # Return in chronological order
        return [{"role": m.role, "content": m.content} for m in reversed(messages)]

    async def add_message(
        self,
        db: AsyncSession,
        session_id: int,
        role: str,
        content: str,
        trust_score: float = None,
        hallucination_risk: str = None,
        sources: list = None,
        verification_data: dict = None,
        correction_iterations: int = 0,
        latency_ms: int = None,
        model_used: str = None,
        token_usage: dict = None,
    ) -> Message:
        msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            trust_score=trust_score,
            hallucination_risk=hallucination_risk,
            sources=sources,
            verification_data=verification_data,
            correction_iterations=correction_iterations,
            latency_ms=latency_ms,
            model_used=model_used,
            token_usage=token_usage,
        )
        db.add(msg)
        
        # Update session message count
        session = await db.get(Session, session_id)
        if session:
            session.message_count += 1
            
        await db.commit()
        await db.refresh(msg)
        return msg


session_service = SessionService()
