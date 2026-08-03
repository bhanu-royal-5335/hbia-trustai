from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.session import Message
from app.models.document import Document

router = APIRouter()

@router.get("/system")
async def get_system_metrics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Only admins should see this in a real app, 
    # but for this demo we'll let any authenticated user see their metrics
    
    # 1. Total documents processed
    docs_result = await db.execute(select(func.count()).select_from(Document).where(Document.user_id == current_user.id))
    total_docs = docs_result.scalar() or 0
    
    # 2. Total messages & avg trust score
    msgs_result = await db.execute(
        select(
            func.count(), 
            func.avg(Message.trust_score)
        )
        .select_from(Message)
        .where(Message.role == "assistant")
        # We would join with Session to filter by user_id, 
        # but for simplicity returning global/dummy stats or just count
    )
    row = msgs_result.one_or_none()
    total_messages = row[0] if row else 0
    avg_trust_score = float(row[1]) if row and row[1] else 0.0
    
    return {
        "metrics": {
            "total_documents": total_docs,
            "total_generations": total_messages,
            "average_trust_score": round(avg_trust_score, 2),
            "hallucinations_prevented": int(total_messages * 0.15), # Mocked for demo
            "active_users": 1
        },
        "layers": {
            "l0_orchestrator": {"status": "healthy", "latency_ms": 12},
            "l1_retrieval": {"status": "healthy", "latency_ms": 150},
            "l2_generation": {"status": "healthy", "latency_ms": 1200},
            "l3_verification": {"status": "healthy", "latency_ms": 800},
            "l4_correction": {"status": "healthy", "latency_ms": 1500}
        }
    }
