import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.document import Document
from app.schemas.auth import DocumentUploadResponse, DocumentListResponse
from app.services.document_service import document_service

router = APIRouter()


@router.post("/ingest", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")
        
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ["pdf", "docx", "txt", "doc"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    doc_id_str = str(uuid.uuid4())
    content = await file.read()
    
    new_doc = Document(
        user_id=current_user.id,
        filename=doc_id_str,
        original_filename=file.filename,
        file_type=ext,
        file_size=len(content),
        status="processing",
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    background_tasks.add_task(
        document_service.process_document,
        db=db,
        doc_id=new_doc.id,
        doc_uuid=doc_id_str,
        file_bytes=content,
        original_filename=file.filename,
    )

    return new_doc


@router.get("/", response_model=List[DocumentListResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(Document.user_id == current_user.id).order_by(Document.created_at.desc())
    )
    return result.scalars().all()


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc = await document_service.get_document(db, doc_id, current_user.id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    await document_service.delete_document(db, doc)
    return {"status": "success", "message": "Document deleted"}
