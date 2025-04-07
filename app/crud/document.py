from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models.document import Document
from app.schemas.document import DocumentCreate
from typing import Optional

def get_document_by_hash(db: Session, file_hash: str) -> Optional[Document]:
    return db.query(Document).filter(Document.file_hash == file_hash).first()

def create_document(db: Session, document: DocumentCreate) -> Document:
    db_document = Document(
        filename=document.filename,
        version=document.version,
        title=document.title,
        content_type=document.content_type,
        file_path=document.file_path,
        file_hash=document.file_hash,
        original_filename=document.original_filename
    )
    try:
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        return db_document
    except IntegrityError:
        db.rollback()
        raise ValueError("Document with this hash already exists")

def get_document(db: Session, document_id: int):
    return db.query(Document).filter(Document.id == document_id).first()

def get_documents(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Document).offset(skip).limit(limit).all() 