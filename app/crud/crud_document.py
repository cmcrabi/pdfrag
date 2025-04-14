from sqlalchemy.orm import Session
from app.models.document import Document
from app.schemas.document import DocumentCreate

def get_document(db: Session, document_id: int) -> Document | None:
    return db.query(Document).filter(Document.id == document_id).first()

def get_document_by_hash(db: Session, file_hash: str) -> Document | None:
    return db.query(Document).filter(Document.file_hash == file_hash).first()

def get_documents(db: Session, skip: int = 0, limit: int = 100) -> list[Document]:
    return db.query(Document).offset(skip).limit(limit).all()

def get_documents_by_product_id(db: Session, product_id: int, skip: int = 0, limit: int = 100) -> list[Document]:
    """Gets documents filtered by product ID."""
    return db.query(Document).filter(Document.product_id == product_id).offset(skip).limit(limit).all()


def create_document(db: Session, document: DocumentCreate) -> Document:
    # Create the document instance directly with product_id from input schema
    db_document = Document(
        filename=document.filename,
        version=document.version,
        title=document.title,
        content_type=document.content_type,
        file_path=document.file_path,
        file_hash=document.file_hash,
        original_filename=document.original_filename,
        product_id=document.product_id  # Use product_id from the input schema
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document 