from sqlalchemy import Column, Integer, String, DateTime, Text, UniqueConstraint
from sqlalchemy.sql import func
from app.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    version = Column(String(50))
    title = Column(String(255))
    content_type = Column(String(50))
    file_path = Column(Text, nullable=False)
    file_hash = Column(String(64), nullable=False)  # SHA-256 hash
    original_filename = Column(String(255), nullable=False)  # Store original filename
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Add unique constraint on file_hash
    __table_args__ = (UniqueConstraint('file_hash', name='uq_document_hash'),)

    def __repr__(self):
        return f"<Document {self.filename} (v{self.version})>" 