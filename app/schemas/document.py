from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Optional


class DocumentBase(BaseModel):
    filename: str
    version: Optional[str] = None
    title: Optional[str] = None
    content_type: Optional[str] = None


class DocumentCreate(DocumentBase):
    file_path: str
    file_hash: str
    original_filename: str

    @field_validator('file_path')
    @classmethod
    def normalize_file_path(cls, v: str) -> str:
        # Convert Windows paths to forward slashes
        return v.replace('\\', '/')


class Document(DocumentCreate):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True 