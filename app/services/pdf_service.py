from typing import Dict
import os
from pathlib import Path
from app.processors.pdf_processor import PDFProcessor
from app.models.document import Document
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self, db: Session):
        self.db = db
    
    async def process_document(self, document_id: int) -> Dict:
        """
        Process a document and store its content.
        """
        try:
            # Get document from database
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document with id {document_id} not found")
            
            # Verify file exists
            file_path = document.file_path.replace('/', os.path.sep)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"PDF file not found at path: {file_path}")
            
            logger.info(f"Processing document {document_id}: {document.filename}")
            logger.info(f"File path: {file_path}")
            
            # Initialize PDF processor
            processor = PDFProcessor(
                pdf_path=file_path,
                document_id=document_id
            )
            
            # Process PDF
            content = processor.process_pdf()
            
            # Store processing results
            result = {
                "document_id": document_id,
                "metadata": content["metadata"],
                "processed_path": f"data/processed/{document_id}",
                "page_count": len(content["pages"]),
                "status": "success"
            }
            
            # Update document with metadata if needed
            if not document.title and content["metadata"].get("title"):
                document.title = content["metadata"]["title"]
                self.db.commit()
            
            logger.info(f"Successfully processed document {document_id}")
            return result
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            raise ValueError(f"PDF file not found: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            raise 