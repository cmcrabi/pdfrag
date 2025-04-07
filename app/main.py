from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import logging

from app.config import settings
from app.database import init_db, get_db
from app.models import document as models
from app.schemas import document as schemas
from app.crud import document as crud
from app.services.pdf_service import PDFService
from app.services.search_service import SearchService
from app.services.embedding_service import EmbeddingService
from app.models import Document, DocumentChunk
from app.services.file_service import FileService
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# Create FastAPI app with proper documentation settings
app = FastAPI(
    title=settings.APP_NAME,
    description="PDF RAG System API for managing and querying technical documents",
    version="1.0.0",
    docs_url="/docs",   # Swagger UI endpoint
    redoc_url="/redoc"  # ReDoc endpoint
)

# Add CORS middleware with both localhost and IP
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Create necessary directories
    settings.create_directories()
    
    # Initialize database
    init_db()

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "PDF RAG System API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "database": "connected"
    }

# Test endpoints for Document
@app.get("/documents/", response_model=List[schemas.Document], tags=["documents"])
def read_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve a list of documents with pagination support.
    """
    documents = crud.get_documents(db, skip=skip, limit=limit)
    return documents

@app.get("/documents/{document_id}", response_model=schemas.Document, tags=["documents"])
def read_document(document_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a specific document by its ID.
    """
    db_document = crud.get_document(db, document_id=document_id)
    if db_document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_document

@app.post("/documents/{document_id}/process", response_model=Dict)
async def process_document(
    document_id: int, 
    process_only: bool = False,  # If True, only process without vectorization
    db: Session = Depends(get_db)
):
    """
    Process a document and optionally create vector embeddings.
    
    Args:
        document_id: ID of the document to process
        process_only: If True, only process the document without creating vectors
    """
    try:
        # Step 1: Process the document
        pdf_service = PDFService(db)
        process_result = await pdf_service.process_document(document_id)
        
        if process_only:
            return {
                "status": "success",
                "message": "Document processed successfully",
                "process_result": process_result
            }
        
        # Step 2: Create vector embeddings
        vector_service = VectorService(db)
        vector_result = await vector_service.vectorize_document(document_id)
        
        return {
            "status": "success",
            "message": "Document processed and vectorized successfully",
            "process_result": process_result,
            "vector_result": vector_result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_documents(
    query: str,
    limit: int = 5,
    threshold: float = 0.7,
    document_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Search for similar content in processed documents.
    
    Parameters:
    - query: Search text
    - limit: Maximum number of results (default: 5)
    - threshold: Minimum similarity score 0-1 (default: 0.7)
    - document_id: Optional filter for specific document
    """
    try:
        search_service = SearchService(db)
        results = await search_service.search(
            query=query,
            limit=limit,
            threshold=threshold,
            document_id=document_id
        )
        return {
            "query": query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/by-example")
async def search_by_example(
    document_id: int,
    page_number: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """
    Search for similar content using a selected region from a document.
    
    Parameters:
    - document_id: Source document ID
    - page_number: Page number
    - x1, y1, x2, y2: Coordinates of selection box
    - limit: Maximum number of results
    """
    try:
        search_service = SearchService(db)
        results = await search_service.search_by_region(
            document_id=document_id,
            page_number=page_number,
            bbox=[x1, y1, x2, y2],
            limit=limit
        )
        return {
            "query": f"Region from document {document_id}, page {page_number}",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload", response_model=schemas.Document)
async def upload_document(
    file: UploadFile = File(...),
    version: Optional[str] = None,
    title: Optional[str] = None,
    content_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Upload a PDF document.
    
    - Saves file with timestamp
    - Checks for duplicates using hash
    - Creates document record
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        file_service = FileService()
        
        # Save file and get info
        file_info = await file_service.save_uploaded_file(file)
        
        # Check if document with same hash exists
        existing_doc = crud.get_document_by_hash(db, file_info["file_hash"])
        if existing_doc:
            # Delete the uploaded file
            file_service.delete_file(file_info["file_path"])
            raise HTTPException(
                status_code=409,
                detail=f"Document already exists with ID: {existing_doc.id}"
            )
        
        # Create document record
        document = schemas.DocumentCreate(
            filename=file_info["saved_filename"],
            version=version,
            title=title,
            content_type=content_type,
            file_path=file_info["file_path"],
            file_hash=file_info["file_hash"],
            original_filename=file_info["original_filename"]
        )
        
        return crud.create_document(db, document)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail="Error uploading document")

@app.get("/search/enhanced")
async def enhanced_search(
    query: str,
    limit: int = 5,
    threshold: float = 0.3,
    document_id: Optional[int] = None,
    context_pages: int = 3,
    include_detailed_results: bool = False,
    db: Session = Depends(get_db)
):
    """
    Enhanced search that combines vector search with LLM-generated responses.
    
    Parameters:
    - query: The search query
    - limit: Maximum number of results to return
    - threshold: Similarity threshold for results
    - document_id: Optional document ID to search within
    - context_pages: Number of pages to include around matches
    - include_detailed_results: If True, includes detailed search results in response
    """
    try:
        # First, perform the vector search
        search_service = SearchService(db)
        search_results = await search_service.search(
            query=query,
            limit=limit,
            threshold=threshold,
            document_id=document_id,
            context_pages=context_pages
        )

        # Then, generate an LLM response
        #llm_service = LLMService()
        llm_service = LLMService(provider="gemini")
        llm_response = await llm_service.generate_response(
            query=query,
            search_results=search_results,
            context_pages=context_pages
        )

        # Prepare response
        response = {
            "query": query,
            "llm_response": llm_response
        }

        # Only include detailed results if requested
        if include_detailed_results:
            response["search_results"] = search_results

        return response

    except Exception as e:
        logger.error(f"Error in enhanced search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )