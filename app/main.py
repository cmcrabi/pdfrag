from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import logging
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.database import init_db, get_db
from app.models import document as models_doc # Renamed to avoid conflict
from app.models import product as models_prod # Import product model
from app.schemas import document as schemas_document # Rename existing schema import
from app.schemas import product as schemas_product # Add product schema import
from app.crud import document as crud_document # Rename existing crud import
from app.crud import crud_product # Add product crud import
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
@app.get("/documents/", response_model=List[schemas_document.Document], tags=["documents"])
def read_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve a list of documents with pagination support.
    """
    documents = crud_document.get_documents(db, skip=skip, limit=limit)
    return documents

@app.get("/documents/{document_id}", response_model=schemas_document.Document, tags=["documents"])
def read_document(document_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a specific document by its ID.
    """
    db_document = crud_document.get_document(db, document_id=document_id)
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
    product_id: int, # Make product_id mandatory (remove Optional and default)
    limit: int = 5,
    threshold: float = 0.7,
    # Remove: document_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Search for similar content within documents of a specific product.
    
    Parameters:
    - query: Search text
    - product_id: ID of the product to search within (mandatory)
    - limit: Maximum number of results (default: 5)
    - threshold: Minimum similarity score 0-1 (default: 0.7)
    # Remove: - document_id: Optional filter for specific document
    """
    # Remove: if document_id and product_id:
    # Remove:     raise HTTPException(status_code=400, detail="Cannot filter by both document_id and product_id simultaneously.")

    try:
        search_service = SearchService(db)
        results = await search_service.search(
            query=query,
            limit=limit,
            threshold=threshold,
            product_id=product_id, # Pass mandatory product_id
            # Remove: document_id=document_id,
        )
        return {
            "query": query,
            "product_id": product_id, # Include product_id in response
            "results": results
        }
    except Exception as e:
        # Consider adding specific exception handling for product not found if SearchService doesn't handle it
        logger.error(f"Error during search for product {product_id}: {str(e)}", exc_info=True)
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

@app.post("/documents/upload", response_model=schemas_document.Document)
async def upload_document(
    product_id: int = Form(...), # Add product_id as required form data
    file: UploadFile = File(...),
    version: Optional[str] = Form(None), # Keep optional fields as Form data
    title: Optional[str] = Form(None),
    content_type: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF document and associate it with a product ID.
    
    - Saves file with timestamp
    - Checks for duplicates using hash
    - Creates document record with product_id
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Optional: Validate product_id exists
    db_product = crud_product.get_product(db, product_id=product_id) # Assuming get_product exists in crud_product
    if not db_product:
        raise HTTPException(status_code=404, detail=f"Product with ID {product_id} not found.")

    try:
        file_service = FileService()
        
        # Save file and get info
        file_info = await file_service.save_uploaded_file(file)
        current_file_hash = file_info["file_hash"]
        logger.info(f"Calculated hash for uploaded file '{file.filename}': {current_file_hash}") # Log hash
        
        # Check if document with same hash exists
        logger.info(f"Checking database for existing document with hash: {current_file_hash}") # Log before check
        existing_doc = crud_document.get_document_by_hash(db, current_file_hash)
        
        if existing_doc:
            logger.warning(f"Found existing document (ID: {existing_doc.id}) with hash {current_file_hash}. Deleting uploaded file.") # Log if found
            # Delete the uploaded file
            file_service.delete_file(file_info["file_path"])
            raise HTTPException(
                status_code=409,
                detail=f"Document already exists with ID: {existing_doc.id}"
            )
        else:
            logger.info(f"No existing document found with hash: {current_file_hash}. Proceeding with creation.") # Log if not found
        
        # Create document record schema
        document_data = schemas_document.DocumentCreate(
            filename=file_info["saved_filename"],
            version=version,
            title=title,
            content_type=content_type or file.content_type, # Use file's content type if not provided
            file_path=file_info["file_path"],
            file_hash=file_info["file_hash"],
            original_filename=file_info["original_filename"],
            product_id=product_id # Pass the validated product_id
        )
        
        # Create document in DB
        return crud_document.create_document(db, document=document_data)
        
    except ValueError as e:
        # Catch potential value errors (though less likely here now)
        logger.error(f"Value error during document upload processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        # Catch DB constraint violations (e.g., non-existent product_id, duplicate hash if check failed somehow)
        logger.error(f"Database integrity error creating document: {str(e)}", exc_info=True)
        db.rollback() # Rollback the session
        # Check if it's a foreign key violation specifically
        if "violates foreign key constraint" in str(e).lower() and "fk_document_product" in str(e).lower():
             raise HTTPException(status_code=400, detail=f"Error creating document: Product ID {product_id} does not exist.")
        # Check if it's the unique hash constraint (should have been caught earlier, but for robustness)
        elif "violates unique constraint" in str(e).lower() and "uq_document_hash" in str(e).lower():
             raise HTTPException(status_code=409, detail=f"Error creating document: Document hash {current_file_hash} already exists (commit level)." )
        else:
             # Generic integrity error
             raise HTTPException(status_code=500, detail=f"Database integrity error: {str(e)}")
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error uploading document: {str(e)}", exc_info=True)
        # Ensure rollback in case of other errors during DB operations
        try:
            db.rollback()
        except Exception as rb_e:
            logger.error(f"Error during rollback attempt: {rb_e}")
        raise HTTPException(status_code=500, detail="Internal server error during document upload")

@app.get("/search/enhanced")
async def enhanced_search(
    query: str,
    product_id: int, # Make product_id mandatory
    limit: int = 5,
    threshold: float = 0.3,
    # Remove: document_id: Optional[int] = None,
    context_pages: int = 3,
    include_detailed_results: bool = False,
    db: Session = Depends(get_db)
):
    """
    Enhanced search within a specific product, combining vector search with LLM.
    
    Parameters:
    - query: The search query
    - product_id: ID of the product to search within (mandatory)
    - limit: Maximum number of results to return
    - threshold: Similarity threshold for results
    # Remove: - document_id: Optional document ID to search within
    - context_pages: Number of pages to include around matches
    - include_detailed_results: If True, includes detailed search results in response
    """
    # Remove: if document_id and product_id:
    # Remove:     raise HTTPException(status_code=400, detail="Cannot search by both document_id and product_id simultaneously.")
        
    try:
        # First, perform the vector search, passing mandatory product_id
        search_service = SearchService(db)
        search_results = await search_service.search(
            query=query,
            limit=limit,
            threshold=threshold,
            product_id=product_id, # Pass mandatory product_id
            # Remove: document_id=document_id,
            context_pages=context_pages
        )
        
        # Check if search returned any results before calling LLM
        if not search_results:
             return {
                 "query": query,
                 "product_id": product_id,
                 "llm_response": "No relevant documents found for this product to generate an answer.",
                 "search_results": [] if include_detailed_results else None
             }

        # Then, generate an LLM response
        llm_service = LLMService(provider="gemini") # Or your configured provider
        llm_response = await llm_service.generate_response(
            query=query,
            search_results=search_results
        )

        # Prepare response
        response = {
            "query": query,
            "product_id": product_id, # Include product_id in response
            "llm_response": llm_response
        }

        # Only include detailed results if requested
        if include_detailed_results:
            response["search_results"] = search_results

        return response

    except Exception as e:
        logger.error(f"Error in enhanced search for product {product_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Product Endpoints ---

@app.post("/products/", response_model=schemas_product.Product, tags=["products"])
def create_product_endpoint(
    product: schemas_product.ProductCreate,
    db: Session = Depends(get_db)
):
    """Creates a new product."""
    db_product = crud_product.get_product_by_name(db, name=product.name)
    if db_product:
        raise HTTPException(status_code=400, detail=f"Product with name '{product.name}' already exists.")
    return crud_product.create_product(db=db, product=product)

@app.get("/products/", response_model=List[schemas_product.Product], tags=["products"])
def list_products_endpoint(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """Lists all products with pagination."""
    # Add a more standard list function in crud_product if needed
    products = db.query(models_prod.Product).offset(skip).limit(limit).all()
    return products

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )