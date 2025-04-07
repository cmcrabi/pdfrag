# PDF RAG System

A Retrieval-Augmented Generation (RAG) system for technical documentation that combines PDF processing, vector embeddings, and large language models to provide intelligent search and question-answering capabilities.

## Features

- PDF document processing and text extraction
- Vector embeddings for semantic search
- Integration with OpenAI and Gemini LLMs
- Streamlit-based web interface
- RESTful API endpoints
- PostgreSQL database with pgvector extension
- Document versioning and management

## Prerequisites

- Python 3.8+
- PostgreSQL 12+ with pgvector extension
- OpenAI API key
- Gemini API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdfRAG
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```env
DB_HOST=your_db_host
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_PORT=your_db_port
DB_NAME=your_db_name
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

5. Database Setup:
   - First, create the database and vector extension:
     ```bash
     python create_db.py
     ```
   - Then, create the necessary tables and schema:
     ```bash
     alembic upgrade head
     ```
   Note: `create_db.py` creates an empty database, while migrations set up the table structure and relationships.

## Running the Application

The application consists of two parts that need to be run separately:

### 1. Backend API Server

```bash
uvicorn app.main:app --reload
```

The API server will be available at `http://localhost:8000`

### 2. Streamlit Frontend

In a separate terminal:
```bash
streamlit run streamlit_app.py
```

The web interface will be available at `http://localhost:8501`

## API Endpoints

### Document Management
- `POST /documents/upload` - Upload a new PDF document
- `GET /documents/` - List all documents
- `GET /documents/{document_id}` - Get document details
- `POST /documents/{document_id}/process` - Process a document

### Search
- `GET /search` - Basic semantic search
- `GET /search/enhanced` - Enhanced search with LLM integration
- `GET /search/by-example` - Search using document region selection

### Health
- `GET /` - Root endpoint
- `GET /health` - Health check endpoint

## Documentation

- API Documentation: `http://localhost:8000/docs` (Swagger UI)
- Alternative API Documentation: `http://localhost:8000/redoc` (ReDoc)

## Project Structure

```
pdfRAG/
├── app/                    # Main application code
│   ├── crud/              # Database operations
│   ├── models/            # SQLAlchemy models
│   ├── schemas/           # Pydantic schemas
│   ├── services/          # Business logic
│   └── processors/        # PDF processing
├── migrations/            # Database migrations
├── data/                  # Processed documents
├── streamlit_app.py       # Streamlit frontend
├── create_db.py           # Database initialization
├── requirements.txt       # Dependencies
└── .env                   # Environment variables
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license here]

## Acknowledgments

- OpenAI for GPT models
- Google for Gemini models
- PostgreSQL and pgvector for vector search
- FastAPI for the backend framework
- Streamlit for the web interface 