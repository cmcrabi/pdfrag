from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from app.models.document_chunk import DocumentChunk
import json
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding service with a specific model.
        Args:
            model_name: The name of the sentence-transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.chunk_size = 512  # Maximum tokens per chunk
            self.chunk_overlap = 50  # Overlap between chunks
            logger.info(f"Initialized EmbeddingService with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {str(e)}")
            raise
        
    def create_embeddings(self, text: str) -> List[float]:
        """
        Create embeddings for a piece of text.
        Args:
            text: The text to create embeddings for
        Returns:
            List of floats representing the embedding vector
        """
        try:
            if not text.strip():
                raise ValueError("Empty text provided for embedding")
            
            embedding = self.model.encode(text)
            
            # Ensure embedding is normalized
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into smaller chunks for processing.
        Args:
            text: The text to split into chunks
        Returns:
            List of text chunks
        """
        try:
            # Remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split into sentences (improved sentence splitting)
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                
                # If adding this sentence would exceed chunk size
                if current_length + sentence_length > self.chunk_size:
                    # If we have a current chunk, add it to chunks
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    # Start new chunk with current sentence
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise

    def process_page_content(self, page_content: Dict[str, Any], document_id: int) -> List[Dict]:
        """
        Process page content and create chunks with metadata.
        Args:
            page_content: Dictionary containing page content (text, images, tables)
            document_id: ID of the document being processed
        Returns:
            List of dictionaries containing chunk data
        """
        try:
            chunks_data = []
            
            # Process text blocks
            for text_block in page_content.get('text', []):
                text = text_block['text']
                if not text.strip():
                    continue
                    
                text_chunks = self.chunk_text(text)
                
                for chunk in text_chunks:
                    chunk_data = {
                        'content': chunk,
                        'document_id': document_id,
                        'chunk_metadata': {  # Changed from 'metadata' to 'chunk_metadata'
                            'page_number': page_content['page_number'],
                            'bbox': text_block['bbox'],
                            'type': 'text',
                            'images': [],
                            'tables': []
                        }
                    }
                    chunks_data.append(chunk_data)
            
            # Add references to nearby images and tables
            for chunk_data in chunks_data:
                chunk_bbox = chunk_data['chunk_metadata']['bbox']
                
                # Add references to images - include all images from the same page
                for img in page_content.get('images', []):
                    # Since we don't have valid bbox for images, include all images from the page
                    # but only include essential information
                    chunk_data['chunk_metadata']['images'].append({
                        'filename': img['filename'],
                        'path': img['path']
                    })
                    logger.debug(f"Added image {img['filename']} to chunk")
                
                # Add references to tables
                for table in page_content.get('tables', []):
                    if self._is_nearby(chunk_bbox, table['bbox']):
                        chunk_data['chunk_metadata']['tables'].append({
                            'filename': table['filename'],
                            'path': table['path'],
                            'bbox': table['bbox']
                        })
            
            return chunks_data
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            raise
    
    def _is_nearby(self, bbox1: List[float], bbox2: List[float], threshold: float = 200) -> bool:
        """
        Check if two bounding boxes are near each other.
        Args:
            bbox1: First bounding box coordinates [x0, y0, x1, y1]
            bbox2: Second bounding box coordinates [x0, y0, x1, y1]
            threshold: Maximum distance to be considered nearby
        Returns:
            Boolean indicating if boxes are nearby
        """
        try:
            # First check if boxes overlap or are very close
            # Check for overlap
            if (bbox1[0] < bbox2[2] and bbox1[2] > bbox2[0] and  # horizontal overlap
                bbox1[1] < bbox2[3] and bbox1[3] > bbox2[1]):    # vertical overlap
                return True

            # Calculate center points
            center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
            center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

            # Calculate distance between centers
            distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

            # Consider boxes nearby if distance is within threshold
            is_nearby = distance < threshold

            # Log the distances for debugging
            logger.debug(f"Checking proximity between boxes:")
            logger.debug(f"Box1: {bbox1}")
            logger.debug(f"Box2: {bbox2}")
            logger.debug(f"Center1: {center1}")
            logger.debug(f"Center2: {center2}")
            logger.debug(f"Distance: {distance}")
            logger.debug(f"Threshold: {threshold}")
            logger.debug(f"Is nearby: {is_nearby}")

            return is_nearby
        except Exception as e:
            logger.error(f"Error checking proximity: {str(e)}")
            return False
    
    def store_chunks(self, db: Session, chunks_data: List[Dict]) -> List[DocumentChunk]:
        """
        Store chunks and their embeddings in the database.
        Args:
            db: Database session
            chunks_data: List of chunk data to store
        Returns:
            List of created DocumentChunk objects
        """
        try:
            chunks = []
            
            for chunk_data in chunks_data:
                # Create embedding for the chunk
                embedding = self.create_embeddings(chunk_data['content'])
                
                # Create chunk record
                chunk = DocumentChunk(
                    document_id=chunk_data['document_id'],
                    content=chunk_data['content'],
                    embedding=embedding,
                    chunk_metadata=chunk_data['chunk_metadata']  # Updated reference
                )
                
                db.add(chunk)
                chunks.append(chunk)
            
            db.commit()
            logger.info(f"Successfully stored {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error storing chunks: {str(e)}")
            raise

    async def find_similar_chunks(
        self,
        db: Session,
        query: str,
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find similar chunks to a query using cosine similarity.
        Args:
            db: Database session
            query: Query text to find similar chunks for
            limit: Maximum number of results to return
            threshold: Minimum similarity score to include in results
        Returns:
            List of similar chunks with their metadata
        """
        try:
            # Create embedding for the query
            query_embedding = self.create_embeddings(query)
            
            # TODO: Implement vector similarity search
            # This will be implemented when we add the similarity search functionality
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {str(e)}")
            raise 