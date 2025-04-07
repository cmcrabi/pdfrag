from sqlalchemy.orm import Session
from app.models.document_chunk import DocumentChunk
from app.models.document import Document
from sentence_transformers import SentenceTransformer
import json
import logging
from typing import List, Dict, Union, Any
import numpy as np

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self, db: Session):
        self.db = db
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Fixed dimension for the model

    async def vectorize_document(self, document_id: int) -> Dict:
        """
        Create vector embeddings for document chunks and store in database.
        Only needs document_id as parameter since processed content is in filesystem.
        """
        try:
            # Get document
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")

            # Read processed content
            processed_path = f"data/processed/{document_id}/metadata.json"
            with open(processed_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            chunks_created = 0
            # Process each page
            for page in content['pages']:
                # Process text blocks
                for text_block in page['text']:
                    if not text_block.get('text'):
                        continue

                    # Create embedding for text
                    embedding = self.model.encode(text_block['text'])
                    
                    # Ensure embedding is the correct dimension and normalized
                    if len(embedding) != self.embedding_dim:
                        logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
                        # Truncate or pad if necessary
                        if len(embedding) > self.embedding_dim:
                            embedding = embedding[:self.embedding_dim]
                        else:
                            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
                    
                    # Normalize the embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    # Convert to list for storage
                    embedding = embedding.tolist()

                    # Get nearby images and tables
                    nearby_images = []
                    nearby_tables = []

                    if isinstance(text_block.get('bbox'), (list, tuple)):
                        # Only look for nearby elements if we have valid bbox
                        nearby_images = [
                            img for img in page.get('images', [])
                            if img.get('type') == 'image'  # Ensure it's an image
                        ]
                        nearby_tables = [
                            table for table in page.get('tables', [])
                            if isinstance(table.get('bbox'), (list, tuple)) and 
                            self._is_nearby(text_block['bbox'], table['bbox'])
                        ]

                    # Create chunk record
                    chunk = DocumentChunk(
                        document_id=document_id,
                        content=text_block['text'],
                        embedding=embedding,
                        chunk_metadata={
                            'page_number': page['page_number'],
                            'bbox': text_block.get('bbox'),
                            'type': 'text',
                            'images': [
                                {
                                    'filename': img.get('filename'),
                                    'path': img.get('path')
                                } for img in nearby_images
                            ],
                            'tables': [
                                {
                                    'filename': table.get('filename'),
                                    'path': table.get('path')
                                } for table in nearby_tables
                            ]
                        }
                    )
                    self.db.add(chunk)
                    chunks_created += 1

                    # Commit in batches to avoid memory issues
                    if chunks_created % 100 == 0:
                        self.db.commit()

            # Final commit for remaining chunks
            self.db.commit()
            logger.info(f"Created {chunks_created} vector embeddings for document {document_id}")

            return {
                "document_id": document_id,
                "chunks_created": chunks_created,
                "status": "success"
            }

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating vector embeddings: {str(e)}")
            raise

    def _is_nearby(self, bbox1: Union[List[float], tuple], bbox2: Union[List[float], tuple], threshold: float = 100) -> bool:
        """
        Check if two bounding boxes are near each other.
        Returns False if either bbox is invalid.
        """
        try:
            if not isinstance(bbox1, (list, tuple)) or not isinstance(bbox2, (list, tuple)):
                return False
            
            if len(bbox1) < 4 or len(bbox2) < 4:
                return False

            # Calculate vertical distance between boxes
            vertical_distance = min(
                abs(float(bbox1[3]) - float(bbox2[1])),  # distance between bottom of box1 and top of box2
                abs(float(bbox2[3]) - float(bbox1[1]))   # distance between bottom of box2 and top of box1
            )
            return vertical_distance < threshold

        except Exception as e:
            logger.error(f"Error checking proximity: {str(e)}")
            return False 