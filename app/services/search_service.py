from typing import List, Dict, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.services.embedding_service import EmbeddingService
import logging
import json
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = EmbeddingService()

    async def search(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.3,
        document_id: Optional[int] = None,
        product_id: Optional[int] = None,
        context_pages: int = 3  # Number of pages to include before and after a match
    ) -> List[Dict]:
        """
        Search for similar content using vector similarity and include context pages.
        Can filter by document_id or product_id.
        """
        try:
            # Debug: Check all documents and their vectorization status
            debug_sql = """
            SELECT 
                d.id as document_id,
                d.filename,
                COUNT(dc.id) as total_chunks,
                COUNT(CASE WHEN dc.embedding IS NOT NULL THEN 1 END) as vectorized_chunks
            FROM documents d
            LEFT JOIN document_chunks dc ON d.id = dc.document_id
            GROUP BY d.id, d.filename
            ORDER BY d.id
            """
            debug_results = self.db.execute(text(debug_sql)).fetchall()
            for row in debug_results:
                logger.info(f"Document {row.document_id} ({row.filename}): {row.total_chunks} total chunks, {row.vectorized_chunks} vectorized chunks")

            # Debug: Check if document exists and has chunks
            if document_id:
                doc_sql = """
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as vectorized_chunks
                FROM document_chunks
                WHERE document_id = :document_id
                """
                doc_result = self.db.execute(text(doc_sql), {"document_id": document_id}).first()
                logger.info(f"Searching document {document_id}: {doc_result.total_chunks} total chunks, {doc_result.vectorized_chunks} vectorized chunks")

            # Create query embedding
            query_embedding = self.embedding_service.create_embeddings(query)
            logger.info(f"Created query embedding of length: {len(query_embedding)}")

            # Debug: Check a sample embedding from the database
            sample_sql = """
            SELECT embedding 
            FROM document_chunks 
            WHERE document_id = :document_id AND embedding IS NOT NULL 
            LIMIT 1
            """
            sample_result = self.db.execute(text(sample_sql), {"document_id": document_id}).first()
            if sample_result:
                logger.info(f"Sample embedding length from DB: {len(sample_result.embedding)}")
            else:
                logger.warning("No vectorized chunks found in database")

            # Build SQL query to find initial matches
            sql = """
            WITH initial_matches AS (
                SELECT 
                    dc.id,
                    dc.document_id,
                    dc.content,
                    dc.chunk_metadata,
                    d.filename,
                    d.version,
                    d.product_id,
                    1 - (dc.embedding <=> CAST(:embedding AS vector(384))) as similarity,
                    (dc.chunk_metadata->>'page_number')::int as page_number
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE 1 - (dc.embedding <=> CAST(:embedding AS vector(384))) > :threshold
                {filter_clause}
                ORDER BY similarity DESC
                LIMIT :limit
            ),
            context_chunks AS (
                SELECT 
                    dc.id,
                    dc.document_id,
                    dc.content,
                    dc.chunk_metadata,
                    d.filename,
                    d.version,
                    d.product_id,
                    (dc.chunk_metadata->>'page_number')::int as page_number,
                    im.similarity as original_similarity
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                JOIN initial_matches im ON dc.document_id = im.document_id
                WHERE (dc.chunk_metadata->>'page_number')::int 
                    BETWEEN (im.page_number - :context_pages) 
                    AND (im.page_number + :context_pages)
                {filter_clause}
            )
            SELECT 
                cc.*,
                COALESCE(
                    (
                        SELECT json_agg(json_build_object(
                            'filename', i.data->>'filename',
                            'path', i.data->>'path'
                        ))
                        FROM json_array_elements(cc.chunk_metadata->'images') as i(data)
                        WHERE i.data->>'filename' IS NOT NULL
                    ),
                    '[]'::json
                ) as images,
                COALESCE(
                    (
                        SELECT json_agg(json_build_object(
                            'filename', t.data->>'filename',
                            'path', t.data->>'path'
                        ))
                        FROM json_array_elements(cc.chunk_metadata->'tables') as t(data)
                        WHERE t.data->>'filename' IS NOT NULL
                    ),
                    '[]'::json
                ) as tables
            FROM context_chunks cc
            ORDER BY cc.original_similarity DESC, cc.page_number ASC
            """

            # Build filter clause and parameters
            params = {
                "embedding": query_embedding,
                "threshold": threshold,
                "limit": limit,
                "context_pages": context_pages,
            }
            filter_clause = ""
            if document_id:
                filter_clause = "AND d.document_id = :document_id"
                params["document_id"] = document_id
            elif product_id:
                filter_clause = "AND d.product_id = :product_id"
                params["product_id"] = product_id

            # Add filter clause to SQL
            sql = sql.format(filter_clause=filter_clause)

            # Log the complete query with parameters for manual testing
            test_query = sql.replace(":embedding", f"'{query_embedding}'::vector")
            test_query = test_query.replace(":threshold", str(threshold))
            test_query = test_query.replace(":limit", str(limit))
            if document_id:
                test_query = test_query.replace(":document_id", str(document_id))
            elif product_id:
                test_query = test_query.replace(":product_id", str(product_id))
            #logger.info("Complete SQL query for manual testing:\n" + test_query)

            # Execute query
            result = self.db.execute(text(sql), params)

            # Process results and group by context
            results = []
            current_context = None
            current_group = []
            seen_chunks = set()  # Track seen chunk IDs to avoid duplicates
            seen_images = set()  # Track seen image filenames

            for row in result:
                # Skip if we've already seen this chunk
                if row.id in seen_chunks:
                    continue
                seen_chunks.add(row.id)

                # Process images to remove duplicates based on filename (original logic)
                unique_images = []
                for img in (row.images or []):
                    if img['filename'] not in seen_images:
                        unique_images.append(img)
                        seen_images.add(img['filename'])

                result_item = {
                    "content": row.content,
                    "document": {
                        "id": row.document_id,
                        "filename": row.filename,
                        "version": row.version,
                        "product_id": row.product_id
                    },
                    "metadata": {
                        "page_number": row.page_number,
                        "bbox": row.chunk_metadata.get("bbox")
                    },
                    "similarity": float(row.original_similarity),
                    "images": unique_images,
                    "tables": row.tables or []
                }

                # If this is a high-similarity match, start a new context group
                if float(row.original_similarity) > threshold:
                    if current_group:
                        # Remove duplicate pages within the group
                        unique_pages = []
                        seen_page_numbers = set()
                        for page in current_group:
                            if page['metadata']['page_number'] not in seen_page_numbers:
                                unique_pages.append(page)
                                seen_page_numbers.add(page['metadata']['page_number'])
                        
                        results.append({
                            "context": current_context,
                            "pages": unique_pages
                        })
                    current_context = result_item
                    current_group = [result_item]
                # Only add context pages if they belong to a relevant context
                elif current_context: 
                    current_group.append(result_item)

            # Add the last group if it exists
            if current_group and current_context:
                # Remove duplicate pages within the last group
                unique_pages = []
                seen_page_numbers = set()
                for page in current_group:
                    # Avoid adding the context page itself to the 'pages' list if it's the only item
                    if current_context and page['content'] == current_context['content'] and len(current_group) == 1:
                         continue
                    if page['metadata']['page_number'] not in seen_page_numbers:
                        unique_pages.append(page)
                        seen_page_numbers.add(page['metadata']['page_number'])
                
                results.append({
                    "context": current_context,
                    "pages": unique_pages
                })

            # Log the number of results
            logger.info(f"Found {len(results)} context groups for query: {query}")
            logger.info(f"Total unique pages: {sum(len(group['pages']) for group in results)}")
            logger.info(f"Total unique images: {len(seen_images)}")

            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
            raise

    async def search_by_region(
        self,
        document_id: int,
        page_number: int,
        bbox: List[float],
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for similar content using a selected region from a document,
        constraining the search to the same product.
        """
        try:
            # Step 1: Find the document and its product_id
            doc_sql = """
            SELECT product_id 
            FROM documents 
            WHERE id = :document_id
            """
            doc_result = self.db.execute(text(doc_sql), {"document_id": document_id}).first()
            if not doc_result:
                 raise ValueError(f"Document with ID {document_id} not found.")
            source_product_id = doc_result.product_id
            logger.info(f"Searching region from document ID {document_id} (Product ID: {source_product_id})")
            
            # Step 2: Find the chunk content that contains this region
            chunk_sql = """
            SELECT content
            FROM document_chunks
            WHERE document_id = :document_id
            AND (chunk_metadata->>'page_number')::int = :page_number
            AND chunk_metadata->>'bbox' @> CAST(:bbox AS jsonb)
            LIMIT 1
            """
            
            chunk_result = self.db.execute(
                text(chunk_sql),
                {
                    "document_id": document_id,
                    "page_number": page_number,
                    "bbox": json.dumps(bbox) # Ensure bbox is proper JSON string for query
                }
            ).first()

            if not chunk_result or not chunk_result.content:
                logger.warning(f"No content found for region in doc {document_id}, page {page_number}, bbox {bbox}")
                return []

            # Step 3: Use the found content as query, filtering by the original product_id
            logger.info(f"Using content from region as query, searching within product {source_product_id}")
            return await self.search(
                query=chunk_result.content,
                limit=limit,
                product_id=source_product_id, # Pass the fetched product_id
                document_id=None # Ensure we don't filter by the source document ID itself
            )

        except ValueError as e:
             logger.error(f"Value error in region search: {str(e)}")
             # Re-raise or return empty list depending on desired API behavior for non-existent doc
             raise HTTPException(status_code=404, detail=str(e)) 
        except Exception as e:
            logger.error(f"Error in region search: {str(e)}", exc_info=True)
            raise 