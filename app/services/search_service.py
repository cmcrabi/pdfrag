from typing import List, Dict, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.services.embedding_service import EmbeddingService
import logging

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
        context_pages: int = 3  # Number of pages to include before and after a match
    ) -> List[Dict]:
        """
        Search for similar content using vector similarity and include context pages.
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
                    1 - (dc.embedding <=> CAST(:embedding AS vector(384))) as similarity,
                    (dc.chunk_metadata->>'page_number')::int as page_number
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE 1 - (dc.embedding <=> CAST(:embedding AS vector(384))) > :threshold
                {document_filter}
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
                    (dc.chunk_metadata->>'page_number')::int as page_number,
                    im.similarity as original_similarity
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                JOIN initial_matches im ON dc.document_id = im.document_id
                WHERE (dc.chunk_metadata->>'page_number')::int 
                    BETWEEN (im.page_number - :context_pages) 
                    AND (im.page_number + :context_pages)
                {document_filter}
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

            # Add document filter if specified
            document_filter = "AND dc.document_id = :document_id" if document_id else ""
            sql = sql.format(document_filter=document_filter)

            # Log the complete query with parameters for manual testing
            test_query = sql.replace(":embedding", f"'{query_embedding}'::vector")
            test_query = test_query.replace(":threshold", str(threshold))
            test_query = test_query.replace(":limit", str(limit))
            if document_id:
                test_query = test_query.replace(":document_id", str(document_id))
            #logger.info("Complete SQL query for manual testing:\n" + test_query)

            # Execute query
            result = self.db.execute(
                text(sql),
                {
                    "embedding": query_embedding,
                    "threshold": threshold,
                    "limit": limit,
                    "context_pages": context_pages,
                    "document_id": document_id
                }
            )

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

                # Process images to remove duplicates
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
                        "version": row.version
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
                else:
                    current_group.append(result_item)

            # Add the last group if it exists
            if current_group:
                # Remove duplicate pages within the last group
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

            # Log the number of results
            logger.info(f"Found {len(results)} context groups for query: {query}")
            logger.info(f"Total unique pages: {sum(len(group['pages']) for group in results)}")
            logger.info(f"Total unique images: {len(seen_images)}")

            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    async def search_by_region(
        self,
        document_id: int,
        page_number: int,
        bbox: List[float],
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for similar content using a selected region from a document.
        """
        try:
            # Find the chunk that contains this region
            sql = """
            SELECT content
            FROM document_chunks
            WHERE document_id = :document_id
            AND (chunk_metadata->>'page_number')::int = :page_number
            AND chunk_metadata->>'bbox' @> :bbox::jsonb
            LIMIT 1
            """
            
            result = self.db.execute(
                text(sql),
                {
                    "document_id": document_id,
                    "page_number": page_number,
                    "bbox": bbox
                }
            ).first()

            if not result:
                return []

            # Use the found content as query
            return await self.search(
                query=result.content,
                limit=limit,
                document_id=None  # Search across all documents
            )

        except Exception as e:
            logger.error(f"Error in region search: {str(e)}")
            raise 