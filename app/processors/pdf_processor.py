import fitz  # PyMuPDF
import camelot
import hashlib
from pathlib import Path
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, pdf_path: str, document_id: int):
        """
        Initialize PDF processor.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: ID of the document in database
        """
        self.pdf_path = pdf_path
        self.document_id = document_id
        self.doc = None
        
        # Set up directory structure
        self.base_dir = Path(f"data/processed/{document_id}")
        self.images_dir = self.base_dir / "images"
        self.tables_dir = self.base_dir / "tables"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized PDFProcessor for document {document_id}")

    @contextmanager
    def open_pdf(self):
        """Context manager for handling PDF document opening and closing."""
        try:
            self.doc = fitz.open(self.pdf_path)
            yield self.doc
        finally:
            if self.doc:
                self.doc.close()
                self.doc = None

    def process_pdf(self) -> Dict:
        """Process PDF and extract text, images, and tables."""
        try:
            with self.open_pdf() as doc:
                pdf_content = {
                    "metadata": self._extract_metadata(doc),
                    "pages": []
                }
                
                for page_num in range(len(doc)):
                    page_content = self._process_page(doc[page_num], page_num)
                    pdf_content["pages"].append(page_content)
                
                # Save processing metadata
                self._save_metadata(pdf_content)
                
                return pdf_content
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def _extract_metadata(self, doc: fitz.Document) -> Dict:
        """Extract PDF metadata."""
        metadata = doc.metadata
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
            "page_count": len(doc)
        }

    def _process_page(self, page: fitz.Page, page_num: int) -> Dict:
        """Process a single page."""
        page_content = {
            "page_number": page_num + 1,
            "text": self._extract_text(page),
            "images": self._extract_images(page),
            "tables": self._extract_tables(page_num)
        }
        return page_content

    def _extract_text(self, page: fitz.Page) -> List[Dict]:
        """Extract text blocks with positions."""
        text_blocks = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                text = " ".join([span["text"] for line in block["lines"] 
                               for span in line["spans"]])
                if text.strip():
                    text_blocks.append({
                        "text": text,
                        "bbox": block["bbox"],
                        "type": "text"
                    })
        return text_blocks

    def _extract_images(self, page: fitz.Page) -> List[Dict]:
        """Extract and save images from page."""
        images = []
        try:
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_md5 = hashlib.md5(image_bytes).hexdigest()
                    image_filename = f"page_{page.number + 1}_img_{img_index}_{image_md5}.{image_ext}"
                    
                    # Use absolute paths
                    image_path = Path(os.getcwd()) / self.images_dir / image_filename
                    
                    # Ensure the path is within the project directory
                    if not str(image_path).startswith(str(Path(os.getcwd()))):
                        raise ValueError(f"Invalid path: {image_path}")
                    
                    # Create directory if it doesn't exist
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save image
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Store relative path for database
                    rel_path = str(image_path.relative_to(Path(os.getcwd())))
                    
                    images.append({
                        "filename": image_filename,
                        "path": rel_path,
                        "bbox": img[-1],  # bounding box
                        "type": "image"
                    })
                    
                    logger.info(f"Saved image: {rel_path}")
                
        except Exception as e:
            logger.error(f"Error extracting image: {str(e)}")
        
        return images

    def _extract_tables(self, page_num: int) -> List[Dict]:
        """Extract and save tables from page."""
        tables = []
        try:
            # Extract tables using camelot
            extracted_tables = camelot.read_pdf(
                self.pdf_path,
                pages=str(page_num + 1),
                flavor='stream'
            )
            
            for idx, table in enumerate(extracted_tables):
                table_data = table.df.to_dict('records')
                table_filename = f"page_{page_num + 1}_table_{idx}.json"
                
                # Use absolute paths
                table_path = Path(os.getcwd()) / self.tables_dir / table_filename
                
                # Ensure the path is within the project directory
                if not str(table_path).startswith(str(Path(os.getcwd()))):
                    raise ValueError(f"Invalid path: {table_path}")
                
                # Create directory if it doesn't exist
                table_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save table as JSON
                with open(table_path, 'w', encoding='utf-8') as f:
                    json.dump(table_data, f, ensure_ascii=False, indent=2)
                
                # Store relative path for database
                rel_path = str(table_path.relative_to(Path(os.getcwd())))
                
                tables.append({
                    "filename": table_filename,
                    "path": rel_path,
                    "data": table_data,
                    "bbox": table._bbox,
                    "type": "table"
                })
                
                logger.info(f"Saved table: {rel_path}")
                
        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num + 1}: {str(e)}")
        
        return tables

    def _save_metadata(self, content: Dict):
        """Save processing metadata."""
        metadata_path = self.base_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2) 