import hashlib
from pathlib import Path
from datetime import datetime
import shutil
import os
from typing import Dict, Optional
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    async def save_uploaded_file(self, file: UploadFile) -> Dict[str, str]:
        """
        Save uploaded file with timestamp and calculate hash.
        Returns file info including path and hash.
        """
        try:
            # Get file extension
            original_filename = file.filename
            file_extension = Path(original_filename).suffix
            
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = Path(original_filename).stem
            new_filename = f"{base_filename}_{timestamp}{file_extension}"
            
            # Create full file path
            file_path = self.raw_dir / new_filename
            
            # Calculate hash while saving file
            sha256_hash = hashlib.sha256()
            
            # Save file and calculate hash
            with open(file_path, "wb") as buffer:
                while chunk := await file.read(8192):  # Read in chunks
                    sha256_hash.update(chunk)
                    buffer.write(chunk)
            
            return {
                "original_filename": original_filename,
                "saved_filename": new_filename,
                "file_path": str(file_path),
                "file_hash": sha256_hash.hexdigest()
            }
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            # Clean up if file was partially saved
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            raise

    async def check_file_exists(self, file_hash: str) -> bool:
        """Check if a file with the same hash exists in the database."""
        # This will be implemented in the document service
        pass

    def delete_file(self, file_path: str):
        """Delete a file from the raw directory."""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"Deleted file: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            raise 