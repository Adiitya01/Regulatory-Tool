import os
import shutil
from pathlib import Path
import logging
from typing import Optional, Union

# Try importing supabase, but don't fail if not installed yet
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

from . import config

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Manages file storage operations, switching between Local and Cloud (Supabase/S3)
    based on configuration.
    """
    
    def __init__(self):
        self.provider = os.getenv("STORAGE_PROVIDER", "local").lower()
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.bucket_name = os.getenv("SUPABASE_BUCKET", "dhf-reports")
        
        self.client: Optional[Client] = None
        
        if self.provider == "supabase":
            if not SUPABASE_AVAILABLE:
                logger.error("❌ Supabase provider selected but 'supabase' package not installed.")
                logger.warning("⚠️ Falling back to LOCAL storage.")
                self.provider = "local"
            elif not self.supabase_url or not self.supabase_key:
                logger.error("❌ Supabase credentials missing (SUPABASE_URL, SUPABASE_KEY).")
                logger.warning("⚠️ Falling back to LOCAL storage.")
                self.provider = "local"
            else:
                try:
                    self.client = create_client(self.supabase_url, self.supabase_key)
                    logger.info("✅ Supabase Storage initialized.")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Supabase: {e}")
                    self.provider = "local"

    def save_file(self, source_path: Union[str, Path], destination_name: str) -> str:
        """
        Saves a file to the configured storage provider.
        Returns the path or public URL of the saved file.
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if self.provider == "supabase" and self.client:
            try:
                # Read file binary
                with open(source_path, "rb") as f:
                    file_content = f.read()
                
                # Upload to bucket (overwrite if exists)
                # Note: upsert=True is supported in newer supabase-py
                self.client.storage.from_(self.bucket_name).upload(
                    path=destination_name,
                    file=file_content,
                    file_options={"upsert": "true", "content-type": "application/pdf" if destination_name.endswith(".pdf") else "text/plain"}
                )
                
                # Get public URL
                public_url = self.client.storage.from_(self.bucket_name).get_public_url(destination_name)
                logger.info(f"☁️ Uploaded to Supabase: {destination_name}")
                return public_url
                
            except Exception as e:
                logger.error(f"❌ Supabase upload failed: {e}. Saving locally instead.")
                # Fallback to local
                return self._save_local(source_path, destination_name)
        
        else:
            return self._save_local(source_path, destination_name)

    def _save_local(self, source_path: Path, destination_name: str) -> str:
        """Helper to save file locally to OUTPUTS_DIR"""
        dest_path = config.OUTPUTS_DIR / destination_name
        
        # If source is same as dest (already in outputs), just return path
        if source_path.resolve() == dest_path.resolve():
            return str(dest_path)
            
        shutil.copy2(source_path, dest_path)
        logger.info(f"💾 Saved locally: {destination_name}")
        return str(dest_path)

    def get_file_url(self, filename: str) -> Optional[str]:
        """Get the URL/Path for a file"""
        if self.provider == "supabase" and self.client:
             return self.client.storage.from_(self.bucket_name).get_public_url(filename)
        
        local_path = config.OUTPUTS_DIR / filename
        if local_path.exists():
            return str(local_path)
        return None

    _file_list_cache = {}  # {bucket_name: ([files], timestamp)}
    CACHE_DURATION = 10    # seconds

    def exists(self, filename: str) -> bool:
        """Check if a file exists in the storage provider with local caching"""
        if self.provider == "supabase" and self.client:
            import time
            current_time = time.time()
            
            # Check if we have a fresh cache for this bucket
            cached_data = self._file_list_cache.get(self.bucket_name)
            if cached_data and (current_time - cached_data[1] < self.CACHE_DURATION):
                files = cached_data[0]
            else:
                try:
                    # Refresh cache
                    files = self.client.storage.from_(self.bucket_name).list()
                    self._file_list_cache[self.bucket_name] = (files, current_time)
                except Exception as e:
                    logger.error(f"Error listing Supabase bucket: {e}")
                    return False
            
            return any(f['name'] == filename for f in files)
        
        return (config.OUTPUTS_DIR / filename).exists()

    def fetch_file_content(self, filename: str) -> Optional[str]:
        """Fetch file content as string (from local or cloud)"""
        # Try local first
        local_path = config.OUTPUTS_DIR / filename
        if local_path.exists():
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                pass

        # Try Cloud
        if self.provider == "supabase" and self.client:
            try:
                response = self.client.storage.from_(self.bucket_name).download(filename)
                return response.decode('utf-8')
            except Exception as e:
                logger.error(f"Error fetching file from Supabase: {e}")
        
        return None

    def ensure_local(self, filename: str) -> Optional[Path]:
        """Ensure file exists locally, downloading from cloud if necessary"""
        local_path = config.OUTPUTS_DIR / filename
        if local_path.exists():
            return local_path
        
        if self.provider == "supabase" and self.client:
            try:
                logger.info(f"☁️ Downloading {filename} from Supabase for local processing...")
                response = self.client.storage.from_(self.bucket_name).download(filename)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(response)
                return local_path
            except Exception as e:
                logger.error(f"Error downloading from Supabase: {e}")
        
        return None

# Singleton instance
storage = StorageManager()
