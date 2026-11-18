import logging
from typing import Optional
from supabase import create_client, Client
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

_supabase_client: Optional[Client] = None

def get_supabase_client() -> Optional[Client]:
    """
    Returns a singleton Supabase Client instance initialized with the service role key.
    Using the service role key ensures backend ML pipelines can reliably persist records
    and artifacts without exposing sensitive keys to client-side code.
    Returns None if Supabase credentials are not configured.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if not settings.is_supabase_configured:
        logger.debug("Supabase credentials not configured or placeholder detected. Operating in local mode.")
        return None

    try:
        _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
        logger.info(f"Supabase client initialized successfully for project: {settings.SUPABASE_URL}")
        return _supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None

def reset_supabase_client():
    """Resets the singleton client (useful for unit tests with mocked configurations)."""
    global _supabase_client
    _supabase_client = None
