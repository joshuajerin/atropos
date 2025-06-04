import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# API key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Default to a random secure API key if not set
DEFAULT_API_KEY = os.environ.get("ATROPOS_API_KEY", secrets.token_urlsafe(32))

# Store active API keys with expiration (in-memory for simplicity)
# In production, consider using a more persistent and secure storage
api_keys: Dict[str, datetime] = {
    DEFAULT_API_KEY: datetime.now() + timedelta(days=30),
}


class ApiKeyInfo(BaseModel):
    """API key information for the client"""
    key: str
    expires: datetime


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate the API key"""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Check if key exists and is not expired
    if api_key in api_keys and api_keys[api_key] > datetime.now():
        return api_key
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )


def create_api_key(expiration_days: int = 30) -> ApiKeyInfo:
    """Generate a new API key"""
    new_key = secrets.token_urlsafe(32)
    expiration = datetime.now() + timedelta(days=expiration_days)
    api_keys[new_key] = expiration
    return ApiKeyInfo(key=new_key, expires=expiration)


def get_default_api_key() -> str:
    """Return the default API key for local development"""
    return DEFAULT_API_KEY