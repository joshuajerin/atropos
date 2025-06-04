"""
Client module for interacting with the AtroposLib API securely
"""

import os
import urllib.parse
from typing import Any, Dict, List, Optional, Union

import requests
from pydantic import BaseModel


class ApiClientError(Exception):
    """Exception raised for API client errors"""
    pass


class AtroposApiClient:
    """
    Client for interacting with the AtroposLib API
    
    This client handles authentication and secure communication with the API
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000", 
        api_key: Optional[str] = None,
        verify_ssl: bool = True
    ):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL of the AtroposLib API
            api_key: API key for authentication (defaults to ATROPOS_API_KEY env var)
            verify_ssl: Whether to verify SSL certificates (set to False for self-signed certs in dev)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("ATROPOS_API_KEY")
        self.verify_ssl = verify_ssl
        
        if not self.api_key:
            raise ApiClientError(
                "No API key provided. Either pass api_key parameter or set ATROPOS_API_KEY environment variable."
            )
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]], BaseModel, List[BaseModel]]] = None
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to the API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON data for request body
            
        Returns:
            API response as dictionary
            
        Raises:
            ApiClientError: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Convert Pydantic models to dictionaries
        if isinstance(json_data, BaseModel):
            json_data = json_data.dict()
        elif isinstance(json_data, list) and all(isinstance(item, BaseModel) for item in json_data):
            json_data = [item.dict() for item in json_data]
        
        headers = {"X-API-Key": self.api_key}
        
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=headers,
                verify=self.verify_ssl
            )
            
            # Raise exception for HTTP errors
            response.raise_for_status()
            
            # Return JSON response or empty dict for non-JSON responses
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.json()
            return {"status": "success", "status_code": response.status_code}
            
        except requests.RequestException as e:
            raise ApiClientError(f"API request failed: {str(e)}")
    
    # API endpoint methods
    
    def check_api_key(self) -> Dict[str, Any]:
        """Check if the API key is valid"""
        return self._make_request("GET", "/api-key")
    
    def create_api_key(self, days: int = 30) -> Dict[str, Any]:
        """Create a new API key"""
        return self._make_request("POST", "/create-api-key", params={"days": days})
    
    def register(self, registration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a trainer with the API"""
        return self._make_request("POST", "/register", json_data=registration_data)
    
    def register_env(self, env_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register an environment with the API"""
        return self._make_request("POST", "/register-env", json_data=env_data)
    
    def disconnect_env(self, env_id: int) -> Dict[str, Any]:
        """Disconnect an environment from the API"""
        return self._make_request("POST", "/disconnect-env", json_data={"env_id": env_id})
    
    def get_wandb_info(self) -> Dict[str, Any]:
        """Get Weights & Biases information"""
        return self._make_request("GET", "/wandb_info")
    
    def get_info(self) -> Dict[str, Any]:
        """Get API configuration information"""
        return self._make_request("GET", "/info")
    
    def get_batch(self) -> Dict[str, Any]:
        """Get a batch of data from the API"""
        return self._make_request("GET", "/batch")
    
    def get_latest_example(self) -> Dict[str, Any]:
        """Get the latest example from the API"""
        return self._make_request("GET", "/latest_example")
    
    def submit_scored_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit scored data to the API"""
        return self._make_request("POST", "/scored_data", json_data=data)
    
    def submit_scored_data_list(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submit a list of scored data to the API"""
        return self._make_request("POST", "/scored_data_list", json_data=data_list)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the API"""
        return self._make_request("GET", "/status")
    
    def get_env_status(self, env_id: int) -> Dict[str, Any]:
        """Get the status of an environment"""
        return self._make_request("GET", "/status-env", params={"env_id": env_id})
    
    def reset_data(self) -> Dict[str, Any]:
        """Reset all data in the API (use with caution)"""
        return self._make_request("GET", "/reset_data")


# Example usage
if __name__ == "__main__":
    # Initialize the client (using environment variable for API key)
    client = AtroposApiClient(base_url="http://localhost:8000")
    
    # Check if API key is valid
    try:
        result = client.check_api_key()
        print(f"API key is valid: {result}")
    except ApiClientError as e:
        print(f"Error: {e}")