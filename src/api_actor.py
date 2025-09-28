# api_actor.py - REST API tool
import requests
import time
import json
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

class APIActor:
    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL", "https://jsonplaceholder.typicode.com")
        self.api_key = os.getenv("API_KEY")  # Optional API key
        
    def call_api(self, endpoint: str, method: str = "GET", params: Dict = None, data: Dict = None) -> Tuple[Dict, float]:
        """
        Make REST API call.
        Returns: (response_data, latency)
        """
        start_time = time.perf_counter()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            headers = {"Content-Type": "application/json"}
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=10)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, headers=headers, timeout=10)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"text": response.text}
            
            latency = time.perf_counter() - start_time
            return response_data, latency
            
        except requests.exceptions.RequestException as e:
            print(f"API call failed: {e}")
            latency = time.perf_counter() - start_time
            return {"error": str(e)}, latency
        except Exception as e:
            print(f"Unexpected error in API call: {e}")
            latency = time.perf_counter() - start_time
            return {"error": str(e)}, latency
    
    def get_user_info(self, user_id: int) -> Tuple[Dict, float]:
        """Get user information by ID"""
        return self.call_api(f"users/{user_id}")
    
    def get_posts(self, user_id: int = None) -> Tuple[List[Dict], float]:
        """Get posts, optionally filtered by user ID"""
        endpoint = "posts"
        params = {"userId": user_id} if user_id else None
        return self.call_api(endpoint, params=params)
    
    def get_weather(self, city: str) -> Tuple[Dict, float]:
        """Simulated weather API call"""
        # This is a mock implementation - in reality you'd use a real weather API
        mock_weather = {
            "city": city,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%",
            "source": "Mock Weather API"
        }
        latency = 0.1  # Simulated latency
        return mock_weather, latency
