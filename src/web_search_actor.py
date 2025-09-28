# web_search_actor.py - web search tool
import requests
import time
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

class WebSearchActor:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_KEY")  # Optional: SerpAPI key
        self.base_url = "https://api.serpapi.com/search"
        
    def search(self, query: str, num_results: int = 3) -> Tuple[List[Dict], float]:
        """
        Perform web search using SerpAPI or fallback to DuckDuckGo.
        Returns: (results, latency)
        """
        start_time = time.perf_counter()
        
        if self.api_key:
            return self._serpapi_search(query, num_results, start_time)
        else:
            return self._duckduckgo_search(query, num_results, start_time)
    
    def _serpapi_search(self, query: str, num_results: int, start_time: float) -> Tuple[List[Dict], float]:
        """Search using SerpAPI (requires API key)"""
        try:
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": num_results,
                "engine": "google"
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if "organic_results" in data:
                for item in data["organic_results"][:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", ""),
                        "source": "SerpAPI"
                    })
            
            latency = time.perf_counter() - start_time
            return results, latency
            
        except Exception as e:
            print(f"SerpAPI search failed: {e}")
            return self._duckduckgo_search(query, num_results, start_time)
    
    def _duckduckgo_search(self, query: str, num_results: int, start_time: float) -> Tuple[List[Dict], float]:
        """Fallback search using DuckDuckGo (no API key required)"""
        try:
            # Simple DuckDuckGo search simulation
            # In a real implementation, you'd use duckduckgo-search library
            results = [
                {
                    "title": f"Search result for: {query}",
                    "snippet": f"This is a simulated search result for '{query}'. In a real implementation, this would fetch actual web results.",
                    "link": f"https://example.com/search?q={query}",
                    "source": "DuckDuckGo (simulated)"
                }
            ]
            
            latency = time.perf_counter() - start_time
            return results, latency
            
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
            latency = time.perf_counter() - start_time
            return [], latency
