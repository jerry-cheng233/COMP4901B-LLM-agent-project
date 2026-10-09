"""
Google Maps Agent Tool - Provides location search, directions, and place information.

Features:
1. Search for places (restaurants, landmarks, addresses)
2. Get directions between locations
3. Get detailed place information (ratings, hours, reviews)
4. Generate Google Maps links for navigation

Uses Google Maps Platform APIs:
- Places API (Text Search, Place Details)
- Directions API
"""

import requests
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus


class GoogleMapsClient:
    """Client for Google Maps Platform APIs."""
    
    BASE_URL = "https://maps.googleapis.com/maps/api"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def search_places(self, query: str, location: str = None) -> Dict[str, Any]:
        """
        Search for places using Google Maps Text Search API.
        
        Args:
            query: Search query (e.g., "restaurants in Hong Kong", "HKUST")
            location: Optional location bias (e.g., "Hong Kong")
        
        Returns:
            Dictionary with search results including name, address, rating, etc.
        """
        url = f"{self.BASE_URL}/place/textsearch/json"
        params = {
            "query": query if not location else f"{query} near {location}",
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") != "OK":
                return {"error": data.get("status"), "message": data.get("error_message", "Search failed")}
            
            results = []
            for place in data.get("results", [])[:5]:  # Limit to top 5
                results.append({
                    "name": place.get("name"),
                    "address": place.get("formatted_address"),
                    "rating": place.get("rating"),
                    "user_ratings_total": place.get("user_ratings_total"),
                    "place_id": place.get("place_id"),
                    "types": place.get("types", [])[:3],
                    "location": place.get("geometry", {}).get("location"),
                    "open_now": place.get("opening_hours", {}).get("open_now")
                })
            
            return {"status": "success", "results": results, "count": len(results)}
        
        except Exception as e:
            return {"error": str(e)}
    
    def get_place_details(self, place_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a place.
        
        Args:
            place_id: Google Place ID
        
        Returns:
            Detailed place information including hours, phone, website, reviews
        """
        url = f"{self.BASE_URL}/place/details/json"
        params = {
            "place_id": place_id,
            "fields": "name,formatted_address,formatted_phone_number,website,rating,reviews,opening_hours,price_level,url",
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") != "OK":
                return {"error": data.get("status")}
            
            result = data.get("result", {})
            
            # Extract reviews (top 3)
            reviews = []
            for review in result.get("reviews", [])[:3]:
                reviews.append({
                    "author": review.get("author_name"),
                    "rating": review.get("rating"),
                    "text": review.get("text", "")[:200]
                })
            
            return {
                "status": "success",
                "name": result.get("name"),
                "address": result.get("formatted_address"),
                "phone": result.get("formatted_phone_number"),
                "website": result.get("website"),
                "rating": result.get("rating"),
                "google_maps_url": result.get("url"),
                "opening_hours": result.get("opening_hours", {}).get("weekday_text"),
                "price_level": result.get("price_level"),
                "reviews": reviews
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def get_directions(
        self,
        origin: str,
        destination: str,
        mode: str = "driving"
    ) -> Dict[str, Any]:
        """
        Get directions between two locations.
        
        Args:
            origin: Starting location (address or place name)
            destination: Ending location (address or place name)
            mode: Travel mode - driving, walking, bicycling, transit
        
        Returns:
            Directions with steps, duration, distance, and Google Maps link
        """
        url = f"{self.BASE_URL}/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") != "OK":
                return {"error": data.get("status"), "message": data.get("error_message", "Directions failed")}
            
            route = data.get("routes", [{}])[0]
            leg = route.get("legs", [{}])[0]
            
            # Extract step-by-step directions
            steps = []
            for step in leg.get("steps", [])[:10]:  # Limit to 10 steps
                steps.append({
                    "instruction": step.get("html_instructions", "").replace("<b>", "").replace("</b>", ""),
                    "distance": step.get("distance", {}).get("text"),
                    "duration": step.get("duration", {}).get("text")
                })
            
            # Generate Google Maps direction link
            maps_link = self.generate_directions_link(origin, destination, mode)
            
            return {
                "status": "success",
                "origin": leg.get("start_address"),
                "destination": leg.get("end_address"),
                "distance": leg.get("distance", {}).get("text"),
                "duration": leg.get("duration", {}).get("text"),
                "mode": mode,
                "steps": steps,
                "google_maps_link": maps_link
            }
        
        except Exception as e:
            return {"error": str(e)}

    def generate_directions_link(
        self,
        origin: str,
        destination: str,
        mode: str = "driving"
    ) -> str:
        """
        Generate a Google Maps directions link.

        Args:
            origin: Starting location
            destination: Ending location
            mode: Travel mode

        Returns:
            Google Maps URL that opens directions
        """
        origin_encoded = quote_plus(origin)
        dest_encoded = quote_plus(destination)
        return f"https://www.google.com/maps/dir/?api=1&origin={origin_encoded}&destination={dest_encoded}&travelmode={mode}"

    def generate_place_link(self, query: str) -> str:
        """
        Generate a Google Maps search link for a place.

        Args:
            query: Place name or address

        Returns:
            Google Maps URL that opens the place search
        """
        query_encoded = quote_plus(query)
        return f"https://www.google.com/maps/search/?api=1&query={query_encoded}"


# ============================================================================
# Tool Definitions for LLM Function Calling
# ============================================================================

def get_google_maps_tool_definitions() -> List[Dict]:
    """Return tool definitions for Google Maps functions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_location",
                "description": "Search for a place or location on Google Maps. Use this to find restaurants, landmarks, addresses, or any place. Returns name, address, rating, and Google Maps link.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'HKUST', 'restaurants near Central Hong Kong', 'Hong Kong Science Park')"
                        },
                        "location": {
                            "type": "string",
                            "description": "Optional location to bias results (e.g., 'Hong Kong', 'Kowloon')"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_place_info",
                "description": "Get detailed information about a specific place including opening hours, phone, website, reviews. Requires a place_id from search_location results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "place_id": {
                            "type": "string",
                            "description": "The Google Place ID from search_location results"
                        }
                    },
                    "required": ["place_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_directions",
                "description": "Get directions between two locations. Returns distance, duration, step-by-step directions, and a Google Maps link to navigate.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {
                            "type": "string",
                            "description": "Starting location (address or place name, e.g., 'HKUST', 'my current location')"
                        },
                        "destination": {
                            "type": "string",
                            "description": "Destination location (address or place name)"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["driving", "walking", "bicycling", "transit"],
                            "description": "Travel mode (default: driving)"
                        }
                    },
                    "required": ["origin", "destination"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_map_link",
                "description": "Generate a clickable Google Maps link for a location or directions. Use this when you need to share a location or navigation link via email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location/address to generate a link for"
                        },
                        "from_location": {
                            "type": "string",
                            "description": "Optional: If provided, generates a directions link from this location to the destination"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["driving", "walking", "bicycling", "transit"],
                            "description": "Travel mode for directions (default: driving)"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
