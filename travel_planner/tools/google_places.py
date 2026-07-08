import requests
from typing import List, Dict, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from travel_planner.config import GOOGLE_PLACES_API_KEY

# ========== ARGS SCHEMAS ==========
class TextSearchInput(BaseModel):
    query: str = Field(description="Search query (e.g., 'restaurants in Jaipur')")
    page_size: int = Field(default=10, description="Number of results")

class PlaceDetailsInput(BaseModel):
    place_id: str = Field(description="Google Place ID to fetch details for")

class NearbySearchInput(BaseModel):
    location: str = Field(description="Location (e.g., 'Jaipur, India')")
    radius: int = Field(default=5000, description="Search radius in meters")
    place_type: str = Field(default="", description="Place type (e.g., 'restaurant', 'tourist_attraction')")
    max_results: int = Field(default=10, description="Maximum results")

# ========== CORE FUNCTIONS ==========

def search_places_text(query: str, page_size: int = 10) -> List[Dict]:
    """Search places using Google Places API (New) Text Search."""
    url = "https://places.googleapis.com/v1/places:searchText"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.types,places.priceLevel,places.websiteUri,places.regularOpeningHours"
    }
    
    data = {
        "textQuery": query,
        "pageSize": page_size
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result.get('places', [])
    except Exception as e:
        print(f"❌ Error searching places: {str(e)}")
        return []

def get_place_details(place_id: str) -> Dict:
    """Get detailed information for a place using its Place ID."""
    url = f"https://places.googleapis.com/v1/places/{place_id}"
    
    headers = {
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "displayName,formattedAddress,location,rating,reviews,priceLevel,websiteUri,regularOpeningHours,types,photos,addressComponents"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error getting place details: {str(e)}")
        return {}

def get_place_id_by_city(city_name: str) -> Optional[str]:
    """Get Place ID for a city name."""
    places = search_places_text(city_name, page_size=1)
    if places:
        return places[0].get('id')
    return None

# ========== TOOLS ==========

@tool(args_schema=TextSearchInput)
def search_places(query: str, page_size: int = 10) -> List[Dict]:
    """
    Search for places using natural language query.
    
    Args:
        query: Natural language query (e.g., "Italian restaurants in Jaipur")
        page_size: Number of results to return
    
    Returns:
        List of places with basic information
    """
    print(f"\n🔍 Searching: {query}")
    results = search_places_text(query, page_size)
    print(f"✅ Found {len(results)} places")
    return results

@tool(args_schema=PlaceDetailsInput)
def get_place_details_by_id(place_id: str) -> Dict:
    """
    Get complete details for a specific place using its Place ID.
    
    Args:
        place_id: Google Place ID
    
    Returns:
        Complete place details including reviews, photos, hours
    """
    print(f"\n📋 Getting details for place: {place_id}")
    details = get_place_details(place_id)
    if details:
        print(f"✅ Found details for {details.get('displayName', {}).get('text', 'Unknown')}")
    return details

@tool(args_schema=NearbySearchInput)
def search_nearby_places(
    location: str,
    radius: int = 5000,
    place_type: str = "",
    max_results: int = 10
) -> List[Dict]:
    """
    Search for places near a location with optional type filter.
    
    Args:
        location: Location name (e.g., "Jaipur, India")
        radius: Search radius in meters
        place_type: Type of place (restaurant, tourist_attraction, hotel, etc.)
        max_results: Maximum results to return
    
    Returns:
        List of nearby places
    """
    query = f"{place_type} near {location}" if place_type else f"places near {location}"
    print(f"\n📍 Searching nearby: {query}")
    
    results = search_places_text(query, page_size=max_results)
    print(f"✅ Found {len(results)} nearby places")
    return results

# ========== SPECIALIZED TOOLS ==========

@tool
def search_restaurants(city: str, cuisine: str = "", max_results: int = 10) -> List[Dict]:
    """
    Search restaurants in a city.
    
    Args:
        city: City name (e.g., "Jaipur")
        cuisine: Cuisine type (e.g., "Italian", "Mexican")
        max_results: Maximum results
    
    Returns:
        List of restaurants with details
    """
    query = f"{cuisine} restaurants in {city}" if cuisine else f"restaurants in {city}"
    return search_places.invoke({
        'query': query,
        'page_size': max_results
    })

@tool
def search_attractions(city: str, max_results: int = 10) -> List[Dict]:
    """
    Search tourist attractions in a city.
    
    Args:
        city: City name
        max_results: Maximum results
    
    Returns:
        List of attractions
    """
    query = f"tourist attractions in {city}"
    return search_places.invoke({
        'query': query,
        'page_size': max_results
    })

@tool
def search_hotels_google(city: str, max_results: int = 10) -> List[Dict]:
    """
    Search hotels in a city (fallback for RapidAPI).
    
    Args:
        city: City name
        max_results: Maximum results
    
    Returns:
        List of hotels
    """
    query = f"hotels in {city}"
    return search_places.invoke({
        'query': query,
        'page_size': max_results
    })

@tool
def get_city_place_id(city_name: str) -> str:
    """
    Get Google Place ID for a city.
    
    Args:
        city_name: Name of the city
    
    Returns:
        Place ID string
    """
    print(f"\n🔍 Getting Place ID for: {city_name}")
    place_id = get_place_id_by_city(city_name)
    if place_id:
        print(f"✅ Place ID: {place_id}")
        return place_id
    else:
        return "City not found"

# ========== GOOGLE DISTANCE MATRIX API ==========
class DistanceMatrixInput(BaseModel):
    origins: List[str] = Field(description="List of origin locations or coordinates (lat,lng) (e.g. ['26.9124,75.7873'])")
    destinations: List[str] = Field(description="List of destination locations or coordinates (lat,lng) (e.g. ['26.9855,75.8513'])")
    mode: str = Field(default="driving", description="Travel mode: driving, walking, bicycling, transit")

@tool(args_schema=DistanceMatrixInput)
def get_distance_matrix(origins: List[str], destinations: List[str], mode: str = "driving") -> Dict:
    """
    Get travel distance and duration between origins and destinations using Google Distance Matrix API.
    
    Args:
        origins: List of origins (names or lat,lng strings)
        destinations: List of destinations (names or lat,lng strings)
        mode: Travel mode (driving, walking, bicycling, transit)
        
    Returns:
        JSON response from Google Distance Matrix API containing distances and travel times
    """
    print(f"\n🚗 Calling Distance Matrix for: {origins} to {destinations}")
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": "|".join(origins),
        "destinations": "|".join(destinations),
        "mode": mode,
        "key": GOOGLE_PLACES_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        res = response.json()
        print(f"   ✅ Distance Matrix status: {res.get('status')}")
        return res
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return {"status": "ERROR", "error_message": str(e)}

