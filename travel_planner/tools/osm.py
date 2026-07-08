import requests
from typing import List, Dict, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# OSM Nominatim and Overpass endpoints
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Compliant User-Agent to respect OSM usage policies
OSM_HEADERS = {
    "User-Agent": "TravelPlannerAgent/1.0 (travelagentcontactdev@gmail.com)"
}

# ========== ARGS SCHEMAS ==========
class OSMSearchInput(BaseModel):
    query: str = Field(description="Search query (e.g., 'tourist attractions in Jaipur', 'restaurants in London')")
    limit: int = Field(default=10, description="Max number of results to return")

class OSMNearbyInput(BaseModel):
    latitude: float = Field(description="Latitude coordinate")
    longitude: float = Field(description="Longitude coordinate")
    place_type: str = Field(default="tourism", description="OSM Key class to search (e.g., 'tourism' for sights, 'amenity' for restaurants/cafes)")
    value_type: str = Field(default="attraction", description="OSM Value tag (e.g., 'attraction' for sights, 'restaurant' for dining)")
    radius_meters: int = Field(default=3000, description="Search radius in meters")

# ========== CORE FUNCTIONS ==========

def search_nominatim(query: str, limit: int = 10) -> List[Dict]:
    """Search features using OpenStreetMap Nominatim API."""
    params = {
        "q": query,
        "format": "json",
        "limit": limit,
        "addressdetails": 1
    }
    try:
        response = requests.get(NOMINATIM_URL, headers=OSM_HEADERS, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        
        places = []
        for r in results:
            places.append({
                "name": r.get("display_name", "").split(",")[0],
                "formatted_address": r.get("display_name", ""),
                "latitude": float(r.get("lat", 0.0)),
                "longitude": float(r.get("lon", 0.0)),
                "type": r.get("type", "N/A"),
                "class": r.get("class", "N/A"),
                "importance": r.get("importance", 0.0)
            })
        return places
    except Exception as e:
        print(f"❌ Error in OSM Nominatim search: {str(e)}")
        return []

def query_overpass_nearby(
    lat: float,
    lon: float,
    place_type: str = "tourism",
    value_type: str = "attraction",
    radius: int = 3000
) -> List[Dict]:
    """Query OSM elements around a lat/lon using Overpass API."""
    # Build Overpass QL query searching nodes and ways matching the tag around the center
    overpass_query = f"""
    [out:json][timeout:15];
    (
      node["{place_type}"="{value_type}"](around:{radius},{lat},{lon});
      way["{place_type}"="{value_type}"](around:{radius},{lat},{lon});
    );
    out center;
    """
    try:
        response = requests.post(OVERPASS_URL, headers=OSM_HEADERS, data={"data": overpass_query}, timeout=15)
        response.raise_for_status()
        data = response.json()
        elements = data.get("elements", [])
        
        results = []
        for elem in elements[:12]:
            tags = elem.get("tags") or {}
            name = tags.get("name") or tags.get("name:en") or "Unnamed Point of Interest"
            
            # Extract center coordinates for ways/areas, or direct lat/lon for nodes
            elem_lat = elem.get("lat") or (elem.get("center") or {}).get("lat") or lat
            elem_lon = elem.get("lon") or (elem.get("center") or {}).get("lon") or lon
            
            results.append({
                "name": name,
                "latitude": float(elem_lat),
                "longitude": float(elem_lon),
                "type": tags.get("tourism") or tags.get("amenity") or value_type,
                "opening_hours": tags.get("opening_hours", "N/A"),
                "website": tags.get("website", "N/A"),
                "address": tags.get("addr:street", "N/A")
            })
        return results
    except Exception as e:
        print(f"❌ Error in Overpass API call: {str(e)}")
        return []

# ========== TOOLS ==========

@tool(args_schema=OSMSearchInput)
def search_osm_places(query: str, limit: int = 10) -> List[Dict]:
    """
    Search tourist sights, attractions or dining spots globally using OpenStreetMap.
    Doesn't require any API keys.
    
    Args:
        query: Search string (e.g., 'attractions in Jaipur', 'restaurants in London')
        limit: Max results to return
    """
    print(f"\n🗺️ OSM Search: {query}")
    return search_nominatim(query, limit)

@tool(args_schema=OSMNearbyInput)
def search_osm_nearby(
    latitude: float,
    longitude: float,
    place_type: str = "tourism",
    value_type: str = "attraction",
    radius_meters: int = 3000
) -> List[Dict]:
    """
    Query OpenStreetMap elements located near specific GPS coordinates.
    Use class 'tourism' / value 'attraction' for sights, or class 'amenity' / value 'restaurant' for dining.
    
    Args:
        latitude: Center latitude
        longitude: Center longitude
        place_type: OSM tag key (e.g. tourism, amenity, shop)
        value_type: OSM tag value (e.g. attraction, restaurant, museum, cafe)
        radius_meters: Search radius limit
    """
    print(f"\n📍 OSM Nearby Search: ({latitude}, {longitude}) - Type: {place_type}={value_type}")
    return query_overpass_nearby(latitude, longitude, place_type, value_type, radius_meters)
