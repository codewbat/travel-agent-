import requests
from travel_planner.config import NUMBEO_API_KEY
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ========== HELPERS ==========
def get_city_budget_estimate(city_name: str, country: str = "") -> dict:
    """Numbeo API se city ka estimated daily budget nikaalein."""
    from travel_planner.config import USE_OPENVAN
    if not NUMBEO_API_KEY:
        if not USE_OPENVAN:
            print("⚠️ Numbeo API Key is missing. Please add it to your .env file.")
        return {}
        
    url = "https://www.numbeo.com/api/city_prices"
    params = {
        "api_key": NUMBEO_API_KEY,
        "query": f"{city_name}, {country}" if country else city_name,
        "currency": "USD"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # Extract relevant price data
        budget_estimate = {
            "city": data.get("name"),
            "currency": data.get("currency"),
            "meal_inexpensive": _find_price(data, "Meal, Inexpensive Restaurant"),
            "meal_midrange": _find_price(data, "Meal for 2, Mid-range Restaurant"),
            "transport_one_way": _find_price(data, "One-way Ticket, Local Transport"),
            "taxi_start": _find_price(data, "Taxi Start"),
            "basic_utilities": _find_price(data, "Basic Utilities for 85m2 Apartment"),
        }
        return budget_estimate
    except Exception as e:
        print(f"Numbeo API Error: {e}")
        return {}

def _find_price(data, item_name):
    """Helper function to find price for a specific item."""
    for item in data.get("prices", []):
        if item.get("item_name") == item_name:
            return item.get("average_price", "N/A")
    return "N/A"

# ========== SCHEMAS & TOOLS ==========
class CityBudgetInput(BaseModel):
    city: str = Field(description="Name of the city")
    country: str = Field(default="", description="Country name (optional)")

@tool(args_schema=CityBudgetInput)
def estimate_city_budget(city: str, country: str = "") -> str:
    """
    Numbeo API se city ka estimated daily budget nikaalein.
    
    Returns formatted budget estimate string.
    """
    data = get_city_budget_estimate(city, country)
    if not data:
        return f"Budget data not available for this city ({city}). Please verify if the NUMBEO_API_KEY is configured in your .env."
    
    output = f"\n💰 Budget Estimate for {data['city']} (in {data['currency']}):\n"
    output += f"   🍽️ Inexpensive Meal: {data.get('meal_inexpensive', 'N/A')}\n"
    output += f"   🍽️ Mid-range Meal (for 2): {data.get('meal_midrange', 'N/A')}\n"
    output += f"   🚌 Local Transport (one-way): {data.get('transport_one_way', 'N/A')}\n"
    output += f"   🚕 Taxi Start: {data.get('taxi_start', 'N/A')}\n"
    output += f"   💡 Basic Utilities: {data.get('basic_utilities', 'N/A')}\n"
    return output
