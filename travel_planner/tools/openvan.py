import requests
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# OpenVan.camp Public API Base URL
BASE_URL = "https://openvan.camp"

# ========== ARGS SCHEMAS ==========
class FoodCostCompareInput(BaseModel):
    from_country: str = Field(description="Origin ISO 3166-1 alpha-2 country code (e.g. 'DE' for Germany, 'US' for USA)")
    to_country: str = Field(description="Destination ISO 3166-1 alpha-2 country code (e.g. 'IN' for India, 'TR' for Turkey)")

class CountryCodeInput(BaseModel):
    country_code: str = Field(description="ISO 3166-1 alpha-2 country code (e.g. 'DE', 'IN')")

# ========== CORE FUNCTIONS ==========

def compare_food_basket(from_country: str, to_country: str) -> Dict[str, Any]:
    """Compare food basket prices between two countries using OpenVan.camp API."""
    url = f"{BASE_URL}/api/vanbasket/compare"
    params = {
        "from": from_country.upper(),
        "to": to_country.upper(),
        "source": "antigravity_travel_planner"
    }
    try:
        response = requests.get(url, params=params, timeout=10, headers={"Accept": "application/json"})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ OpenVan compare API error: {str(e)}")
        return {}

def get_country_fuel_prices() -> Dict[str, Any]:
    """Fetch global retail fuel prices from OpenVan.camp API."""
    url = f"{BASE_URL}/api/fuel/prices"
    params = {
        "source": "antigravity_travel_planner"
    }
    try:
        response = requests.get(url, params=params, timeout=10, headers={"Accept": "application/json"})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ OpenVan fuel prices API error: {str(e)}")
        return {}

# ========== TOOLS ==========

@tool(args_schema=FoodCostCompareInput)
def compare_country_food_costs(from_country: str, to_country: str) -> str:
    """
    Compare the relative food basket prices and cost index between two countries using OpenVan.camp API.
    Does not require any API Key.
    
    Args:
        from_country: ISO 2-letter origin country code (e.g., 'US', 'DE')
        to_country: ISO 2-letter destination country code (e.g., 'IN', 'TR')
    """
    print(f"\n🌽 Comparing Food Costs: {from_country} -> {to_country}")
    res = compare_food_basket(from_country, to_country)
    if not res or not res.get("success"):
        return f"Unable to fetch food cost comparison between {from_country} and {to_country}."
        
    data = res.get("data", {})
    from_info = data.get("from", {})
    to_info = data.get("to", {})
    diff = data.get("diff_percent", 0)
    budget_eq = data.get("budget_100", 100)
    
    cheaper_txt = "cheaper" if diff < 0 else "more expensive"
    
    output = f"\n🛒 Food Cost Comparison ({from_info.get('country_name')} ➔ {to_info.get('country_name')}):\n"
    output += f"   📊 Food index in origin: {from_info.get('vanbasket_index')} (World Avg = 100)\n"
    output += f"   📊 Food index in destination: {to_info.get('vanbasket_index')} (World Avg = 100)\n"
    output += f"   📈 Destination is {abs(diff)}% {cheaper_txt} for grocery/food.\n"
    output += f"   💶 Spending €100 in {from_info.get('country_name')} is equivalent to spending €{budget_eq} in {to_info.get('country_name')} for food.\n"
    return output

@tool(args_schema=CountryCodeInput)
def get_fuel_prices_by_country(country_code: str) -> str:
    """
    Get retail fuel prices (gasoline, diesel, LPG in EUR/liter or gallon) for a country using OpenVan.camp API.
    Does not require any API Key.
    
    Args:
        country_code: ISO 2-letter country code (e.g. 'DE' for Germany, 'US' for USA, 'IN' for India)
    """
    print(f"\n⛽ Fetching Fuel Prices for: {country_code}")
    res = get_country_fuel_prices()
    if not res or not res.get("success"):
        return f"Unable to fetch fuel prices data."
        
    country_data = res.get("data", {}).get(country_code.upper())
    if not country_data:
        return f"Fuel price data not available for country code: {country_code}."
        
    prices = country_data.get("prices", {})
    currency = country_data.get("currency", "EUR")
    unit = country_data.get("unit", "liter")
    
    output = f"\n⛽ Fuel Prices in {country_data.get('country_name')} (per {unit} in {currency}):\n"
    output += f"   🟢 Gasoline: {prices.get('gasoline') or 'N/A'}\n"
    output += f"   ⚫ Diesel: {prices.get('diesel') or 'N/A'}\n"
    output += f"   🟡 LPG: {prices.get('lpg') or 'N/A'}\n"
    output += f"   📅 Updated At: {country_data.get('fetched_at')}\n"
    return output
