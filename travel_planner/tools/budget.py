from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from travel_planner.config import llm
from travel_planner.models import BudgetCategoryResponse

class BudgetInput(BaseModel):
    total_budget: int = Field(description="Total budget in rupees")
    total_days: int = Field(description="Total number of days")
    city: str = Field(description="Name of the city")

budget_parser = PydanticOutputParser(pydantic_object=BudgetCategoryResponse)

from langsmith import traceable

@tool(args_schema=BudgetInput)
@traceable(name="get_categorize_budget")
def get_categorize_budget(
    total_budget: int,
    total_days: int,
    city: str
):
    """
    Get a realistic categorized budget breakdown for the trip using real cost indicators from Numbeo and OpenVan.
    """
    real_prices_context = "No real-time index data found for this city. Use historical standard costs."
    
    from travel_planner.config import USE_OPENVAN
    if USE_OPENVAN:
        try:
            from openvancamp import OpenVan
            from travel_planner.tools.osm import search_nominatim
            osm_places = search_nominatim(city, limit=1)
            if osm_places:
                address = osm_places[0].get('formatted_address', '')
                parts = [p.strip() for p in address.split(',')]
                country_name = parts[-1] if parts else "US"
                
                # Dictionary mapping for common countries
                country_codes = {
                    "India": "IN", "Germany": "DE", "United States": "US", "United Kingdom": "GB",
                    "Turkey": "TR", "Georgia": "GE", "Switzerland": "CH", "Thailand": "TH",
                    "France": "FR", "Italy": "IT", "Spain": "ES", "Japan": "JP"
                }
                c_code = country_codes.get(country_name, "US")
                
                # Fetch relative cost index using OpenVan SDK
                ov = OpenVan()
                comparison = ov.basket.compare("IN", c_code)
                
                real_prices_context = f"""
                REAL COST OF LIVING COMPARISON (Relative to India - World Bank Index):
                - Food cost index in destination: {comparison.get('cost_index', 100)} (World Average = 100)
                - Relative price difference: India is {comparison.get('percentage_cheaper', 0)}% cheaper for grocery/food.
                - Spending power equivalent: Spending ₹100 in India is equivalent to ₹{comparison.get('spending_power', 100)} in the destination.
                """
                print("🟢 [DATA SOURCE] BUDGET: Cost of living retrieved via openvancamp SDK wrapper.")
        except Exception as e:
            print(f"⚠️ OpenVan SDK call failed: {e}")
            print("🟡 [DATA SOURCE] BUDGET: Falling back to LLM standard defaults.")
    else:
        try:
            # 1. Fetch real price lists from Numbeo
            from travel_planner.tools.numbeo_budget import get_city_budget_estimate
            numbeo_data = get_city_budget_estimate(city)
            if numbeo_data:
                real_prices_context = f"""
                REAL PRICE GUIDELINES FOR {city} (in {numbeo_data.get('currency', 'USD')}):
                - Inexpensive Restaurant Meal: {numbeo_data.get('meal_inexpensive')}
                - Mid-range Restaurant Meal (for 2): {numbeo_data.get('meal_midrange')}
                - Local Transport One-way Ticket: {numbeo_data.get('transport_one_way')}
                - Taxi Start: {numbeo_data.get('taxi_start')}
                """
                print(f"🟢 [DATA SOURCE] BUDGET: Successfully fetched real cost indices from Numbeo.")
        except Exception as e:
            print(f"⚠️ Numbeo API indexes fetch failed: {e}.")
            print("🟡 [DATA SOURCE] BUDGET: No real price index found. Using standard LLM defaults.")

    categorize_budget_prompt = PromptTemplate(
        template="""
You are a travel budget expert for {city}.

REAL COST & PRICE INDICATORS FOR THE DESTINATION:
{real_prices_context}

BUDGET INFORMATION:
- Total Budget: ₹{total_budget}
- Number of Days: {total_days}
- City: {city}

**YOUR TASK - YOU MUST DECIDE EVERYTHING:**

1. Calculate Per Day Budget (Total Budget ÷ Number of Days)
2. What category would you put this in? (Budget/Mid-Range/Luxury)
3. What type of hotels can they afford in {city}?
4. What type of restaurants can they afford in {city}?
5. What transport can they afford in {city}?
6. What activities can they do in {city}?
7. How should they split their per-day budget?

**IMPORTANT:**
- Per Day Budget = Total Budget ÷ Days (YOU calculate this)
- per_day_breakdown should sum to approximately your calculated Per Day Budget
- Consider the REAL price indicators listed above to split the budget realistically
- **CRITICAL**: Do NOT output the schema definition, "properties", "defs", or "type" definitions. Fill in actual values!

Example JSON structure to output:
{{
  "category": "Mid-Range",
  "description": "Comfortable travel with decent hotels and dining",
  "hotel_type": "3-star hotel",
  "restaurant_type": "Casual dining",
  "transport_type": "Cab/Metro",
  "activity_type": "Sightseeing",
  "per_day_breakdown": {{
    "hotel": 3000,
    "food": 1500,
    "transport": 500,
    "activities": 1000,
    "miscellaneous": 500
  }},
  "reasoning": "Fits the budget of ₹6500 per day"
}}

{instruction}
""",
        input_variables=['total_budget', 'total_days', 'city', 'real_prices_context'],
        partial_variables={
            'instruction': budget_parser.get_format_instructions()
        }
    )
    
    budget_chain = categorize_budget_prompt | llm | budget_parser
    from travel_planner.utils.callbacks import TokenUsageTracker
    budget_chain_result = budget_chain.invoke({
        'total_budget': total_budget,
        'total_days': total_days,
        'city': city,
        'real_prices_context': real_prices_context
    }, config={"callbacks": [TokenUsageTracker("Budget Splitter")]})
    
    return budget_chain_result
