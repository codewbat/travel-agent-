from datetime import datetime
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from travel_planner.config import llm
from travel_planner.models import (
    PackingTipsResponse, TravelPlanResponse, BudgetCategoryResponse,
    HotelRecommendation, DailyActivity, FoodRecommendation, DailyBudget
)

class PackingTipsInput(BaseModel):
    city: str = Field(description="City name")
    weather_data: list = Field(description="Weather records list")
    budget_category: BudgetCategoryResponse = Field(description="Budget category Response")
    total_days: int = Field(description="Total days of trip")

class TravelPlanInput(BaseModel):
    city: str = Field(description="Name of the city")
    start_date: str = Field(description="Start date of trip (YYYY-MM-DD)")
    end_date: str = Field(description="End date of trip (YYYY-MM-DD)")
    budget_category: BudgetCategoryResponse = Field(description="Budget category response")
    hotels: list = Field(description="List of hotel recommendations")
    activities: list = Field(description="List of daily activities")
    food: list = Field(description="List of food recommendations")
    packing_tips: PackingTipsResponse = Field(description="Packing tips response")

packing_parser = PydanticOutputParser(pydantic_object=PackingTipsResponse)

@tool(args_schema=PackingTipsInput)
def get_packing_tips_recommendation(
    city: str,
    weather_data: list,
    budget_category: BudgetCategoryResponse,
    total_days: int
) -> PackingTipsResponse:
    """Get packing list, tips, and final recommendation using Pydantic parser."""
    rain_probs = [d.get('rain_probability', 0) for d in weather_data]
    avg_rain = sum(rain_probs) / len(rain_probs) if rain_probs else 0
    is_rainy = avg_rain > 0.4
    rainy_days = sum(1 for d in weather_data if d.get('rain_probability', 0) > 0.4) // 8
    weather_pattern = 'Rainy' if is_rainy else 'Sunny'
    
    prompt = PromptTemplate(
        template="""
You are a travel expert for {city}.

WEATHER INFORMATION:
- Total Days: {total_days}
- Rainy Days: {rainy_days}/{total_days}
- Weather Pattern: {weather_pattern}

BUDGET CATEGORY: {category}

Based on the weather and budget:

1. **PACKING LIST**: Provide at least 6 essential items
   - Consider weather (rainy/sunny)
   - Consider activities
   - Consider budget constraints

2. **TRAVEL TIPS**: Provide at least 6 practical tips
   - Weather-related tips
   - Safety tips
   - Money-saving tips
   - Cultural tips

3. **FINAL RECOMMENDATION**:
   - If rainy days >= total_days/2 → "DON'T GO"
   - If rainy days < total_days/2 → "GO"
   - Give clear reasoning
   - If "DON'T GO", suggest alternative timing\n
{instructions}
""",
        input_variables=['city', 'total_days', 'rainy_days', 'weather_pattern', 'category'],
        partial_variables={
            'instructions': packing_parser.get_format_instructions()
        }
    )
    
    chain = prompt | llm | packing_parser
    result = chain.invoke({
        'city': city,
        'total_days': total_days,
        'rainy_days': rainy_days,
        'weather_pattern': weather_pattern,
        'category': budget_category.category
    })
    
    return result

@tool(args_schema=TravelPlanInput)
def create_travel_plan(
    city: str,
    start_date: str,
    end_date: str,
    budget_category: BudgetCategoryResponse,
    hotels: list,
    activities: list,
    food: list,
    packing_tips: PackingTipsResponse
) -> TravelPlanResponse:
    """Combine all results into final TravelPlanResponse."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1
    
    db = DailyBudget(
        hotel=budget_category.per_day_breakdown.hotel,
        food=budget_category.per_day_breakdown.food,
        transport=budget_category.per_day_breakdown.transport,
        activities=budget_category.per_day_breakdown.activities,
        miscellaneous=budget_category.per_day_breakdown.miscellaneous
    )
    
    return TravelPlanResponse(
        city=city,
        start_date=start_date,
        end_date=end_date,
        total_days=total_days,
        budget_category=budget_category.category,
        hotels=hotels,
        daily_plan=activities,
        food_recommendations=food,
        packing_list=packing_tips.packing_list,
        tips=packing_tips.tips,
        daily_budget=[db],
        recommendation=packing_tips.recommendation,
        recommendation_reason=packing_tips.recommendation_reason,
        alternative=packing_tips.alternative
    )
