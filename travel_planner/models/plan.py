from typing import List, Optional, Annotated
from pydantic import BaseModel, Field
from travel_planner.models.hotels import HotelRecommendation
from travel_planner.models.activities import DailyActivity
from travel_planner.models.food import FoodRecommendation
from travel_planner.models.budget import DailyBudget

class TravelPlanResponse(BaseModel):
    city: Annotated[str, Field(description="City name")]
    start_date: Annotated[str, Field(description="Start date")]
    end_date: Annotated[str, Field(description="End date")]
    total_days: Annotated[int, Field(description="Total days")]
    budget_category: Annotated[str, Field(description="Budget category")]
    hotels: Annotated[List[HotelRecommendation], Field(description="Hotel recommendations")]
    daily_plan: Annotated[List[DailyActivity], Field(description="Day-wise activities")]
    food_recommendations: Annotated[List[FoodRecommendation], Field(description="Day-wise food")]
    packing_list: Annotated[List[str], Field(description="Items to pack")]
    daily_budget: Annotated[List[DailyBudget], Field(description="Daily budget breakdown")]
    tips: Annotated[List[str], Field(description="Travel tips")]
    recommendation: Annotated[str, Field(description="GO or DON'T GO")]
    recommendation_reason: Annotated[str, Field(description="Why this recommendation")]
    alternative: Annotated[Optional[str], Field(description="Alternative suggestion")]
