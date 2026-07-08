import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from travel_planner.tools import (
    get_weather_report_of_city,
    get_categorize_budget,
    get_hotel_recommendations,
    get_activity_recommendations,
    get_food_recommendations,
    get_packing_tips_recommendation,
    create_travel_plan
)
from travel_planner.models import TravelPlanResponse

app = FastAPI(
    title="🌍 AI Travel Agent API",
    description="FastAPI backend for sequential agentic travel planning using Langchain & Groq",
    version="1.0.0"
)

class TravelPlanRequest(BaseModel):
    city: str = Field(..., example="Jaipur", description="Name of the destination city")
    start_date: str = Field(..., example="2026-07-08", description="Start date of the trip (YYYY-MM-DD)")
    end_date: str = Field(..., example="2026-07-10", description="End date of the trip (YYYY-MM-DD)")
    total_budget: int = Field(..., example=10000, description="Total budget for the trip in rupees")

@app.post("/plan-trip", response_model=TravelPlanResponse, summary="Generate sequential travel plan")
async def plan_trip(request: TravelPlanRequest):
    try:
        # Step 1: Validate dates
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
        total_days = (end - start).days + 1
        if total_days <= 0:
            raise HTTPException(status_code=400, detail="End date must be after or equal to start date.")
        
        # Step 2: Fetch Weather Data
        print(f"🌤️ Fetching weather data for {request.city}...")
        weather_data = get_weather_report_of_city.invoke({
            'city_name': request.city.lower(),
            'startdate': request.start_date,
            'enddate': request.end_date
        })
        
        # Step 3: Categorize and split budget
        print("💰 Calculating budget splits...")
        budget = get_categorize_budget.invoke({
            'total_budget': request.total_budget,
            'total_days': total_days,
            'city': request.city
        })
        
        # Step 4: Fetch Hotel Recommendations
        print("🏨 Querying hotel recommendations...")
        hotels = get_hotel_recommendations.invoke({
            'city': request.city,
            'budget_category': budget,
            'weather_data': weather_data
        })
        
        # Step 5: Fetch Activity Recommendations
        print("🗺️ Getting activity recommendations...")
        activities = get_activity_recommendations.invoke({
            'city': request.city,
            'weather_data': weather_data,
            'budget_category': budget,
            'start_date': request.start_date,
            'end_date': request.end_date
        })
        
        # Step 6: Fetch Food Recommendations
        print("🍽️ Getting food recommendations...")
        food = get_food_recommendations.invoke({
            'city': request.city,
            'budget_category': budget,
            'total_days': total_days
        })
        
        # Step 7: Get Packing, Tips, and Recommendation
        print("🧳 Finalizing packing list & tips...")
        packing_tips = get_packing_tips_recommendation.invoke({
            'city': request.city,
            'weather_data': weather_data,
            'budget_category': budget,
            'total_days': total_days
        })
        
        travel_plan = create_travel_plan.invoke({
            'city': request.city,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'budget_category': budget,
            'hotels': hotels,
            'activities': activities,
            'food': food,
            'packing_tips': packing_tips
        })
        
        return travel_plan

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating travel plan: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_fastapi:app", host="127.0.0.1", port=8000, reload=True)
