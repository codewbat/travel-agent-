import os
from dotenv import load_dotenv
import requests
from langchain_core.tools import tool , InjectedToolArg
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import Annotated , Dict , List , Literal ,Optional
from pydantic import BaseModel , Field
from datetime import datetime
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()

weather_api_key = os.getenv("Weather_API_KEY")
groq_key = os.getenv("GROQAI_API_Key")


llm = ChatGroq(
    api_key=groq_key,
    model="llama-3.3-70b-versatile",
    temperature=0.2
)


@tool
def get_weather_report_of_city(
  city_name : str ,
  startdate: str,
  enddate:str
):
    """
    this fuction is to get city weather data form api on a targeted date 
    """
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={weather_api_key}"
    
    respose = requests.get(url)
    data = respose.json()
    filter_by_date = [
        item for item in data["list"] 
        if startdate <= item["dt_txt"].split()[0] <= enddate
    ]
    
    weather_summary = []
    
    for item in filter_by_date :
        weather_summary.append({
            "time": item["dt_txt"],
            "temperature" : item["main"]["temp"],
            "feels_like" : item["main"]["feels_like"],
            "humidity" : item["main"]["humidity"],
            "weather" : item["weather"][0]["description"],
            "wind" : item["wind"]["speed"],
            "rain_probability" : item["pop"],

        })
    return weather_summary

weather_data = get_weather_report_of_city.invoke({
    'city_name':'jaipur',
    'startdate' : '2026-07-08',
    'enddate' : '2026-07-10'
})

print(weather_data)

class BudgetBreakdown(BaseModel):
    """Per day budget breakdown"""
    hotel: int = Field(description="Amount to spend on hotel per day")
    food: int = Field(description="Amount to spend on food per day")
    transport: int = Field(description="Amount to spend on transport per day")
    activities: int = Field(description="Amount to spend on activities/sightseeing per day")
    miscellaneous: int = Field(description="Amount to spend on miscellaneous items per day")

class BudgetCategoryResponse(BaseModel):
    """Budget categorization response from LLM"""
    category: Annotated[
        str,
        Field(description="Budget category: Budget/Mid-Range/Luxury")
    ]
    description: Annotated[
        str,
        Field(description="Brief description of what this budget means")
    ]
    hotel_type: Annotated[
        str,
        Field(description="What type of hotels they can afford")
    ]
    restaurant_type: Annotated[
        str,
        Field(description="What type of restaurants they can afford")
    ]
    transport_type: Annotated[
        str,
        Field(description="What type of transport they can use")
    ]
    activity_type: Annotated[
        str,
        Field(description="What type of activities they can do")
    ]
    per_day_breakdown: Annotated[
        BudgetBreakdown,
        Field(description="Per day budget breakdown")
    ]
    reasoning: Annotated[
        str,
        Field(description="Why this budget category was chosen")
    ]
budget_parser = PydanticOutputParser(pydantic_object=BudgetCategoryResponse)
@tool
def get_categorize_budget(
    total_budget: str,
    total_days: int,
    city: str
) : 
    """
        LLM will decide and breakdown budger category.
    """
    categorize_budget_prompt = PromptTemplate(
        template="""
    You are a travel budget expert for {city}.

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
    - Consider local prices in {city}
    IMPORTANT: per_day_breakdown should sum to approximately ₹ per_day\n
    {instruction}
    """,
    input_variables=['total_budget','total_days','city'],
    partial_variables = {
        'instruction' : budget_parser.get_format_instructions()
    }
    )
    budget_chain = categorize_budget_prompt | llm | budget_parser
    budget_chain_result = budget_chain.invoke({
        'total_budget':total_budget,
        'total_days':total_days,
        'city':city
    })
    
    return budget_chain_result


budget = get_categorize_budget.invoke({
        'total_budget':"10000 ruppes",
        'total_days':2,
        'city':"jaipur"
})

print(budget)

class ActivityDetail(BaseModel):
    """Individual activity detail"""
    name: str = Field(description="Name of the place/activity")
    location: str = Field(description="Area/location in the city")
    latitude: float = Field(description="Latitude of the place")
    longitude: float = Field(description="Longitude of the place")
    entry_fee: int = Field(description="Entry fee in rupees (0 if free)")
    timing: str = Field(description="Timing (e.g., 9:00 AM - 5:00 PM)")
    description: str = Field(description="Brief description of the place")
    why_recommend: str = Field(description="Why this place is recommended for today's weather")
    duration: str = Field(description="Suggested duration to spend (e.g., 1-2 hours)")

class HotelRecommendation(BaseModel):
    """Hotel recommendation with complete services"""
    name: str = Field(description="Hotel name")
    location: str = Field(description="Hotel location/area")
    latitude: float = Field(description="Latitude of hotel")
    longitude: float = Field(description="Longitude of hotel")
    price: int = Field(description="Price per night in rupees")
    star_rating: float = Field(description="Star rating (1-5)")
    
    # ============ Services Categories ============
    room_services: List[str] = Field(
        description="Room services: 24/7 Room Service, Laundry, Housekeeping, etc."
    )
    dining_services: List[str] = Field(
        description="Dining services: Restaurant, Cafe, Bar, In-room Dining, etc."
    )
    wellness_services: List[str] = Field(
        description="Wellness & Fitness: Swimming Pool, Gym, Spa, Massage, Sauna, etc."
    )
    internet_services: List[str] = Field(
        description="Internet services: Free WiFi, Paid WiFi, Business Center, etc."
    )
    family_services: List[str] = Field(
        description="Family services: Kids Club, Baby Sitting, Family Rooms, etc."
    )
    transport_services: List[str] = Field(
        description="Transport services: Airport Shuttle, Car Rental, Parking, etc."
    )
    accessibility_services: List[str] = Field(
        description="Accessibility: Wheelchair Access, Elevator, Disabled Parking, etc."
    )
    other_services: List[str] = Field(
        description="Other services: Pet Friendly, Concierge, Tour Desk, etc."
    )
    
    # ============ Additional Info ============
    weather_recommendation: str = Field(
        description="Why this hotel is good for the weather"
    )
class DailyActivity(BaseModel):
    """Daily activity plan with details"""
    date: str = Field(description="Date")
    day: str = Field(description="Day name")
    weather: str = Field(description="Weather condition")
    is_rainy: bool = Field(description="Whether day is rainy")
    morning: List[ActivityDetail] = Field(description="Morning activities (8 AM - 12 PM)")
    afternoon: List[ActivityDetail] = Field(description="Afternoon activities (12 PM - 5 PM)")
    evening: List[ActivityDetail] = Field(description="Evening activities (5 PM - 9 PM)")

class DishDetail(BaseModel):
    """Individual dish detail"""
    name: str = Field(description="Name of the dish")
    price: int = Field(description="Price of this dish in rupees")
    description: str = Field(description="Brief description of the dish")
    why_special: str = Field(description="Why this dish is special here")

class FoodRecommendation(BaseModel):
    """Food recommendation with multiple dishes and location"""
    day: int = Field(description="Day number (1, 2, 3, etc.)")
    meal: str = Field(description="Meal type: Breakfast/Lunch/Snack/Dinner")
    place_name: str = Field(description="Restaurant/Cafe/Street Food Stall name")
    location: str = Field(description="Area/location in the city")
    latitude: float = Field(description="Latitude of the place")
    longitude: float = Field(description="Longitude of the place")
    place_type: str = Field(description="Type: Restaurant/Cafe/Street Food/Vendor")
    dishes: List[DishDetail] = Field(description="List of recommended dishes at this place")
    weather_suitability: str = Field(description="Why this food suits today's weather")

class TravelPlanResponse(BaseModel):
    """Complete travel plan response"""
    city: Annotated[str, Field(description="City name")]
    start_date: Annotated[str, Field(description="Start date")]
    end_date: Annotated[str, Field(description="End date")]
    total_days: Annotated[int, Field(description="Total days")]
    budget_category: Annotated[str, Field(description="Budget category")]
    hotels: Annotated[List[HotelRecommendation], Field(description="Hotel recommendations")]
    daily_plan: Annotated[List[DailyActivity], Field(description="Day-wise activities with details")]
    food_recommendations: Annotated[List[FoodRecommendation], Field(description="Day-wise food with multiple dishes")]
    packing_list: Annotated[List[str], Field(description="Items to pack")]
    tips: Annotated[List[str], Field(description="Travel tips")]
    daily_budget: Annotated[dict, Field(description="Daily budget breakdown")]
    recommendation: Annotated[str, Field(description="GO or DON'T GO")]
    recommendation_reason: Annotated[str, Field(description="Why this recommendation")]
    alternative: Annotated[Optional[str], Field(description="Alternative suggestion")]


travel_plan_parser = PydanticOutputParser(pydantic_object=TravelPlanResponse)
def convert_weather_to_text(weather_data: List[dict]) -> str:
    """
    Convert weather data to readable text format.
    Raw data bhejo - LLM khud samajh lega!
    """
    if not weather_data:
        return "No weather data available."
    
    text = "📊 WEATHER DATA (Raw)\n"
    text += "=" * 50 + "\n\n"
    
    for item in weather_data:
        # Sirf raw data - NO CONVERSION!
        time = item.get('time', 'Unknown')
        temp = item.get('temperature', 0)
        feels_like = item.get('feels_like', 0)
        humidity = item.get('humidity', 0)
        weather = item.get('weather', 'Unknown')
        wind = item.get('wind', 0)
        rain_prob = item.get('rain_probability', 0)
        
        text += f"   Time: {time}\n"
        text += f"   Temperature: {temp} K\n"
        text += f"   Feels Like: {feels_like} K\n"
        text += f"   Humidity: {humidity}%\n"
        text += f"   Weather: {weather}\n"
        text += f"   Wind Speed: {wind} m/s\n"
        text += f"   Rain Probability: {rain_prob}\n"
        text += "\n"
    
    return text

def convert_budget_to_text(budget_category) -> str:
    """
    Convert budget category to readable text format.
    RAW DATA ONLY - NO FORMATTING! LLM will handle it.
    """
    if not budget_category:
        return "No budget information available."
    
    breakdown = budget_category.per_day_breakdown
    
    text = "💰 BUDGET DATA (Raw)\n"
    text += "=" * 50 + "\n\n"
    
    text += f"Category: {budget_category.category}\n"
    text += f"Description: {budget_category.description}\n\n"
    
    text += "Per Day Breakdown:\n"
    text += f"  Hotel: {breakdown.hotel}\n"
    text += f"  Food: {breakdown.food}\n"
    text += f"  Transport: {breakdown.transport}\n"
    text += f"  Activities: {breakdown.activities}\n"
    text += f"  Miscellaneous: {breakdown.miscellaneous}\n\n"
    
    text += f"Hotel Type: {budget_category.hotel_type}\n"
    text += f"Restaurant Type: {budget_category.restaurant_type}\n"
    text += f"Transport Type: {budget_category.transport_type}\n"
    text += f"Activity Type: {budget_category.activity_type}\n\n"
    
    text += f"Reasoning: {budget_category.reasoning}\n"
    
    return text
@tool
def get_travel_plan(
    weather_data: List[dict],
    budget_category: BudgetCategoryResponse,
    city: str,
    start_date: str,
    end_date: str
):
    """
        Generate complete travel plan using weather and budget data.
    
    Args:
        weather_data: List of weather records with time, temperature, etc.
        budget_category: Budget category with per-day breakdown
        city: Name of the city
        start_date: Trip start date (YYYY-MM-DD)
        end_date: Trip end date (YYYY-MM-DD)
    
    Returns:
        TravelPlanResponse: Complete travel plan with hotels, activities, food, packing, tips, and recommendation
 
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1

    travel_plan_prompt = PromptTemplate(
        template="""
You are a SENIOR TRAVEL EXPERT in {city}.

⚠️ **CRITICAL INSTRUCTION**: You MUST complete ALL fields. Incomplete responses will be rejected.

═══════════════════════════════════════════════════════════════
INPUT 1: WEATHER DATA
═══════════════════════════════════════════════════════════════
{weather_text}

═══════════════════════════════════════════════════════════════
INPUT 2: BUDGET CATEGORY
═══════════════════════════════════════════════════════════════
{budget_text}

═══════════════════════════════════════════════════════════════
YOUR TASK: Create COMPLETE TRAVEL PLAN for {total_days} days
═══════════════════════════════════════════════════════════════

**MANDATORY FIELDS - YOU MUST FILL ALL:**

1. 🏨 **HOTELS**: Exactly 3 hotels with ALL fields:
   - name, location, latitude, longitude, price, star_rating
   - room_services, dining_services, wellness_services, internet_services
   - family_services, transport_services, accessibility_services, other_services
   - weather_recommendation (MUST fill)

2. 🗺️ **DAY-WISE ACTIVITIES**: For EACH day ({total_days} days):
   - date, day, weather, is_rainy
   - morning: EXACTLY 2 places with ALL fields
   - afternoon: EXACTLY 2 places with ALL fields
   - evening: EXACTLY 1 place with ALL fields

3. 🍽️ **DAY-WISE FOOD**: For EACH day, for EACH meal (Breakfast, Lunch, Snack, Dinner):
   - day, meal, place_name, location, latitude, longitude, place_type
   - dishes: MUST have AT LEAST 2 dishes, EACH with name, price, description, why_special (ALL 4 REQUIRED)
   - weather_suitability (MUST fill)

4. 🧳 **PACKING LIST**: MUST have AT LEAST 5 items (List of strings)

5. ⚠️ **TRAVEL TIPS**: MUST have AT LEAST 5 tips (List of strings)

6. 💰 **DAILY BUDGET**: MUST be a dictionary with keys: hotel, food, transport, activities, miscellaneous

7. 🎯 **FINAL RECOMMENDATION**: 
   - recommendation: MUST be "GO" or "DON'T GO"
   - recommendation_reason: MUST be a string explaining why
   - alternative: MUST be a string or null \n
    {instructions}
    """,
        input_variables=['city', 'weather_text', 'budget_text' , 'total_days'],
        partial_variables={
            'instructions': travel_plan_parser.get_format_instructions()
        }
    )
    weather_text = convert_weather_to_text(weather_data=weather_data)
    budget_text = convert_budget_to_text(budget_category=budget_category)
    travel_chain = travel_plan_prompt | llm | travel_plan_parser
    travel_chain_result = travel_chain.invoke({
        'city' : city,
        'weather_text' : weather_text,
        'budget_text' : budget_text,
        'total_days' : total_days
    })
    
    return travel_chain_result


travel_plan = get_travel_plan.invoke({
    'weather_data' : weather_data,
    'budget_category' : budget,
    'city' : "jaipur",
    'start_date' : '2026-07-08',
    'end_date' : '2026-07-10',

})

print(travel_plan)

