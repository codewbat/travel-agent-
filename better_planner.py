import os
from dotenv import load_dotenv
import requests
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from typing import Annotated, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import json

load_dotenv()

weather_api_key = os.getenv("Weather_API_KEY")
groq_key = os.getenv("GROQAI_API_Key")

llm = ChatGroq(
    api_key=groq_key,
    model="llama-3.3-70b-versatile",
    temperature=1.2
)


# ==================== PYDANTIC MODELS ====================

class BudgetBreakdown(BaseModel):
    hotel: int = Field(description="Amount to spend on hotel per day")
    food: int = Field(description="Amount to spend on food per day")
    transport: int = Field(description="Amount to spend on transport per day")
    activities: int = Field(description="Amount to spend on activities per day")
    miscellaneous: int = Field(description="Amount to spend on miscellaneous items per day")


class BudgetCategoryResponse(BaseModel):
    category: Annotated[str, Field(description="Budget category: Budget/Mid-Range/Luxury")]
    description: Annotated[str, Field(description="Brief description of what this budget means")]
    hotel_type: Annotated[str, Field(description="What type of hotels they can afford")]
    restaurant_type: Annotated[str, Field(description="What type of restaurants they can afford")]
    transport_type: Annotated[str, Field(description="What type of transport they can use")]
    activity_type: Annotated[str, Field(description="What type of activities they can do")]
    per_day_breakdown: Annotated[BudgetBreakdown, Field(description="Per day budget breakdown")]
    reasoning: Annotated[str, Field(description="Why this budget category was chosen")]


class ActivityDetail(BaseModel):
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
    name: str = Field(description="Hotel name")
    location: str = Field(description="Hotel location/area")
    latitude: float = Field(description="Latitude of hotel")
    longitude: float = Field(description="Longitude of hotel")
    price: int = Field(description="Price per night in rupees")
    star_rating: float = Field(description="Star rating (1-5)")
    room_services: List[str] = Field(description="Room services")
    dining_services: List[str] = Field(description="Dining services")
    wellness_services: List[str] = Field(description="Wellness services")
    internet_services: List[str] = Field(description="Internet services")
    family_services: List[str] = Field(description="Family services")
    transport_services: List[str] = Field(description="Transport services")
    accessibility_services: List[str] = Field(description="Accessibility services")
    other_services: List[str] = Field(description="Other services")
    weather_recommendation: str = Field(description="Why this hotel is good for the weather")


class DailyActivity(BaseModel):
    date: str = Field(description="Date")
    day: str = Field(description="Day name")
    weather: str = Field(description="Weather condition")
    is_rainy: bool = Field(description="Whether day is rainy")
    morning: List[ActivityDetail] = Field(description="Morning activities (8 AM - 12 PM)")
    afternoon: List[ActivityDetail] = Field(description="Afternoon activities (12 PM - 5 PM)")
    evening: List[ActivityDetail] = Field(description="Evening activities (5 PM - 9 PM)")


class DishDetail(BaseModel):
    name: str = Field(description="Name of the dish")
    price: int = Field(description="Price of this dish in rupees")
    description: str = Field(description="Brief description of the dish")
    why_special: str = Field(description="Why this dish is special here")


class FoodRecommendation(BaseModel):
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
    city: Annotated[str, Field(description="City name")]
    start_date: Annotated[str, Field(description="Start date")]
    end_date: Annotated[str, Field(description="End date")]
    total_days: Annotated[int, Field(description="Total days")]
    budget_category: Annotated[str, Field(description="Budget category")]
    hotels: Annotated[List[HotelRecommendation], Field(description="Hotel recommendations")]
    daily_plan: Annotated[List[DailyActivity], Field(description="Day-wise activities")]
    food_recommendations: Annotated[List[FoodRecommendation], Field(description="Day-wise food")]
    packing_list: Annotated[List[str], Field(description="Items to pack")]
    tips: Annotated[List[str], Field(description="Travel tips")]
    daily_budget: Annotated[dict, Field(description="Daily budget breakdown")]
    recommendation: Annotated[str, Field(description="GO or DON'T GO")]
    recommendation_reason: Annotated[str, Field(description="Why this recommendation")]
    alternative: Annotated[Optional[str], Field(description="Alternative suggestion")]


# ==================== WRAPPER CLASSES FOR LIST OUTPUT ====================
class HotelRecommendationList(BaseModel):
    hotels: List[HotelRecommendation] = Field(description="List of hotel recommendations")


class ActivityRecommendationList(BaseModel):
    activities: List[DailyActivity] = Field(description="List of day-wise activities")


class FoodRecommendationList(BaseModel):
    foods: List[FoodRecommendation] = Field(description="List of food recommendations")


# ==================== TOOL 1: Weather Data ====================
@tool
def get_weather_report_of_city(
    city_name: str,
    startdate: str,
    enddate: str
):
    """Get city weather data from API between two dates"""
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={weather_api_key}"
    
    response = requests.get(url)
    data = response.json()
    
    filter_by_date = [
        item for item in data["list"] 
        if startdate <= item["dt_txt"].split()[0] <= enddate
    ]
    
    weather_summary = []
    
    for item in filter_by_date:
        weather_summary.append({
            "time": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "feels_like": item["main"]["feels_like"],
            "humidity": item["main"]["humidity"],
            "weather": item["weather"][0]["description"],
            "wind": item["wind"]["speed"],
            "rain_probability": item["pop"],
        })
    return weather_summary


# ==================== TOOL 2: Budget Category ====================
budget_parser = PydanticOutputParser(pydantic_object=BudgetCategoryResponse)

@tool
def get_categorize_budget(
    total_budget: int,
    total_days: int,
    city: str
):
    """LLM will decide and breakdown budget category."""
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

{instruction}
""",
        input_variables=['total_budget', 'total_days', 'city'],
        partial_variables={
            'instruction': budget_parser.get_format_instructions()
        }
    )
    
    budget_chain = categorize_budget_prompt | llm | budget_parser
    budget_chain_result = budget_chain.invoke({
        'total_budget': total_budget,
        'total_days': total_days,
        'city': city
    })
    
    return budget_chain_result


# ==================== TOOL 3: Hotel Recommendations ====================
hotel_parser = PydanticOutputParser(pydantic_object=HotelRecommendationList)

@tool
def get_hotel_recommendations(
    city: str,
    budget_category: BudgetCategoryResponse,
    weather_data: List[dict]
):
    """Get hotel recommendations based on budget and weather."""
    
    rainy_count = sum(1 for d in weather_data if d.get('rain_probability', 0) > 0.4)
    total_days = len(weather_data) // 8 or 1
    weather_type = 'rainy' if rainy_count > total_days/2 else 'sunny'
    
    prompt = PromptTemplate(
        template=f"""
You are a hotel expert in {city}.

BUDGET INFO:
- Category: {{category}}
- Hotel Budget: ₹{{hotel_budget}}/night
- Hotel Type: {{hotel_type}}

WEATHER:
- Rainy Days: {{rainy_days}}/{{total_days}} days
- Weather Type: {weather_type}

Suggest 3 hotels in {city} that:
1. Fit within ₹{{hotel_budget}}/night
2. Have good amenities for {weather_type} weather
3. Include all services

{{format_instructions}}
""",
        input_variables=['city', 'category', 'hotel_budget', 'hotel_type', 'rainy_days', 'total_days', 'format_instructions'],
        partial_variables={
            'format_instructions': hotel_parser.get_format_instructions()
        }
    )
    
    chain = prompt | llm | hotel_parser
    
    result = chain.invoke({
        'city': city,
        'category': budget_category.category,
        'hotel_budget': budget_category.per_day_breakdown.hotel,
        'hotel_type': budget_category.hotel_type,
        'rainy_days': rainy_count,
        'total_days': total_days,
        'format_instructions': hotel_parser.get_format_instructions()
    })
    
    return result.hotels


# ==================== TOOL 4: Activity Recommendations ====================
activity_parser = PydanticOutputParser(pydantic_object=ActivityRecommendationList)

@tool
def get_activity_recommendations(
    city: str,
    weather_data: List[dict],
    budget_category: BudgetCategoryResponse,
    start_date: str,
    end_date: str
):
    """Get day-wise activity recommendations based on weather."""
    
    weather_by_date = {}
    for item in weather_data:
        date = item.get('time', '').split()[0] if 'time' in item else 'Unknown'
        if date not in weather_by_date:
            weather_by_date[date] = []
        weather_by_date[date].append(item)
    
    daily_weather = []
    for date, items in weather_by_date.items():
        rain_probs = [i.get('rain_probability', 0) for i in items]
        conditions = [i.get('weather', '') for i in items]
        avg_rain = sum(rain_probs) / len(rain_probs) if rain_probs else 0
        is_rainy = avg_rain > 0.4
        dominant_condition = max(set(conditions), key=conditions.count) if conditions else 'Unknown'
        
        daily_weather.append({
            'date': date,
            'day': datetime.strptime(date, "%Y-%m-%d").strftime("%A"),
            'is_rainy': is_rainy,
            'condition': dominant_condition,
            'rain_chance': round(avg_rain * 100, 0)
        })
    
    daily_weather_str = json.dumps(daily_weather, indent=2)
    
    prompt = PromptTemplate(
        template="""
You are a travel expert in {city}.

WEATHER FORECAST:
{daily_weather}

BUDGET: {budget_category}

Create a day-wise activity plan for {total_days} days:
- Morning: 2 places
- Afternoon: 2 places
- Evening: 1 place
- Based on weather (indoor if rainy, outdoor if sunny)
- Include all details

{format_instructions}
""",
        input_variables=['city', 'daily_weather', 'budget_category', 'total_days', 'format_instructions'],
        partial_variables={
            'format_instructions': activity_parser.get_format_instructions()
        }
    )
    
    chain = prompt | llm | activity_parser
    
    result = chain.invoke({
        'city': city,
        'daily_weather': daily_weather_str,
        'budget_category': budget_category.category,
        'total_days': len(daily_weather),
        'format_instructions': activity_parser.get_format_instructions()
    })
    
    return result.activities


# ==================== TOOL 5: Food Recommendations ====================
food_parser = PydanticOutputParser(pydantic_object=FoodRecommendationList)

@tool
def get_food_recommendations(
    city: str,
    budget_category: BudgetCategoryResponse,
    total_days: int
):
    """Get day-wise food recommendations based on budget."""
    
    prompt = PromptTemplate(
        template="""
You are a food expert in {city}.

BUDGET:
- Category: {category}
- Food Budget per day: ₹{food_budget}

Create food recommendations for {total_days} days:
- For each day: Breakfast, Lunch, Snack, Dinner
- Each meal: place_name, location, lat/long, place_type
- Each place: 2 dishes with name, price, description, why_special
- Include weather_suitability for each meal

{format_instructions}
""",
        input_variables=['city', 'category', 'food_budget', 'total_days', 'format_instructions'],
        partial_variables={
            'format_instructions': food_parser.get_format_instructions()
        }
    )
    
    chain = prompt | llm | food_parser
    
    result = chain.invoke({
        'city': city,
        'category': budget_category.category,
        'food_budget': budget_category.per_day_breakdown.food,
        'total_days': total_days,
        'format_instructions': food_parser.get_format_instructions()
    })
    
    return result.foods


# ==================== TOOL 6: Packing + Tips + Recommendation ====================
@tool
def get_packing_tips_recommendation(
    city: str,
    weather_data: List[dict],
    budget_category: BudgetCategoryResponse,
    total_days: int
):
    """Get packing list, tips, and final recommendation."""
    
    rain_probs = [d.get('rain_probability', 0) for d in weather_data]
    avg_rain = sum(rain_probs) / len(rain_probs) if rain_probs else 0
    is_rainy = avg_rain > 0.4
    rainy_days = sum(1 for d in weather_data if d.get('rain_probability', 0) > 0.4) // 8
    weather_pattern = 'Rainy' if is_rainy else 'Sunny'
    
    prompt = PromptTemplate(
        template=f"""
You are a travel expert for {city}.

WEATHER:
- Rainy Days: {{rainy_days}}/{{total_days}} days
- Weather Pattern: {weather_pattern}

BUDGET: {{category}}

Based on the weather and budget:

1. PACKING LIST: At least 6 items
2. TRAVEL TIPS: At least 6 tips
3. FINAL RECOMMENDATION: 
   - If rainy days >= total_days/2 → DON'T GO
   - If rainy days < total_days/2 → GO
   - Give reason and alternative if not going

Return in this JSON format:
{{{{ 
    "packing_list": ["item1", "item2", ...],
    "tips": ["tip1", "tip2", ...],
    "recommendation": "GO or DON'T GO",
    "recommendation_reason": "reason",
    "alternative": "alternative suggestion or null"
}}}}
""",
        input_variables=['city', 'rainy_days', 'total_days', 'category']
    )
    
    chain = prompt | llm
    
    result = chain.invoke({
        'city': city,
        'rainy_days': rainy_days,
        'total_days': total_days,
        'category': budget_category.category
    })
    
    try:
        return json.loads(result.content)
    except:
        return {
            "packing_list": ["Umbrella", "Raincoat", "Light clothes", "Comfortable shoes", "Sunscreen", "Power bank"],
            "tips": ["Carry umbrella", "Start early", "Stay hydrated", "Use Ola/Uber", "Carry cash", "Book in advance"],
            "recommendation": "GO" if not is_rainy else "DON'T GO",
            "recommendation_reason": "Weather is favorable" if not is_rainy else "Too much rain",
            "alternative": "Try October-November" if is_rainy else None
        }


# ==================== AGGREGATOR ====================
def create_travel_plan(
    city: str,
    start_date: str,
    end_date: str,
    weather_data: List[dict],
    budget_category: BudgetCategoryResponse,
    hotels: List[HotelRecommendation],
    activities: List[DailyActivity],
    food: List[FoodRecommendation],
    packing_tips: dict
) -> TravelPlanResponse:
    """Combine all results into final TravelPlanResponse."""
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1
    
    return TravelPlanResponse(
        city=city,
        start_date=start_date,
        end_date=end_date,
        total_days=total_days,
        budget_category=budget_category.category,
        hotels=hotels,
        daily_plan=activities,
        food_recommendations=food,
        packing_list=packing_tips.get("packing_list", []),
        tips=packing_tips.get("tips", []),
        daily_budget={
            "hotel": budget_category.per_day_breakdown.hotel,
            "food": budget_category.per_day_breakdown.food,
            "transport": budget_category.per_day_breakdown.transport,
            "activities": budget_category.per_day_breakdown.activities,
            "miscellaneous": budget_category.per_day_breakdown.miscellaneous
        },
        recommendation=packing_tips.get("recommendation", "GO"),
        recommendation_reason=packing_tips.get("recommendation_reason", "Weather is favorable"),
        alternative=packing_tips.get("alternative")
    )


# ==================== MAIN ====================
if __name__ == "__main__":
    print("🌍 TRAVEL PLANNING ASSISTANT (Parallel Tools)")
    print("=" * 70)
    
    city = "Jaipur"
    start_date = "2026-07-08"
    end_date = "2026-07-10"
    total_budget = 10000
    
    # STEP 1: Get Weather
    print("\n🌤️ Fetching Weather Data...")
    weather_data = get_weather_report_of_city.invoke({
        'city_name': city.lower(),
        'startdate': start_date,
        'enddate': end_date
    })
    print(f"✅ Weather: {len(weather_data)} records")
    
    # STEP 2: Get Budget Category
    print("\n💰 Getting Budget Category...")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1
    
    budget = get_categorize_budget.invoke({
        'total_budget': total_budget,
        'total_days': total_days,
        'city': city
    })
    print(f"✅ Budget: {budget.category}")
    
    # STEP 3: Get Hotel Recommendations
    print("\n🏨 Getting Hotel Recommendations...")
    hotels = get_hotel_recommendations.invoke({
        'city': city,
        'budget_category': budget,
        'weather_data': weather_data
    })
    print(f"✅ Hotels: {len(hotels)} found")
    
    # STEP 4: Get Activity Recommendations
    print("\n🗺️ Getting Activity Recommendations...")
    activities = get_activity_recommendations.invoke({
        'city': city,
        'weather_data': weather_data,
        'budget_category': budget,
        'start_date': start_date,
        'end_date': end_date
    })
    print(f"✅ Activities: {len(activities)} days planned")
    
    # STEP 5: Get Food Recommendations
    print("\n🍽️ Getting Food Recommendations...")
    food = get_food_recommendations.invoke({
        'city': city,
        'budget_category': budget,
        'total_days': total_days
    })
    print(f"✅ Food: {len(food)} recommendations")
    
    # STEP 6: Get Packing + Tips + Recommendation
    print("\n🧳 Getting Packing List, Tips & Recommendation...")
    packing_tips = get_packing_tips_recommendation.invoke({
        'city': city,
        'weather_data': weather_data,
        'budget_category': budget,
        'total_days': total_days
    })
    print(f"✅ Packing & Tips: Done")
    
    # STEP 7: Combine All Results
    print("\n📦 Combining All Results...")
    travel_plan = create_travel_plan(
        city=city,
        start_date=start_date,
        end_date=end_date,
        weather_data=weather_data,
        budget_category=budget,
        hotels=hotels,
        activities=activities,
        food=food,
        packing_tips=packing_tips
    )
    
    # Display Results
    print("\n" + "=" * 70)
    print("🌍 COMPLETE TRAVEL PLAN")
    print("=" * 70)
    
    print(f"\n📍 {travel_plan.city}")
    print(f"📅 {travel_plan.start_date} to {travel_plan.end_date} ({travel_plan.total_days} days)")
    print(f"💰 Category: {travel_plan.budget_category}")
    
    # Hotels
    print("\n" + "-" * 40)
    print("🏨 HOTELS")
    print("-" * 40)
    for hotel in travel_plan.hotels:
        print(f"\n🏨 {hotel.name}")
        print(f"   📍 Location: {hotel.location}")
        print(f"   💰 ₹{hotel.price}/night")
        print(f"   ⭐ {hotel.star_rating}/5")
        print(f"   🌤️ {hotel.weather_recommendation}")
    
    # Activities
    print("\n" + "-" * 40)
    print("🗺️ ACTIVITIES")
    print("-" * 40)
    for day in travel_plan.daily_plan:
        emoji = "🌧️" if day.is_rainy else "☀️"
        print(f"\n📅 {day.date} ({day.day}) {emoji}")
        print(f"   Morning: {', '.join([a.name for a in day.morning])}")
        print(f"   Afternoon: {', '.join([a.name for a in day.afternoon])}")
        print(f"   Evening: {', '.join([a.name for a in day.evening])}")
    
    # Food
    print("\n" + "-" * 40)
    print("🍽️ FOOD")
    print("-" * 40)
    for food_item in travel_plan.food_recommendations:
        print(f"\nDay {food_item.day} - {food_item.meal}")
        print(f"   {food_item.place_name} ({food_item.place_type})")
        print(f"   Dishes: {', '.join([d.name for d in food_item.dishes])}")
    
    # Packing
    print("\n" + "-" * 40)
    print("🧳 PACKING LIST")
    print("-" * 40)
    for item in travel_plan.packing_list:
        print(f"   • {item}")
    
    # Tips
    print("\n" + "-" * 40)
    print("⚠️ TIPS")
    print("-" * 40)
    for tip in travel_plan.tips:
        print(f"   • {tip}")
    
    # Budget
    print("\n" + "-" * 40)
    print("💰 DAILY BUDGET")
    print("-" * 40)
    for key, value in travel_plan.daily_budget.items():
        print(f"   {key.capitalize()}: ₹{value}")
    
    # Recommendation
    print("\n" + "-" * 40)
    print("🎯 RECOMMENDATION")
    print("-" * 40)
    if travel_plan.recommendation == "GO":
        print("✅ GO")
    else:
        print("❌ DON'T GO")
    print(f"   {travel_plan.recommendation_reason}")
    if travel_plan.alternative:
        print(f"   Alternative: {travel_plan.alternative}")
    
    print("\n" + "=" * 70)
    print("✅ Complete!")
    print("=" * 70)