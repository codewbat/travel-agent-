from typing import List, Optional
from pydantic import BaseModel, Field

class DishDetail(BaseModel):
    name: str = Field(description="Name of the dish")
    price: int = Field(description="Price of this dish in rupees")
    description: Optional[str] = Field(default="Delicious dish", description="Brief description of the dish")
    why_special: Optional[str] = Field(default="Local specialty", description="Why this dish is special here")

class FoodRecommendation(BaseModel):
    day: int = Field(description="Day number (1, 2, 3, etc.)")
    meal: str = Field(description="Meal type: Breakfast/Lunch/Snack/Dinner")
    place_name: str = Field(description="Restaurant/Cafe/Street Food Stall name")
    location: str = Field(description="Area/location in the city")
    latitude: float = Field(description="Latitude of the place")
    longitude: float = Field(description="Longitude of the place")
    place_type: Optional[str] = Field(default="Restaurant", description="Type: Restaurant/Cafe/Street Food/Vendor")
    dishes: List[DishDetail] = Field(default=[], description="List of recommended dishes at this place")
    weather_suitability: Optional[str] = Field(default="Suitable for all weather", description="Why this food suits today's weather")

class FoodRecommendationList(BaseModel):
    foods: List[FoodRecommendation] = Field(description="List of food recommendations")
