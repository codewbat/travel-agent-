from typing import List, Optional
from pydantic import BaseModel, Field

class ActivityDetail(BaseModel):
    name: str = Field(description="Name of the place/activity")
    location: str = Field(description="Area/location in the city")
    latitude: float = Field(description="Latitude of the place")
    longitude: float = Field(description="Longitude of the place")
    entry_fee: int = Field(default=0, description="Entry fee in rupees (0 if free)")
    timing: str = Field(description="Timing (e.g., 9:00 AM - 5:00 PM)")
    description: Optional[str] = Field(default="No description available", description="Brief description of the place")
    why_recommend: Optional[str] = Field(default="Recommended for the trip", description="Why this place is recommended for today's weather")
    duration: Optional[str] = Field(default="1-2 hours", description="Suggested duration to spend (e.g., 1-2 hours)")

class DailyActivity(BaseModel):
    date: str = Field(description="Date")
    day: str = Field(description="Day name")
    weather: str = Field(description="Weather condition")
    is_rainy: bool = Field(description="Whether day is rainy")
    morning: List[ActivityDetail] = Field(default=[], description="Morning activities (8 AM - 12 PM)")
    afternoon: List[ActivityDetail] = Field(default=[], description="Afternoon activities (12 PM - 5 PM)")
    evening: List[ActivityDetail] = Field(default=[], description="Evening activities (5 PM - 9 PM)")

class ActivityRecommendationList(BaseModel):
    activities: List[DailyActivity] = Field(description="List of day-wise activities")
