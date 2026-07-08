from pydantic import BaseModel, Field

class BudgetBreakdown(BaseModel):
    hotel: int = Field(description="Amount to spend on hotel per day")
    food: int = Field(description="Amount to spend on food per day")
    transport: int = Field(description="Amount to spend on transport per day")
    activities: int = Field(description="Amount to spend on activities per day")
    miscellaneous: int = Field(description="Amount to spend on miscellaneous items per day")

class BudgetCategoryResponse(BaseModel):
    category: str = Field(default="Mid-Range", description="Budget category: Budget/Mid-Range/Luxury")
    description: str = Field(default="Standard budget allocation", description="Brief description of what this budget means")
    hotel_type: str = Field(default="3-star hotel", description="What type of hotels they can afford")
    restaurant_type: str = Field(default="Casual dining", description="What type of restaurants they can afford")
    transport_type: str = Field(default="Metro/Cab", description="What type of transport they can use")
    activity_type: str = Field(default="Sightseeing", description="What type of activities they can do")
    per_day_breakdown: BudgetBreakdown = Field(description="Per day budget breakdown")
    reasoning: str = Field(default="Calculated from total budget", description="Why this budget category was chosen")

class DailyBudget(BaseModel):
    hotel: int = Field(description="Amount to spend on hotel per day")
    food: int = Field(description="Amount to spend on food per day")
    transport: int = Field(description="Amount to spend on transport per day")
    activities: int = Field(description="Amount to spend on activities per day")
    miscellaneous: int = Field(description="Amount to spend on miscellaneous items per day")
