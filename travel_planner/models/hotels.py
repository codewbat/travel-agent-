from typing import List
from pydantic import BaseModel, Field

class RoomType(BaseModel):
    """Room type information"""
    name: str = Field(description="Room name")
    price: str = Field(description="Room price")
    beds: List[str] = Field(default=[], description="Bed options")
    max_occupancy: int = Field(default=0, description="Maximum occupancy")
    cancellation: str = Field(default="N/A", description="Cancellation policy")
    cancellation_free: bool = Field(default=False, description="Is cancellation free")

class TransportLocation(BaseModel):
    """Transportation location"""
    name: str = Field(description="Location name")
    time: str = Field(description="Travel time")

class TrustYouReview(BaseModel):
    """TrustYou review category"""
    category_name: str = Field(description="Review category")
    percentage: str = Field(description="Positive percentage")
    text: str = Field(description="Review text")
    sentiment: str = Field(description="Sentiment (pos/neg)")

class HotelData(BaseModel):
    """Complete hotel data model"""
    name: str = Field(description="Hotel name")
    star_rating: int = Field(description="Star rating (1-5)")
    address: str = Field(description="Full address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    neighborhood: str = Field(description="Neighborhood name")
    current_price: str = Field(description="Current price per night")
    original_price: str = Field(description="Original price per night")
    currency: str = Field(description="Currency code")
    price_numeric: int = Field(default=0, description="Numeric price")
    guest_rating: str = Field(description="Guest rating out of 10")
    rating_badge: str = Field(description="Rating badge text")
    reviews_count: int = Field(default=0, description="Number of reviews")
    tripadvisor_rating: str = Field(default="N/A", description="TripAdvisor rating")
    tripadvisor_reviews: int = Field(default=0, description="TripAdvisor review count")
    trustyou_reviews: List[TrustYouReview] = Field(default=[], description="TrustYou review categories")
    amenities: List[str] = Field(default=[], description="Main amenities")
    freebies: List[str] = Field(default=[], description="Free amenities")
    room_amenities: List[str] = Field(default=[], description="In-room amenities")
    tagline: str = Field(default="", description="Hotel tagline")
    room_types: List[RoomType] = Field(default=[], description="Available room types")
    latitude: float = Field(default=0.0, description="Latitude")
    longitude: float = Field(default=0.0, description="Longitude")
    location_name: str = Field(default="", description="Location name")
    nearby_attractions: List[str] = Field(default=[], description="Nearby attractions")
    airports: List[TransportLocation] = Field(default=[], description="Nearby airports")
    train_stations: List[TransportLocation] = Field(default=[], description="Nearby train stations")
    metro_stations: List[TransportLocation] = Field(default=[], description="Nearby metro stations")
    check_in_time: str = Field(default="N/A", description="Check-in time")
    check_out_time: str = Field(default="N/A", description="Check-out time")
    required_at_checkin: List[str] = Field(default=[], description="Required documents")
    hotel_size: List[str] = Field(default=[], description="Hotel size information")
    pets_policy: List[str] = Field(default=[], description="Pets policy")
    policies: List[str] = Field(default=[], description="Hotel policies")
    optional_extras: List[str] = Field(default=[], description="Optional extras")
    mandatory_fees: List[str] = Field(default=[], description="Mandatory fees")
    dining_info: List[str] = Field(default=[], description="Dining information")
    awards: List[str] = Field(default=[], description="Awards and affiliations")
    badge_type: str = Field(default="", description="Hotel badge type")
    badge_tooltip: str = Field(default="", description="Badge tooltip")

class HotelRecommendation(BaseModel):
    name: str = Field(description="Hotel name")
    location: str = Field(description="Hotel location/area")
    latitude: float = Field(description="Latitude of hotel")
    longitude: float = Field(description="Longitude of hotel")
    price: int = Field(description="Price per night in rupees")
    star_rating: float = Field(description="Star rating (1-5)")
    room_services: List[str] = Field(default=[], description="Room services")
    dining_services: List[str] = Field(default=[], description="Dining services")
    wellness_services: List[str] = Field(default=[], description="Wellness services")
    internet_services: List[str] = Field(default=[], description="Internet services")
    family_services: List[str] = Field(default=[], description="Family services")
    transport_services: List[str] = Field(default=[], description="Transport services")
    accessibility_services: List[str] = Field(default=[], description="Accessibility services")
    other_services: List[str] = Field(default=[], description="Other services")
    weather_recommendation: str = Field(description="Why this hotel is good for the weather")

class HotelRecommendationList(BaseModel):
    hotels: List[HotelRecommendation] = Field(description="List of hotel recommendations")
