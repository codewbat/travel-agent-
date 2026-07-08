import re
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from travel_planner.config import llm, llm_fallback, RAPIDAPI_KEY, RAPIDAPI_HOST
from travel_planner.models import (
    RoomType, TransportLocation, TrustYouReview, HotelData,
    HotelRecommendation, HotelRecommendationList, BudgetCategoryResponse
)
from travel_planner.rag import get_hotel_rag, retrieve_hotel_info

def clean_and_parse_json(raw_json: str, parser) -> Any:
    """Helper to strip comments and parse JSON manually with the parser."""
    # Strip inline comments but keep URLs safe
    cleaned = re.sub(r'(?<!https:)(?<!http:)//.*$', '', raw_json, flags=re.MULTILINE)
    # Strip block comments if any
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    # Extract JSON if wrapped in markdown code blocks
    if cleaned.startswith("```"):
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1:
            cleaned = cleaned[first_brace:last_brace+1]
    return parser.parse(cleaned)


# Reuse connection session for high performance
http_session = requests.Session()

# ========== SCHEMAS FOR TOOLS ==========
class HotelExtractInput(BaseModel):
    hotel_id: str = Field(description="Hotel ID to fetch details for")
    check_in: str = Field(description="Check-in date (YYYY-MM-DD)")
    check_out: str = Field(description="Check-out date (YYYY-MM-DD)")
    adults: int = Field(default=2, description="Number of adults")
    currency: str = Field(default="USD", description="Currency code")

class HotelSearchInput(BaseModel):
    city_name: str = Field(description="Name of the city")
    check_in: str = Field(description="Check-in date (YYYY-MM-DD)")
    check_out: str = Field(description="Check-out date (YYYY-MM-DD)")
    adults: int = Field(default=2, description="Number of adults")
    currency: str = Field(default="USD", description="Currency code")

class HotelInput(BaseModel):
    city: str = Field(description="City name")
    budget_category: BudgetCategoryResponse = Field(description="Budget category Response")
    weather_data: List[dict] = Field(description="Weather records list")

# ========== HELPERS ==========
def _get_destination_id(city_name: str) -> Optional[str]:
    """Get destination ID for a city."""
    try:
        url = "https://hotels4.p.rapidapi.com/locations/v3/search"
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": RAPIDAPI_HOST
        }
        params = {
            "q": city_name,
            "locale": "en_US"
        }
        response = http_session.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        
        if 'sr' in data and len(data['sr']) > 0:
            for result in data['sr']:
                if result.get('type') in ['CITY', 'REGION']:
                    gaia_id = result.get('gaiaId')
                    if gaia_id:
                        return gaia_id
            
            first_result = data['sr'][0]
            gaia_id = first_result.get('gaiaId')
            if gaia_id:
                return gaia_id
        return None
    except Exception as e:
        print(f"   ❌ Error getting destination ID: {str(e)}")
        return None

def _search_hotels(
    destination_id: str,
    check_in: str,
    check_out: str,
    adults: int = 2,
    currency: str = "USD"
) -> List[Dict]:
    """Search for hotels using destination ID."""
    try:
        url = "https://hotels4.p.rapidapi.com/properties/list"
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": RAPIDAPI_HOST
        }
        params = {
            "destinationId": destination_id,
            "pageNumber": "1",
            "pageSize": "25",
            "checkIn": check_in,
            "checkOut": check_out,
            "adults1": str(adults),
            "sortOrder": "PRICE",
            "currency": currency
        }
        response = http_session.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        
        hotels = []
        if 'data' in data and 'body' in data['data'] and 'searchResults' in data['data']['body']:
            search_results = data['data']['body']['searchResults']['results']
            for result in search_results:
                hotel = {
                    'id': result.get('id'),
                    'name': result.get('name'),
                    'star_rating': result.get('starRating', 0),
                    'price': result.get('ratePlan', {}).get('price', {}).get('current', 'N/A'),
                    'currency': result.get('ratePlan', {}).get('price', {}).get('currency', 'USD'),
                    'address': result.get('address', {}).get('fullAddress', ''),
                    'city': result.get('address', {}).get('locality', ''),
                    'guest_rating': result.get('guestRating', 'N/A'),
                    'reviews_count': result.get('reviews', {}).get('total', 0),
                }
                hotels.append(hotel)
        return hotels
    except Exception as e:
        print(f"   ❌ Error searching hotels: {str(e)}")
        return []

def _extract_hotel_data_to_model(hotel_response: Dict) -> HotelData:
    """Extract all data from API response and create HotelData object."""
    data = hotel_response.get('data') or {}
    body = data.get('body') or {}
    
    prop_desc = body.get('propertyDescription') or {}
    name = prop_desc.get('name', 'Unknown Hotel')
    star_rating = prop_desc.get('starRating', 0)
    address = (prop_desc.get('address') or {}).get('fullAddress', 'N/A')
    city = (prop_desc.get('address') or {}).get('cityName', 'N/A')
    country = (prop_desc.get('address') or {}).get('countryName', 'N/A')
    neighborhood = (hotel_response.get('neighborhood') or {}).get('neighborhoodName', 'N/A')
    
    featured_price = prop_desc.get('featuredPrice') or {}
    current_price_obj = featured_price.get('currentPrice') or {}
    current_price = current_price_obj.get('formatted', 'N/A')
    original_price = featured_price.get('oldPrice', 'N/A')
    currency = (body.get('pdpHeader') or {}).get('currencyCode', 'USD')
    price_numeric = current_price_obj.get('plain', 0)
    
    guest_reviews = body.get('guestReviews') or {}
    brands = guest_reviews.get('brands') or {}
    guest_rating = brands.get('formattedRating', 'N/A')
    rating_badge = brands.get('badgeText', 'N/A')
    reviews_count = brands.get('total', 0)
    
    tripadvisor = guest_reviews.get('tripAdvisor') or {}
    tripadvisor_rating = str(tripadvisor.get('rating', 'N/A'))
    tripadvisor_reviews = tripadvisor.get('total', 0)
    
    trustyou_data = guest_reviews.get('trustYouReviews') or []
    trustyou_reviews = []
    for item in trustyou_data[:3]:
        trustyou_reviews.append(TrustYouReview(
            category_name=item.get('categoryName', ''),
            percentage=item.get('percentage', ''),
            text=item.get('text', ''),
            sentiment=item.get('sentiment', '')
        ))
    
    overview = body.get('overview') or {}
    overview_sections = overview.get('overviewSections') or []
    amenities = []
    freebies = []
    nearby_attractions = []
    tagline = ""
    
    for section in overview_sections:
        section_type = section.get('type', '')
        content = section.get('content') or []
        if section_type == 'HOTEL_FEATURE':
            amenities = content[:10]
        elif section_type == 'LOCATION_SECTION':
            nearby_attractions = content[:5]
        elif section_type == 'HOTEL_FREEBIES':
            freebies = content
        elif section_type == 'TAGLINE':
            tagline = content[0] if content else ""
            
    amenities_data = body.get('amenities') or []
    room_amenities = []
    for amenity_group in amenities_data:
        if amenity_group.get('heading') == 'In the room':
            for item in amenity_group.get('listItems') or []:
                if item.get('listItems'):
                    room_amenities.extend(item['listItems'][:5])
    room_amenities = room_amenities[:10]
    
    rooms_and_rates = body.get('roomsAndRates') or {}
    rooms = rooms_and_rates.get('rooms') or []
    room_types = []
    for room in rooms[:3]:
        rate_plan = room.get('ratePlans', [{}])[0] if room.get('ratePlans') else {}
        room_types.append(RoomType(
            name=room.get('name', ''),
            price=(rate_plan.get('price') or {}).get('current', 'N/A'),
            beds=(room.get('bedChoices') or {}).get('mainOptions', []),
            max_occupancy=(room.get('maxOccupancy') or {}).get('total', 0),
            cancellation=(rate_plan.get('cancellation') or {}).get('title', 'N/A'),
            cancellation_free=(rate_plan.get('cancellation') or {}).get('free', False)
        ))
        
    pdp_header = body.get('pdpHeader') or {}
    hotel_location = pdp_header.get('hotelLocation') or {}
    coordinates = hotel_location.get('coordinates') or {}
    latitude = coordinates.get('latitude', 0.0)
    longitude = coordinates.get('longitude', 0.0)
    location_name = hotel_location.get('locationName', '')
    
    transportation = hotel_response.get('transportation') or {}
    transport_locations = transportation.get('transportLocations') or []
    airports = []
    train_stations = []
    metro_stations = []
    
    for loc in transport_locations:
        category = loc.get('category', '')
        locations = loc.get('locations') or []
        if category == 'airport':
            for l in locations[:3]:
                airports.append(TransportLocation(name=l.get('name', '').split('-')[0].strip(), time=l.get('distanceInTime', '')))
        elif category == 'train-station':
            for l in locations[:3]:
                train_stations.append(TransportLocation(name=l.get('name', '').split('-')[0].strip(), time=l.get('distanceInTime', '')))
        elif category == 'metro':
            for l in locations[:3]:
                metro_stations.append(TransportLocation(name=l.get('name', '').split('-')[0].strip(), time=l.get('distanceInTime', '')))
                
    at_glance = body.get('atAGlance') or {}
    key_facts = at_glance.get('keyFacts') or {}
    arriving_leaving = key_facts.get('arrivingLeaving') or []
    check_in_time = arriving_leaving[0] if len(arriving_leaving) > 0 else 'N/A'
    check_out_time = arriving_leaving[1] if len(arriving_leaving) > 1 else 'N/A'
    required_at_checkin = key_facts.get('requiredAtCheckIn') or []
    hotel_size = key_facts.get('hotelSize') or []
    
    travelling = (at_glance.get('travellingOrInternet') or {}).get('travelling') or {}
    pets_policy = travelling.get('pets') or []
    
    small_print = body.get('smallPrint') or {}
    policies = small_print.get('policies') or []
    policies = policies[:2]
    optional_extras = small_print.get('optionalExtras') or []
    optional_extras = optional_extras[:3]
    mandatory_fees = small_print.get('mandatoryFees') or []
    
    special_features = (body.get('specialFeatures') or {}).get('sections') or []
    dining_info = []
    awards = []
    for section in special_features:
        heading = section.get('heading', '')
        free_text = section.get('freeText', '')
        if 'dining' in heading.lower():
            dining_info.append(free_text)
        elif 'award' in heading.lower() or 'affiliation' in heading.lower():
            awards.append(free_text)
            
    hotel_badge = body.get('hotelBadge') or {}
    badge_type = hotel_badge.get('label', '')
    badge_tooltip = hotel_badge.get('tooltipTitle', '')
    
    return HotelData(
        name=name, star_rating=star_rating, address=address, city=city, country=country, neighborhood=neighborhood,
        current_price=current_price, original_price=original_price, currency=currency, price_numeric=price_numeric,
        guest_rating=guest_rating, rating_badge=rating_badge, reviews_count=reviews_count,
        tripadvisor_rating=tripadvisor_rating, tripadvisor_reviews=tripadvisor_reviews, trustyou_reviews=trustyou_reviews,
        amenities=amenities, freebies=freebies, room_amenities=room_amenities, tagline=tagline, room_types=room_types,
        latitude=latitude, longitude=longitude, location_name=location_name, nearby_attractions=nearby_attractions,
        airports=airports, train_stations=train_stations, metro_stations=metro_stations, check_in_time=check_in_time,
        check_out_time=check_out_time, required_at_checkin=required_at_checkin, hotel_size=hotel_size, pets_policy=pets_policy,
        policies=policies, optional_extras=optional_extras, mandatory_fees=mandatory_fees, dining_info=dining_info,
        awards=awards, badge_type=badge_type, badge_tooltip=badge_tooltip
    )

# ========== TOOLS ==========
@tool(args_schema=HotelExtractInput)
def get_hotel_details_extracted(
    hotel_id: str,
    check_in: str,
    check_out: str,
    adults: int = 2,
    currency: str = "USD"
) -> HotelData:
    """Get complete hotel details and extract all essential data."""
    try:
        url = "https://hotels4.p.rapidapi.com/properties/get-details"
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": RAPIDAPI_HOST
        }
        params = {
            "id": hotel_id,
            "checkIn": check_in,
            "checkOut": check_out,
            "adults1": str(adults),
            "currency": currency,
            "locale": "en_US"
        }
        response = http_session.get(url, headers=headers, params=params, timeout=12)
        hotel_response = response.json()
        hotel_data = _extract_hotel_data_to_model(hotel_response)
        return hotel_data
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return HotelData(
            name="Error", address="N/A", city="N/A", country="N/A", neighborhood="N/A",
            current_price="N/A", original_price="N/A", currency="USD", guest_rating="N/A", rating_badge="N/A"
        )

@tool(args_schema=HotelSearchInput)
def search_and_extract_hotels(
    city_name: str,
    check_in: str,
    check_out: str,
    adults: int = 2,
    currency: str = "USD"
) -> List[HotelData]:
    """Search hotels by city and extract all essential data in parallel."""
    destination_id = _get_destination_id(city_name)
    if not destination_id:
        return []
    hotels = _search_hotels(destination_id, check_in, check_out, adults, currency)
    if not hotels:
        return []
    
    extracted_hotels = []
    
    # High-Performance Optimization: Fetch details for 5 hotels in parallel
    def fetch_single_hotel_details(hotel):
        hotel_id = hotel.get('id')
        if not hotel_id:
            return None
        try:
            hotel_data = get_hotel_details_extracted.invoke({
                'hotel_id': hotel_id,
                'check_in': check_in,
                'check_out': check_out,
                'adults': adults,
                'currency': currency
            })
            if hotel_data and hotel_data.name != "Error":
                return hotel_data
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=5) as executor:
        threads_results = executor.map(fetch_single_hotel_details, hotels[:5])
        
    for item in threads_results:
        if item:
            extracted_hotels.append(item)
            
    return extracted_hotels


hotel_parser = PydanticOutputParser(pydantic_object=HotelRecommendationList)

from langsmith import traceable

@tool(args_schema=HotelInput)
@traceable(name="get_hotel_recommendations")
def get_hotel_recommendations(
    city: str,
    budget_category: BudgetCategoryResponse,
    weather_data: List[dict]
):
    """
    Get hotel recommendations using RAG for intelligent semantic retrieval.
    
    Flow:
    1. Fetch real hotels from API
    2. Store in vector DB
    3. Use RAG to retrieve most relevant hotels
    4. LLM recommends based on retrieved data
    """
    rainy_count = sum(1 for d in weather_data if d.get('rain_probability', 0) > 0.4)
    total_days = len(weather_data) // 8 or 1
    weather_type = 'rainy' if rainy_count > total_days/2 else 'sunny'

    # Get dates from weather data
    if weather_data:
        dates = sorted(list(set([d['time'].split()[0] for d in weather_data if 'time' in d])))
        check_in = dates[0]
        check_out = dates[-1]
    else:
        check_in = datetime.now().strftime("%Y-%m-%d")
        check_out = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")

    # ========== STEP 1: Fetch real hotels from API ==========
    print(f"\n🌐 Fetching real hotels for {city}...")
    real_hotels_data = []
    try:
        real_hotels_data = search_and_extract_hotels.invoke({
            "city_name": city, 
            "check_in": check_in, 
            "check_out": check_out, 
            "adults": 2, 
            "currency": "INR"
        })
        if real_hotels_data:
            print(f"🟢 [DATA SOURCE] HOTELS: Successfully fetched {len(real_hotels_data)} real hotels from Booking API.")
        else:
            print(f"🟡 [DATA SOURCE] HOTELS: API returned 0 hotels. Using LLM Simulation fallback.")
    except Exception as e:
        print(f"⚠️ API call failed: {e}.")
        print(f"🟡 [DATA SOURCE] HOTELS: Falling back to LLM Simulation.")

    # ========== STEP 2: Create Documents and store in Vector DB ==========
    if real_hotels_data:
        try:
            print(f"\n📝 Creating RAG documents for {len(real_hotels_data)} hotels...")
            rag = get_hotel_rag()
            
            # Set city-specific DB path
            city_db_path = f"./chroma_hotels_{city.lower().replace(' ', '_')}_db"
            rag.persist_dir = city_db_path
            
            # Create documents
            documents = rag.create_documents(real_hotels_data, city)
            chunks = rag.chunk_documents(documents)
            rag.store_in_vector_db(chunks)
            
            # ========== STEP 3: Retrieve relevant hotels using RAG ==========
            print(f"\n🔍 Using RAG to find best hotels for budget ₹{budget_category.per_day_breakdown.hotel}...")
            
            search_query = f"""
            Hotels in {city} with:
            - Budget: ₹{budget_category.per_day_breakdown.hotel} per night
            - Star rating: {budget_category.hotel_type}
            - Weather: {weather_type}
            - Good guest ratings
            - Amenities suitable for {weather_type} weather
            """
            
            # ========== STEP 4: Format for LLM ==========
            rag_context = rag.retrieve_and_format_for_llm(search_query, k=3)
            
            # ========== STEP 5: LLM makes recommendations based on RAG data ==========
            prompt = PromptTemplate(
                template="""
You are a hotel expert in {city}.

REAL HOTEL DATA FROM RAG RETRIEVAL:
{rag_context}

USER REQUIREMENTS:
- Budget: ₹{hotel_budget}/night
- Hotel Type: {hotel_type}
- Weather: {weather_type}
- Rainy Days: {rainy_days}/{total_days}

Based on the REAL hotel data above, recommend 3 hotels that best match the user's requirements.

For EACH hotel, provide:
- name: Hotel name (MUST match the data)
- location: Location/neighborhood
- latitude: Latitude from data
- longitude: Longitude from data
- price: Price per night in rupees
- star_rating: Star rating
- room_services: List of room services
- dining_services: List of dining services
- wellness_services: List of wellness services
- internet_services: List of internet services
- family_services: List of family services
- transport_services: List of transport services
- accessibility_services: List of accessibility services
- other_services: List of other services
**CRITICAL JSON RULE**: Do NOT include any inline comments (like // or /* */) inside the JSON string response. The output must be strictly valid, clean JSON only.

{instructions}
""",
                input_variables=['city', 'hotel_budget', 'hotel_type', 'weather_type', 'rainy_days', 'total_days', 'rag_context'],
                partial_variables={'instructions': hotel_parser.get_format_instructions()}
            )
            
            try:
                chain = prompt | llm | hotel_parser
                from travel_planner.utils.callbacks import TokenUsageTracker
                result = chain.invoke({
                    'city': city,
                    'hotel_budget': budget_category.per_day_breakdown.hotel,
                    'hotel_type': budget_category.hotel_type,
                    'weather_type': weather_type,
                    'rainy_days': rainy_count,
                    'total_days': total_days,
                    'rag_context': rag_context
                }, config={"callbacks": [TokenUsageTracker("Hotels RAG Generation")]})
                return result.hotels
            except Exception as parse_error:
                print(f"⚠️ Primary hotel RAG chain failed/truncated: {parse_error}. Retrying with fallback Llama-3.3-70b...")
                chain = prompt | llm_fallback | hotel_parser
                from travel_planner.utils.callbacks import TokenUsageTracker
                result = chain.invoke({
                    'city': city,
                    'hotel_budget': budget_category.per_day_breakdown.hotel,
                    'hotel_type': budget_category.hotel_type,
                    'weather_type': weather_type,
                    'rainy_days': rainy_count,
                    'total_days': total_days,
                    'rag_context': rag_context
                }, config={"callbacks": [TokenUsageTracker("Hotels RAG Fallback Llama-70b")]})
                return result.hotels
        except Exception as e:
            print(f"⚠️ RAG processing failed: {e}. Falling back to standard API context formatting.")
            # Fallback to direct context mapping without vector storage
            real_hotels_str = ""
            for i, h in enumerate(real_hotels_data[:3]):
                real_hotels_str += f"""
Hotel {i+1}:
Name: {h.name}
Star Rating: {h.star_rating}
Address: {h.address}
Price: {h.current_price}
Neighborhood: {h.neighborhood}
Amenities: {", ".join(h.amenities[:8])}
Room Amenities: {", ".join(h.room_amenities[:8])}
Latitude: {h.latitude}
Longitude: {h.longitude}
"""
            prompt = PromptTemplate(
                template="""
You are a hotel expert in {city}.

We fetched these real hotels in {city} using our API:
{real_hotels_str}

WEATHER:
- Rainy Days: {rainy_days}/{total_days} days
- Weather Type: {weather_type}

Format these real hotels to match our required schema output.
For EACH hotel, map the details and also write a weather suitability recommendation:
- name: Match real hotel name
- location: Match neighborhood or address
- latitude: Match real latitude
- longitude: Match real longitude
- price: Parse the price numeric value in rupees (e.g., if price is ₹4,500, write 4500)
- star_rating: Star rating
- room_services: Suggest services based on amenities
- dining_services: Suggest dining services based on amenities
- wellness_services: Suggest wellness services
- internet_services: Suggest internet services
- family_services: Suggest family services
- transport_services: Suggest transport services
- accessibility_services: Suggest accessibility services
- other_services: Suggest other services

**IMPORTANT**: You must output a JSON object containing the actual list of recommended hotels under the key "hotels". DO NOT output the schema metadata, schema properties, type definitions, or schema fields themselves. Output only the concrete hotel details inside the JSON object.

**CRITICAL JSON RULE**: Do NOT include any inline comments (like // or /* */) inside the JSON string response. The output must be strictly valid, clean JSON only.

{instructions}
""",
                input_variables=['city', 'real_hotels_str', 'rainy_days', 'total_days', 'weather_type'],
                partial_variables={'instructions': hotel_parser.get_format_instructions()}
            )
            try:
                chain = prompt | llm | hotel_parser
                from travel_planner.utils.callbacks import TokenUsageTracker
                result = chain.invoke({
                    'city': city, 'real_hotels_str': real_hotels_str, 'rainy_days': rainy_count, 'total_days': total_days, 'weather_type': weather_type
                }, config={"callbacks": [TokenUsageTracker("Hotels Direct Mapping")]})
                return result.hotels
            except Exception as parse_error:
                print(f"⚠️ Primary direct mapping hotel chain failed: {parse_error}. Retrying with fallback Llama-3.3-70b...")
                chain = prompt | llm_fallback | hotel_parser
                from travel_planner.utils.callbacks import TokenUsageTracker
                result = chain.invoke({
                    'city': city, 'real_hotels_str': real_hotels_str, 'rainy_days': rainy_count, 'total_days': total_days, 'weather_type': weather_type
                }, config={"callbacks": [TokenUsageTracker("Hotels Direct Mapping Fallback Llama-70b")]})
                return result.hotels

    # ========== FALLBACK: No real data, use LLM simulation ==========
    prompt = PromptTemplate(
        template="""
You are a hotel expert in {city}.

BUDGET INFO:
- Category: {category}
- Hotel Budget: ₹{hotel_budget}/night
- Hotel Type: {hotel_type}

WEATHER:
- Rainy Days: {rainy_days}/{total_days} days
- Weather Type: {weather_type}

Suggest 3 hotels in {city} that:
1. Fit within ₹{hotel_budget}/night
2. Have good amenities for {weather_type} weather
3. Include all services

For EACH hotel, provide:
- name: Hotel name
- location: Area/locality name (e.g., "Sindhi Camp")
- latitude: Approximate latitude (e.g., 26.9124)
- longitude: Approximate longitude (e.g., 75.7873)
- price: Price per night in rupees
- star_rating: 1-5
- room_services: List of room services
- dining_services: List of dining services
- wellness_services: List of wellness services
- internet_services: List of internet services
- family_services: List of family services
- transport_services: List of transport services
- accessibility_services: List of accessibility services
- other_services: List of other services

**IMPORTANT**: You must output a JSON object containing the actual list of recommended hotels under the key "hotels". DO NOT output the schema metadata, schema properties, type definitions, or schema fields themselves. Output only the concrete hotel details inside the JSON object.

**CRITICAL JSON RULE**: Do NOT include any inline comments (like // or /* */) inside the JSON string response. The output must be strictly valid, clean JSON only.

{instructions}
""",
        input_variables=['city', 'category', 'hotel_budget', 'hotel_type', 'rainy_days', 'total_days', 'weather_type'],
        partial_variables={'instructions': hotel_parser.get_format_instructions()}
    )
    try:
        chain = prompt | llm | hotel_parser
        from travel_planner.utils.callbacks import TokenUsageTracker
        result = chain.invoke({
            'city': city, 'category': budget_category.category, 'hotel_budget': budget_category.per_day_breakdown.hotel,
            'hotel_type': budget_category.hotel_type, 'rainy_days': rainy_count, 'total_days': total_days, 'weather_type': weather_type
        }, config={"callbacks": [TokenUsageTracker("Hotels Simulation")]})
        return result.hotels
    except Exception as parse_error:
        print(f"⚠️ Primary simulation hotel chain failed: {parse_error}. Retrying with fallback Llama-3.3-70b...")
        chain = prompt | llm_fallback | hotel_parser
        from travel_planner.utils.callbacks import TokenUsageTracker
        result = chain.invoke({
            'city': city, 'category': budget_category.category, 'hotel_budget': budget_category.per_day_breakdown.hotel,
            'hotel_type': budget_category.hotel_type, 'rainy_days': rainy_count, 'total_days': total_days, 'weather_type': weather_type
        }, config={"callbacks": [TokenUsageTracker("Hotels Simulation Fallback Llama-70b")]})
        return result.hotels
