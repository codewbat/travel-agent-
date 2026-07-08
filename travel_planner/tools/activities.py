import re
import json
from datetime import datetime
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from travel_planner.config import llm, llm_fallback
from travel_planner.models import ActivityRecommendationList, BudgetCategoryResponse
from travel_planner.rag import get_activity_rag
from travel_planner.utils.callbacks import TokenUsageTracker
from langsmith import traceable


class ActivityInput(BaseModel):
    city: str = Field(description="City name")
    weather_data: list = Field(description="Weather records list")
    budget_category: BudgetCategoryResponse = Field(description="Budget category Response")
    start_date: str = Field(description="Start date of trip")
    end_date: str = Field(description="End date of trip")

activity_parser = PydanticOutputParser(pydantic_object=ActivityRecommendationList)

def clean_and_parse_json(raw_json: str, parser) -> Any:
    """Helper to strip comments and parse JSON manually with the parser."""
    # Strip inline comments (e.g. // Approximate latitude) but keep URLs safe
    cleaned = re.sub(r'(?<!https:)(?<!http:)//.*$', '', raw_json, flags=re.MULTILINE)
    # Strip block comments if any
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    return parser.parse(cleaned)

@tool(args_schema=ActivityInput)
@traceable(name="get_activity_recommendations")
def get_activity_recommendations(
    city: str,
    weather_data: list,
    budget_category: BudgetCategoryResponse,
    start_date: str,
    end_date: str
):
    """
    Get day-wise activity recommendations based on weather using RAG semantic search.
    """
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
    
    # ========== STEP 1: Fetch raw attractions from multiple API sources (Double-Fallback) ==========
    real_attractions_data = []
    
    # Source A: Google Places API
    try:
        from travel_planner.tools.google_places import search_places_text
        print(f"🗺️ Calling Google Places API to search tourist attractions in {city}...")
        real_attractions_data = search_places_text(f"tourist attractions in {city}", page_size=20)
    except Exception as e:
        print(f"⚠️ Google Places API call failed: {e}. Trying OpenStreetMap fallback.")

    # Source B: OSM Nominatim Fallback
    if not real_attractions_data:
        try:
            from travel_planner.tools.osm import search_nominatim
            print(f"🗺️ Calling OpenStreetMap Nominatim to search tourist attractions in {city}...")
            real_attractions_data = search_nominatim(f"tourist attractions in {city}", limit=20)
        except Exception as e:
            print(f"⚠️ OSM Nominatim call failed: {e}. Falling back to simulation.")

    # ========== STEP 2: Index Attractions in local RAG Database ==========
    if real_attractions_data:
        try:
            print(f"📝 Indexing {len(real_attractions_data)} attractions in Chroma Vector DB...")
            rag = get_activity_rag()
            
            # City-specific persistence path
            city_db_path = f"./chroma_activities_{city.lower().replace(' ', '_')}_db"
            rag.persist_dir = city_db_path
            
            # Create docs, chunk and save
            documents = rag.create_documents(real_attractions_data, city)
            chunks = rag.chunk_documents(documents)
            rag.store_in_vector_db(chunks)
            
            # ========== STEP 3: Semantic Retrieve for Itinerary Selection ==========
            search_query = f"""
            Top tourist attractions, landmarks, museums, and points of interest in {city} suitable for:
            - Budget: {budget_category.category}
            - Weather: {'indoor sights' if daily_weather[0]['is_rainy'] else 'outdoor sights and parks'}
            - Top rated sightseeing destinations
            """
            
            total_days_count = len(daily_weather)
            k_needed = max(5, total_days_count * 5 + 2)
            print(f"🔍 Performing semantic search for attractions query in local database (k={k_needed})...")
            rag_context = rag.retrieve_and_format_for_llm(search_query, k=k_needed)
            # Truncate context to avoid exceeding API payload limits
            rag_context = rag_context[:3500] if len(rag_context) > 3500 else rag_context
            prompt = PromptTemplate(
                template="""
You are a travel expert in {city}.

REAL ATTRACTIONS DATA FROM SEMANTIC SEARCH (RAG):
{rag_context}

WEATHER FORECAST:
{daily_weather}

BUDGET: {budget_category}

Create a day-wise activity plan for {total_days} days using the REAL tourist attractions retrieved above:
- Morning: 2 places
- Afternoon: 2 places
- Evening: 1 place
- Based on weather (indoor if rainy, outdoor if sunny)
- Include all details

**CRITICAL UNIQUENESS RULE**: Each activity in the entire itinerary MUST be completely unique. DO NOT repeat or revisit any attraction or place across any day or within the same day. Every single morning, afternoon, and evening activity across the entire {total_days}-day itinerary must specify a distinct, unique place/attraction.

For EACH activity/place:
- name: Match one of the real attraction names from the database
- location: Address matching the data (Keep it concise, e.g. a neighborhood or area in {city}. Do NOT write full street addresses)
- latitude: Match real latitude from the list (Do NOT write 0.0, use the exact real latitude float)
- longitude: Match real longitude from the list (Do NOT write 0.0, use the exact real longitude float)
- entry_fee: Estimate a realistic entry fee in rupees (INR) for this monument/sight (e.g., 50 or 80, or 0 if it is a free park, public space, or temple)

**CRITICAL BREVITY RULE**: To prevent output truncation, you MUST keep the `description` and `why_recommend` fields extremely brief (less than 10 words each). Do not write long paragraphs.

**CRITICAL JSON RULE**: Do NOT include any inline comments (like // or /* */) inside the JSON string response. The output must be strictly valid, clean JSON only.

{format_instructions}
""",
                input_variables=['city', 'daily_weather', 'budget_category', 'total_days', 'rag_context'],
                partial_variables={
                    'format_instructions': activity_parser.get_format_instructions()
                }
            )
            
            try:
                chain = prompt | llm
                raw_response = chain.invoke({
                    'city': city,
                    'daily_weather': daily_weather_str,
                    'budget_category': budget_category.category,
                    'total_days': len(daily_weather),
                    'rag_context': rag_context
                }, config={"callbacks": [TokenUsageTracker("Activities RAG Generation")]})
                result = clean_and_parse_json(raw_response.content, activity_parser)
                return result.activities
            except Exception as parse_error:
                print(f"⚠️ Primary model parsing failed: {parse_error}. Retrying with fallback Llama-3.3-70b-versatile...")
                chain = prompt | llm_fallback
                raw_response = chain.invoke({
                    'city': city,
                    'daily_weather': daily_weather_str,
                    'budget_category': budget_category.category,
                    'total_days': len(daily_weather),
                    'rag_context': rag_context
                }, config={"callbacks": [TokenUsageTracker("Activities RAG Fallback Llama-70b")]})
                result = clean_and_parse_json(raw_response.content, activity_parser)
                return result.activities
            
        except Exception as e:
            print(f"⚠️ RAG index/retrieval failed: {e}. Falling back to standard context format.")
            # Fallback to direct mapping without vector queries
            total_days_count = len(daily_weather)
            k_needed = max(5, total_days_count * 5 + 2)
            real_attractions_str = ""
            seen_fallback_names = set()
            count = 1
            for attr in real_attractions_data:
                disp_name = attr.get('name') or attr.get('displayName', {}).get('text') or 'Unknown'
                if disp_name in seen_fallback_names:
                    continue
                seen_fallback_names.add(disp_name)
                address = attr.get('formattedAddress') or attr.get('address') or 'N/A'
                real_attractions_str += f"\nAttraction {count}: Name: {disp_name} | Address: {address} | Lat: {attr.get('latitude', 0.0)} | Lng: {attr.get('longitude', 0.0)}"
                count += 1
                if count > k_needed:
                    break
            
            prompt = PromptTemplate(
                template="""
You are a travel expert in {city}.

We fetched these real tourist attractions:
{real_attractions_str}

WEATHER FORECAST:
{daily_weather}

BUDGET: {budget_category}

Create a day-wise activity plan for {total_days} days using these attractions:
- Morning: 2 places
- Afternoon: 2 places
- Evening: 1 place
- Based on weather (indoor if rainy, outdoor if sunny)

**CRITICAL UNIQUENESS RULE**: Each activity in the entire itinerary MUST be completely unique. DO NOT repeat or revisit any attraction or place across any day or within the same day. Every single morning, afternoon, and evening activity across the entire {total_days}-day itinerary must specify a distinct, unique place/attraction.

For EACH activity/place:
- name: Match real attraction name
- location: General area/neighborhood matching the address
- latitude: Match real latitude
- longitude: Match real longitude

**CRITICAL BREVITY RULE**: To prevent output truncation, you MUST keep all dish `description` and `why_special` fields extremely brief (less than 10 words each).

**CRITICAL JSON RULE**: Do NOT include any inline comments (like // or /* */) inside the JSON string response. The output must be strictly valid, clean JSON only.

{format_instructions}
""",
                input_variables=['city', 'daily_weather', 'budget_category', 'total_days', 'real_attractions_str'],
                partial_variables={
                    'format_instructions': activity_parser.get_format_instructions()
                }
            )
            try:
                chain = prompt | llm
                raw_response = chain.invoke({
                    'city': city,
                    'daily_weather': daily_weather_str,
                    'budget_category': budget_category.category,
                    'total_days': len(daily_weather),
                    'real_attractions_str': real_attractions_str
                }, config={"callbacks": [TokenUsageTracker("Activities Direct Mapping")]})
                result = clean_and_parse_json(raw_response.content, activity_parser)
                return result.activities
            except Exception as parse_error:
                print(f"⚠️ Fallback to direct mapping failed on primary model. Retrying direct mapping on Llama-3.3-70b...")
                chain = prompt | llm_fallback
                raw_response = chain.invoke({
                    'city': city,
                    'daily_weather': daily_weather_str,
                    'budget_category': budget_category.category,
                    'total_days': len(daily_weather),
                    'real_attractions_str': real_attractions_str
                }, config={"callbacks": [TokenUsageTracker("Activities Direct Mapping Fallback Llama-70b")]})
                result = clean_and_parse_json(raw_response.content, activity_parser)
                return result.activities

    # ========== STEP 5: Fallback to simulated places (No API data) ==========
    print("🟡 [DATA SOURCE] ACTIVITIES: No API data available. Falling back to LLM Simulated attractions.")
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

**CRITICAL UNIQUENESS RULE**: Each activity in the entire itinerary MUST be completely unique. DO NOT repeat or revisit any attraction or place across any day or within the same day. Every single morning, afternoon, and evening activity across the entire {total_days}-day itinerary must specify a distinct, unique place/attraction.

For EACH activity/place, provide approximate coordinates:
- latitude: Approximate latitude (e.g., 26.9124)
- longitude: Approximate longitude (e.g., 75.7873)

**CRITICAL BREVITY RULE**: To prevent output truncation, you MUST keep the `description` and `why_recommend` fields extremely brief (less than 10 words each). Do not write long paragraphs.

**CRITICAL JSON RULE**: Do NOT include any inline comments (like // or /* */) inside the JSON string response. The output must be strictly valid, clean JSON only.

{format_instructions}
""",
        input_variables=['city', 'daily_weather', 'budget_category', 'total_days'],
        partial_variables={
            'format_instructions': activity_parser.get_format_instructions()
        }
    )
    
    try:
        chain = prompt | llm
        raw_response = chain.invoke({
            'city': city,
            'daily_weather': daily_weather_str,
            'budget_category': budget_category.category,
            'total_days': len(daily_weather)
        }, config={"callbacks": [TokenUsageTracker("Activities Simulation")]})
        result = clean_and_parse_json(raw_response.content, activity_parser)
        return result.activities
    except Exception as parse_error:
        print(f"⚠️ Simulated fallback failed on primary model. Retrying on Llama-3.3-70b...")
        chain = prompt | llm_fallback
        raw_response = chain.invoke({
            'city': city,
            'daily_weather': daily_weather_str,
            'budget_category': budget_category.category,
            'total_days': len(daily_weather)
        }, config={"callbacks": [TokenUsageTracker("Activities Simulation Fallback Llama-70b")]})
        result = clean_and_parse_json(raw_response.content, activity_parser)
        return result.activities
