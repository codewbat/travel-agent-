from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from travel_planner.config import llm, llm_fallback
from travel_planner.models import FoodRecommendationList, BudgetCategoryResponse
from travel_planner.rag import get_food_rag

class FoodInput(BaseModel):
    city: str = Field(description="City name")
    budget_category: BudgetCategoryResponse = Field(description="Budget category Response")
    total_days: int = Field(description="Total days")

food_parser = PydanticOutputParser(pydantic_object=FoodRecommendationList)

from langsmith import traceable

@tool(args_schema=FoodInput)
@traceable(name="get_food_recommendations")
def get_food_recommendations(
    city: str,
    budget_category: BudgetCategoryResponse,
    total_days: int
):
    """
    Get day-wise food and restaurant recommendations based on budget using RAG semantic search.
    """
    # ========== STEP 1: Fetch raw restaurants from API sources (Double-Fallback) ==========
    real_food_data = []
    
    # Source A: Google Places API
    try:
        from travel_planner.tools.google_places import search_places_text
        print(f"🗺️ Calling Google Places API to search restaurants in {city}...")
        real_food_data = search_places_text(f"restaurants in {city}", page_size=20)
        if real_food_data:
            print(f"🟢 [DATA SOURCE] FOOD: Successfully fetched {len(real_food_data)} restaurants from Google Places API.")
    except Exception as e:
        print(f"⚠️ Google Places API call failed: {e}. Trying OpenStreetMap fallback.")

    # Source B: OSM Nominatim Fallback
    if not real_food_data:
        try:
            from travel_planner.tools.osm import search_nominatim
            print(f"🗺️ Calling OpenStreetMap Nominatim to search restaurants in {city}...")
            real_food_data = search_nominatim(f"restaurants in {city}", limit=20)
            if real_food_data:
                print(f"🟢 [DATA SOURCE] FOOD: Successfully fetched {len(real_food_data)} restaurants from OpenStreetMap Nominatim.")
        except Exception as e:
            print(f"⚠️ OSM Nominatim call failed: {e}. Falling back to simulation.")

    # ========== STEP 2: Index Restaurants in local RAG Database ==========
    if real_food_data:
        try:
            print(f"📝 Indexing {len(real_food_data)} restaurants in Chroma Vector DB...")
            rag = get_food_rag()
            
            # City-specific persistence path
            city_db_path = f"./chroma_food_{city.lower().replace(' ', '_')}_db"
            rag.persist_dir = city_db_path
            
            # Create docs, chunk and save
            documents = rag.create_documents(real_food_data, city)
            chunks = rag.chunk_documents(documents)
            rag.store_in_vector_db(chunks)
            
            # ========== STEP 3: Semantic Retrieve for Dining Selection ==========
            search_query = f"""
            Top rated restaurants, cafes, food joints and dining options in {city} suitable for:
            - Budget Level: {budget_category.category}
            - Daily food budget target: ₹{budget_category.per_day_breakdown.food}
            - Family friendly, good local and international cuisines
            """
            
            k_needed = max(3, total_days * 3 + 2)
            print(f"🔍 Performing semantic search for restaurants in local database (k={k_needed})...")
            rag_context = rag.retrieve_and_format_for_llm(search_query, k=k_needed)
            # Truncate context to avoid exceeding API payload limits
            rag_context = rag_context[:3500] if len(rag_context) > 3500 else rag_context
            prompt = PromptTemplate(
                template="""
You are a food expert in {city}.

REAL RESTAURANTS DATA FROM SEMANTIC SEARCH (RAG):
{rag_context}

BUDGET:
- Category: {category}
- Food Budget per day: ₹{food_budget}

Create food recommendations for {total_days} days using the REAL restaurants retrieved above:
- For each day: Breakfast, Lunch, Dinner
- Each meal: place_name (MUST match one of the restaurant names from the database), location (Keep it extremely short, e.g. a neighborhood or area in {city}. Do NOT write full street addresses), latitude, longitude, place_type
- Each place: exactly 1 dish with name, price, description (less than 5 words), why_special (less than 5 words)
- **CRITICAL**: For EACH meal, include weather_suitability field (e.g. "Indoor" or "Outdoor terrace" - less than 5 words)

**CRITICAL UNIQUENESS RULE**: Each food recommendation/dining place in the entire itinerary MUST be completely unique. DO NOT repeat or revisit any restaurant, cafe, street food stall, or dining place across any day or within the same day (e.g., Breakfast, Lunch, and Dinner must all be at different places, and you must not reuse a place on another day). Every single meal across the entire {total_days}-day itinerary must specify a distinct, unique dining place.

**CRITICAL BREVITY RULE**: To prevent output truncation, you MUST keep all dish `description`, `why_special`, `location`, and `weather_suitability` fields extremely brief (less than 5 words each). Do not write long paragraphs.

**CRITICAL JSON RULE**: Do NOT include any inline comments (like // or /* */) inside the JSON string response. The output must be strictly valid, clean JSON only.

{instructions}
""",
                input_variables=['city', 'category', 'food_budget', 'total_days', 'rag_context'],
                partial_variables={
                    'instructions': food_parser.get_format_instructions()
                }
            )
            
            try:
                chain = prompt | llm | food_parser
                from travel_planner.utils.callbacks import TokenUsageTracker
                result = chain.invoke({
                    'city': city,
                    'category': budget_category.category,
                    'food_budget': budget_category.per_day_breakdown.food,
                    'total_days': total_days,
                    'rag_context': rag_context
                }, config={"callbacks": [TokenUsageTracker("Food RAG Generation")]})
                return result.foods
            except Exception as parse_error:
                print(f"⚠️ Primary model food parsing failed: {parse_error}. Retrying with fallback Llama-3.3-70b...")
                chain = prompt | llm_fallback | food_parser
                from travel_planner.utils.callbacks import TokenUsageTracker
                result = chain.invoke({
                    'city': city,
                    'category': budget_category.category,
                    'food_budget': budget_category.per_day_breakdown.food,
                    'total_days': total_days,
                    'rag_context': rag_context
                }, config={"callbacks": [TokenUsageTracker("Food RAG Fallback Llama-70b")]})
                return result.foods
            
        except Exception as e:
            print(f"⚠️ RAG index/retrieval failed: {e}. Falling back to standard context format.")
            # Fallback to direct mapping without vector queries
            k_needed = max(3, total_days * 3 + 2)
            real_food_str = ""
            seen_fallback_names = set()
            count = 1
            for rest in real_food_data:
                disp_name = rest.get('name') or rest.get('displayName', {}).get('text') or 'Unknown'
                if disp_name in seen_fallback_names:
                    continue
                seen_fallback_names.add(disp_name)
                address = rest.get('formattedAddress') or rest.get('address') or 'N/A'
                real_food_str += f"\nRestaurant {count}: Name: {disp_name} | Address: {address} | Lat: {rest.get('latitude', 0.0)} | Lng: {rest.get('longitude', 0.0)}"
                count += 1
                if count > k_needed:
                    break
            
            prompt = PromptTemplate(
                template="""
You are a food expert in {city}.

We fetched these real restaurants:
{real_food_str}

BUDGET:
- Category: {category}
- Food Budget per day: ₹{food_budget}

Create food recommendations for {total_days} days using these restaurants:
- For each day: Breakfast, Lunch, Dinner
- Each meal: place_name, location (Keep it extremely short, e.g. a neighborhood or area in {city}. Do NOT write full street addresses), latitude, longitude, place_type
- Each place: exactly 1 dish with name, price, description (less than 5 words), why_special (less than 5 words)
- **CRITICAL**: For EACH meal, include weather_suitability field (less than 5 words)

**CRITICAL UNIQUENESS RULE**: Each food recommendation/dining place in the entire itinerary MUST be completely unique. DO NOT repeat or revisit any restaurant, cafe, street food stall, or dining place across any day or within the same day (e.g., Breakfast, Lunch, and Dinner must all be at different places, and you must not reuse a place on another day). Every single meal across the entire {total_days}-day itinerary must specify a distinct, unique dining place.

**CRITICAL BREVITY RULE**: To prevent output truncation, you MUST keep all dish `description`, `why_special`, `location`, and `weather_suitability` fields extremely brief (less than 5 words each). Do not write long paragraphs.

**CRITICAL JSON RULE**: Do NOT include any inline comments (like // or /* */) inside the JSON string response. The output must be strictly valid, clean JSON only.

{instructions}
""",
                input_variables=['city', 'category', 'food_budget', 'total_days', 'real_food_str'],
                partial_variables={
                    'instructions': food_parser.get_format_instructions()
                }
            )
            try:
                chain = prompt | llm | food_parser
                from travel_planner.utils.callbacks import TokenUsageTracker
                result = chain.invoke({
                    'city': city,
                    'category': budget_category.category,
                    'food_budget': budget_category.per_day_breakdown.food,
                    'total_days': total_days,
                    'real_food_str': real_food_str
                }, config={"callbacks": [TokenUsageTracker("Food Direct Mapping")]})
                return result.foods
            except Exception as parse_error:
                print(f"⚠️ Food fallback direct mapping failed. Retrying on Llama-3.3-70b...")
                chain = prompt | llm_fallback | food_parser
                from travel_planner.utils.callbacks import TokenUsageTracker
                result = chain.invoke({
                    'city': city,
                    'category': budget_category.category,
                    'food_budget': budget_category.per_day_breakdown.food,
                    'total_days': total_days,
                    'real_food_str': real_food_str
                }, config={"callbacks": [TokenUsageTracker("Food Direct Mapping Fallback Llama-70b")]})
                return result.foods

    # ========== STEP 5: Fallback to simulated foods (No API data) ==========
    print("🟡 [DATA SOURCE] FOOD: No API data available. Falling back to LLM Simulated dining spots.")
    prompt = PromptTemplate(
        template="""
You are a food expert in {city}.

BUDGET:
- Category: {category}
- Food Budget per day: ₹{food_budget}

Create food recommendations for {total_days} days:
- For each day: Breakfast, Lunch, Dinner
- Each meal: place_name, location (Keep it extremely short, e.g. a neighborhood or area in {city}. Do NOT write full street addresses), latitude, longitude, place_type
- Each place: exactly 1 dish with name, price, description (less than 5 words), why_special (less than 5 words)
- **CRITICAL**: For EACH meal, include weather_suitability field (less than 5 words)

**CRITICAL UNIQUENESS RULE**: Each food recommendation/dining place in the entire itinerary MUST be completely unique. DO NOT repeat or revisit any restaurant, cafe, street food stall, or dining place across any day or within the same day (e.g., Breakfast, Lunch, and Dinner must all be at different places, and you must not reuse a place on another day). Every single meal across the entire {total_days}-day itinerary must specify a distinct, unique dining place.

**CRITICAL BREVITY RULE**: To prevent output truncation, you MUST keep all dish `description`, `why_special`, `location`, and `weather_suitability` fields extremely brief (less than 5 words each). Do not write long paragraphs.

**CRITICAL JSON RULE**: Do NOT include any inline comments (like // or /* */) inside the JSON string response. The output must be strictly valid, clean JSON only.

{instructions}
""",
        input_variables=['city', 'category', 'food_budget', 'total_days'],
        partial_variables={
            'instructions': food_parser.get_format_instructions()
        }
    )
    
    try:
        chain = prompt | llm | food_parser
        from travel_planner.utils.callbacks import TokenUsageTracker
        result = chain.invoke({
            'city': city,
            'category': budget_category.category,
            'food_budget': budget_category.per_day_breakdown.food,
            'total_days': total_days
        }, config={"callbacks": [TokenUsageTracker("Food Simulation")]})
        return result.foods
    except Exception as parse_error:
        print(f"⚠️ Food simulation failed on primary model. Retrying on Llama-3.3-70b...")
        chain = prompt | llm_fallback | food_parser
        from travel_planner.utils.callbacks import TokenUsageTracker
        result = chain.invoke({
            'city': city,
            'category': budget_category.category,
            'food_budget': budget_category.per_day_breakdown.food,
            'total_days': total_days
        }, config={"callbacks": [TokenUsageTracker("Food Simulation Fallback Llama-70b")]})
        return result.foods
