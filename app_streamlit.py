import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime, date
from collections import defaultdict
from travel_planner.config import llm
from travel_planner.tools import (
    get_weather_report_of_city,
    get_categorize_budget,
    get_hotel_recommendations,
    get_activity_recommendations,
    get_food_recommendations,
    get_packing_tips_recommendation,
    create_travel_plan
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

import re
import json

# Pydantic model for parameter extraction
class ExtractedTripParams(BaseModel):
    city: str = Field(description="Name of the destination city")
    start_date: str = Field(description="Start date of the trip in YYYY-MM-DD format. Assume the current year is 2026 if not specified.")
    end_date: str = Field(description="End date of the trip in YYYY-MM-DD format. Calculate this based on start date and duration.")
    total_budget: int = Field(default=50000, description="Total budget for the trip in rupees. If the user doesn't specify any budget, assume a default of 50000.")

def _extract_json_from_text(text: str) -> dict:
    """Try to extract a JSON object from mixed LLM output."""
    # Try to find JSON block in markdown code fences
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try to find a raw JSON object
    match = re.search(r'\{[^{}]*"city"[^{}]*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON object found in LLM response")

def extract_trip_params(user_query: str) -> ExtractedTripParams:
    parser = PydanticOutputParser(pydantic_object=ExtractedTripParams)
    prompt = PromptTemplate(
        template="""You are a JSON extraction bot. Extract trip parameters from the user query below.
Current date: Wednesday, July 8, 2026. Use year 2026 for all dates.
If no budget is mentioned, use 50000. Convert "lakhs" to actual number (1 lakh = 100000).

User Query: {query}

RESPOND WITH ONLY A SINGLE JSON OBJECT. NO explanation, NO code, NO markdown.
Output exactly this format:
{{"city": "...", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "total_budget": 00000}}""",
        input_variables=["query"]
    )
    chain = prompt | llm
    raw_response = chain.invoke({"query": user_query})
    raw_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)

    # Try standard parser first
    try:
        return parser.parse(raw_text)
    except Exception:
        pass

    # Fallback: extract JSON from mixed text
    try:
        data = _extract_json_from_text(raw_text)
        return ExtractedTripParams(**data)
    except Exception:
        pass

    # Last resort: regex extraction of individual fields
    city_match = re.search(r'"city"\s*:\s*"([^"]+)"', raw_text)
    start_match = re.search(r'"start_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"', raw_text)
    end_match = re.search(r'"end_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"', raw_text)
    budget_match = re.search(r'"total_budget"\s*:\s*(\d+)', raw_text)

    if city_match and start_match and end_match:
        return ExtractedTripParams(
            city=city_match.group(1),
            start_date=start_match.group(1),
            end_date=end_match.group(1),
            total_budget=int(budget_match.group(1)) if budget_match else 50000
        )

    raise ValueError(f"Could not parse trip parameters from LLM response: {raw_text[:200]}")

# Page Configuration
st.set_page_config(
    page_title="🌍 AI Travel Agent Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2b6cb0;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1a365d;
        color: #ffffff;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #2b6cb0;
    }
    .hotel-card {
        background-color: white;
        color: #2d3748;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-top: 4px solid #319795;
    }
    .hotel-card h3 {
        color: #1a202c !important;
        margin-top: 0px;
        margin-bottom: 10px;
    }
    .hotel-card p, .hotel-card span, .hotel-card b, .hotel-card i {
        color: #4a5568 !important;
    }
    .food-card {
        background-color: white;
        color: #2d3748;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 4px solid #dd6b20;
    }
    .food-card h4 {
        color: #1a202c !important;
        margin-top: 0px;
        margin-bottom: 8px;
    }
    .food-card p, .food-card span, .food-card b, .food-card i {
        color: #4a5568 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Agentic AI Travel Planner")
st.write("Plan your dream vacation dynamically with Natural Language input, Weather Integration, and Smart Budget Allocation.")

# Sidebar Settings
st.sidebar.header("✈️ Information")
st.sidebar.write("Simply describe your trip in plain English (e.g. city, dates, budget) and our agent will automatically understand it and plan it for you.")

# User prompt query input
user_query = st.text_area(
    "Describe your trip query here:",
    value="give me a trip planner for delhi of 5 days from 15 july 2026 my budget is 50000 ruppes",
    height=100
)

plan_button = st.button("Generate Travel Plan 🚀")

if plan_button:
    if not user_query.strip():
        st.error("Please enter a trip query first!")
    else:
        # Collapsible real-time logs console
        with st.expander("⚙️ Agent Execution Logs (Real-time)", expanded=True):
            log_placeholder = st.empty()
            log_lines = []
            
            def log(msg: str):
                log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                log_placeholder.code("\n".join(log_lines))

        with st.spinner("🕵️‍♂️ Agent is planning your trip..."):
            try:
                # Step 0: Extract params
                log("🔍 Analyzing natural language query...")
                params = extract_trip_params(user_query)
                city = params.city
                start_str = params.start_date
                end_str = params.end_date
                total_budget = params.total_budget
                
                # Parse days
                dt_start = datetime.strptime(start_str, "%Y-%m-%d")
                dt_end = datetime.strptime(end_str, "%Y-%m-%d")
                total_days = (dt_end - dt_start).days + 1
                
                log(f"✅ Extracted Parameters:\n   - City: {city.capitalize()}\n   - Dates: {start_str} to {end_str} ({total_days} days)\n   - Budget: ₹{total_budget:,}")
                
                # Step 1: Weather
                log("🌤️ Calling get_weather_report_of_city...")
                weather_data = get_weather_report_of_city.invoke({
                    'city_name': city.lower(),
                    'startdate': start_str,
                    'enddate': end_str
                })
                log(f"   ✅ Weather fetched: {len(weather_data)} records")
                
                # Step 2: Budget split
                log("💰 Calling get_categorize_budget...")
                budget = get_categorize_budget.invoke({
                    'total_budget': total_budget,
                    'total_days': total_days,
                    'city': city
                })
                log(f"   ✅ Budget split determined:\n      Hotel: ₹{budget.per_day_breakdown.hotel}/day\n      Food: ₹{budget.per_day_breakdown.food}/day")
                
                # Step 3: Hotels
                log("🏨 Calling get_hotel_recommendations...")
                hotels = get_hotel_recommendations.invoke({
                    'city': city,
                    'budget_category': budget,
                    'weather_data': weather_data
                })
                log(f"   ✅ Found {len(hotels)} suitable hotels matching budget.")
                
                # Step 4: Activities
                log("🗺️ Calling get_activity_recommendations...")
                activities = get_activity_recommendations.invoke({
                    'city': city,
                    'weather_data': weather_data,
                    'budget_category': budget,
                    'start_date': start_str,
                    'end_date': end_str
                })
                log(f"   ✅ Day-wise activities generated successfully for {len(activities)} days.")
                
                # Step 5: Food
                log("🍽️ Calling get_food_recommendations...")
                food = get_food_recommendations.invoke({
                    'city': city,
                    'budget_category': budget,
                    'total_days': total_days
                })
                log(f"   ✅ Selected {len(food)} restaurants/local dining spots.")
                
                # Step 6: Packing & Tips
                log("🧳 Calling get_packing_tips_recommendation...")
                packing_tips = get_packing_tips_recommendation.invoke({
                    'city': city,
                    'weather_data': weather_data,
                    'budget_category': budget,
                    'total_days': total_days
                })
                log("   ✅ Packing list & travel tips generated.")
                
                # Step 7: Combine
                log("📦 Calling create_travel_plan to compile final response...")
                travel_plan = create_travel_plan.invoke({
                    'city': city,
                    'start_date': start_str,
                    'end_date': end_str,
                    'budget_category': budget,
                    'hotels': hotels,
                    'activities': activities,
                    'food': food,
                    'packing_tips': packing_tips
                })
                log("🎉 Final compilation complete! Rendering travel plan...")
                
                st.success("🎉 Travel plan generated successfully!")
                
                # Metrics Row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Destination", travel_plan.city.capitalize())
                with col2:
                    st.metric("Duration", f"{travel_plan.total_days} Days")
                with col3:
                    st.metric("Category", travel_plan.budget_category)
                with col4:
                    st.metric("Verdict", travel_plan.recommendation)
                
                # Recommendation Callout
                if travel_plan.recommendation == "GO":
                    st.info(f"🟢 **Recommendation Reason**: {travel_plan.recommendation_reason}")
                else:
                    st.warning(f"🔴 **Recommendation Reason**: {travel_plan.recommendation_reason}")
                    if travel_plan.alternative:
                        st.write(f"💡 **Alternative Suggested**: {travel_plan.alternative}")
                
                # Create Tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🏨 Hotels", "🗺️ Daily Itinerary", "🍽️ Culinary Recommendations", "💰 Budget Allocation", "🧳 Packing & Tips"
                ])
                
                # Tab 1: Hotels
                with tab1:
                    st.subheader("Recommended Accommodations")
                    map_data = []
                    for hotel in travel_plan.hotels:
                        st.markdown(f"""
                        <div style="background-color: #ffffff !important; color: #1a202c !important; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); margin-bottom: 20px; border-top: 4px solid #319795;">
                            <h3 style="color: #2b6cb0 !important; margin-top: 0px; margin-bottom: 10px; font-size: 1.3rem;">🏨 {hotel.name}</h3>
                            <p style="color: #2d3748 !important; margin: 4px 0px; font-size: 0.95rem;">📍 <b style="color: #1a202c !important;">Location:</b> {hotel.location} (Lat: {hotel.latitude}, Lon: {hotel.longitude})</p>
                            <p style="color: #2d3748 !important; margin: 4px 0px; font-size: 0.95rem;">💰 <b style="color: #1a202c !important;">Price:</b> ₹{hotel.price}/night | ⭐ <b style="color: #1a202c !important;">Star Rating:</b> {hotel.star_rating}/5</p>
                            <p style="color: #2d3748 !important; margin: 4px 0px; font-size: 0.95rem;">🌤️ <b style="color: #1a202c !important;">Weather Suitability:</b> {hotel.weather_recommendation}</p>
                            <p style="color: #2d3748 !important; margin: 4px 0px; font-size: 0.95rem;">ℹ️ <b style="color: #1a202c !important;">Amenities:</b> Room Services: {", ".join(hotel.room_services[:4])} | Dining: {", ".join(hotel.dining_services[:4])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        map_data.append({"name": hotel.name, "latitude": hotel.latitude, "longitude": hotel.longitude, "type": "Hotel"})
                    
                    # Hotel Map with names
                    if map_data:
                        df_map = pd.DataFrame(map_data)
                        st.subheader("Hotel Location Map")
                        st.pydeck_chart(pdk.Deck(
                            initial_view_state=pdk.ViewState(
                                latitude=df_map['latitude'].mean(),
                                longitude=df_map['longitude'].mean(),
                                zoom=12, pitch=0
                            ),
                            layers=[
                                pdk.Layer('ScatterplotLayer', data=df_map, get_position='[longitude, latitude]',
                                    get_radius=120, get_fill_color=[49, 151, 149, 200], pickable=True),
                                pdk.Layer('TextLayer', data=df_map, get_position='[longitude, latitude]',
                                    get_text='name', get_size=14, get_color=[255, 255, 255, 255],
                                    get_alignment_baseline="'bottom'", get_pixel_offset='[0, -15]')
                            ], tooltip={"text": "{name}"}
                        ))
                
                # Tab 2: Itinerary
                with tab2:
                    st.subheader("Day-by-Day Activity Flow")
                    all_activity_map_data = []
                    
                    # Create swipeable day tabs
                    day_tab_labels = [f"📅 Day {idx+1}" for idx in range(len(travel_plan.daily_plan))]
                    if day_tab_labels:
                        day_tabs = st.tabs(day_tab_labels)
                        for idx, (day_tab, day) in enumerate(zip(day_tabs, travel_plan.daily_plan)):
                            with day_tab:
                                st.markdown(f"### 📅 Day {idx+1}: {day.date} ({day.day})")
                                day_map_data = []
                                
                                col_m, col_a, col_e = st.columns(3)
                                with col_m:
                                    st.markdown("🌅 **Morning Activities**")
                                    for act in day.morning:
                                        st.write(f"**{act.name}** ({act.location})")
                                        st.write(f"🕒 {act.timing} | 🎟️ Fee: ₹{act.entry_fee}")
                                        st.caption(act.description)
                                        day_map_data.append({"name": act.name, "latitude": act.latitude, "longitude": act.longitude})
                                
                                with col_a:
                                    st.markdown("☀️ **Afternoon Activities**")
                                    for act in day.afternoon:
                                        st.write(f"**{act.name}** ({act.location})")
                                        st.write(f"🕒 {act.timing} | 🎟️ Fee: ₹{act.entry_fee}")
                                        st.caption(act.description)
                                        day_map_data.append({"name": act.name, "latitude": act.latitude, "longitude": act.longitude})
                                
                                with col_e:
                                    st.markdown("🌆 **Evening Activities**")
                                    for act in day.evening:
                                        st.write(f"**{act.name}** ({act.location})")
                                        st.write(f"🕒 {act.timing} | 🎟️ Fee: ₹{act.entry_fee}")
                                        st.caption(act.description)
                                        day_map_data.append({"name": act.name, "latitude": act.latitude, "longitude": act.longitude})
                                
                                # Per-day map with names
                                if day_map_data:
                                    df_day = pd.DataFrame(day_map_data)
                                    st.markdown(f"#### 🗺️ Day {idx+1} Spots")
                                    st.pydeck_chart(pdk.Deck(
                                        initial_view_state=pdk.ViewState(
                                            latitude=df_day['latitude'].mean(),
                                            longitude=df_day['longitude'].mean(),
                                            zoom=13, pitch=0
                                        ),
                                        layers=[
                                            pdk.Layer('ScatterplotLayer', data=df_day, get_position='[longitude, latitude]',
                                                get_radius=100, get_fill_color=[66, 135, 245, 200], pickable=True),
                                            pdk.Layer('TextLayer', data=df_day, get_position='[longitude, latitude]',
                                                get_text='name', get_size=14, get_color=[255, 255, 255, 255],
                                                get_alignment_baseline="'bottom'", get_pixel_offset='[0, -15]')
                                        ], tooltip={"text": "{name}"}
                                    ))
                                all_activity_map_data.extend(day_map_data)
                
                # Tab 3: Culinary
                with tab3:
                    st.subheader("Localized Dining Recommendations")
                    all_food_map_data = []
                    
                    # Group food by day
                    food_by_day = defaultdict(list)
                    for meal in travel_plan.food_recommendations:
                        food_by_day[meal.day].append(meal)
                    
                    sorted_food_days = sorted(food_by_day.keys())
                    
                    # Swipeable day tabs for food
                    if sorted_food_days:
                        food_day_tabs = st.tabs([f"📅 Day {d}" for d in sorted_food_days])
                        for food_tab, day_num in zip(food_day_tabs, sorted_food_days):
                            with food_tab:
                                st.markdown(f"### 🍽️ Day {day_num} Dining")
                                day_food_map = []
                                for meal in food_by_day[day_num]:
                                    meal_emoji = {"Breakfast": "🍳", "Lunch": "🍛", "Dinner": "🍽️", "Snack": "☕"}.get(meal.meal, "🍴")
                                    st.markdown(f"""
                                    <div style="background-color: #ffffff !important; color: #1a202c !important; padding: 14px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.08); margin-bottom: 12px; border-left: 4px solid #dd6b20;">
                                        <h4 style="color: #dd6b20 !important; margin-top: 0px; margin-bottom: 6px; font-size: 1rem;">{meal_emoji} {meal.meal}: {meal.place_name}</h4>
                                        <p style="color: #718096 !important; margin: 2px 0; font-size: 0.85rem;">({meal.place_type or 'Restaurant'})</p>
                                        <p style="color: #2d3748 !important; margin: 4px 0; font-size: 0.9rem;">📍 {meal.location}</p>
                                        <p style="color: #2d3748 !important; margin: 4px 0; font-size: 0.9rem;">🥣 <b style="color: #1a202c !important;">Dishes:</b> {", ".join([d.name for d in meal.dishes])}</p>
                                        <p style="color: #2d3748 !important; margin: 4px 0; font-size: 0.9rem;">🌤️ {meal.weather_suitability}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    day_food_map.append({"name": meal.place_name, "latitude": meal.latitude, "longitude": meal.longitude})
                                
                                # Per-day food map with names
                                if day_food_map:
                                    df_food = pd.DataFrame(day_food_map)
                                    st.markdown(f"#### 🗺️ Day {day_num} Dining Map")
                                    st.pydeck_chart(pdk.Deck(
                                        initial_view_state=pdk.ViewState(
                                            latitude=df_food['latitude'].mean(),
                                            longitude=df_food['longitude'].mean(),
                                            zoom=13, pitch=0
                                        ),
                                        layers=[
                                            pdk.Layer('ScatterplotLayer', data=df_food, get_position='[longitude, latitude]',
                                                get_radius=100, get_fill_color=[221, 107, 32, 200], pickable=True),
                                            pdk.Layer('TextLayer', data=df_food, get_position='[longitude, latitude]',
                                                get_text='name', get_size=14, get_color=[255, 255, 255, 255],
                                                get_alignment_baseline="'bottom'", get_pixel_offset='[0, -15]')
                                        ], tooltip={"text": "{name}"}
                                    ))
                                all_food_map_data.extend(day_food_map)
                
                # Tab 4: Budget
                with tab4:
                    st.subheader("Daily Budget Allocation Details")
                    for db_item in travel_plan.daily_budget:
                        st.markdown(f"""
                        - 🏨 **Hotel:** ₹{db_item.hotel} per day
                        - 🍽️ **Food:** ₹{db_item.food} per day
                        - 🚗 **Transport:** ₹{db_item.transport} per day
                        - 🗺️ **Activities:** ₹{db_item.activities} per day
                        - 🛍️ **Miscellaneous:** ₹{db_item.miscellaneous} per day
                        """)
                
                # Tab 5: Packing & Tips
                with tab5:
                    col_pack, col_tips = st.columns(2)
                    with col_pack:
                        st.subheader("🧳 Packing Checklist")
                        for item in travel_plan.packing_list:
                            st.checkbox(item, value=True, key=item)
                    
                    with col_tips:
                        st.subheader("⚠️ Essential Travel Tips")
                        for tip in travel_plan.tips:
                            st.info(tip)
                
            except Exception as e:
                st.error(f"Failed to generate travel plan: {str(e)}")
else:
    st.info("👈 Fill out the settings in the sidebar and click **Generate Travel Plan** to start planning!")
