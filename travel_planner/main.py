import json
import sys
from datetime import datetime
from langchain_core.messages import HumanMessage, ToolMessage
from travel_planner.config import llm
from travel_planner.tools import (
    get_weather_report_of_city, get_categorize_budget, get_hotel_recommendations,
    get_activity_recommendations, get_food_recommendations, get_packing_tips_recommendation,
    create_travel_plan, search_restaurants, search_attractions, search_hotels_google,
    get_place_details_by_id, get_city_place_id, search_nearby_places, search_places,
    get_distance_matrix, search_osm_places, search_osm_nearby, estimate_city_budget,
    compare_country_food_costs, get_fuel_prices_by_country
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Bind all planning tools and Google Places tools
llm_with_tools = llm.bind_tools(
    [
        get_weather_report_of_city,      
        get_categorize_budget,            
        get_hotel_recommendations,        
        get_activity_recommendations,    
        get_food_recommendations,        
        get_packing_tips_recommendation, 
        create_travel_plan,
        search_restaurants,
        search_attractions,
        search_hotels_google,
        get_place_details_by_id,
        get_city_place_id,
        search_nearby_places,
        search_places,
        get_distance_matrix,
        search_osm_places,
        search_osm_nearby,
        estimate_city_budget,
        compare_country_food_costs,
        get_fuel_prices_by_country
    ]
)

def run_agent(message: str):
    """Run sequential tool agent loop for a user query."""
    chatconvo = []
    
    user_message = f"""
{message}

IMPORTANT: To create a complete travel plan, you MUST call ALL these tools in sequence:
1. get_weather_report_of_city - Get weather data
2. get_categorize_budget - Analyze budget
3. get_hotel_recommendations - Suggest hotels
4. get_activity_recommendations - Plan activities
5. get_food_recommendations - Suggest food
6. get_packing_tips_recommendation - Get packing tips
7. create_travel_plan - Create final plan

Google Places tools (search_restaurants, search_attractions, search_nearby_places, search_places) can be used to query real restaurants/sights during planning.
"""
    chatconvo.append(HumanMessage(content=user_message))
    ai_message = llm_with_tools.invoke(chatconvo)
    chatconvo.append(ai_message)
    
    print("\n🔧 LLM Tool Calls:")
    print(json.dumps([tc['name'] for tc in ai_message.tool_calls], indent=2))
    
    results = {}
    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            print(f"\n🔧 Executing: {tool_name}")
            print(f"   Args: {json.dumps(tool_args, indent=2)}")
            
            if tool_name == 'get_weather_report_of_city':
                result = get_weather_report_of_city.invoke(tool_args)  
                results['weather'] = result
                chatconvo.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call['id']))
                print(f"   ✅ Weather: {len(result)} records")
                
            elif tool_name == 'get_categorize_budget':
                result = get_categorize_budget.invoke(tool_args)  
                results['budget'] = result
                chatconvo.append(ToolMessage(content=result.model_dump_json(), tool_call_id=tool_call['id']))
                print(f"   ✅ Budget: {result.category}")
                
            elif tool_name == 'get_hotel_recommendations':
                if results.get('budget'):
                    tool_args['budget_category'] = results['budget'] 
                if results.get('weather'):
                    tool_args['weather_data'] = results['weather']
                result = get_hotel_recommendations.invoke(tool_args) 
                results['hotels'] = result
                chatconvo.append(ToolMessage(content=json.dumps([h.model_dump() for h in result]), tool_call_id=tool_call['id']))
                print(f"   ✅ Hotels: {len(result)} found")
                
            elif tool_name == 'get_activity_recommendations':
                if results.get('budget'):
                    tool_args['budget_category'] = results['budget']
                if results.get('weather'):
                    tool_args['weather_data'] = results['weather']
                result = get_activity_recommendations.invoke(tool_args)  
                results['activities'] = result
                chatconvo.append(ToolMessage(content=json.dumps([a.model_dump() for a in result]), tool_call_id=tool_call['id']))
                print(f"   ✅ Activities: {len(result)} days")
                
            elif tool_name == 'get_food_recommendations':
                if results.get('budget'):
                    tool_args['budget_category'] = results['budget']
                result = get_food_recommendations.invoke(tool_args)  
                results['food'] = result
                chatconvo.append(ToolMessage(content=json.dumps([f.model_dump() for f in result]), tool_call_id=tool_call['id']))
                print(f"   ✅ Food: {len(result)} recommendations")
                
            elif tool_name == 'get_packing_tips_recommendation':
                if results.get('budget'):
                    tool_args['budget_category'] = results['budget']
                if results.get('weather'):
                    tool_args['weather_data'] = results['weather']
                result = get_packing_tips_recommendation.invoke(tool_args)  
                results['packing'] = result
                chatconvo.append(ToolMessage(content=result.model_dump_json(), tool_call_id=tool_call['id']))
                print(f"   ✅ Packing: Done")
                
            elif tool_name == 'create_travel_plan':
                if results.get('budget'):
                    tool_args['budget_category'] = results['budget']       
                if results.get('hotels'):
                    tool_args['hotels'] = results['hotels']                
                if results.get('activities'):
                    tool_args['activities'] = results['activities']
                if results.get('food'):
                    tool_args['food'] = results['food']                      
                if results.get('packing'):
                    tool_args['packing_tips'] = results['packing']           
                
                result = create_travel_plan.invoke(tool_args) 
                results['travel_plan'] = result
                chatconvo.append(ToolMessage(content=result.model_dump_json(), tool_call_id=tool_call['id']))
                print(f"   ✅ Travel Plan: Created")
                
            # Google Places API Tools
            elif tool_name == 'search_restaurants':
                result = search_restaurants.invoke(tool_args)
                chatconvo.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call['id']))
                print(f"   ✅ Google Restaurants: {len(result)} found")
                
            elif tool_name == 'search_attractions':
                result = search_attractions.invoke(tool_args)
                chatconvo.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call['id']))
                print(f"   ✅ Google Attractions: {len(result)} found")
                
            elif tool_name == 'search_hotels_google':
                result = search_hotels_google.invoke(tool_args)
                chatconvo.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call['id']))
                print(f"   ✅ Google Hotels fallback: {len(result)} found")
                
            elif tool_name == 'get_city_place_id':
                result = get_city_place_id.invoke(tool_args)
                chatconvo.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call['id']))
                print(f"   ✅ Google City Place ID: {result}")
                
            elif tool_name == 'get_place_details_by_id':
                result = get_place_details_by_id.invoke(tool_args)
                chatconvo.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call['id']))
                print(f"   ✅ Google Place Details: fetched")
                
            elif tool_name == 'search_nearby_places':
                result = search_nearby_places.invoke(tool_args)
                chatconvo.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call['id']))
                print(f"   ✅ Google Nearby Places: {len(result)} found")

            elif tool_name == 'search_places':
                result = search_places.invoke(tool_args)
                chatconvo.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call['id']))
                print(f"   ✅ Google Text Search: {len(result)} found")
                
            elif tool_name == 'get_distance_matrix':
                result = get_distance_matrix.invoke(tool_args)
                chatconvo.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call['id']))
                print(f"   ✅ Google Distance Matrix: fetched")
                
            elif tool_name == 'search_osm_places':
                result = search_osm_places.invoke(tool_args)
                chatconvo.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call['id']))
                print(f"   ✅ OSM Search: {len(result)} found")
                
            elif tool_name == 'search_osm_nearby':
                result = search_osm_nearby.invoke(tool_args)
                chatconvo.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call['id']))
                print(f"   ✅ OSM Nearby Search: {len(result)} found")
                
            elif tool_name == 'estimate_city_budget':
                result = estimate_city_budget.invoke(tool_args)
                chatconvo.append(ToolMessage(content=result, tool_call_id=tool_call['id']))
                print(f"   ✅ Numbeo Budget: fetched")
                
            elif tool_name == 'compare_country_food_costs':
                result = compare_country_food_costs.invoke(tool_args)
                chatconvo.append(ToolMessage(content=result, tool_call_id=tool_call['id']))
                print(f"   ✅ OpenVan Compare: fetched")
                
            elif tool_name == 'get_fuel_prices_by_country':
                result = get_fuel_prices_by_country.invoke(tool_args)
                chatconvo.append(ToolMessage(content=result, tool_call_id=tool_call['id']))
                print(f"   ✅ OpenVan Fuel Prices: fetched")
                
        if results.get('travel_plan'):
            return results['travel_plan']
    return None

if __name__ == "__main__":
    print("🌍 TRAVEL PLANNING ASSISTANT")
    print("=" * 70)
    query = input("Enter your prompt : \n")
    plan = run_agent(query)
    if plan:
        print("\n🎉 Complete travel plan created successfully for", plan.city)
    else:
        print("\n⚠️ Failed to generate travel plan.")
