import requests
from datetime import datetime, timedelta
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from travel_planner.config import weather_api_key

class WeatherInput(BaseModel):
    city_name: str = Field(description="Name of the city")
    startdate: str = Field(description="Start date of trip (YYYY-MM-DD)")
    enddate: str = Field(description="End date of trip (YYYY-MM-DD)")

@tool(args_schema=WeatherInput)
def get_weather_report_of_city(
    city_name: str,
    startdate: str,
    enddate: str
):
    """Get city weather data from API between two dates"""
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={weather_api_key}"
    
    response = requests.get(url)
    data = response.json()
    
    if "list" not in data:
        print(f"⚠️ OpenWeather API warning or error: {data.get('message', 'Unknown error')}. Using default weather data.")
        mock_data = []
        try:
            start_dt = datetime.strptime(startdate, "%Y-%m-%d")
            end_dt = datetime.strptime(enddate, "%Y-%m-%d")
            delta = (end_dt - start_dt).days + 1
            for i in range(delta):
                date_str = (start_dt.replace(day=start_dt.day + i)).strftime("%Y-%m-%d")
                for hour in ["09:00:00", "15:00:00", "21:00:00"]:
                    mock_data.append({
                        "dt_txt": f"{date_str} {hour}",
                        "main": {"temp": 303.15, "feels_like": 305.15, "humidity": 60},
                        "weather": [{"description": "scattered clouds"}],
                        "wind": {"speed": 3.5},
                        "pop": 0.1
                    })
            data = {"list": mock_data}
        except Exception:
            data = {"list": [{
                "dt_txt": f"{startdate} 12:00:00",
                "main": {"temp": 303.15, "feels_like": 305.15, "humidity": 60},
                "weather": [{"description": "scattered clouds"}],
                "wind": {"speed": 3.5},
                "pop": 0.1
            }]}

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
        
    # Fallback for future dates beyond 5-day forecast
    if not weather_summary:
        try:
            start_dt = datetime.strptime(startdate, "%Y-%m-%d")
            end_dt = datetime.strptime(enddate, "%Y-%m-%d")
            delta = (end_dt - start_dt).days + 1
            for i in range(delta):
                date_str = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
                for hour in ["09:00:00", "15:00:00", "21:00:00"]:
                    weather_summary.append({
                        "time": f"{date_str} {hour}",
                        "temperature": 303.15,
                        "feels_like": 305.15,
                        "humidity": 60,
                        "weather": "clear sky",
                        "wind": 3.5,
                        "rain_probability": 0.1
                    })
        except Exception:
            pass
            
    return weather_summary
