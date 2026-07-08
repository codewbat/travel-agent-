import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
NUMBEO_API_KEY = os.getenv("NUMBEO_API_KEY") or os.getenv("Numbeo_API_KEY")
USE_OPENVAN = True

# LangSmith Tracing Setup
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "travel_planner")
# Explicitly disable LangSmith API key to avoid tracing uploads
os.environ["LANGCHAIN_API_KEY"] = ""


weather_api_key = os.getenv("Weather_API_KEY") or os.getenv("WEATHER_API_KEY")
groq_key = os.getenv("GROQAI_API_Key") or os.getenv("GROQ_API_KEY")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = "hotels4.p.rapidapi.com"

llm_primary = ChatGroq(
    groq_api_key=groq_key,
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=4096,
    max_retries=2
)

llm_fallback = ChatGroq(
    groq_api_key=groq_key,
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=4096,
    max_retries=2
)

llm = llm_primary.with_fallbacks([llm_fallback])
