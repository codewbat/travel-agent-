# 🌍 Agentic AI Travel Planner

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentic-travel-planner.streamlit.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)

## 🌐 Live Demo

**Try the app here:** [**https://czfze5bhrmcbun5ckofe9z.streamlit.app/**]((https://czfze5bhrmcbun5ckofe9z.streamlit.app/))

---

An AI-powered travel planning assistant that leverages Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and multiple real-world APIs to generate personalized, day-wise travel itineraries.

It intelligently combines weather forecasts, budget analysis, hotel and attraction recommendations, and food suggestions to create a comprehensive travel plan.

## ✨ Key Features

- **🤖 Agentic Workflow**: Uses a multi-step process with specialized agents for weather, budget, hotels, and activities.
- **🧠 RAG Integration**: Employs a Chroma vector store and semantic search to retrieve relevant attraction, food, and hotel data, reducing LLM hallucinations.
- **🌐 Real-World Data**: Integrates with multiple APIs:
    - **Weather**: OpenWeatherMap
    - **Hotels**: HotelsCombined (via RapidAPI)
    - **Attractions & Places**: Google Places API
    - **Budget & Cost of Living**: Numbeo API (with free alternatives like `openvancamp.py`)
- **🗣️ Natural Language Understanding**: Powered by Groq's fast Llama models (`groq/compound`), it understands user prompts like "Plan a 3-day luxury trip to California for a family of four with a budget of 500,000 INR."
- **🖥️ Interactive UI**: Built with Streamlit for easy exploration and a FastAPI backend for potential API integration.
- **🪵 Observability**: Optional integration with LangSmith for tracing and debugging LLM calls.

## 🛠️ Tech Stack

- **Core Framework**: Python 3.11+, LangChain
- **LLM Provider**: Groq (`groq/compound`, `groq/compound-mini`)
- **Vector Database**: ChromaDB (for RAG)
- **UI Layer**: Streamlit
- **API Framework**: FastAPI
- **APIs Integrated**: OpenWeatherMap, RapidAPI (Hotels), Google Places, Numbeo
- **Other Libraries**: Pydantic, python-dotenv, requests

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- pip
- Git

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/codewbat/travel-agent-.git
    cd travel-agent-
