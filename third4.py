from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# ---------------- TOOLS ---------------- #

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str):
    """Get current weather for a city"""
    url = f"https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}"
    response = requests.get(url)
    return response.json()

# ---------------- LLM ---------------- #

api_key = os.getenv("CHAT_GROQ_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key
)

# ---------------- REACT AGENT ---------------- #

agent = create_react_agent(
    model=llm,
    tools=[search_tool, get_weather_data]
)

# ---------------- INVOKE ---------------- #

response = agent.invoke(
    {
        "messages": [
            ("user", "what is the current temperature of gurgaon?")
        ]
    }
)

print(response["messages"][-1].content)