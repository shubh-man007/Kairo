import requests
import os
from dotenv import load_dotenv

load_dotenv()

ACCUWEATHER_API = os.getenv("ACCUWEATHER_API_KEY", "")

BASE_URL = "https://dataservice.accuweather.com"
LOCATION_BASE_URL = f"{BASE_URL}/locations/v1/cities/search"
WEATHER_BASE_URL = f"{BASE_URL}/forecasts/v1/daily/5day"

headers = {"Authorization": f"Bearer {ACCUWEATHER_API}"}

LOC = "Ahmedabad"
LOCATION_URL = f"{LOCATION_BASE_URL}?q={LOC}"

locRes = requests.get(LOCATION_URL, headers = headers)
locKey = locRes.json()[0].get("Key")

WEATHER_URL = f"{WEATHER_BASE_URL}/{locKey}"

foreRes = requests.get(WEATHER_URL, headers = headers)
foreRes = foreRes.json()
print(foreRes)