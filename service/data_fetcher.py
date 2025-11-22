import requests

BASE_URL = "https://api.mfapi.in/mf"

def get_nav_history(scheme_code):
    url = f"{BASE_URL}/{scheme_code}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

def search_funds(query):
    url = f"{BASE_URL}/search"
    response = requests.get(url, params={"q": query}, timeout=10)
    response.raise_for_status()
    return response.json()
