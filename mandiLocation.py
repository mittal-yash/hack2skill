import sys
import re
from geopy.distance import geodesic

sys.path.append("..")

from callback_logging import log_query_to_model, log_model_response
from google.adk import Agent
from google.adk.tools import VertexAiSearchTool, BaseTool

# 🔹 Tool to search crop prices from price datastore
price_search_tool = VertexAiSearchTool(
    data_store_id="projects/kissan-465113/locations/global/collections/default_collection/dataStores/test_1752998730816"
)

# 🔹 Tool to search mandi coordinates + nearby mandis
location_search_tool = VertexAiSearchTool(
    data_store_id="projects/kissan-465113/locations/global/collections/default_collection/dataStores/mandilocation_1753124267693_gcs_store"
)

# ✅ Utility: Normalize snippet to consistent format
def normalize_price_snippet(snippet: str, mandi: str, distance_km: int = None) -> str:
    # Extract prices & date from snippet
    date_match = re.search(r"On\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})", snippet)
    date_str = date_match.group(1) if date_match else "Unknown date"

    max_match = re.search(r"maximum price\s+was\s+₹?(\d+)", snippet)
    min_match = re.search(r"minimum price\s+was\s+₹?(\d+)", snippet)
    modal_match = re.search(r"modal price\s+was\s+₹?(\d+)", snippet)

    max_price = f"₹{max_match.group(1)}" if max_match else "N/A"
    min_price = f"₹{min_match.group(1)}" if min_match else "N/A"
    modal_price = f"₹{modal_match.group(1)}" if modal_match else "N/A"

    resp = f"✅ Apple Price in {mandi}\n"
    resp += f"📅 Date: {date_str}\n"
    resp += f"🔼 Max Price: {max_price}\n"
    resp += f"🔽 Min Price: {min_price}\n"
    resp += f"📊 Modal Price: {modal_price}"

    if distance_km is not None:
        resp += f"\n(Closest available mandi, {distance_km} km away)"

    return resp

# ✅ Strictly fetch price for exact mandi
def fetch_price_for_exact_mandi(crop: str, mandi: str):
    results = price_search_tool.search(f"{crop} price in {mandi}")
    if not results:
        return None  # no results → fallback
    
    for r in results:
        doc_struct = r.document.struct_data
        mandi_name_in_price = doc_struct.get("market_name", "").strip().lower()
        # ✅ Only return if mandi matches exactly
        if mandi_name_in_price == mandi.strip().lower():
            return normalize_price_snippet(r.snippet, mandi)
    
    # If none match exactly → treat as no data
    return None

# 🔹 Custom Tool: Fallback to nearest mandi with available price
class LocationFallbackTool(BaseTool):
    name = "location_based_mandi_fallback"
    description = "If price not available, finds crop price from geographically closest mandi using coordinates."

    def __init__(self):
        super().__init__(name=self.name, description=self.description)

    def __call__(self, mandi_name: str, crop: str = "apple"):
        try:
            # ✅ Step 1: Get all mandi coordinates
            all_results = location_search_tool.search("")  
            mandi_coords = None
            mandi_locations = {}

            for result in all_results:
                doc = result.document.struct_data
                name = doc.get("Mandi_Name", "").strip()
                try:
                    lat = float(doc["Latitude"])
                    lon = float(doc["Longitude"])
                except (KeyError, ValueError):
                    continue

                mandi_locations[name.lower()] = (name, (lat, lon))
                if name.lower() == mandi_name.lower():
                    mandi_coords = (lat, lon)

            if not mandi_coords:
                return f"❌ Sorry, I couldn’t find coordinates for '{mandi_name.strip()}' in the location database."

            # ✅ Step 2: Get all available prices
            all_price_results = price_search_tool.search(crop.strip())
            if not all_price_results:
                return f"❌ No {crop.strip()} prices found in any mandi."

            # Extract only mandi names that have prices
            priced_mandis = []
            for result in all_price_results:
                snippet = result.snippet
                doc_struct = result.document.struct_data
                mandi_name_in_price = doc_struct.get("market_name", "").strip()
                if mandi_name_in_price:
                    priced_mandis.append((mandi_name_in_price, snippet))

            # ✅ Step 3: Find nearest mandi among those with prices
            best_match = None
            min_distance = float("inf")

            for priced_mandi, snippet in priced_mandis:
                priced_lower = priced_mandi.lower()
                if priced_lower in mandi_locations:
                    _, coords = mandi_locations[priced_lower]
                    dist = geodesic(mandi_coords, coords).km
                    if dist < min_distance:
                        min_distance = dist
                        best_match = (priced_mandi, snippet, dist)

            if best_match:
                mandi, snippet, dist = best_match
                return normalize_price_snippet(snippet, mandi, round(dist))

            return f"❌ Sorry, I couldn’t find {crop.strip()} prices in '{mandi_name.strip()}' or any nearby mandis."

        except Exception as e:
            return f"❌ An error occurred during fallback price lookup: {str(e)}"

# ✅ StrictPriceTool: Checks exact mandi, else triggers fallback
class StrictPriceTool(BaseTool):
    name = "strict_price_lookup"
    description = "Looks up crop price for EXACT mandi. If not found, automatically triggers nearest mandi fallback."

    def __init__(self):
        super().__init__(name=self.name, description=self.description)
        self.fallback_tool = LocationFallbackTool()

    def __call__(self, mandi_name: str, crop: str = "apple"):
        # ✅ Try exact mandi first
        exact_price = fetch_price_for_exact_mandi(crop, mandi_name)
        if exact_price:
            return exact_price
        else:
            # ✅ If no match → fallback
            return self.fallback_tool(mandi_name, crop)

# 🔹 Agent Setup with Strict Logic
root_agent = Agent(
    name="farm",
    model="gemini-2.0-flash-001",
    description="Helps Indian farmers understand market trends using mandi data.",
    instruction="""
When a farmer asks for crop prices:

1️⃣ Always use `strict_price_lookup` (which ensures exact mandi match). 
2️⃣ If no exact match is found, it automatically triggers `location_based_mandi_fallback`.
3️⃣ Always respond in SAME language as the user.
4️⃣ Always format like:

✅ <Crop> Price in <Mandi>
📅 Date: <date>
🔼 Max Price: ₹xxxx
🔽 Min Price: ₹xxxx
📊 Modal Price: ₹xxxx

If fallback is used:
(Closest available mandi, <distance> km away)

If still no data:
❌ Sorry, no data available for <crop> in <mandi>.

Do NOT repeat the same snippet multiple times. Always give a clean single response.
""",
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[
        StrictPriceTool(),       # ✅ always use strict match first
        location_search_tool     # needed for fallback calculations
    ]
)
