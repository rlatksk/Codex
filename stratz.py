import requests
import json
import time
import pandas as pd # Make sure you have run "pip install pandas"
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Configuration ---
# Get API key from environment variable
STRATZ_API_KEY = os.getenv('STRATZ_API_KEY')
if not STRATZ_API_KEY:
    raise ValueError("STRATZ_API_KEY not found in environment variables. Please check your .env file.")
STRATZ_GRAPHQL_ENDPOINT = "https://api.stratz.com/graphql"

# --- Analysis Configuration ---
# Set the desired rank bracket.
# Valid values: HERALD_GUARDIAN, CRUSADER_ARCHON, LEGEND_ANCIENT, DIVINE_IMMORTAL
# You can also use a list like: ["HERALD_GUARDIAN", "CRUSADER_ARCHON"]
TARGET_BRACKET = ["HERALD_GUARDIAN", "CRUSADER_ARCHON", "LEGEND_ANCIENT", "DIVINE_IMMORTAL"]

# --- Request Headers ---
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {STRATZ_API_KEY}",
    "User-Agent": "STRATZ_API"
}

# --- GraphQL Queries ---

# 1. Query to get all heroes
GET_ALL_HEROES_QUERY = """
{
  constants {
    heroes {
      id
      displayName
    }
  }
}
"""

# 2. Updated query to get hero matchups and synergies for a specific rank bracket
GET_HERO_INTERACTIONS_QUERY = """
query GetHeroData($heroId: Short!, $bracketBasicIds: [RankBracketBasicEnum]) {
  heroStats {
    heroVsHeroMatchup(heroId: $heroId, bracketBasicIds: $bracketBasicIds) {
      advantage {
        vs {
          heroId2
          synergy
        }
        with {
          heroId2
          synergy
        }
      }
    }
    winMonth {
      heroId
      matchCount
      winCount
    }
  }
}
"""


def fetch_graphql_data(query, variables=None):
    """Generic function to send a GraphQL request to the Stratz API."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    try:
        response = requests.post(STRATZ_GRAPHQL_ENDPOINT, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        if 'errors' in data:
            print("GraphQL Error:", data['errors'])
            return None
        return data.get('data', {})
    except requests.exceptions.HTTPError as e:
        print(f"Request Error: {e.response.text}") # More detailed error
        return None
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response. Raw Response: {response.text}")
        return None

# --- Step 1: Get All Heroes ---
print("✅ Step 1: Fetching all hero definitions from Stratz...")
all_heroes_data = fetch_graphql_data(GET_ALL_HEROES_QUERY)

if not all_heroes_data:
    print("❌ Failed to fetch hero list. Stopping script.")
else:
    heroes = all_heroes_data.get('constants', {}).get('heroes', [])
    hero_id_to_name = {hero['id']: hero['displayName'] for hero in heroes}
    print(f"   > Successfully found {len(heroes)} heroes.")

    # --- Step 2: Iterate and Fetch Interaction Data for Each Hero ---
    all_interactions = []
    total_heroes = len(heroes)

    print(f"\n⏳ Step 2: Fetching interaction data for the '{', '.join(TARGET_BRACKET)}' bracket...")

    for i, hero in enumerate(heroes):
        hero_id = hero['id']
        hero_name = hero['displayName']
        
        print(f"   ({i+1}/{total_heroes}) Fetching data for: {hero_name}")

        # Prepare variables for the interaction query
        query_variables = {
            "heroId": hero_id,
            "bracketBasicIds": TARGET_BRACKET
        }

        # Fetch interaction data for the current hero
        interaction_data = fetch_graphql_data(GET_HERO_INTERACTIONS_QUERY, query_variables)

        if interaction_data and 'heroStats' in interaction_data:
            hero_stats = interaction_data['heroStats']
            
            win_stats_map = {
                item['heroId']: {
                    'matchCount': item['matchCount'],
                    'winCount': item['winCount']
                }
                for item in hero_stats.get('winMonth', [])
            }

            hero_1_stats = win_stats_map.get(hero_id)

            advantage_data = interaction_data.get('heroStats', {}).get('heroVsHeroMatchup', {}).get('advantage', [{}])
            
            if not advantage_data:
                continue

            if hero_1_stats:
              # These are the general stats for heroId1.
              match_count = hero_1_stats['matchCount']
              win_count = hero_1_stats['winCount']
              win_rate = win_count / match_count if match_count > 0 else 0

        advantage_data = hero_stats.get('heroVsHeroMatchup', {}).get('advantage', [{}])
        
        if not advantage_data:
            continue

        # --- Process Opponent Data ('vs') ---
        vs_data = advantage_data[0].get('vs', [])
        for opponent in vs_data:
            all_interactions.append({
                "hero_1_id": hero_id,
                "hero_1_name": hero_name,
                "type": "vs",
                "hero_2_id": opponent['heroId2'],
                "hero_2_name": hero_id_to_name.get(opponent['heroId2'], "Unknown"),
                # 3. Use the pre-fetched stats for heroId1 in every row.
                "match_count": match_count,
                "win_count": win_count,
                "win_rate": round(win_rate, 4),
                "advantage": round(opponent['synergy'] / 100, 5) 
            })

        # --- Process Teammate Data ('with') ---
        with_data = advantage_data[0].get('with', [])
        for teammate in with_data:
            all_interactions.append({
                "hero_1_id": hero_id,
                "hero_1_name": hero_name,
                "type": "with",
                "hero_2_id": teammate['heroId2'],
                "hero_2_name": hero_id_to_name.get(teammate['heroId2'], "Unknown"),
                # Also use heroId1's stats here.
                "match_count": match_count,
                "win_count": win_count,
                "win_rate": round(win_rate, 4),
                "advantage": round(teammate['synergy'] / 100, 5)
            })

        # Add a delay to be respectful to the API
        time.sleep(1)

    # --- Step 3: Save All Data to a CSV File ---
    if all_interactions:
        print("\n✅ Step 3: Saving all interaction data to a CSV file...")
        
        df = pd.DataFrame(all_interactions)
        
        output_filename = f"dota_hero_interactions_{'_'.join(TARGET_BRACKET)}.csv"
        df.to_csv(output_filename, index=False, encoding='utf-8')
        
        print(f"   > Success! Data saved to '{output_filename}'")
        print(f"   > A total of {len(df)} interaction rows were written.")
    else:
        print("❌ No interaction data was collected. The output file was not created.")