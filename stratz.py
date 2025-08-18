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
  }
}
"""

# 3. New query specifically for position-based statistics
GET_HERO_POSITION_STATS_QUERY = """
query GetHeroPositionStats($heroIds: [Short], $bracketBasicIds: [RankBracketBasicEnum]) {
  heroStats {
    stats(
      heroIds: $heroIds, 
      bracketBasicIds: $bracketBasicIds, 
      groupByPosition: true
    ) {
      heroId
      position
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

    # --- Step 1.5: Fetch Position-Based Statistics for All Heroes ---
    print("\n⏳ Step 1.5: Fetching position-based win rates for all heroes...")
    
    position_variables = {
        "heroIds": [hero['id'] for hero in heroes],
        "bracketBasicIds": TARGET_BRACKET
    }
    
    position_data = fetch_graphql_data(GET_HERO_POSITION_STATS_QUERY, position_variables)
    position_stats_map = {}
    
    if position_data and 'heroStats' in position_data:
        stats_data = position_data['heroStats'].get('stats', [])
        
        # Group position stats by hero ID and position
        for stat in stats_data:
            hero_id = stat['heroId']
            position = stat['position']
            
            if hero_id not in position_stats_map:
                position_stats_map[hero_id] = {}
            
            position_stats_map[hero_id][position] = {
                'matchCount': stat['matchCount'],
                'winCount': stat['winCount'],
                'winRate': stat['winCount'] / stat['matchCount'] if stat['matchCount'] > 0 else 0
            }
        
        print(f"   > Successfully fetched position data for {len(position_stats_map)} heroes.")
    else:
        print("   > Warning: Could not fetch position-based statistics.")
    
    # Position mapping for readable names
    POSITION_NAMES = {
        "POSITION_1": "Safelane",
        "POSITION_2": "Midlane", 
        "POSITION_3": "Offlane",
        "POSITION_4": "Soft Support",
        "POSITION_5": "Hard Support"
    }

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
            
            advantage_data = hero_stats.get('heroVsHeroMatchup', {}).get('advantage', [{}])
            
            if not advantage_data:
                continue

        # --- Process Opponent Data ('vs') ---
        vs_data = advantage_data[0].get('vs', [])
        for opponent in vs_data:
            interaction_row = {
                "hero_1_id": hero_id,
                "hero_1_name": hero_name,
                "type": "vs",
                "hero_2_id": opponent['heroId2'],
                "hero_2_name": hero_id_to_name.get(opponent['heroId2'], "Unknown"),
                "advantage": round(opponent['synergy'] / 100, 5)
            }
            
            # Add position-specific win rates for hero_1
            hero_positions = position_stats_map.get(hero_id, {})
            for pos_key, pos_name in POSITION_NAMES.items():
                pos_stats = hero_positions.get(pos_key, {'matchCount': 0, 'winCount': 0, 'winRate': 0})
                interaction_row[f"{pos_name.lower().replace(' ', '_')}_matches"] = pos_stats['matchCount']
                interaction_row[f"{pos_name.lower().replace(' ', '_')}_wins"] = pos_stats['winCount']
                interaction_row[f"{pos_name.lower().replace(' ', '_')}_winrate"] = round(pos_stats['winRate'], 4)
            
            all_interactions.append(interaction_row)

        # --- Process Teammate Data ('with') ---
        with_data = advantage_data[0].get('with', [])
        for teammate in with_data:
            interaction_row = {
                "hero_1_id": hero_id,
                "hero_1_name": hero_name,
                "type": "with",
                "hero_2_id": teammate['heroId2'],
                "hero_2_name": hero_id_to_name.get(teammate['heroId2'], "Unknown"),
                "advantage": round(teammate['synergy'] / 100, 5)
            }
            
            # Add position-specific win rates for hero_1
            hero_positions = position_stats_map.get(hero_id, {})
            for pos_key, pos_name in POSITION_NAMES.items():
                pos_stats = hero_positions.get(pos_key, {'matchCount': 0, 'winCount': 0, 'winRate': 0})
                interaction_row[f"{pos_name.lower().replace(' ', '_')}_matches"] = pos_stats['matchCount']
                interaction_row[f"{pos_name.lower().replace(' ', '_')}_wins"] = pos_stats['winCount']
                interaction_row[f"{pos_name.lower().replace(' ', '_')}_winrate"] = round(pos_stats['winRate'], 4)
            
            all_interactions.append(interaction_row)

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