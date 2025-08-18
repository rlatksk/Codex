import pandas as pd
from docplex.mp.model import Model
import numpy as np
import time

start_time = time.time()

# --- 1. Data Loading and Processing ---

# Load the hero interaction data from the CSV file
try:
    df = pd.read_csv("dota_hero_interactions_HERALD_GUARDIAN_CRUSADER_ARCHON_LEGEND_ANCIENT_DIVINE_IMMORTAL.csv")
    print("✅ Successfully loaded the CSV data.")
except FileNotFoundError:
    print("❌ Error: The CSV file was not found. Please make sure it's in the same directory as the script.")
    exit()

# Get a sorted, unique list of all heroes
heroes_name = sorted(list(set(df['hero_1_name'].unique()) | set(df['hero_2_name'].unique())))
num_heroes = len(heroes_name)
print(f"📊 Found {num_heroes} unique heroes.")

# Create a mapping from hero name to its index for quick lookups
hero_to_idx = {name: i for i, name in enumerate(heroes_name)}

# Define positions
positions = ['safelane', 'midlane', 'offlane', 'soft_support', 'hard_support']
num_positions = len(positions)
pos_to_idx = {pos: i for i, pos in enumerate(positions)}

# Initialize the matrices and lists with zeros or default values
heroes_synergy = np.zeros((num_heroes, num_heroes))
heroes_counter = np.zeros((num_heroes, num_heroes))

# Initialize position-based win rates matrix: heroes x positions
hero_position_winrates = np.zeros((num_heroes, num_positions))

# Extract position-based win rates for each hero
for hero_name in heroes_name:
    hero_idx = hero_to_idx[hero_name]
    hero_data = df[df['hero_1_name'] == hero_name].iloc[0] if len(df[df['hero_1_name'] == hero_name]) > 0 else None
    
    if hero_data is not None:
        for pos in positions:
            winrate_col = f"{pos}_winrate"
            if winrate_col in hero_data:
                hero_position_winrates[hero_idx][pos_to_idx[pos]] = hero_data[winrate_col]

# Separate the 'with' (synergy) and 'vs' (counter) data
synergy_df = df[df['type'] == 'with']
counter_df = df[df['type'] == 'vs']

# Populate the synergy matrix
for index, row in synergy_df.iterrows():
    h1_idx = hero_to_idx[row['hero_1_name']]
    h2_idx = hero_to_idx[row['hero_2_name']]
    # Synergy is positive, higher is better
    heroes_synergy[h1_idx][h2_idx] = row['advantage']

# Populate the counter matrix
for index, row in counter_df.iterrows():
    h1_idx = hero_to_idx[row['hero_1_name']]
    h2_idx = hero_to_idx[row['hero_2_name']]
    # Advantage in a 'vs' matchup is disadvantage.
    # The model subtracts this, so a higher value means it's a better counter for the enemy.
    # We store it such that counter[enemy][our_hero] = disadvantage_value
    heroes_counter[h2_idx][h1_idx] = row['advantage']
    
print("✅ Synergy and Counter matrices have been built from the CSV data.")

# --- 2. CPLEX Model for Hero Selection ---

# Define your team composition and the enemy team
# You can change these lists to match your draft
# Format: (hero_name, position_index) or just hero_name (if position unknown)
teammates = [('Lycan', 2), ('Kez', 1)]
enemies = ['Axe', 'Meepo', 'Huskar']

# Get available heroes and positions
occupied_positions = []
for hero in teammates:
    if isinstance(hero, tuple):
        occupied_positions.append(hero[1])  # Extract position from tuple

available_positions = [i for i in range(num_positions) if i not in occupied_positions]
print(f"🔒 Occupied positions: {[positions[p] for p in occupied_positions]}")
print(f"🆓 Available positions: {[positions[p] for p in available_positions]}")

# Get indices for the heroes
teammate_hero_indices = [hero_to_idx[hero[0] if isinstance(hero, tuple) else hero] for hero in teammates]
enemy_hero_indices = [hero_to_idx[hero[0] if isinstance(hero, tuple) else hero] for hero in enemies]

# For faster solving, we can optionally limit to top heroes by position
# Enable this for significantly faster solving while maintaining good quality
LIMIT_HEROES = False  # Changed to True for better performance
HEROES_PER_POSITION = 20  # Use top 15 heroes per position (adjustable)

if LIMIT_HEROES:
    # Get top heroes per position based on win rate
    top_heroes_per_position = set()
    for p in available_positions:
        position_winrates = [(i, hero_position_winrates[i][p]) for i in range(num_heroes)]
        position_winrates.sort(key=lambda x: x[1], reverse=True)
        top_heroes_per_position.update([i for i, _ in position_winrates[:HEROES_PER_POSITION]])
    
    # Also include heroes that have high synergy with teammates
    for teammate_idx in teammate_hero_indices:
        synergy_scores = [(i, heroes_synergy[i][teammate_idx]) for i in range(num_heroes)]
        synergy_scores.sort(key=lambda x: x[1], reverse=True)
        top_heroes_per_position.update([i for i, _ in synergy_scores[:10]])  # Top 10 synergistic heroes
    
    # Include heroes that counter enemies well
    for enemy_idx in enemy_hero_indices:
        counter_scores = [(i, heroes_counter[enemy_idx][i]) for i in range(num_heroes)]
        counter_scores.sort(key=lambda x: x[1], reverse=True)
        top_heroes_per_position.update([i for i, _ in counter_scores[:10]])  # Top 10 counter heroes
    
    available_hero_indices = list(top_heroes_per_position)
    print(f"🔥 Speed optimization: Using top {len(available_hero_indices)} heroes")
    print(f"   - Top {HEROES_PER_POSITION} per position + synergy/counter heroes")
else:
    available_hero_indices = list(range(num_heroes))

# Create the optimization model
mdl = Model('Dota2HeroSelection_PositionBased')

# --- Decision Variables ---
# Binary variable x[i][p] = 1 if hero i is picked for position p, 0 otherwise
x = {}
for i in available_hero_indices:
    for p in available_positions:  # Only create variables for available positions
        x[i, p] = mdl.binary_var(name=f'x_{i}_{p}')  # Shorter names for speed

# --- Objective Function ---
# 1. Position-based win rate value
position_value = mdl.sum(hero_position_winrates[i][p] * x[i, p] 
                        for i in available_hero_indices 
                        for p in available_positions)

# 2. Synergy value between picked heroes and existing teammates
synergy_with_teammates = mdl.sum(heroes_synergy[i][j] * x[i, p] 
                                for i in available_hero_indices 
                                for p in available_positions
                                for j in teammate_hero_indices)

# 3. Synergy between newly picked heroes (optimized quadratic terms)
# Filter out weak synergies to reduce problem size
MIN_SYNERGY_THRESHOLD = 0.05  # Only consider synergies above this threshold
synergy_between_new_heroes = mdl.sum(heroes_synergy[i][j] * x[i, p1] * x[j, p2] 
                                    for i in available_hero_indices 
                                    for j in available_hero_indices
                                    for p1 in available_positions
                                    for p2 in available_positions
                                    if i < j and p1 != p2 and abs(heroes_synergy[i][j]) > MIN_SYNERGY_THRESHOLD)

# 4. Counter value against enemy heroes
counter_value = mdl.sum(heroes_counter[j][i] * x[i, p] 
                       for i in available_hero_indices 
                       for p in available_positions
                       for j in enemy_hero_indices)

# Maximize the total value (now includes synergy between new heroes)
mdl.maximize(position_value + synergy_with_teammates + synergy_between_new_heroes + counter_value)

# --- Constraints ---
# 1. Each available position must have exactly one hero
for p in available_positions:
    mdl.add_constraint(mdl.sum(x[i, p] for i in available_hero_indices if (i, p) in x) == 1)

# 2. Each hero can be picked for at most one position
for i in available_hero_indices:
    mdl.add_constraint(mdl.sum(x[i, p] for p in available_positions if (i, p) in x) <= 1)

# 3. Don't pick heroes that are already teammates or enemies
for teammate_idx in teammate_hero_indices:
    if teammate_idx in available_hero_indices:
        for p in available_positions:
            if (teammate_idx, p) in x:
                mdl.add_constraint(x[teammate_idx, p] == 0)

for enemy_idx in enemy_hero_indices:
    if enemy_idx in available_hero_indices:
        for p in available_positions:
            if (enemy_idx, p) in x:
                mdl.add_constraint(x[enemy_idx, p] == 0)

# Add solver parameters for faster solving
mdl.parameters.timelimit = 45  # Balanced time limit
mdl.parameters.mip.tolerances.mipgap = 0.01  # 10% optimality gap for faster solving
mdl.parameters.mip.strategy.heuristicfreq = 5  # Use heuristics more frequently
mdl.parameters.preprocessing.presolve = 1  # Enable presolve

# --- Solve the Model ---
print(f"\n🚀 Solving optimized quadratic model (max 45 seconds)...")
print(f"   Variables: {len(x)} binary variables")
print(f"   Available positions: {[positions[p] for p in available_positions]}")
print(f"   Hero pool: {len(available_hero_indices)}/{num_heroes} heroes")
solution = mdl.solve()

# --- Print the Results ---
if solution:
    print("\n🎉 Optimal Team Found! 🎉")
    
    # Extract the picked heroes and their positions
    picked_heroes = []
    for i in available_hero_indices:
        for p in available_positions:
            if (i, p) in x and x[i, p].solution_value > 0.9:
                picked_heroes.append((heroes_name[i], positions[p], i, p))
    
    # Display current teammates with their positions
    print("\nCurrent Team:")
    for hero in teammates:
        if isinstance(hero, tuple):
            hero_name, pos_idx = hero
            print(f" - {hero_name} ({positions[pos_idx].replace('_', ' ').title()}) ✅")
        else:
            print(f" - {hero} (Position Unknown) ✅")
    
    print(f"\nRecommended Picks for remaining positions:")
    for hero_name, position, hero_idx, pos_idx in picked_heroes:
        print(f" - {hero_name} ({position.replace('_', ' ').title()}) 🆕")

    # --- Individual Hero Contributions ---
    print("\n--- Individual Hero Contributions ---")
    for hero_name, position, hero_idx, pos_idx in picked_heroes:
        
        # 1. Position-based win rate value for this hero
        hero_position_val = hero_position_winrates[hero_idx][pos_idx]
        
        # 2. Synergy value of this hero with teammates
        hero_synergy_teammates = sum(heroes_synergy[hero_idx][j] for j in teammate_hero_indices)
        
        # 3. Synergy value of this hero with other picked heroes
        hero_synergy_new = sum(heroes_synergy[hero_idx][other_idx] + heroes_synergy[other_idx][hero_idx] 
                              for _, _, other_idx, _ in picked_heroes if other_idx != hero_idx)
        
        # 4. Counter value of this hero against enemies
        hero_counter_val = sum(heroes_counter[j][hero_idx] for j in enemy_hero_indices)
        
        total_individual_score = hero_position_val + hero_synergy_teammates + hero_synergy_new + hero_counter_val
        
        print(f"🦸 {hero_name} - {position.replace('_', ' ').title()} (Score: {total_individual_score:.4f})")
        print(f"   - Position Win Rate:     {hero_position_val:.4f}")
        print(f"   - Synergy with Team:     {hero_synergy_teammates:.4f}")
        print(f"   - Synergy with New Picks: {hero_synergy_new:.4f}")
        print(f"   - Counter vs Enemies:    {hero_counter_val:.4f}")

    # Calculate overall team value breakdown
    total_position_val = sum(hero_position_winrates[hero_idx][pos_idx] for _, _, hero_idx, pos_idx in picked_heroes)
    
    total_synergy_teammates = sum(heroes_synergy[hero_idx][j] 
                                 for _, _, hero_idx, _ in picked_heroes 
                                 for j in teammate_hero_indices)
    
    total_synergy_new = sum(heroes_synergy[i][j] + heroes_synergy[j][i]
                           for _, _, i, _ in picked_heroes 
                           for _, _, j, _ in picked_heroes if i < j)
    
    total_counter_val = sum(heroes_counter[j][hero_idx] 
                           for _, _, hero_idx, _ in picked_heroes 
                           for j in enemy_hero_indices)

    print("\n--- Overall Team Value Breakdown ---")
    print(f"Objective Value (Total Score):  {mdl.objective_value:.4f}")
    print(f"  - Total Position Value:       {total_position_val:.4f}")
    print(f"  - Synergy with Teammates:     {total_synergy_teammates:.4f}")
    print(f"  - Synergy between New Picks:  {total_synergy_new:.4f}")
    print(f"  - Total Counter Value:        {total_counter_val:.4f}")

else:
    print("No solution found.")

# --- 4. End Timer and Print Duration ---
end_time = time.time()
processing_time = end_time - start_time
print(f"\nProcessing took: {processing_time:.4f} seconds.")