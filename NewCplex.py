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

# Initialize the matrices and lists with zeros or default values
heroes_synergy = np.zeros((num_heroes, num_heroes))
heroes_counter = np.zeros((num_heroes, num_heroes))
# Calculate base value 'v' as the average win rate for each hero
v = df.groupby('hero_1_name')['win_rate'].mean().reindex(heroes_name, fill_value=0).values
# Set a default weight 'w' of 1 for all heroes, as it's not in the CSV
w = np.ones(num_heroes)

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
teammates = []
enemies = ['Wraith King']
available_heroes = [hero for hero in heroes_name if hero not in teammates and hero not in enemies]

# Get indices for the heroes
teammate_indices = [hero_to_idx[hero] for hero in teammates]
enemy_indices = [hero_to_idx[hero] for hero in enemies]
available_indices = [hero_to_idx[hero] for hero in available_heroes]

# Create the optimization model
mdl = Model('Dota2HeroSelection')

# --- Decision Variables ---
# A binary variable for each available hero, to decide whether to pick them or not
x = mdl.binary_var_dict(available_indices, name='x')

# --- Objective Function ---
# Maximize the total value from base win rate, synergy, and countering enemies
# 1. Base value of picked heroes
base_value = mdl.sum(v[i] * x[i] for i in available_indices)

# 2. Synergy value between picked heroes and existing teammates
synergy_with_teammates = mdl.sum(heroes_synergy[i][j] * x[i] for i in available_indices for j in teammate_indices)
synergy_between_new_heroes = mdl.sum(heroes_synergy[i][j] * x[i] * x[j] for i in available_indices for j in available_indices if i < j)

# 3. Counter value against enemy heroes
counter_value = mdl.sum(heroes_counter[j][i] * x[i] for i in available_indices for j in enemy_indices)

# The model will maximize the sum of these values
mdl.maximize(base_value + synergy_with_teammates + synergy_between_new_heroes + counter_value)


# --- Constraints ---
# The number of heroes to pick for the remaining slots on the team
num_heroes_to_pick = 5 - len(teammates)
mdl.add_constraint(mdl.sum(x[i] for i in available_indices) == num_heroes_to_pick)

# --- Solve the Model ---
print("\n🚀 Solving for the optimal hero picks...")
solution = mdl.solve()

# --- Print the Results ---
if solution:
    print("\n🎉 Optimal Team Found! 🎉")
    picked_heroes_indices = [i for i in available_indices if x[i].solution_value > 0.9]
    final_team_indices = teammate_indices + picked_heroes_indices
    
    print(f"\nYour Teammates: {', '.join(teammates) if teammates else 'None'}")
    print("Recommended Picks:")
    for i in picked_heroes_indices:
        print(f" - {heroes_name[i]}")

    # --- NEW: Individual Hero Contributions ---
    print("\n--- Individual Hero Contributions ---")
    for i in picked_heroes_indices:
        hero_name = heroes_name[i]
        
        # 1. Base value for this hero
        hero_base_val = v[i]
        
        # 2. Synergy value of this hero with all other final team members
        hero_synergy_val = sum(heroes_synergy[i][j] + heroes_synergy[j][i] for j in final_team_indices if i != j)
        
        # 3. Counter value of this hero against all enemies
        hero_counter_val = sum(heroes_counter[j][i] for j in enemy_indices)
        
        total_individual_score = hero_base_val + hero_synergy_val + hero_counter_val
        
        print(f"🦸 {hero_name} (Individual Score: {total_individual_score:.4f})")
        print(f"   - Base Win Rate Value: {hero_base_val:.4f}")
        print(f"   - Synergy with Team:   {hero_synergy_val:.4f}")
        print(f"   - Counter vs Enemies:  {hero_counter_val:.4f}")

    # Recalculate and show the breakdown of the objective value for the whole team
    total_base_val = sum(v[i] for i in final_team_indices)
    total_synergy_val = sum(heroes_synergy[i][j] + heroes_synergy[j][i] for i in final_team_indices for j in final_team_indices if i < j)
    total_counter_val = sum(heroes_counter[j][i] for i in final_team_indices for j in enemy_indices)

    print("\n--- Overall Team Value Breakdown ---")
    print(f"Objective Value (Total Score):  {mdl.objective_value:.4f}")
    print(f"  - Total Base Value:           {total_base_val:.4f}")
    print(f"  - Total Synergy Value:        {total_synergy_val:.4f}")
    print(f"  - Total Counter Value:        {total_counter_val:.4f}")

else:
    print("No solution found.")

# --- 4. End Timer and Print Duration ---
end_time = time.time()
processing_time = end_time - start_time
print(f"\nProcessing took: {processing_time:.4f} seconds.")