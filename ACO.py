import pandas as pd
import numpy as np
import time
import random

start_time = time.time()

# --- 1. Data Loading and Processing ---
print("🐜 Ant Colony Optimization for Hero Selection")

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
hero_to_idx = {name: i for i, name in enumerate(heroes_name)}

# Define positions
positions = ['safelane', 'midlane', 'offlane', 'soft_support', 'hard_support']
pos_to_idx = {pos: i for i, pos in enumerate(positions)}

# Initialize matrices
heroes_synergy = np.zeros((num_heroes, num_heroes))
heroes_counter = np.zeros((num_heroes, num_heroes))
hero_position_winrates = np.zeros((num_heroes, len(positions)))

# Extract position-based win rates
for hero_name in heroes_name:
    hero_idx = hero_to_idx[hero_name]
    hero_data = df[df['hero_1_name'] == hero_name].iloc[0] if len(df[df['hero_1_name'] == hero_name]) > 0 else None
    
    if hero_data is not None:
        for pos in positions:
            winrate_col = f"{pos}_winrate"
            if winrate_col in hero_data:
                hero_position_winrates[hero_idx][pos_to_idx[pos]] = hero_data[winrate_col]

# Build synergy and counter matrices
synergy_df = df[df['type'] == 'with']
counter_df = df[df['type'] == 'vs']

for index, row in synergy_df.iterrows():
    h1_idx = hero_to_idx[row['hero_1_name']]
    h2_idx = hero_to_idx[row['hero_2_name']]
    heroes_synergy[h1_idx][h2_idx] = row['advantage']

for index, row in counter_df.iterrows():
    h1_idx = hero_to_idx[row['hero_1_name']]
    h2_idx = hero_to_idx[row['hero_2_name']]
    heroes_counter[h2_idx][h1_idx] = row['advantage']

print("✅ Data processed in", f"{time.time() - start_time:.3f}s")

# --- 2. Configuration ---
teammates = [('Lycan', 2), ('Kez', 1)]  # (hero_name, position_index)
enemies = ['Axe', 'Meepo', 'Huskar']

# Get occupied and available positions
occupied_positions = [hero[1] for hero in teammates if isinstance(hero, tuple)]
available_positions = [i for i in range(len(positions)) if i not in occupied_positions]

# Get hero indices
teammate_hero_indices = [hero_to_idx[hero[0] if isinstance(hero, tuple) else hero] for hero in teammates]
enemy_hero_indices = [hero_to_idx[hero[0] if isinstance(hero, tuple) else hero] for hero in enemies]

print(f"🔒 Occupied positions: {[positions[p] for p in occupied_positions]}")
print(f"🆓 Available positions: {[positions[p] for p in available_positions]}")

# Option to use ALL heroes or limit to top N per position
USE_ALL_HEROES = True  # Set to True to include all heroes, False for top N only

if USE_ALL_HEROES:
    # Include ALL viable heroes for each position
    viable_heroes_per_position = {}
    
    for pos_idx in available_positions:
        candidates = []
        for hero_idx in range(num_heroes):
            # Skip teammates and enemies
            if hero_idx in teammate_hero_indices or hero_idx in enemy_hero_indices:
                continue
            # Only require that the hero has some win rate data for this position
            winrate = hero_position_winrates[hero_idx][pos_idx]
            if winrate > 0.001:  # Very low threshold to include almost all heroes
                candidates.append(hero_idx)
        
        viable_heroes_per_position[pos_idx] = candidates
    
    total_candidates = sum(len(heroes) for heroes in viable_heroes_per_position.values())
    print(f"🌟 Using ALL viable heroes: {total_candidates} total candidates")
    
else:
    # Use top N heroes per position (faster but limited)
    TOP_HEROES_PER_POSITION = 20
    viable_heroes_per_position = {}

    for pos_idx in available_positions:
        candidates = []
        for hero_idx in range(num_heroes):
            # Skip teammates and enemies
            if hero_idx in teammate_hero_indices or hero_idx in enemy_hero_indices:
                continue
            winrate = hero_position_winrates[hero_idx][pos_idx]
            if winrate > 0:
                candidates.append(hero_idx)
        
        # Sort by position win rate and take top candidates
        candidates.sort(key=lambda h: hero_position_winrates[h][pos_idx], reverse=True)
        viable_heroes_per_position[pos_idx] = candidates[:TOP_HEROES_PER_POSITION]

    print(f"🎯 Using top {TOP_HEROES_PER_POSITION} heroes per position for speed")

# --- 3. Ant Colony Optimization Implementation ---

class AntColonyOptimizer:
    def __init__(self, num_ants=20, num_iterations=50, alpha=1.0, beta=2.0, 
                 evaporation_rate=0.1, pheromone_deposit=1.0):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        
        # Initialize pheromone matrix: [position][hero]
        self.pheromones = {}
        for pos_idx in available_positions:
            self.pheromones[pos_idx] = {}
            for hero_idx in viable_heroes_per_position[pos_idx]:
                self.pheromones[pos_idx][hero_idx] = 1.0
        
        self.best_solution = None
        self.best_score = -float('inf')
    
    def calculate_heuristic(self, hero_idx, pos_idx, current_solution):
        """Calculate heuristic value for hero at position"""
        score = 0
        
        # Position win rate (normalized)
        score += hero_position_winrates[hero_idx][pos_idx] * 10
        
        # Synergy with teammates
        for teammate_idx in teammate_hero_indices:
            score += heroes_synergy[hero_idx][teammate_idx] * 8
        
        # Synergy with already selected heroes
        for selected_pos, selected_hero in current_solution.items():
            if selected_hero is not None:
                score += heroes_synergy[hero_idx][selected_hero] * 6
                score += heroes_synergy[selected_hero][hero_idx] * 6
        
        # Counter value against enemies
        for enemy_idx in enemy_hero_indices:
            score += heroes_counter[enemy_idx][hero_idx] * 7
        
        return max(score, 0.1)  # Ensure positive
    
    def calculate_solution_score(self, solution):
        """Calculate total score for a complete solution"""
        selected_heroes = [hero for hero in solution.values() if hero is not None]
        total_score = 0
        
        # Position win rates
        for pos_idx, hero_idx in solution.items():
            if hero_idx is not None:
                total_score += hero_position_winrates[hero_idx][pos_idx]
        
        # Synergy with teammates
        for hero_idx in selected_heroes:
            for teammate_idx in teammate_hero_indices:
                total_score += heroes_synergy[hero_idx][teammate_idx] * 0.8
        
        # Internal synergy
        for i, hero1 in enumerate(selected_heroes):
            for hero2 in selected_heroes[i+1:]:
                total_score += heroes_synergy[hero1][hero2] * 0.6
                total_score += heroes_synergy[hero2][hero1] * 0.6
        
        # Counter value
        for hero_idx in selected_heroes:
            for enemy_idx in enemy_hero_indices:
                total_score += heroes_counter[enemy_idx][hero_idx] * 0.7
        
        return total_score
    
    def construct_solution(self):
        """Construct a solution using ACO probabilistic selection"""
        solution = {pos: None for pos in available_positions}
        
        for pos_idx in available_positions:
            available_heroes = viable_heroes_per_position[pos_idx].copy()
            
            # Remove already selected heroes
            available_heroes = [h for h in available_heroes if h not in solution.values()]
            
            if not available_heroes:
                continue
            
            # Calculate probabilities for each hero
            probabilities = []
            total_prob = 0
            
            for hero_idx in available_heroes:
                pheromone = self.pheromones[pos_idx].get(hero_idx, 1.0)
                heuristic = self.calculate_heuristic(hero_idx, pos_idx, solution)
                
                prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities.append(prob)
                total_prob += prob
            
            # Normalize probabilities
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
                
                # Select hero based on probabilities
                rand = random.random()
                cumulative = 0
                selected_hero = available_heroes[0]  # fallback
                
                for i, prob in enumerate(probabilities):
                    cumulative += prob
                    if rand <= cumulative:
                        selected_hero = available_heroes[i]
                        break
                
                solution[pos_idx] = selected_hero
        
        return solution
    
    def update_pheromones(self, solutions_and_scores):
        """Update pheromone levels based on solution quality"""
        # Evaporation
        for pos_idx in self.pheromones:
            for hero_idx in self.pheromones[pos_idx]:
                self.pheromones[pos_idx][hero_idx] *= (1 - self.evaporation_rate)
        
        # Deposit pheromones
        for solution, score in solutions_and_scores:
            deposit_amount = self.pheromone_deposit * (score / abs(self.best_score) if self.best_score != 0 else 1)
            
            for pos_idx, hero_idx in solution.items():
                if hero_idx is not None and hero_idx in self.pheromones[pos_idx]:
                    self.pheromones[pos_idx][hero_idx] += deposit_amount
    
    def optimize(self):
        """Run the ACO algorithm"""
        print(f"\n🐜 Running ACO with {self.num_ants} ants for {self.num_iterations} iterations...")
        
        for iteration in range(self.num_iterations):
            solutions_and_scores = []
            
            # Generate solutions with ants
            for ant in range(self.num_ants):
                solution = self.construct_solution()
                score = self.calculate_solution_score(solution)
                solutions_and_scores.append((solution, score))
                
                # Update best solution
                if score > self.best_score:
                    self.best_score = score
                    self.best_solution = solution.copy()
            
            # Update pheromones
            self.update_pheromones(solutions_and_scores)
            
            # Progress update
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"   Iteration {iteration + 1}/{self.num_iterations}: Best Score = {self.best_score:.4f}")
        
        return self.best_solution, self.best_score

# --- 4. Run ACO Algorithm ---
aco_start = time.time()

# Adjust parameters based on search space size
if USE_ALL_HEROES:
    # Parameters for full hero search (slower but more comprehensive)
    aco = AntColonyOptimizer(
        num_ants=25,           # More ants for larger search space
        num_iterations=40,     # More iterations for convergence
        alpha=1.2,             # Slightly higher pheromone importance
        beta=2.0,              # Balanced heuristic importance
        evaporation_rate=0.12, # Slower evaporation for more exploration
        pheromone_deposit=1.2  # Stronger pheromone deposits
    )
    
    search_space_size = 1
    for pos_idx in available_positions:
        search_space_size *= len(viable_heroes_per_position[pos_idx])
    
    print(f"🌟 Full search mode: ~{search_space_size:,} total combinations")
    
else:
    # Parameters for limited hero search (faster)
    aco = AntColonyOptimizer(
        num_ants=15,           # Fewer ants for smaller search space
        num_iterations=30,     # Standard iterations
        alpha=1.0,             # Standard pheromone importance
        beta=2.5,              # Higher heuristic importance (more greedy)
        evaporation_rate=0.15, # Standard evaporation
        pheromone_deposit=1.0  # Standard pheromone deposits
    )
    
    search_space_size = 1
    for pos_idx in available_positions:
        search_space_size *= len(viable_heroes_per_position[pos_idx])
    
    print(f"🎯 Limited search mode: ~{search_space_size:,} total combinations")

best_solution, best_score = aco.optimize()
aco_time = time.time() - aco_start

print(f"\n⚡ ACO completed in {aco_time:.3f}s")

# --- 5. Display Results ---
print("\n🎉 Optimal Team Found with ACO! 🎉")

print("\nCurrent Team:")
for hero in teammates:
    if isinstance(hero, tuple):
        hero_name, pos_idx = hero
        print(f" - {hero_name} ({positions[pos_idx].replace('_', ' ').title()}) ✅")

print(f"\nACO Recommended Picks:")
for pos_idx, hero_idx in best_solution.items():
    if hero_idx is not None:
        hero_name = heroes_name[hero_idx]
        position = positions[pos_idx]
        print(f" - {hero_name} ({position.replace('_', ' ').title()}) 🐜")

# --- 6. Individual Contributions ---
print("\n--- Individual Hero Contributions ---")
for pos_idx, hero_idx in best_solution.items():
    if hero_idx is not None:
        hero_name = heroes_name[hero_idx]
        position = positions[pos_idx]
        
        # Calculate detailed breakdown
        position_val = hero_position_winrates[hero_idx][pos_idx]
        synergy_teammates = sum(heroes_synergy[hero_idx][j] for j in teammate_hero_indices)
        
        # Synergy with other ACO picks
        synergy_new = 0
        for other_pos, other_hero in best_solution.items():
            if other_hero is not None and other_hero != hero_idx:
                synergy_new += heroes_synergy[hero_idx][other_hero]
                synergy_new += heroes_synergy[other_hero][hero_idx]
        
        counter_val = sum(heroes_counter[j][hero_idx] for j in enemy_hero_indices)
        
        total_score = position_val + synergy_teammates*0.8 + synergy_new*0.6 + counter_val*0.7
        
        print(f"🐜 {hero_name} - {position.replace('_', ' ').title()} (Score: {total_score:.4f})")
        print(f"   - Position Win Rate:     {position_val:.4f}")
        print(f"   - Synergy with Team:     {synergy_teammates:.4f}")
        print(f"   - Synergy with ACO Picks: {synergy_new:.4f}")
        print(f"   - Counter vs Enemies:    {counter_val:.4f}")

# --- 7. Performance Summary ---
total_time = time.time() - start_time
print(f"\n⏱️  Total processing time: {total_time:.3f} seconds")
print(f"   - Data loading: {aco_start - start_time:.3f}s")
print(f"   - ACO algorithm: {aco_time:.3f}s")
print(f"   - Best solution score: {best_score:.4f}")
print(f"\n🐜 ACO Algorithm Details:")
print(f"   - {aco.num_ants} ants × {aco.num_iterations} iterations = {aco.num_ants * aco.num_iterations} solutions explored")
print(f"   - Search space: ~{search_space_size:,} combinations")
print(f"   - Mode: {'Full hero search' if USE_ALL_HEROES else 'Limited hero search'}")
print(f"   - Heroes per position: {[len(viable_heroes_per_position[p]) for p in available_positions]}")
