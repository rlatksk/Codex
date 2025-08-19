package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"time"
)

// Hero represents a Dota 2 hero
type Hero struct {
	ID   int
	Name string
}

// Position represents game positions
type Position struct {
	ID   int
	Name string
}

// GameData holds all the game data
type GameData struct {
	Heroes               []Hero
	Positions            []Position
	HeroToIdx            map[string]int
	PosToIdx             map[string]int
	HeroSynergy          [][]float64
	HeroCounter          [][]float64
	HeroPositionWinrates [][]float64
	TeammateIndices      []int
	EnemyIndices         []int
	OccupiedPositions    []int
	AvailablePositions   []int
	ViableHeroesPerPos   map[int][]int
}

// Solution represents a team composition
type Solution struct {
	Assignment map[int]int // position -> hero index
	Score      float64
}

// AntColonyOptimizer represents the ACO algorithm
type AntColonyOptimizer struct {
	NumAnts          int
	NumIterations    int
	Alpha            float64
	Beta             float64
	EvaporationRate  float64
	PheromoneDeposit float64
	Pheromones       map[int]map[int]float64 // position -> hero -> pheromone
	BestSolution     *Solution
	BestScore        float64
	GameData         *GameData
}

func main() {
	fmt.Println("🐜 Ant Colony Optimization for Hero Selection")
	startTime := time.Now()

	// Load and process data
	gameData, err := loadData()
	if err != nil {
		fmt.Printf("❌ Error loading data: %v\n", err)
		return
	}

	dataLoadTime := time.Since(startTime)
	fmt.Printf("✅ Data processed in %.3fs\n", dataLoadTime.Seconds())

	// Configuration
	teammates := []struct {
		name string
		pos  int
	}{
		{"Invoker", 3}, // (hero_name, position_index)
	}

	enemies := []string{"Axe", "Meepo", "Huskar"}

	// Process configuration
	err = processConfiguration(gameData, teammates, enemies)
	if err != nil {
		fmt.Printf("❌ Error processing configuration: %v\n", err)
		return
	}

	fmt.Printf("🔒 Occupied positions: %v\n", getPositionNames(gameData, gameData.OccupiedPositions))
	fmt.Printf("🆓 Available positions: %v\n", getPositionNames(gameData, gameData.AvailablePositions))

	// Setup viable heroes per position
	useAllHeroes := true // Set to true to include all heroes, false for top N only
	setupViableHeroes(gameData, useAllHeroes)

	// Calculate search space
	searchSpaceSize := calculateSearchSpace(gameData)

	if useAllHeroes {
		fmt.Printf("🌟 Full search mode: ~%d total combinations\n", searchSpaceSize)
	} else {
		fmt.Printf("🎯 Limited search mode: ~%d total combinations\n", searchSpaceSize)
	}

	// Run ACO algorithm
	acoStartTime := time.Now()

	var aco *AntColonyOptimizer
	if useAllHeroes {
		// Parameters for full hero search (slower but more comprehensive)
		aco = NewAntColonyOptimizer(gameData, 25, 40, 1.2, 2.0, 0.12, 1.2)
	} else {
		// Parameters for limited hero search (faster)
		aco = NewAntColonyOptimizer(gameData, 15, 30, 1.0, 2.5, 0.15, 1.0)
	}

	bestSolution, bestScore := aco.Optimize()
	acoTime := time.Since(acoStartTime)

	fmt.Printf("\n⚡ ACO completed in %.3fs\n", acoTime.Seconds())

	// Display results
	displayResults(gameData, teammates, bestSolution, bestScore, aco, useAllHeroes, searchSpaceSize)

	totalTime := time.Since(startTime)
	fmt.Printf("\n⏱️  Total processing time: %.3fs\n", totalTime.Seconds())
	fmt.Printf("   - Data loading: %.3fs\n", dataLoadTime.Seconds())
	fmt.Printf("   - ACO algorithm: %.3fs\n", acoTime.Seconds())
}

func loadData() (*GameData, error) {
	file, err := os.Open("dota_hero_interactions_HERALD_GUARDIAN_CRUSADER_ARCHON_LEGEND_ANCIENT_DIVINE_IMMORTAL.csv")
	if err != nil {
		return nil, fmt.Errorf("CSV file not found: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("insufficient data in CSV")
	}

	// Parse header
	header := records[0]
	heroNameCol := findColumn(header, "hero_1_name")
	hero2NameCol := findColumn(header, "hero_2_name")
	typeCol := findColumn(header, "type")
	advantageCol := findColumn(header, "advantage")

	// Position columns
	positionCols := map[string]int{
		"safelane":     findColumn(header, "safelane_winrate"),
		"midlane":      findColumn(header, "midlane_winrate"),
		"offlane":      findColumn(header, "offlane_winrate"),
		"soft_support": findColumn(header, "soft_support_winrate"),
		"hard_support": findColumn(header, "hard_support_winrate"),
	}

	// Collect unique heroes
	heroSet := make(map[string]bool)
	for _, record := range records[1:] {
		heroSet[record[heroNameCol]] = true
		heroSet[record[hero2NameCol]] = true
	}

	// Create hero list and mapping
	var heroNames []string
	for name := range heroSet {
		heroNames = append(heroNames, name)
	}
	sort.Strings(heroNames)

	heroes := make([]Hero, len(heroNames))
	heroToIdx := make(map[string]int)
	for i, name := range heroNames {
		heroes[i] = Hero{ID: i, Name: name}
		heroToIdx[name] = i
	}

	// Create positions
	positions := []Position{
		{0, "safelane"},
		{1, "midlane"},
		{2, "offlane"},
		{3, "soft_support"},
		{4, "hard_support"},
	}

	posToIdx := map[string]int{
		"safelane":     0,
		"midlane":      1,
		"offlane":      2,
		"soft_support": 3,
		"hard_support": 4,
	}

	numHeroes := len(heroes)
	numPositions := len(positions)

	// Initialize matrices
	heroSynergy := make([][]float64, numHeroes)
	heroCounter := make([][]float64, numHeroes)
	heroPositionWinrates := make([][]float64, numHeroes)

	for i := range heroSynergy {
		heroSynergy[i] = make([]float64, numHeroes)
		heroCounter[i] = make([]float64, numHeroes)
		heroPositionWinrates[i] = make([]float64, numPositions)
	}

	// Process data
	heroWinrateData := make(map[string]map[string]float64)

	for _, record := range records[1:] {
		heroName := record[heroNameCol]
		hero2Name := record[hero2NameCol]
		interactionType := record[typeCol]
		advantage, _ := strconv.ParseFloat(record[advantageCol], 64)

		h1Idx := heroToIdx[heroName]
		h2Idx := heroToIdx[hero2Name]

		// Store position winrates (only from first occurrence of each hero)
		if _, exists := heroWinrateData[heroName]; !exists {
			heroWinrateData[heroName] = make(map[string]float64)
			for pos, col := range positionCols {
				if col >= 0 && col < len(record) {
					if winrate, err := strconv.ParseFloat(record[col], 64); err == nil {
						heroWinrateData[heroName][pos] = winrate
						heroPositionWinrates[h1Idx][posToIdx[pos]] = winrate
					}
				}
			}
		}

		// Process synergy and counter data
		if interactionType == "with" {
			heroSynergy[h1Idx][h2Idx] = advantage
		} else if interactionType == "vs" {
			heroCounter[h2Idx][h1Idx] = advantage
		}
	}

	gameData := &GameData{
		Heroes:               heroes,
		Positions:            positions,
		HeroToIdx:            heroToIdx,
		PosToIdx:             posToIdx,
		HeroSynergy:          heroSynergy,
		HeroCounter:          heroCounter,
		HeroPositionWinrates: heroPositionWinrates,
		ViableHeroesPerPos:   make(map[int][]int),
	}

	return gameData, nil
}

func processConfiguration(gameData *GameData, teammates []struct {
	name string
	pos  int
}, enemies []string) error {

	// Process teammates
	for _, teammate := range teammates {
		if idx, exists := gameData.HeroToIdx[teammate.name]; exists {
			gameData.TeammateIndices = append(gameData.TeammateIndices, idx)
			gameData.OccupiedPositions = append(gameData.OccupiedPositions, teammate.pos)
		}
	}

	// Process enemies
	for _, enemy := range enemies {
		if idx, exists := gameData.HeroToIdx[enemy]; exists {
			gameData.EnemyIndices = append(gameData.EnemyIndices, idx)
		}
	}

	// Calculate available positions
	occupiedSet := make(map[int]bool)
	for _, pos := range gameData.OccupiedPositions {
		occupiedSet[pos] = true
	}

	for i := 0; i < len(gameData.Positions); i++ {
		if !occupiedSet[i] {
			gameData.AvailablePositions = append(gameData.AvailablePositions, i)
		}
	}

	return nil
}

func setupViableHeroes(gameData *GameData, useAllHeroes bool) {
	topHeroesPerPosition := 20

	for _, posIdx := range gameData.AvailablePositions {
		var candidates []int

		for heroIdx := range gameData.Heroes {
			// Skip teammates and enemies
			if contains(gameData.TeammateIndices, heroIdx) || contains(gameData.EnemyIndices, heroIdx) {
				continue
			}

			winrate := gameData.HeroPositionWinrates[heroIdx][posIdx]

			if useAllHeroes {
				// Only require that the hero has some win rate data for this position
				if winrate > 0.001 { // Very low threshold to include almost all heroes
					candidates = append(candidates, heroIdx)
				}
			} else {
				// Use only heroes with positive winrate for sorting
				if winrate > 0 {
					candidates = append(candidates, heroIdx)
				}
			}
		}

		if !useAllHeroes {
			// Sort by position win rate and take top candidates
			sort.Slice(candidates, func(i, j int) bool {
				return gameData.HeroPositionWinrates[candidates[i]][posIdx] > gameData.HeroPositionWinrates[candidates[j]][posIdx]
			})

			if len(candidates) > topHeroesPerPosition {
				candidates = candidates[:topHeroesPerPosition]
			}
		}

		gameData.ViableHeroesPerPos[posIdx] = candidates
	}

	totalCandidates := 0
	for _, heroes := range gameData.ViableHeroesPerPos {
		totalCandidates += len(heroes)
	}

	if useAllHeroes {
		fmt.Printf("🌟 Using ALL viable heroes: %d total candidates\n", totalCandidates)
	} else {
		fmt.Printf("🎯 Using top %d heroes per position for speed\n", topHeroesPerPosition)
	}
}

func calculateSearchSpace(gameData *GameData) int {
	searchSpace := 1
	for _, posIdx := range gameData.AvailablePositions {
		searchSpace *= len(gameData.ViableHeroesPerPos[posIdx])
	}
	return searchSpace
}

func NewAntColonyOptimizer(gameData *GameData, numAnts, numIterations int, alpha, beta, evaporationRate, pheromoneDeposit float64) *AntColonyOptimizer {
	pheromones := make(map[int]map[int]float64)

	// Initialize pheromones for available positions
	for _, pos := range gameData.AvailablePositions {
		pheromones[pos] = make(map[int]float64)
		for _, heroIdx := range gameData.ViableHeroesPerPos[pos] {
			pheromones[pos][heroIdx] = 1.0
		}
	}

	return &AntColonyOptimizer{
		NumAnts:          numAnts,
		NumIterations:    numIterations,
		Alpha:            alpha,
		Beta:             beta,
		EvaporationRate:  evaporationRate,
		PheromoneDeposit: pheromoneDeposit,
		Pheromones:       pheromones,
		GameData:         gameData,
		BestScore:        -math.Inf(1),
		BestSolution:     &Solution{Assignment: make(map[int]int)},
	}
}

func (aco *AntColonyOptimizer) CalculateHeuristic(heroIdx, posIdx int, currentSolution map[int]int) float64 {
	score := 0.0

	// Position win rate (normalized)
	score += aco.GameData.HeroPositionWinrates[heroIdx][posIdx] * 10

	// Synergy with teammates
	for _, teammateIdx := range aco.GameData.TeammateIndices {
		score += aco.GameData.HeroSynergy[heroIdx][teammateIdx] * 8
	}

	// Synergy with already selected heroes
	for _, selectedHero := range currentSolution {
		if selectedHero != -1 {
			score += aco.GameData.HeroSynergy[heroIdx][selectedHero] * 6
			score += aco.GameData.HeroSynergy[selectedHero][heroIdx] * 6
		}
	}

	// Counter value against enemies
	for _, enemyIdx := range aco.GameData.EnemyIndices {
		score += aco.GameData.HeroCounter[enemyIdx][heroIdx] * 7
	}

	return math.Max(score, 0.1) // Ensure positive
}

func (aco *AntColonyOptimizer) CalculateSolutionScore(solution map[int]int) float64 {
	selectedHeroes := make([]int, 0)
	for _, hero := range solution {
		if hero != -1 {
			selectedHeroes = append(selectedHeroes, hero)
		}
	}

	totalScore := 0.0

	// Position win rates
	for posIdx, heroIdx := range solution {
		if heroIdx != -1 {
			totalScore += aco.GameData.HeroPositionWinrates[heroIdx][posIdx]
		}
	}

	// Synergy with teammates
	for _, heroIdx := range selectedHeroes {
		for _, teammateIdx := range aco.GameData.TeammateIndices {
			totalScore += aco.GameData.HeroSynergy[heroIdx][teammateIdx] * 0.8
		}
	}

	// Internal synergy
	for i, hero1 := range selectedHeroes {
		for j, hero2 := range selectedHeroes {
			if i < j {
				totalScore += aco.GameData.HeroSynergy[hero1][hero2] * 0.6
				totalScore += aco.GameData.HeroSynergy[hero2][hero1] * 0.6
			}
		}
	}

	// Counter value
	for _, heroIdx := range selectedHeroes {
		for _, enemyIdx := range aco.GameData.EnemyIndices {
			totalScore += aco.GameData.HeroCounter[enemyIdx][heroIdx] * 0.7
		}
	}

	return totalScore
}

func (aco *AntColonyOptimizer) ConstructSolution() map[int]int {
	solution := make(map[int]int)
	for _, pos := range aco.GameData.AvailablePositions {
		solution[pos] = -1 // Initialize with invalid hero index
	}

	for _, posIdx := range aco.GameData.AvailablePositions {
		availableHeroes := make([]int, 0)

		// Get available heroes for this position
		for _, heroIdx := range aco.GameData.ViableHeroesPerPos[posIdx] {
			// Check if hero already assigned
			alreadyAssigned := false
			for _, assignedHero := range solution {
				if assignedHero == heroIdx {
					alreadyAssigned = true
					break
				}
			}
			if !alreadyAssigned {
				availableHeroes = append(availableHeroes, heroIdx)
			}
		}

		if len(availableHeroes) == 0 {
			continue
		}

		// Calculate probabilities
		probabilities := make([]float64, len(availableHeroes))
		totalProb := 0.0

		for i, heroIdx := range availableHeroes {
			pheromone := aco.Pheromones[posIdx][heroIdx]
			heuristic := aco.CalculateHeuristic(heroIdx, posIdx, solution)

			prob := math.Pow(pheromone, aco.Alpha) * math.Pow(heuristic, aco.Beta)
			probabilities[i] = prob
			totalProb += prob
		}

		// Normalize and select hero
		if totalProb > 0 {
			for i := range probabilities {
				probabilities[i] /= totalProb
			}

			// Roulette wheel selection
			randVal := rand.Float64()
			cumulative := 0.0
			selectedHero := availableHeroes[0]

			for i, prob := range probabilities {
				cumulative += prob
				if randVal <= cumulative {
					selectedHero = availableHeroes[i]
					break
				}
			}

			solution[posIdx] = selectedHero
		}
	}

	return solution
}

func (aco *AntColonyOptimizer) UpdatePheromones(solutionsAndScores []struct {
	solution map[int]int
	score    float64
}) {
	// Evaporation
	for posIdx := range aco.Pheromones {
		for heroIdx := range aco.Pheromones[posIdx] {
			aco.Pheromones[posIdx][heroIdx] *= (1 - aco.EvaporationRate)
		}
	}

	// Deposit pheromones
	for _, solutionScore := range solutionsAndScores {
		var depositAmount float64
		if aco.BestScore != 0 {
			depositAmount = aco.PheromoneDeposit * (solutionScore.score / math.Abs(aco.BestScore))
		} else {
			depositAmount = aco.PheromoneDeposit
		}

		for posIdx, heroIdx := range solutionScore.solution {
			if heroIdx != -1 && aco.Pheromones[posIdx][heroIdx] != 0 {
				aco.Pheromones[posIdx][heroIdx] += depositAmount
			}
		}
	}
}

func (aco *AntColonyOptimizer) Optimize() (map[int]int, float64) {
	fmt.Printf("\n🐜 Running ACO with %d ants for %d iterations...\n", aco.NumAnts, aco.NumIterations)

	for iteration := 0; iteration < aco.NumIterations; iteration++ {
		solutionsAndScores := make([]struct {
			solution map[int]int
			score    float64
		}, 0)

		// Generate solutions with ants
		for ant := 0; ant < aco.NumAnts; ant++ {
			solution := aco.ConstructSolution()
			score := aco.CalculateSolutionScore(solution)
			solutionsAndScores = append(solutionsAndScores, struct {
				solution map[int]int
				score    float64
			}{solution, score})

			// Update best solution
			if score > aco.BestScore {
				aco.BestScore = score
				aco.BestSolution.Assignment = make(map[int]int)
				for k, v := range solution {
					aco.BestSolution.Assignment[k] = v
				}
				aco.BestSolution.Score = score
			}
		}

		// Update pheromones
		aco.UpdatePheromones(solutionsAndScores)

		// Progress update
		if (iteration+1)%10 == 0 || iteration == 0 {
			fmt.Printf("   Iteration %d/%d: Best Score = %.4f\n", iteration+1, aco.NumIterations, aco.BestScore)
		}
	}

	return aco.BestSolution.Assignment, aco.BestScore
}

func displayResults(gameData *GameData, teammates []struct {
	name string
	pos  int
}, bestSolution map[int]int, bestScore float64, aco *AntColonyOptimizer, useAllHeroes bool, searchSpaceSize int) {

	fmt.Println("\n🎉 Optimal Team Found with ACO! 🎉")

	fmt.Println("\nCurrent Team:")
	for _, teammate := range teammates {
		positionName := capitalizePosition(gameData.Positions[teammate.pos].Name)
		fmt.Printf(" - %s (%s) ✅\n", teammate.name, positionName)
	}

	fmt.Println("\nACO Recommended Picks:")
	for posIdx, heroIdx := range bestSolution {
		if heroIdx != -1 {
			heroName := gameData.Heroes[heroIdx].Name
			positionName := capitalizePosition(gameData.Positions[posIdx].Name)
			fmt.Printf(" - %s (%s) 🐜\n", heroName, positionName)
		}
	}

	// Individual Contributions
	fmt.Println("\n--- Individual Hero Contributions ---")
	for posIdx, heroIdx := range bestSolution {
		if heroIdx != -1 {
			heroName := gameData.Heroes[heroIdx].Name
			positionName := capitalizePosition(gameData.Positions[posIdx].Name)

			// Calculate detailed breakdown
			positionVal := gameData.HeroPositionWinrates[heroIdx][posIdx]

			synergyTeammates := 0.0
			for _, teammateIdx := range gameData.TeammateIndices {
				synergyTeammates += gameData.HeroSynergy[heroIdx][teammateIdx]
			}

			// Synergy with other ACO picks
			synergyNew := 0.0
			for _, otherHero := range bestSolution {
				if otherHero != -1 && otherHero != heroIdx {
					synergyNew += gameData.HeroSynergy[heroIdx][otherHero]
					synergyNew += gameData.HeroSynergy[otherHero][heroIdx]
				}
			}

			counterVal := 0.0
			for _, enemyIdx := range gameData.EnemyIndices {
				counterVal += gameData.HeroCounter[enemyIdx][heroIdx]
			}

			totalScore := positionVal + synergyTeammates*0.8 + synergyNew*0.6 + counterVal*0.7

			fmt.Printf("🐜 %s - %s (Score: %.4f)\n", heroName, positionName, totalScore)
			fmt.Printf("   - Position Win Rate:      %.4f\n", positionVal)
			fmt.Printf("   - Synergy with Team:      %.4f\n", synergyTeammates)
			fmt.Printf("   - Synergy with ACO Picks: %.4f\n", synergyNew)
			fmt.Printf("   - Counter vs Enemies:     %.4f\n", counterVal)
		}
	}

	// Performance Summary
	fmt.Printf("   - Best solution score: %.4f\n", bestScore)
	fmt.Printf("\n🐜 ACO Algorithm Details:\n")
	fmt.Printf("   - %d ants × %d iterations = %d solutions explored\n", aco.NumAnts, aco.NumIterations, aco.NumAnts*aco.NumIterations)
	fmt.Printf("   - Search space: ~%d combinations\n", searchSpaceSize)
	if useAllHeroes {
		fmt.Printf("   - Mode: Full hero search\n")
	} else {
		fmt.Printf("   - Mode: Limited hero search\n")
	}

	heroesPerPos := make([]int, 0)
	for _, posIdx := range gameData.AvailablePositions {
		heroesPerPos = append(heroesPerPos, len(gameData.ViableHeroesPerPos[posIdx]))
	}
	fmt.Printf("   - Heroes per position: %v\n", heroesPerPos)
}

// Helper functions
func findColumn(header []string, columnName string) int {
	for i, col := range header {
		if col == columnName {
			return i
		}
	}
	return -1
}

func contains(slice []int, item int) bool {
	for _, v := range slice {
		if v == item {
			return true
		}
	}
	return false
}

func getPositionNames(gameData *GameData, positions []int) []string {
	names := make([]string, len(positions))
	for i, pos := range positions {
		names[i] = gameData.Positions[pos].Name
	}
	return names
}

func capitalizePosition(pos string) string {
	switch pos {
	case "safelane":
		return "Safelane"
	case "midlane":
		return "Midlane"
	case "offlane":
		return "Offlane"
	case "soft_support":
		return "Soft Support"
	case "hard_support":
		return "Hard Support"
	default:
		return pos
	}
}
