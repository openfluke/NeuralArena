package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"time"

	"paragon"

	"github.com/gocarina/gocsv"
)

// StockData represents a single day’s stock data from Alpha Vantage CSV
type StockData struct {
	Timestamp string  `csv:"timestamp"`
	Open      float64 `csv:"open"`
	High      float64 `csv:"high"`
	Low       float64 `csv:"low"`
	Close     float64 `csv:"close"`
	Volume    int     `csv:"volume"`
}

// discretizePriceChange converts a price change into a discrete token (0 = down, 1 = flat, 2 = up)
func discretizePriceChange(prev, curr float64) int {
	change := (curr - prev) / prev * 100 // Percentage change
	if change < -0.5 {
		return 0 // Down
	} else if change > 0.5 {
		return 2 // Up
	}
	return 1 // Flat
}

// fetchStockData downloads daily stock data and returns it as a slice of StockData
func fetchStockData(symbol, apiKey string) ([]StockData, error) {
	url := fmt.Sprintf("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&apikey=%s&datatype=csv", symbol, apiKey)
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch data: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status: %s", resp.Status)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	// Save to temporary file for CSV parsing
	tmpFile, err := os.CreateTemp("", "stockdata-*.csv")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	if _, err := tmpFile.Write(data); err != nil {
		return nil, fmt.Errorf("failed to write to temp file: %v", err)
	}
	tmpFile.Close()

	file, err := os.Open(tmpFile.Name())
	if err != nil {
		return nil, fmt.Errorf("failed to open temp file: %v", err)
	}
	defer file.Close()

	var stockData []StockData
	if err := gocsv.UnmarshalFile(file, &stockData); err != nil {
		return nil, fmt.Errorf("failed to parse CSV: %v", err)
	}

	return stockData, nil
}

// prepareTrainingData converts stock data into a sequence of discrete tokens
func prepareTrainingData(stockData []StockData, seqLength int) [][]int {
	// Reverse data to have oldest first
	for i, j := 0, len(stockData)-1; i < j; i, j = i+1, j-1 {
		stockData[i], stockData[j] = stockData[j], stockData[i]
	}

	// Convert to price change directions
	changes := make([]int, len(stockData)-1)
	for i := 1; i < len(stockData); i++ {
		changes[i-1] = discretizePriceChange(stockData[i-1].Close, stockData[i].Close)
	}

	// Split into sequences of seqLength
	sequences := make([][]int, 0)
	for i := 0; i <= len(changes)-seqLength; i++ {
		sequences = append(sequences, changes[i:i+seqLength])
	}
	return sequences
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Configuration
	symbol := "AAPL"                   // Apple stock
	apiKey := "YOUR_ALPHA_VANTAGE_KEY" // Replace with your Alpha Vantage API key
	seqLength := 30                    // Use 30 days as input sequence

	// Fetch stock data
	fmt.Println("Fetching stock data...")
	stockData, err := fetchStockData(symbol, apiKey)
	if err != nil {
		fmt.Printf("Error fetching data: %v\n", err)
		return
	}
	fmt.Printf("Downloaded %d days of data\n", len(stockData))

	// Prepare training data
	trainData := prepareTrainingData(stockData, seqLength)
	if len(trainData) == 0 {
		fmt.Println("Not enough data to create training sequences")
		return
	}
	fmt.Printf("Prepared %d training sequences of length %d\n", len(trainData), seqLength)

	// Transformer configuration
	tConfig := paragon.TransformerConfig{
		DModel:      64,
		NHeads:      4,
		NLayers:     1,
		FeedForward: 128,
		VocabSize:   4, // 0=down, 1=flat, 2=up, 3=[MASK]
		MaxLength:   seqLength,
		Activation:  "relu",
		GridRows:    0, // 1D sequence
		GridCols:    0,
	}

	// Diffusion configuration
	dConfig := paragon.DiffusionConfig{
		NumTimesteps:      50,
		MaxLength:         seqLength,
		LearningRate:      0.001,
		Epochs:            200,
		Temperature:       1.0,
		TopK:              3,
		MaskScheduleStart: 0.1,
		MaskScheduleEnd:   0.9,
	}

	// Initialize model
	network := paragon.NewTransformerEncoder(tConfig)
	tokenizer := paragon.CustomTokenizer{
		Vocab:         map[string]int{"down": 0, "flat": 1, "up": 2, "[MASK]": 3},
		ReverseVocab:  map[int]string{0: "down", 1: "flat", 2: "up", 3: "[MASK]"},
		VocabSize:     4,
		SpecialTokens: map[int]bool{3: true},
	}
	model := paragon.NewDiffusionModel(network, dConfig, nil)
	model.Tokenizer = &tokenizer

	// Train the model
	fmt.Println("Starting training...")
	trainDiffusion(model, trainData, tConfig)

	// Generate predictions
	fmt.Println("\nGenerating predictions:")
	predictNextWeek(model, tConfig)
	predictNextQuarter(model, tConfig)
}

// trainDiffusion trains the model on sequences of price change directions
func trainDiffusion(model *paragon.DiffusionModel, samples [][]int, tConfig paragon.TransformerConfig) {
	data := make([][]int, len(samples))
	copy(data, samples)

	for epoch := 0; epoch < model.Config.Epochs; epoch++ {
		totalLoss := 0.0
		lr := model.Config.LearningRate * (1.0 - float64(epoch)/float64(model.Config.Epochs)) // Decay LR

		rand.Shuffle(len(data), func(i, j int) {
			data[i], data[j] = data[j], data[i]
		})

		for _, x0 := range data {
			t := rand.Intn(model.Config.NumTimesteps)
			xt := model.BetterAddNoise(x0, t)

			// Build one-hot input [MaxLength][VocabSize]
			batchInput := make([][]float64, model.Config.MaxLength)
			for i, tok := range xt {
				batchInput[i] = make([]float64, tConfig.VocabSize)
				if tok >= 0 && tok < tConfig.VocabSize {
					batchInput[i][tok] = 1.0
				}
			}

			// Forward pass
			output2D := model.Network.ForwardTransformer(batchInput)
			preds := output2D[0]

			// Compute loss on masked positions
			loss := 0.0
			maskedCount := 0
			errorTerms := make([]float64, model.Config.MaxLength*tConfig.VocabSize)
			maskID := model.Tokenizer.Vocab["[MASK]"]
			for i, tok := range xt {
				if tok == maskID {
					maskedCount++
					start := i * tConfig.VocabSize
					end := start + tConfig.VocabSize
					probs := paragon.Softmax(preds[start:end])
					target := x0[i]
					loss -= math.Log(math.Max(probs[target], 1e-10))
					for m := 0; m < tConfig.VocabSize; m++ {
						delta := probs[m]
						if m == target {
							delta -= 1.0
						}
						if delta > 5.0 {
							delta = 5.0
						} else if delta < -5.0 {
							delta = -5.0
						}
						errorTerms[start+m] = delta
					}
				}
			}
			if maskedCount > 0 {
				totalLoss += loss / float64(maskedCount)
			}

			// Reshape error terms
			shaped := make([][]float64, model.Config.MaxLength)
			for i := 0; i < model.Config.MaxLength; i++ {
				start := i * tConfig.VocabSize
				shaped[i] = errorTerms[start : start+tConfig.VocabSize]
			}
			model.Network.BackwardExternal(shaped, lr)
		}

		avgLoss := totalLoss / float64(len(data))
		if epoch%20 == 0 || epoch == model.Config.Epochs-1 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, avgLoss)
		}
	}
}

// predictNextWeek generates a 7-day prediction
func predictNextWeek(model *paragon.DiffusionModel, tConfig paragon.TransformerConfig) {
	fmt.Println("Predicting next week (7 days):")
	generated := model.GenerateBetter()
	weekPrediction := generated[len(generated)-7:] // Last 7 days
	for i, token := range weekPrediction {
		fmt.Printf("Day %d: %s\n", i+1, model.Tokenizer.ReverseVocab[token])
	}
}

// predictNextQuarter generates a 90-day prediction
func predictNextQuarter(model *paragon.DiffusionModel, tConfig paragon.TransformerConfig) {
	fmt.Println("\nPredicting next quarter (90 days):")
	generated := model.GenerateBetter()
	// Since seqLength might be less than 90, extrapolate by generating multiple sequences
	quarterPrediction := make([]int, 0, 90)
	for len(quarterPrediction) < 90 {
		generated = model.GenerateBetter()
		quarterPrediction = append(quarterPrediction, generated...)
	}
	quarterPrediction = quarterPrediction[:90] // Trim to 90 days
	for i := 0; i < len(quarterPrediction); i += 7 {
		end := i + 7
		if end > len(quarterPrediction) {
			end = len(quarterPrediction)
		}
		fmt.Printf("Week %d: %v\n", i/7+1, quarterPrediction[i:end])
	}
}
