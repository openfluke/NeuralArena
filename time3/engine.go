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

type StockData struct {
	Date  string  `csv:"DATE"`
	Open  float64 `csv:"OPEN"`
	High  float64 `csv:"HIGH"`
	Low   float64 `csv:"LOW"`
	Close float64 `csv:"CLOSE"`
}

func discretizePriceChange(prev, curr float64) int {
	change := (curr - prev) / prev * 100
	if math.IsNaN(change) || math.IsInf(change, 0) {
		return 1
	}
	if change < -2.0 {
		return 0
	} else if change > 2.0 {
		return 2
	}
	return 1
}

func downloadStockData(url, filename string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download %s: %v", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %v", filename, err)
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func loadStockData(filename string) ([]StockData, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	var stockData []StockData
	if err := gocsv.UnmarshalFile(file, &stockData); err != nil {
		return nil, fmt.Errorf("failed to parse CSV: %v", err)
	}
	return stockData, nil
}

func prepareTrainingData(stockData []StockData, seqLength int) [][]int {
	for i, j := 0, len(stockData)-1; i < j; i, j = i+1, j-1 {
		stockData[i], stockData[j] = stockData[j], stockData[i]
	}

	changes := make([]int, len(stockData)-1)
	for i := 1; i < len(stockData); i++ {
		changes[i-1] = discretizePriceChange(stockData[i-1].Close, stockData[i].Close)
	}

	for i := 0; i < 9 && i < len(changes); i++ {
		fmt.Printf("Day %d: Prev %.2f, Curr %.2f, Change %.2f%%, Class %d\n",
			i+1, stockData[i].Close, stockData[i+1].Close, (stockData[i+1].Close-stockData[i].Close)/stockData[i].Close*100, changes[i])
	}

	counts := [3]int{}
	for _, c := range changes {
		counts[c]++
	}
	fmt.Printf("Class distribution - Down: %d, Flat: %d, Up: %d\n", counts[0], counts[1], counts[2])

	sequences := make([][]int, 0)
	for i := 0; i <= len(changes)-seqLength; i++ {
		sequences = append(sequences, changes[i:i+seqLength])
	}
	return sequences
}

func main() {
	rand.Seed(time.Now().UnixNano())

	filename := "vix-daily.csv"
	dataURL := "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"
	seqLength := 20

	os.Remove("AAPL.csv")

	if _, err := os.Stat(filename); os.IsNotExist(err) {
		fmt.Println("Downloading stock data...")
		if err := downloadStockData(dataURL, filename); err != nil {
			fmt.Printf("Error downloading data: %v\nFalling back to local file or exiting.\n", err)
			return
		}
		fmt.Println("Download complete.")
	}

	fmt.Println("Loading stock data...")
	stockData, err := loadStockData(filename)
	if err != nil {
		fmt.Printf("Error loading data: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d days of data\n", len(stockData))

	trainData := prepareTrainingData(stockData, seqLength)
	if len(trainData) == 0 {
		fmt.Println("Not enough data to create training sequences")
		return
	}
	fmt.Printf("Prepared %d training sequences of length %d\n", len(trainData), seqLength)

	tConfig := paragon.TransformerConfig{
		DModel:      32,
		NHeads:      4,
		NLayers:     1,
		FeedForward: 64,
		VocabSize:   4,
		MaxLength:   seqLength,
		Activation:  "relu",
		GridRows:    0,
		GridCols:    0,
	}

	dConfig := paragon.DiffusionConfig{
		NumTimesteps:      15,
		MaxLength:         seqLength,
		LearningRate:      0.001,
		Epochs:            20,
		Temperature:       1.0,
		TopK:              3,
		MaskScheduleStart: 0.1,
		MaskScheduleEnd:   0.9,
	}

	network := paragon.NewTransformerEncoder(tConfig)
	tokenizer := paragon.CustomTokenizer{
		Vocab:         map[string]int{"down": 0, "flat": 1, "up": 2, "[MASK]": 3},
		ReverseVocab:  map[int]string{0: "down", 1: "flat", 2: "up", 3: "[MASK]"},
		VocabSize:     4,
		SpecialTokens: map[int]bool{3: true},
	}
	model := paragon.NewDiffusionModel(network, dConfig, nil)
	model.Tokenizer = &tokenizer

	fmt.Println("Starting training with Diffusion...")
	startTime := time.Now()
	trainDiffusion(model, trainData, tConfig)
	fmt.Printf("Training took %v\n", time.Since(startTime))

	fmt.Println("\nGenerating predictions:")
	predictNextWeek(model, tConfig)
	predictNextQuarter(model, tConfig)
}

func trainDiffusion(model *paragon.DiffusionModel, samples [][]int, tConfig paragon.TransformerConfig) {
	data := make([][]int, len(samples))
	copy(data, samples)

	sampleSize := 100
	if sampleSize > len(data) {
		sampleSize = len(data)
	}

	for epoch := 0; epoch < model.Config.Epochs; epoch++ {
		totalLoss := 0.0
		lr := model.Config.LearningRate * (1.0 - float64(epoch)/float64(model.Config.Epochs))

		rand.Shuffle(len(data), func(i, j int) {
			data[i], data[j] = data[j], data[i]
		})

		for _, x0 := range data {
			t := rand.Intn(model.Config.NumTimesteps)
			xt := model.BetterAddNoise(x0, t)

			batchInput := make([][]float64, model.Config.MaxLength)
			for i, tok := range xt {
				batchInput[i] = make([]float64, tConfig.VocabSize)
				if tok >= 0 && tok < tConfig.VocabSize {
					batchInput[i][tok] = 1.0
				}
			}

			output2D := model.Network.ForwardTransformer(batchInput)
			preds := output2D[0]

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

			shaped := make([][]float64, model.Config.MaxLength)
			for i := 0; i < model.Config.MaxLength; i++ {
				start := i * tConfig.VocabSize
				shaped[i] = errorTerms[start : start+tConfig.VocabSize]
			}
			model.Network.BackwardExternal(shaped, lr)
		}

		correct := 0
		total := 0
		for i := 0; i < sampleSize; i++ {
			generated := model.GenerateBetter()
			if len(generated) != len(samples[i]) {
				continue
			}
			for j := 0; j < len(generated); j++ {
				if generated[j] == samples[i][j] {
					correct++
				}
				total++
			}
		}
		avgLoss := totalLoss / float64(len(data))
		accuracy := float64(correct) / float64(total) * 100
		fmt.Printf("Epoch %d, Loss: %.4f, Train Acc: %.2f%%\n", epoch, avgLoss, accuracy)
	}
}

func predictNextWeek(model *paragon.DiffusionModel, tConfig paragon.TransformerConfig) {
	fmt.Println("Predicting next week (7 days):")
	generated := model.GenerateBetter()
	weekPrediction := generated
	if len(generated) > 7 {
		weekPrediction = generated[len(generated)-7:]
	}
	for i, token := range weekPrediction {
		fmt.Printf("Day %d: %s\n", i+1, model.Tokenizer.ReverseVocab[token])
	}
}

func predictNextQuarter(model *paragon.DiffusionModel, tConfig paragon.TransformerConfig) {
	fmt.Println("\nPredicting next quarter (90 days):")
	quarterPrediction := make([]int, 0, 90)
	for len(quarterPrediction) < 90 {
		generated := model.GenerateBetter()
		quarterPrediction = append(quarterPrediction, generated...)
	}
	quarterPrediction = quarterPrediction[:90]
	for i := 0; i < len(quarterPrediction); i += 7 {
		end := i + 7
		if end > len(quarterPrediction) {
			end = len(quarterPrediction)
		}
		fmt.Printf("Week %d: ", i/7+1)
		for j := i; j < end; j++ {
			fmt.Printf("%s ", model.Tokenizer.ReverseVocab[quarterPrediction[j]])
		}
		fmt.Println()
	}
}
