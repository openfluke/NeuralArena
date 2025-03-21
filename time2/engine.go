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

// StockData matches the VIX CSV structure exactly
type StockData struct {
	Date  string  `csv:"DATE"`
	Open  float64 `csv:"OPEN"`
	High  float64 `csv:"HIGH"`
	Low   float64 `csv:"LOW"`
	Close float64 `csv:"CLOSE"`
}

// discretizePriceChange converts a price change into a discrete token (0 = down, 1 = flat, 2 = up)
func discretizePriceChange(prev, curr float64) int {
	change := (curr - prev) / prev * 100
	if math.IsNaN(change) || math.IsInf(change, 0) {
		return 1 // Default to flat for invalid changes
	}
	if change < -2.0 {
		return 0 // Down
	} else if change > 2.0 {
		return 2 // Up
	}
	return 1 // Flat
}

// downloadStockData downloads a CSV from a URL
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

// loadStockData reads the CSV file
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

// prepareData converts stock data into input-output pairs
func prepareData(stockData []StockData, seqLength int) (inputs [][][]float64, targets [][][]float64) {
	for i, j := 0, len(stockData)-1; i < j; i, j = i+1, j-1 {
		stockData[i], stockData[j] = stockData[j], stockData[i]
	}

	changes := make([]int, len(stockData)-1)
	for i := 1; i < len(stockData); i++ {
		change := (stockData[i].Close - stockData[i-1].Close) / stockData[i-1].Close * 100
		changes[i-1] = discretizePriceChange(stockData[i-1].Close, stockData[i].Close)
		if i < 10 {
			fmt.Printf("Day %d: Prev %.2f, Curr %.2f, Change %.2f%%, Class %d\n",
				i, stockData[i-1].Close, stockData[i].Close, change, changes[i-1])
		}
	}

	counts := [3]int{}
	for _, c := range changes {
		counts[c]++
	}
	fmt.Printf("Class distribution - Down: %d, Flat: %d, Up: %d\n", counts[0], counts[1], counts[2])

	inputs = make([][][]float64, 0)
	targets = make([][][]float64, 0)
	for i := seqLength; i < len(changes); i++ {
		input := make([][]float64, seqLength)
		for j := 0; j < seqLength; j++ {
			input[j] = make([]float64, 3)
			input[j][changes[i-seqLength+j]] = 1.0
		}
		target := make([][]float64, 1)
		target[0] = make([]float64, 3)
		target[0][changes[i]] = 1.0
		inputs = append(inputs, input)
		targets = append(targets, target)
	}
	return inputs, targets
}

// argMax finds the index of the maximum value in a slice
func argMax(arr []float64) int {
	maxIdx := 0
	for i := 1; i < len(arr); i++ {
		if arr[i] > arr[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

func main() {
	rand.Seed(time.Now().UnixNano())

	filename := "vix-daily.csv"
	dataURL := "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"
	seqLength := 30

	// Remove any stale AAPL.csv to avoid confusion
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

	inputs, targets := prepareData(stockData, seqLength)
	if len(inputs) == 0 {
		fmt.Println("Not enough data to create training sequences")
		return
	}
	fmt.Printf("Prepared %d training samples\n", len(inputs))

	trainSize := int(0.8 * float64(len(inputs)))
	perm := rand.Perm(len(inputs))
	trainInputs := make([][][]float64, trainSize)
	trainTargets := make([][][]float64, trainSize)
	testInputs := make([][][]float64, len(inputs)-trainSize)
	testTargets := make([][][]float64, len(inputs)-trainSize)
	for i, p := range perm {
		if i < trainSize {
			trainInputs[i] = inputs[p]
			trainTargets[i] = targets[p]
		} else {
			testInputs[i-trainSize] = inputs[p]
			testTargets[i-trainSize] = targets[p]
		}
	}
	fmt.Printf("Training samples: %d, Test samples: %d\n", len(trainInputs), len(testInputs))

	layerSizes := []struct{ Width, Height int }{
		{3, seqLength},
		{128, 1},
		{3, 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	nn := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	fmt.Println("Starting training with Backward...")
	/*for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		perm := rand.Perm(len(trainInputs))
		for i, p := range perm {
			nn.Forward(trainInputs[p])
			loss := nn.ComputeLoss(trainTargets[p])
			if math.IsNaN(loss) {
				fmt.Printf("NaN loss at epoch %d, sample %d\n", epoch, i)
				continue
			}
			totalLoss += loss
			nn.Backward(trainTargets[p], learningRate)
		}
		avgLoss := totalLoss / float64(len(trainInputs))
		if epoch%2 == 0 || epoch == epochs-1 {
			trainAcc := computeAccuracy(nn, trainInputs, trainTargets)
			testAcc := computeAccuracy(nn, testInputs, testTargets)
			fmt.Printf("Epoch %d, Loss: %.4f, Train Acc: %.2f%%, Test Acc: %.2f%%\n",
				epoch, avgLoss, trainAcc*100, testAcc*100)
		}
	}*/

	// Improved sub-network hyperparameters:
	// Improved configuration using vector input (assuming layer 1 width is 128)
	// Improved sub-network hyperparameters:
	// Define sub-network architecture for dimensional neurons
	subLayerSizes := []struct{ Width, Height int }{
		{1, 1},  // Input: single scalar from parent neuron
		{16, 1}, // Hidden layer 1
		{16, 1}, // Hidden layer 2
		{1, 1},  // Output: single scalar
	}
	subActivations := []string{"linear", "relu", "relu", "linear"}
	subFullyConnected := []bool{true, true, true, true}

	// Define options for sub-network initialization
	opts := paragon.SetLayerDimensionOptions{
		Shared:     false,    // Each neuron gets its own sub-network instance
		InitMethod: "xavier", // Use Xavier initialization for better convergence
	}

	// Assign sub-networks to layer 1 with options
	nn.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	nn.Train(inputs, targets, 10, 0.001)

	// Compute accuracy on training data
	trainAcc := computeAccuracy(nn, trainInputs, trainTargets)

	// Compute accuracy on testing data
	testAcc := computeAccuracy(nn, testInputs, testTargets)

	// Print the accuracies
	fmt.Printf("Training Accuracy: %.2f%%\n", trainAcc*100)
	fmt.Printf("Testing Accuracy: %.2f%%\n", testAcc*100)

	predictFuture(nn, stockData, seqLength, 7, "week")
	predictFuture(nn, stockData, seqLength, 90, "quarter")
}

// computeAccuracy calculates accuracy on a dataset
func computeAccuracy(nn *paragon.Network, inputs [][][]float64, targets [][][]float64) float64 {
	correct := 0
	for i := range inputs {
		nn.Forward(inputs[i])
		output := nn.Layers[nn.OutputLayer].Neurons[0]
		pred := argMax([]float64{output[0].Value, output[1].Value, output[2].Value})
		label := argMax(targets[i][0])
		if pred == label {
			correct++
		}
	}
	return float64(correct) / float64(len(inputs))
}

// predictFuture generates predictions for a given number of days
func predictFuture(nn *paragon.Network, stockData []StockData, seqLength, days int, period string) {
	fmt.Printf("\nPredicting next %s (%d days):\n", period, days)
	changes := make([]int, len(stockData)-1)
	for i := 1; i < len(stockData); i++ {
		changes[i-1] = discretizePriceChange(stockData[i-1].Close, stockData[i].Close)
	}
	if len(changes) < seqLength {
		fmt.Println("Not enough historical data for prediction")
		return
	}

	currentSequence := make([]int, seqLength)
	copy(currentSequence, changes[len(changes)-seqLength:])
	predictions := make([]int, days)

	for i := 0; i < days; i++ {
		input := make([][]float64, seqLength)
		for j := 0; j < seqLength; j++ {
			input[j] = make([]float64, 3)
			input[j][currentSequence[j]] = 1.0
		}
		nn.Forward(input)
		output := nn.Layers[nn.OutputLayer].Neurons[0]
		probs := []float64{output[0].Value, output[1].Value, output[2].Value}
		pred := argMax(probs)
		predictions[i] = pred
		copy(currentSequence[:seqLength-1], currentSequence[1:])
		currentSequence[seqLength-1] = pred
	}

	for i, pred := range predictions {
		var movement string
		switch pred {
		case 0:
			movement = "down"
		case 1:
			movement = "flat"
		case 2:
			movement = "up"
		}
		if days <= 7 {
			fmt.Printf("Day %d: %s\n", i+1, movement)
		} else {
			if i%7 == 0 {
				fmt.Printf("Week %d: ", i/7+1)
			}
			fmt.Printf("%s ", movement)
			if i%7 == 6 || i == len(predictions)-1 {
				fmt.Println()
			}
		}
	}
}
