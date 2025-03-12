package main

import (
	"archive/zip"
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"

	"paragon" // Replace with your Paragon package path
)

func main() {
	// Set random seed for reproducibility
	rand.Seed(42)

	// Download and extract the Bank Marketing dataset
	url := "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
	resp, err := http.Get(url)
	if err != nil {
		panic(fmt.Errorf("failed to download Bank Marketing dataset: %v", err))
	}
	defer resp.Body.Close()

	// Save ZIP file temporarily
	zipFile, err := os.Create("bank.zip")
	if err != nil {
		panic(fmt.Errorf("failed to create zip file: %v", err))
	}
	_, err = io.Copy(zipFile, resp.Body)
	zipFile.Close()
	if err != nil {
		panic(fmt.Errorf("failed to write zip file: %v", err))
	}
	defer os.Remove("bank.zip")

	// Extract bank.csv from ZIP
	r, err := zip.OpenReader("bank.zip")
	if err != nil {
		panic(fmt.Errorf("failed to open zip: %v", err))
	}
	defer r.Close()

	var csvFile io.ReadCloser
	for _, f := range r.File {
		if f.Name == "bank.csv" {
			csvFile, err = f.Open()
			if err != nil {
				panic(fmt.Errorf("failed to open bank.csv: %v", err))
			}
			break
		}
	}
	if csvFile == nil {
		panic("bank.csv not found in zip")
	}
	defer csvFile.Close()

	// Parse CSV
	reader := csv.NewReader(csvFile)
	reader.Comma = ';' // Bank CSV uses semicolon delimiter
	records, err := reader.ReadAll()
	if err != nil {
		panic(fmt.Errorf("failed to parse CSV: %v", err))
	}

	// Prepare data: Use age, balance, and duration as features, predict subscription (yes/no)
	inputs := make([][][]float64, 0)
	targets := make([][][]float64, 0)
	for i, record := range records[1:] { // Skip header
		if len(record) < 17 { // Ensure row has all columns
			continue
		}
		// Columns: age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome, y
		age, err := strconv.ParseFloat(strings.TrimSpace(record[0]), 64)
		if err != nil {
			continue
		}
		balance, err := strconv.ParseFloat(strings.TrimSpace(record[5]), 64)
		if err != nil {
			continue
		}
		duration, err := strconv.ParseFloat(strings.TrimSpace(record[11]), 64)
		if err != nil {
			continue
		}
		outcome := strings.TrimSpace(record[16]) // "yes" or "no"

		// Normalize features
		input := []float64{
			age / 100.0,       // Scale age (e.g., 18-95)
			balance / 10000.0, // Scale balance (can be negative or large)
			duration / 1000.0, // Scale call duration (seconds)
		}
		inputs = append(inputs, [][]float64{input})

		// Target: 1 for "yes", 0 for "no"
		target := make([]float64, 2) // Binary classification
		if outcome == "yes" {
			target[1] = 1.0
		} else {
			target[0] = 1.0
		}
		targets = append(targets, [][]float64{target})

		// Limit to 1000 samples for demo speed
		if i >= 999 {
			break
		}
	}

	// Shuffle and split into training (80%) and testing (20%) sets
	perm := rand.Perm(len(inputs))
	split := int(0.8 * float64(len(inputs)))
	trainingIndices := perm[:split]
	testingIndices := perm[split:]

	// Create training and testing sets
	trainingInputs := make([][][]float64, 0, split)
	trainingTargets := make([][][]float64, 0, split)
	for _, idx := range trainingIndices {
		trainingInputs = append(trainingInputs, inputs[idx])
		trainingTargets = append(trainingTargets, targets[idx])
	}
	testingInputs := make([][][]float64, 0, len(inputs)-split)
	testingTargets := make([][][]float64, 0, len(inputs)-split)
	for _, idx := range testingIndices {
		testingInputs = append(testingInputs, inputs[idx])
		testingTargets = append(testingTargets, targets[idx])
	}

	// Select 5 random samples from the testing set for output display
	sampleIndices := rand.Perm(len(testingInputs))[:5]

	// Define network architecture
	layerSizes := []struct{ Width, Height int }{
		{3, 1}, // Input layer (age, balance, duration)
		{8, 1}, // Hidden layer
		{2, 1}, // Output layer (yes/no subscription)
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	// Create and train the network on the training set
	net := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fmt.Println("Training on Bank Marketing dataset...")
	net.Train(trainingInputs, trainingTargets, 50, 0.01) // 50 epochs, learning rate 0.01

	// Evaluate on both training and testing sets after training
	fmt.Println("---------- Trained Model ----------")
	trainAccuracy := evaluateAccuracy(net, trainingInputs, trainingTargets)
	testAccuracy := evaluateAccuracy(net, testingInputs, testingTargets)
	fmt.Printf("Training Accuracy: %.2f%%\n", trainAccuracy)
	fmt.Printf("Testing Accuracy: %.2f%%\n", testAccuracy)
	printSampleOutputs(net, testingInputs, testingTargets, sampleIndices)

	// Save to JSON and gob
	if err := net.SaveToJSON("model.json"); err != nil {
		panic(fmt.Errorf("failed to save model to JSON: %v", err))
	}
	fmt.Println("Model saved to model.json")
	if err := net.SaveToGob("model.gob"); err != nil {
		panic(fmt.Errorf("failed to save model to gob: %v", err))
	}
	fmt.Println("Model saved to model.gob")

	// Load from JSON and evaluate
	fmt.Println("---------- Loaded from JSON ----------")
	jsonNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	if err := jsonNet.LoadFromJSON("model.json"); err != nil {
		panic(fmt.Errorf("failed to load model from JSON: %v", err))
	}
	jsonTrainAccuracy := evaluateAccuracy(jsonNet, trainingInputs, trainingTargets)
	jsonTestAccuracy := evaluateAccuracy(jsonNet, testingInputs, testingTargets)
	fmt.Printf("Training Accuracy: %.2f%%\n", jsonTrainAccuracy)
	fmt.Printf("Testing Accuracy: %.2f%%\n", jsonTestAccuracy)
	printSampleOutputs(jsonNet, testingInputs, testingTargets, sampleIndices)

	// Load from gob and evaluate
	fmt.Println("---------- Loaded from gob ----------")
	gobNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	if err := gobNet.LoadFromGob("model.gob"); err != nil {
		panic(fmt.Errorf("failed to load model from gob: %v", err))
	}
	gobTrainAccuracy := evaluateAccuracy(gobNet, trainingInputs, trainingTargets)
	gobTestAccuracy := evaluateAccuracy(gobNet, testingInputs, testingTargets)
	fmt.Printf("Training Accuracy: %.2f%%\n", gobTrainAccuracy)
	fmt.Printf("Testing Accuracy: %.2f%%\n", gobTestAccuracy)
	printSampleOutputs(gobNet, testingInputs, testingTargets, sampleIndices)
}

// evaluateAccuracy computes the accuracy of the network on the given inputs and targets.
func evaluateAccuracy(net *paragon.Network, inputs [][][]float64, targets [][][]float64) float64 {
	correct := 0
	for i := range inputs {
		net.Forward(inputs[i])
		output := net.Layers[net.OutputLayer].NeuronsToValues()[0]
		pred := maxIndex(output)
		target := maxIndex(targets[i][0])
		if pred == target {
			correct++
		}
	}
	return float64(correct) / float64(len(inputs)) * 100
}

// printSampleOutputs displays the input features, predicted output neurons, and actual targets for the given sample indices.
func printSampleOutputs(net *paragon.Network, inputs [][][]float64, targets [][][]float64, indices []int) {
	for _, idx := range indices {
		input := inputs[idx][0]
		age := input[0] * 100.0
		balance := input[1] * 10000.0
		duration := input[2] * 1000.0
		net.Forward(inputs[idx])
		output := net.Layers[net.OutputLayer].NeuronsToValues()[0]
		predClass := maxIndex(output)
		predLabel := "no"
		if predClass == 1 {
			predLabel = "yes"
		}
		actualClass := maxIndex(targets[idx][0])
		actualLabel := "no"
		if actualClass == 1 {
			actualLabel = "yes"
		}
		fmt.Printf("Sample %d: Age=%.0f, Balance=%.2f, Duration=%.0f\n", idx+1, age, balance, duration)
		fmt.Printf("  Predicted Output Neurons: %v (class %d, %s)\n", output, predClass, predLabel)
		fmt.Printf("  Actual Output: %v (class %d, %s)\n", targets[idx][0], actualClass, actualLabel)
	}
}

// maxIndex returns the index of the maximum value in a slice.
func maxIndex(slice []float64) int {
	maxVal, maxIdx := slice[0], 0
	for i, val := range slice[1:] {
		if val > maxVal {
			maxVal, maxIdx = val, i+1
		}
	}
	return maxIdx
}
