package main

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"

	"paragon"
)

// Experiment60 performs a quick NLP text generation experiment.
func Experiment60(file *os.File) {
	fmt.Println("\n=== Experiment 60: Quick NLP Text Generation ===")

	// Step 1: Download text data
	fmt.Println("Downloading text data...")
	url := "http://www.gutenberg.org/files/1342/1342-0.txt" // Pride and Prejudice by Jane Austen (small text)
	resp, err := http.Get(url)
	if err != nil {
		panic(fmt.Sprintf("Failed to download text: %v", err))
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(fmt.Sprintf("Failed to read downloaded text: %v", err))
	}

	// Clean the text and split into lines
	text := string(body)
	lines := strings.Split(text, "\n")
	// Filter out empty lines and limit to first 1000 lines for quick training
	var filteredLines []string
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" && i < 1000 {
			filteredLines = append(filteredLines, trimmed)
		}
	}
	fmt.Printf("Loaded %d lines of text\n", len(filteredLines))

	// Step 2: Tokenize the text
	tokenizer := paragon.NewCustomTokenizer(filteredLines)
	fmt.Printf("Vocabulary size: %d\n", tokenizer.VocabSize)

	// Step 3: Set up the transformer configuration
	config := paragon.TransformerConfig{
		DModel:      64,  // Small model dimension
		NHeads:      2,   // Few attention heads
		NLayers:     1,   // Single hidden layer
		FeedForward: 128, // Small feed-forward size
		VocabSize:   tokenizer.VocabSize,
		MaxLength:   10, // Short sequence length
		Activation:  "relu",
	}

	// Create the transformer network
	net := paragon.NewTransformerEncoder(config)

	// Step 4: Prepare training data
	var sequences [][]int
	for _, line := range filteredLines {
		tokens := tokenizer.Encode(line)
		if len(tokens) > config.MaxLength {
			tokens = tokens[:config.MaxLength]
		} else {
			for len(tokens) < config.MaxLength {
				tokens = append(tokens, tokenizer.Vocab["[PAD]"])
			}
		}
		sequences = append(sequences, tokens)
	}

	// Convert sequences to one-hot encoded inputs and targets
	var inputs [][][]float64
	var targets [][][]float64
	for _, seq := range sequences {
		input := make([][]float64, config.MaxLength)
		target := make([][]float64, config.MaxLength)
		for t := 0; t < config.MaxLength; t++ {
			input[t] = make([]float64, config.VocabSize)
			target[t] = make([]float64, config.VocabSize)
			if t < len(seq)-1 {
				input[t][seq[t]] = 1.0
				target[t][seq[t+1]] = 1.0
			} else {
				input[t][tokenizer.Vocab["[PAD]"]] = 1.0
				target[t][tokenizer.Vocab["[PAD]"]] = 1.0
			}
		}
		inputs = append(inputs, input)
		targets = append(targets, target)
	}

	// Step 5: Train the network for 2 epochs
	for epoch := 0; epoch < 2; epoch++ {
		totalLoss := 0.0
		for i := range inputs {
			// Forward pass
			outputFlat := net.ForwardTransformer(inputs[i]) // [1][MaxLength*VocabSize]
			output := make([][]float64, config.MaxLength)
			for t := 0; t < config.MaxLength; t++ {
				start := t * config.VocabSize
				end := start + config.VocabSize
				output[t] = outputFlat[0][start:end]
			}

			// Compute loss and error terms
			loss := 0.0
			errorTerms := make([][]float64, config.MaxLength)
			for t := 0; t < config.MaxLength; t++ {
				probs := paragon.Softmax(output[t])
				target := targets[i][t]
				for j := 0; j < config.VocabSize; j++ {
					loss -= target[j] * math.Log(math.Max(probs[j], 1e-10))
				}
				errorTerms[t] = make([]float64, config.VocabSize)
				for j := 0; j < config.VocabSize; j++ {
					errorTerms[t][j] = probs[j] - target[j]
				}
			}
			totalLoss += loss

			// Backward pass
			net.BackwardExternal(errorTerms, 0.001) // Learning rate 0.001
		}
		avgLoss := totalLoss / float64(len(inputs))
		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, avgLoss)
	}

	// Step 6: Generate text
	fmt.Println("Generating text...")
	seed := sequences[0][:5] // Use first 5 tokens of the first sequence as seed
	current := make([]int, len(seed))
	copy(current, seed)
	for t := len(seed); t < config.MaxLength; t++ {
		input := make([][]float64, config.MaxLength)
		for i := 0; i < config.MaxLength; i++ {
			input[i] = make([]float64, config.VocabSize)
			if i < len(current) {
				input[i][current[i]] = 1.0
			} else {
				input[i][tokenizer.Vocab["[PAD]"]] = 1.0
			}
		}
		outputFlat := net.ForwardTransformer(input)
		output := make([][]float64, config.MaxLength)
		for i := 0; i < config.MaxLength; i++ {
			start := i * config.VocabSize
			end := start + config.VocabSize
			output[i] = outputFlat[0][start:end]
		}
		probs := paragon.Softmax(output[len(current)-1]) // Predict next token
		nextToken := paragon.ArgMax(probs)
		current = append(current, nextToken)
	}
	generatedText := tokenizer.Decode(current)
	fmt.Println("Generated text:", generatedText)

	// Write result to file
	result := fmt.Sprintf("Experiment 60 Generated Text: %s\n", generatedText)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}
