package main

import (
	"fmt"
	"math"
	"math/rand"
	"paragon"
	"strings"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Sample dataset
	sentences := []string{
		"the quick brown fox jumps over the lazy dog",
		"a journey of a thousand miles begins with a single step",
		"to be or not to be that is the question",
	}

	// Build vocabulary
	vocab, reverseVocab := buildVocabulary(sentences)
	vocabSize := len(vocab)
	maxSeqLength := 10 // Maximum sequence length for simplicity
	totalSteps := 5    // Number of diffusion steps
	epochs := 100      // Training epochs
	learningRate := 0.01

	// Define network architecture: [sequence_length * vocab_size] -> [hidden] -> [sequence_length * vocab_size]
	layerSizes := []struct{ Width, Height int }{
		{Width: vocabSize * maxSeqLength, Height: 1}, // Input: flattened one-hot sequence
		{Width: 128, Height: 1},                      // Hidden layer
		{Width: vocabSize * maxSeqLength, Height: 1}, // Output: flattened probabilities
	}
	activations := []string{"linear", "relu", "linear"} // Changed "softmax" to "linear"; apply softmax manually
	fullyConnected := []bool{true, true, true}
	model := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for _, sentence := range sentences {
			sequence := tokenize(sentence, vocab)
			padded := padSequence(sequence, maxSeqLength, vocab["[PAD]"])
			t := rand.Intn(totalSteps + 1) // Random diffusion step
			masked := maskSequence(padded, t, totalSteps, vocab)
			inputFlat := flattenOneHot(sequenceToOneHot(masked, vocabSize, maxSeqLength))
			targetFlat := flattenOneHot(sequenceToOneHot(padded, vocabSize, maxSeqLength))

			// Forward pass
			model.Forward([][]float64{inputFlat}) // Wrap inputFlat in a 2D slice
			outputFlat := model.GetOutput()       // Retrieve output (assumes GetOutput exists)
			loss := computeLoss(outputFlat, targetFlat)
			totalLoss += loss

			// Backward pass
			model.Backward([][]float64{targetFlat}, learningRate) // Wrap targetFlat in a 2D slice
		}
		avgLoss := totalLoss / float64(len(sentences))
		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, avgLoss)
	}

	// Generate text
	generated := generate(model, vocab, reverseVocab, maxSeqLength, totalSteps, vocabSize)
	fmt.Println("Generated text:", generated)
}

// buildVocabulary creates a mapping of words to indices and vice versa
func buildVocabulary(sentences []string) (map[string]int, map[int]string) {
	vocab := make(map[string]int)
	reverseVocab := make(map[int]string)
	idx := 0
	for _, sentence := range sentences {
		words := strings.Fields(sentence)
		for _, word := range words {
			if _, exists := vocab[word]; !exists {
				vocab[word] = idx
				reverseVocab[idx] = word
				idx++
			}
		}
	}
	vocab["[PAD]"] = idx
	reverseVocab[idx] = "[PAD]"
	idx++
	vocab["[MASK]"] = idx
	reverseVocab[idx] = "[MASK]"
	return vocab, reverseVocab
}

// tokenize converts a sentence to a sequence of token indices
func tokenize(sentence string, vocab map[string]int) []int {
	words := strings.Fields(sentence)
	sequence := make([]int, len(words))
	for i, word := range words {
		sequence[i] = vocab[word]
	}
	return sequence
}

// padSequence pads or truncates a sequence to a fixed length
func padSequence(sequence []int, length int, padToken int) []int {
	padded := make([]int, length)
	for i := 0; i < length; i++ {
		if i < len(sequence) {
			padded[i] = sequence[i]
		} else {
			padded[i] = padToken
		}
	}
	return padded
}

// maskSequence applies a simple diffusion process by masking tokens
func maskSequence(sequence []int, step int, totalSteps int, vocab map[string]int) []int {
	masked := make([]int, len(sequence))
	copy(masked, sequence)
	maskProb := float64(step) / float64(totalSteps)
	for i := range masked {
		if rand.Float64() < maskProb {
			masked[i] = vocab["[MASK]"]
		}
	}
	return masked
}

// sequenceToOneHot converts a sequence to a one-hot encoded 2D slice
func sequenceToOneHot(sequence []int, vocabSize int, maxSeqLength int) [][]float64 {
	oneHot := make([][]float64, maxSeqLength)
	for i := 0; i < maxSeqLength; i++ {
		oneHot[i] = make([]float64, vocabSize)
		if i < len(sequence) {
			token := sequence[i]
			if token >= 0 && token < vocabSize {
				oneHot[i][token] = 1.0
			}
		}
	}
	return oneHot
}

// flattenOneHot flattens a 2D one-hot encoded sequence into a 1D slice
func flattenOneHot(oneHot [][]float64) []float64 {
	flat := make([]float64, len(oneHot)*len(oneHot[0]))
	for i := 0; i < len(oneHot); i++ {
		for j := 0; j < len(oneHot[i]); j++ {
			flat[i*len(oneHot[0])+j] = oneHot[i][j]
		}
	}
	return flat
}

// computeLoss calculates cross-entropy loss between output and target
func computeLoss(output, target []float64) float64 {
	loss := 0.0
	for i := range output {
		outputVal := output[i]
		if outputVal <= 0 {
			outputVal = 1e-10 // Prevent log(0)
		}
		loss += -target[i] * math.Log(outputVal)
	}
	return loss
}

// generate produces text by iteratively denoising a masked sequence
func generate(model *paragon.Network, vocab map[string]int, reverseVocab map[int]string, maxSeqLength, steps, vocabSize int) string {
	sequence := make([]int, maxSeqLength)
	for i := range sequence {
		sequence[i] = vocab["[MASK]"] // Start fully masked
	}
	for step := steps; step > 0; step-- {
		inputFlat := flattenOneHot(sequenceToOneHot(sequence, vocabSize, maxSeqLength))
		model.Forward([][]float64{inputFlat}) // Wrap inputFlat in a 2D slice
		outputFlat := model.GetOutput()       // Retrieve output
		// Reshape output to [maxSeqLength][vocabSize]
		output := make([][]float64, maxSeqLength)
		for i := 0; i < maxSeqLength; i++ {
			output[i] = outputFlat[i*vocabSize : (i+1)*vocabSize]
			probs := softmax(output[i]) // Apply softmax manually
			sequence[i] = argMax(probs)
		}
	}
	// Decode sequence to text
	words := make([]string, 0)
	for _, token := range sequence {
		if token != vocab["[PAD]"] && token != vocab["[MASK]"] {
			words = append(words, reverseVocab[token])
		}
	}
	return strings.Join(words, " ")
}

// softmax applies the softmax function to a slice
func softmax(input []float64) []float64 {
	output := make([]float64, len(input))
	sum := 0.0
	maxVal := input[0]
	for _, v := range input {
		if v > maxVal {
			maxVal = v
		}
	}
	for i, v := range input {
		output[i] = math.Exp(v - maxVal)
		sum += output[i]
	}
	for i := range output {
		output[i] /= sum
	}
	return output
}

// argMax returns the index of the maximum value in a slice
func argMax(slice []float64) int {
	maxIdx := 0
	for i := 1; i < len(slice); i++ {
		if slice[i] > slice[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}
