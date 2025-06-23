package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"paragon"
	"strings"
	"sync"
	"time"
)

// Constants
const (
	dataDir             = "nlp_data"
	dataFile            = "corpus.txt"
	resultsFile         = "results.txt"
	vocab               = "abcdefghijklmnopqrstuvwxyz .,!?" // 30 characters
	vocabSize           = 31                                // Aligned with paragon
	inputWidth          = 100
	inputHeight         = 100
	outputWidth         = 10
	outputHeight        = 10
	hiddenWidth         = 5
	hiddenHeight        = 5
	hiddenLayers        = 3
	epochs              = 10
	learningRate        = 0.005
	maxChunks           = 3
	confidenceThreshold = 0.9
	maxDiffusionSteps   = 10
	varianceThreshold   = 0.01
)

// Global file and mutex for thread-safe writing to results.txt
var (
	file   *os.File
	fileMu sync.Mutex
)

// Character to index mapping
var charToIndex map[rune]int

func init() {
	charToIndex = make(map[rune]int)
	for i, c := range vocab {
		charToIndex[c] = i
	}
	// Add padding character
	charToIndex[0] = 30 // Null or padding
	log.Printf("vocabSize: %d, charToIndex['t']: %d, vocab: %q", vocabSize, charToIndex['t'], vocab)
}

// writeResult writes formatted output to results.txt with thread safety
func writeResult(format string, args ...interface{}) {
	fileMu.Lock()
	defer fileMu.Unlock()
	_, err := fmt.Fprintf(file, format, args...)
	if err != nil {
		log.Printf("Failed to write to results.txt: %v", err)
	}
	file.Sync()
}

// ensureTextCorpus downloads or loads a text corpus
func ensureTextCorpus(dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	path := fmt.Sprintf("%s/%s", dir, dataFile)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		url := "https://www.gutenberg.org/files/11/11-0.txt"
		resp, err := http.Get(url)
		if err != nil {
			return fmt.Errorf("failed to download corpus: %v", err)
		}
		defer resp.Body.Close()

		f, err := os.Create(path)
		if err != nil {
			return fmt.Errorf("failed to create corpus file: %v", err)
		}
		defer f.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := strings.ToLower(scanner.Text())
			var cleaned strings.Builder
			for _, r := range line {
				if _, ok := charToIndex[r]; ok {
					cleaned.WriteRune(r)
				}
			}
			if cleaned.Len() > 0 {
				fmt.Fprintln(f, cleaned.String())
			}
		}
		if err := scanner.Err(); err != nil {
			return fmt.Errorf("error scanning corpus: %v", err)
		}
	}
	return nil
}

// loadTextData loads and preprocesses the text corpus into 100x100 inputs and 10x10 targets
func loadTextData(dir string) ([][][]float64, [][][]float64, error) {
	path := fmt.Sprintf("%s/%s", dir, dataFile)
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	var text strings.Builder
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		text.WriteString(scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, err
	}

	data := text.String()
	minLen := inputWidth*inputHeight + outputWidth*outputHeight
	if len(data) < minLen {
		return nil, nil, fmt.Errorf("text too short")
	}

	inputs := make([][][]float64, 0, len(data)-minLen)
	targets := make([][][]float64, 0, len(data)-minLen)
	for i := 0; i <= len(data)-minLen; i += outputWidth * outputHeight {
		input := make([][]float64, vocabSize)
		for v := 0; v < vocabSize; v++ {
			input[v] = make([]float64, inputWidth*inputHeight)
		}
		for y := 0; y < inputHeight; y++ {
			for x := 0; x < inputWidth; x++ {
				idx := i + y*inputWidth + x
				if idx >= len(data) {
					input[vocabSize-1][y*inputWidth+x] = 1.0 // Padding
					continue
				}
				if charIdx, ok := charToIndex[rune(data[idx])]; ok {
					input[charIdx][y*inputWidth+x] = 1.0
				} else {
					input[vocabSize-1][y*inputWidth+x] = 1.0 // Padding
				}
			}
		}

		target := make([][]float64, vocabSize)
		for v := 0; v < vocabSize; v++ {
			target[v] = make([]float64, outputWidth*outputHeight)
		}
		for y := 0; y < outputHeight; y++ {
			for x := 0; x < outputWidth; x++ {
				idx := i + inputWidth*inputHeight + y*outputWidth + x
				if idx >= len(data) {
					target[vocabSize-1][y*outputWidth+x] = 1.0 // Padding
					continue
				}
				if charIdx, ok := charToIndex[rune(data[idx])]; ok {
					target[charIdx][y*outputWidth+x] = 1.0
				} else {
					target[vocabSize-1][y*outputWidth+x] = 1.0 // Padding
				}
			}
		}

		inputs = append(inputs, input)
		targets = append(targets, target)
	}

	return inputs, targets, nil
}

// createNetwork creates a network for baseline or static replay
func createNetwork(replayType string, seed int64) *paragon.Network[float32] {
	layers := []struct{ Width, Height int }{
		{inputWidth * inputHeight, vocabSize},   // 100x100x31
		{hiddenWidth, hiddenHeight},             // 5x5
		{hiddenWidth, hiddenHeight},             // 5x5
		{hiddenWidth, hiddenHeight},             // 5x5
		{outputWidth * outputHeight, vocabSize}, // 10x10x31
	}
	acts := make([]string, len(layers))
	for i := range acts {
		acts[i] = "leaky_relu"
	}
	acts[len(acts)-1] = "softmax"
	fc := make([]bool, len(layers))
	for i := range fc {
		fc[i] = true
	}

	net := paragon.NewNetwork[float32](layers, acts, fc, seed)

	if replayType == "static" {
		for i := 1; i < len(layers)-1; i++ {
			net.Layers[i].ReplayOffset = -1
			net.Layers[i].ReplayPhase = "after"
			net.Layers[i].MaxReplay = 3
		}
	}
	return net
}

// createNetworkTEMPORAL creates a network for dynamic replay with advanced mechanisms
func createNetworkTEMPORAL(replayType string, seed int64, gateType string, gateThreshold float64, replayBudget int, replayPhase string) *paragon.Network[float32] {
	layers := []struct{ Width, Height int }{
		{inputWidth * inputHeight, vocabSize},
		{hiddenWidth, hiddenHeight},
		{hiddenWidth, hiddenHeight},
		{hiddenWidth, hiddenHeight},
		{outputWidth * outputHeight, vocabSize},
	}
	acts := make([]string, len(layers))
	for i := range acts {
		acts[i] = "leaky_relu"
	}
	acts[len(acts)-1] = "softmax"
	fc := make([]bool, len(layers))
	for i := range fc {
		fc[i] = true
	}
	net := paragon.NewNetwork[float32](layers, acts, fc, seed)

	if replayType == "dynamic" {
		for i := 1; i < len(layers)-1; i++ {
			net.Layers[i].ReplayEnabled = true
			net.Layers[i].ReplayBudget = replayBudget
			net.Layers[i].ReplayPhase = replayPhase
			net.Layers[i].ReplayOffset = -1
			switch gateType {
			case "entropy":
				net.Layers[i].ReplayGateFunc = func(input [][]float32) float64 {
					outputs := net.Layers[i].CachedOutputs
					if len(outputs) == 0 {
						return 0.5
					}
					var sum float64
					for _, v := range outputs {
						v64 := float64(v)
						if v64 > 1e-10 {
							sum += v64 * math.Log(v64)
						}
					}
					entropy := -sum / float64(len(outputs))
					maxEntropy := math.Log(float64(len(outputs)))
					if maxEntropy == 0 {
						return 0.5
					}
					return math.Min(1.0, math.Max(0.0, entropy/maxEntropy))
				}
			case "variance":
				net.Layers[i].ReplayGateFunc = func(input [][]float32) float64 {
					outputs := net.Layers[i].CachedOutputs
					if len(outputs) == 0 {
						return 0.5
					}
					var mean, sumSq float64
					n := float64(len(outputs))
					for _, v := range outputs {
						v64 := float64(v)
						mean += v64 / n
						sumSq += v64 * v64
					}
					variance := sumSq/n - mean*mean
					return math.Min(1.0, math.Max(0.0, variance/0.1))
				}
			case "temporal":
				net.Layers[i].ReplayGateFunc = func(input [][]float32) float64 {
					outputs := net.Layers[i].CachedOutputs
					if len(outputs) == 0 {
						return 0.5
					}
					var mean, sumSq float64
					n := float64(len(outputs))
					for _, v := range outputs {
						v64 := float64(v)
						mean += v64 / n
						sumSq += v64 * v64
					}
					variance := sumSq/n - mean*mean
					return math.Min(1.0, math.Max(0.0, variance/0.1))
				}
			case "hybrid":
				net.Layers[i].ReplayGateFunc = func(input [][]float32) float64 {
					outputs := net.Layers[i].CachedOutputs
					if len(outputs) == 0 {
						return 0.5
					}
					var mean, sumSq float64
					n := float64(len(outputs))
					for _, v := range outputs {
						v64 := float64(v)
						mean += v64 / n
						sumSq += v64 * v64
					}
					variance := sumSq/n - mean*mean
					spatialScore := math.Min(1.0, math.Max(0.0, variance/0.1))
					var sum float64
					for _, v := range outputs {
						v64 := float64(v)
						if v64 > 1e-10 {
							sum += v64 * math.Log(v64)
						}
					}
					entropy := -sum / float64(len(outputs))
					maxEntropy := math.Log(float64(len(outputs)))
					temporalScore := math.Min(1.0, math.Max(0.0, entropy/maxEntropy))
					return 0.6*temporalScore + 0.4*spatialScore
				}
			}
			net.Layers[i].ReplayGateToReps = func(score float64) int {
				if score > gateThreshold {
					return int(math.Min(float64(replayBudget), math.Ceil(score*float64(replayBudget))))
				}
				return 0
			}
		}
	}
	return net
}

// generateChunk generates a 10x10 chunk using a diffusion-like process
func generateChunk(net *paragon.Network[float32], input [][]float64, vocabSize, maxSteps int) (string, bool) {
	chunk := make([]int, outputWidth*outputHeight)
	probs := make([][]float64, outputWidth*outputHeight)
	for i := range probs {
		probs[i] = make([]float64, vocabSize)
	}

	// Initialize with 't' in top-left
	chunk[0] = charToIndex['t']
	currentInput := make([][]float64, vocabSize)
	for v := 0; v < vocabSize; v++ {
		currentInput[v] = make([]float64, inputWidth*inputHeight)
	}
	currentInput[charToIndex['t']][0] = 1.0

	for step := 0; step < maxSteps; step++ {
		net.Forward(currentInput)
		output := net.GetOutput()

		avgConfidence := 0.0
		for i := 0; i < outputWidth*outputHeight; i++ {
			maxProb := 0.0
			maxIdx := 0
			for j := 0; j < vocabSize; j++ {
				prob := output[i*vocabSize+j]
				probs[i][j] = prob
				if prob > maxProb {
					maxProb = prob
					maxIdx = j
				}
			}
			chunk[i] = maxIdx
			avgConfidence += maxProb
		}
		avgConfidence /= float64(outputWidth * outputHeight)

		if avgConfidence >= confidenceThreshold {
			return chunkToText(chunk), true
		}

		// Update input for next iteration
		for v := 0; v < vocabSize; v++ {
			for j := range currentInput[v] {
				currentInput[v][j] = 0
			}
		}
		for i := 0; i < outputWidth*outputHeight; i++ {
			currentInput[chunk[i]][inputWidth*inputHeight-outputWidth*outputHeight+i] = 1.0
		}
	}

	return chunkToText(chunk), false
}

// chunkToText converts a chunk to a formatted string
func chunkToText(chunk []int) string {
	var sb strings.Builder
	for y := 0; y < outputHeight; y++ {
		for x := 0; x < outputWidth; x++ {
			idx := y*outputWidth + x
			if idx < len(chunk) && chunk[idx] >= 0 && chunk[idx] < vocabSize {
				if chunk[idx] == vocabSize-1 {
					sb.WriteRune('0') // Padding character
				} else {
					sb.WriteRune(rune(vocab[chunk[idx]]))
				}
			} else {
				sb.WriteRune('?')
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// computePerplexity calculates perplexity for 10x10 targets
func computePerplexity(net *paragon.Network[float32], inputs [][][]float64, targets [][][]float64) float64 {
	totalLogProb := 0.0
	totalChars := 0

	for i := range inputs {
		net.Forward(inputs[i])
		output := net.GetOutput()
		target := targets[i]
		for j := 0; j < outputWidth*outputHeight; j++ {
			for k := 0; k < vocabSize; k++ {
				if target[k][j] > 0 {
					prob := output[j*vocabSize+k]
					if prob <= 1e-10 {
						prob = 1e-10
					}
					totalLogProb += math.Log(prob)
					totalChars++
				}
			}
		}
	}

	if totalChars == 0 {
		return math.Inf(1)
	}
	return math.Exp(-totalLogProb / float64(totalChars))
}

func main() {
	var err error
	file, err = os.OpenFile(resultsFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Failed to open results.txt: %v", err)
	}
	defer file.Close()

	writeResult("\n=== NLP Test Run: %s ===\n", time.Now().Format("2006-01-02 15:04:05"))

	if err := ensureTextCorpus(dataDir); err != nil {
		log.Fatalf("Text corpus error: %v", err)
	}
	trainX, trainY, err := loadTextData(dataDir)
	if err != nil {
		log.Fatalf("Load data error: %v", err)
	}
	trainX, trainY, testX, testY := paragon.SplitDataset(trainX, trainY, 0.2)

	configs := []struct {
		replayType   string
		gateType     string
		threshold    float64
		replayBudget int
		replayPhase  string
	}{
		{replayType: "baseline"},
		{replayType: "static"},
		{replayType: "dynamic", gateType: "entropy", threshold: 0.5, replayBudget: 2, replayPhase: "before"},
		{replayType: "dynamic", gateType: "variance", threshold: 0.5, replayBudget: 2, replayPhase: "before"},
		{replayType: "dynamic", gateType: "temporal", threshold: 0.5, replayBudget: 2, replayPhase: "before"},
		{replayType: "dynamic", gateType: "hybrid", threshold: 0.5, replayBudget: 2, replayPhase: "before"},
	}

	maxWorkers := 1 // Reduced for 16GB RAM
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	type result struct {
		replayType string
		configDesc string
		epoch      int
		perplexity float64
		chunks     []string
	}
	resultsCh := make(chan result, len(configs)*epochs)

	for _, cfg := range configs {
		wg.Add(1)
		go func(cfg struct {
			replayType   string
			gateType     string
			threshold    float64
			replayBudget int
			replayPhase  string
		}) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			seed := time.Now().UnixNano()
			var net *paragon.Network[float32]
			configDesc := cfg.replayType
			if cfg.replayType == "dynamic" {
				net = createNetworkTEMPORAL(cfg.replayType, seed, cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase)
				configDesc = fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s",
					cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase)
			} else {
				net = createNetwork(cfg.replayType, seed)
				if cfg.replayType == "static" {
					configDesc = "StaticReplay"
				} else {
					configDesc = "NoReplay"
				}
			}

			fmt.Printf("🧠 Training %s (%s) …\n", cfg.replayType, configDesc)
			for epoch := 0; epoch < epochs; epoch++ {
				net.Train(trainX, trainY, 1, learningRate, true, 5, -5)
				perp := computePerplexity(net, testX, testY)

				// Generate up to maxChunks
				chunks := []string{}
				input := make([][]float64, vocabSize)
				for v := 0; v < vocabSize; v++ {
					input[v] = make([]float64, inputWidth*inputHeight)
				}
				input[charToIndex['t']][0] = 1.0 // Seed with 't'
				for chunkIdx := 0; chunkIdx < maxChunks; chunkIdx++ {
					chunkText, done := generateChunk(net, input, vocabSize, maxDiffusionSteps)
					chunks = append(chunks, chunkText)
					if !done {
						break
					}
					// Update input for next chunk
					for v := 0; v < vocabSize; v++ {
						for i := range input[v] {
							input[v][i] = 0
						}
					}
					chunkInts := []int{}
					for _, r := range chunkText {
						if r != '\n' {
							if idx, ok := charToIndex[r]; ok {
								chunkInts = append(chunkInts, idx)
							} else {
								chunkInts = append(chunkInts, vocabSize-1) // Padding
							}
						}
					}
					for i, charIdx := range chunkInts {
						if i < inputWidth*inputHeight-outputWidth*outputHeight {
							input[charIdx][inputWidth*inputHeight-outputWidth*outputHeight+i] = 1.0
						}
					}

					// Check global stopping criterion
					if len(chunks) > 1 {
						var mean, variance float64
						n := float64(len(chunks[0]))
						for _, c := range chunks {
							for _, r := range c {
								if r != '\n' {
									v := float64(r)
									mean += v / n
								}
							}
						}
						for _, c := range chunks {
							for _, r := range c {
								if r != '\n' {
									v := float64(r)
									variance += (v - mean) * (v - mean) / n
								}
							}
						}
						if variance < varianceThreshold {
							break
						}
					}
				}

				fmt.Printf("Epoch %d, %s (%s): Perplexity=%.2f\nChunks:\n%s\n", epoch+1, cfg.replayType, configDesc, perp, strings.Join(chunks, "\n---\n"))
				resultsCh <- result{
					replayType: cfg.replayType,
					configDesc: configDesc,
					epoch:      epoch,
					perplexity: perp,
					chunks:     chunks,
				}
			}
		}(cfg)
	}

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	resultsByConfig := make(map[string][]result)
	for res := range resultsCh {
		key := fmt.Sprintf("%s|%s", res.replayType, res.configDesc)
		resultsByConfig[key] = append(resultsByConfig[key], res)
	}

	for _, cfg := range configs {
		configDesc := cfg.replayType
		if cfg.replayType == "dynamic" {
			configDesc = fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s",
				cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase)
		} else if cfg.replayType == "static" {
			configDesc = "StaticReplay"
		} else {
			configDesc = "NoReplay"
		}
		key := fmt.Sprintf("%s|%s", cfg.replayType, configDesc)
		results := resultsByConfig[key]

		writeResult("\n========= EPOCH-WISE NLP RESULTS (%s) =========\n", configDesc)
		writeResult("Epoch | Perplexity | Sample Chunks\n")
		writeResult("---------------------------------------------\n")
		for _, res := range results {
			writeResult("%5d | %10.2f | \n%s\n",
				res.epoch+1, res.perplexity, strings.Join(res.chunks, "\n---\n"))
		}
		writeResult("=============================================\n")
	}
}
