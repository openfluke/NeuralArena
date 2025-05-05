package main

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"sort"

	"paragon"
)

const (
	baseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir  = "mnist_data"
	modelDir  = "models"
	modelFile = "mnist_model.json"
)

func main() {
	// --- Prepare MNIST ---
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainInputs, trainTargets, err := loadMNISTData(mnistDir, true)
	if err != nil {
		log.Fatalf("Training load failed: %v", err)
	}
	testInputs, testTargets, err := loadMNISTData(mnistDir, false)
	if err != nil {
		log.Fatalf("Test load failed: %v", err)
	}
	trainSetInputs, trainSetTargets, _, _ := paragon.SplitDataset(trainInputs, trainTargets, 0.8)

	// --- Build Model ---
	layerSizes := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}
	//modelPath := filepath.Join(modelDir, modelFile)

	var nn *paragon.Network
	fmt.Println("🧠 No pre-trained model found. Starting training...")
	nn = paragon.NewNetwork(layerSizes, activations, fullyConnected)
	nn.Train(trainSetInputs, trainSetTargets, 10, 0.01, true)
	fmt.Println("✅ Training complete.")

	// --- ADHD Evaluation ---
	var expected, predicted []float64
	for i, input := range testInputs {
		nn.Forward(input)
		out := extractOutput(nn)
		pred := paragon.ArgMax(out)
		trueLabel := paragon.ArgMax(testTargets[i][0])
		expected = append(expected, float64(trueLabel))
		predicted = append(predicted, float64(pred))
	}
	nn.EvaluateModel(expected, predicted)

	// --- Unified ADHD Diagnostics ---
	fmt.Println("\n---------SimplePRINT----------")
	fmt.Printf("🧠 ADHD Score: %.2f\n", nn.Performance.Score)
	fmt.Println("📊 Deviation Buckets:")
	for bucket, stats := range nn.Performance.Buckets {
		fmt.Printf(" - %-7s → %d samples\n", bucket, stats.Count)
	}

	fmt.Println("\n---------PrintFullDiagnostics----------")
	nn.EvaluateFull(expected, predicted)
	nn.PrintFullDiagnostics()

	fmt.Println("\n---------PrintSAMPLEDiagnostics----------")
	expectedVectors := make([][]float64, len(testInputs))
	actualVectors := make([][]float64, len(testInputs))
	for i := range testInputs {
		nn.Forward(testInputs[i])
		actualVectors[i] = nn.ExtractOutput()
		expectedVectors[i] = testTargets[i][0]
	}

	perSample := paragon.ComputePerSamplePerformance(expectedVectors, actualVectors, 0.01, nn)
	paragon.PrintSampleDiagnostics(perSample, 0.01)

	//runStudentDistillation(trainSetInputs, trainSetTargets, nn)

	//lets have some fun lol
	//runStudentDistillationPermuteErrLR(trainSetInputs, trainSetTargets, nn)
	//runStudentDistillationPermuteErrLRExtreme(trainSetInputs, trainSetTargets, nn)
	//studentDistillFromHijackedTargetsSweep(trainSetInputs, trainSetTargets, nn)
	//studentDistillFromHijackedTargetsSweepProxyMod(trainSetInputs, trainSetTargets, nn)
	//experimentalPermutationSweep(trainSetInputs, trainSetTargets, nn)
	hybridStudentDistillationSweep(trainSetInputs, trainSetTargets, nn)

	//projectiveDistillationUpstream(trainSetInputs, trainSetTargets, nn)
	//echoDistillationPulse(trainSetInputs, trainSetTargets, nn)
	//reverseCausalTraceAlign(trainSetInputs, trainSetTargets, nn)
	//errorSculptPropagation(trainSetInputs, trainSetTargets, nn)
	//eventTraceAlignment(trainSetInputs, trainSetTargets, nn)
	//eventTraceAlignTopK(trainSetInputs, trainSetTargets, nn)
	//correlationTraceAdjustment(trainSetInputs, trainSetTargets, nn)
	//correlationTraceReinforceV2(trainSetInputs, trainSetTargets, nn)
	//latentSpacePulseInjection(trainSetInputs, trainSetTargets, nn)
	//multiverseInversionAblation(trainSetInputs, trainSetTargets, nn)
}

func adjustNetworkUpstream(net *paragon.Network, input [][]float64, error float64, lr float64, maxUpdate float64, damping float64) {
	// Use proxy signal: mean pixel value from input
	var proxySignal float64
	count := 0
	for _, row := range input {
		for _, v := range row {
			proxySignal += v
			count++
		}
	}
	if count > 0 {
		proxySignal /= float64(count)
	}

	for layerIndex := net.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := &net.Layers[layerIndex]

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				adj := lr * error * damping

				if adj > maxUpdate {
					adj = maxUpdate
				} else if adj < -maxUpdate {
					adj = -maxUpdate
				}

				neuron.Bias += adj

				for i := range neuron.Inputs {
					neuron.Inputs[i].Weight += adj * proxySignal
				}
			}
		}

		proxySignal *= 0.9
	}
}

func adjustNetworkUpstreamModulated(
	net *paragon.Network,
	input [][]float64,
	error float64,
	lr float64,
	maxUpdate float64,
	damping float64,
	proxyMod float64, // ← new: multiplier on proxy signal direction
) {
	// Step 1: Derive proxy signal from the input
	var proxySignal float64
	count := 0
	for _, row := range input {
		for _, v := range row {
			proxySignal += v
			count++
		}
	}
	if count > 0 {
		proxySignal /= float64(count)
	}

	// Step 2: Modulate signal (directional/intentional)
	proxySignal *= proxyMod // ← direct multiplier (can be <0, >1, etc.)

	// Step 3: Backward layer update using modulated signal
	for layerIndex := net.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := &net.Layers[layerIndex]

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				adj := lr * error * damping
				if adj > maxUpdate {
					adj = maxUpdate
				} else if adj < -maxUpdate {
					adj = -maxUpdate
				}

				neuron.Bias += adj

				for i := range neuron.Inputs {
					neuron.Inputs[i].Weight += adj * proxySignal
				}
			}
		}

		// Optionally decay the signal (mimicking depth falloff)
		proxySignal *= 0.9
	}
}

func createStudentNet() *paragon.Network {

	layerSizes := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}

	nn := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	return nn
}

func projectiveDistillationUpstream(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Projective Distillation Upstream (Black-Box Mimicry)----------")

	student := createStudentNet()
	lr := 0.02
	maxUpdate := 0.5
	damping := 0.3
	lastScore := 0.0

	fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

	for iter := 0; iter < 3; iter++ {
		for i := range trainSetInputs {
			input := trainSetInputs[i]

			// Behavior only: input → output
			teacher.Forward(input)
			teacherOutput := teacher.ExtractOutput()

			student.Forward(input)
			studentOutput := student.ExtractOutput()

			// For each output neuron, compute target delta and backpressure upstream
			for j := range teacherOutput {
				err := teacherOutput[j] - studentOutput[j]
				adj := clamp(err*lr, -maxUpdate, maxUpdate)

				// --- Upstream push (no access to teacher weights!) ---
				for layerIndex := student.OutputLayer; layerIndex > 0; layerIndex-- {
					layer := &student.Layers[layerIndex]
					depthFactor := math.Pow(0.7, float64(student.OutputLayer-layerIndex))

					for y := 0; y < layer.Height; y++ {
						for x := 0; x < layer.Width; x++ {
							neuron := layer.Neurons[y][x]

							biasAdj := adj * damping * depthFactor
							neuron.Bias += clamp(biasAdj, -maxUpdate, maxUpdate)

							for k := range neuron.Inputs {
								pre := &neuron.Inputs[k]
								srcAct := student.Layers[pre.SourceLayer].Neurons[pre.SourceY][pre.SourceX].Value
								wAdj := adj * srcAct * damping * depthFactor
								pre.Weight += clamp(wAdj, -maxUpdate, maxUpdate)
							}
						}
					}
				}
			}
		}

		// Evaluate student
		var studentExpected, studentPredicted []float64
		for i := range trainSetInputs {
			student.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(student.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			studentExpected = append(studentExpected, trueLabel)
			studentPredicted = append(studentPredicted, pred)
		}
		student.EvaluateModel(studentExpected, studentPredicted)
		studentScore := student.Performance.Score

		// Evaluate teacher (as reference)
		var teacherExpected, teacherPredicted []float64
		for i := range trainSetInputs {
			teacher.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(teacher.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			teacherExpected = append(teacherExpected, trueLabel)
			teacherPredicted = append(teacherPredicted, pred)
		}
		teacher.EvaluateModel(teacherExpected, teacherPredicted)
		teacherScore := teacher.Performance.Score

		var symbol string
		if studentScore > lastScore {
			symbol = "⬆"
		} else if studentScore < lastScore {
			symbol = "⬇"
		} else {
			symbol = "="
		}
		fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, studentScore, symbol)
		lastScore = studentScore
	}
}

func clamp(val, min, max float64) float64 {
	if val > max {
		return max
	}
	if val < min {
		return min
	}
	return val
}

func echoDistillationPulse(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Echo Distillation Pulse (Temporal Feedback Mimicry)----------")

	student := createStudentNet()
	lr := 0.02
	maxUpdate := 0.4
	damping := 0.4
	lastScore := 0.0

	fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

	for iter := 0; iter < 3; iter++ {
		for i := range trainSetInputs {
			input := trainSetInputs[i]

			teacher.Forward(input)
			tOut := teacher.ExtractOutput()

			student.Forward(input)
			sOut := student.ExtractOutput()

			for j := range tOut {
				err := tOut[j] - sOut[j]
				if math.Abs(err) < 0.01 {
					continue // skip weak error
				}

				// Temporal echo: amplified for confident teacher answers
				echo := err * tOut[j] * damping

				// Propagate upstream
				for layerIndex := student.OutputLayer; layerIndex > 0; layerIndex-- {
					layer := &student.Layers[layerIndex]
					depthFactor := math.Pow(0.6, float64(student.OutputLayer-layerIndex))

					for y := 0; y < layer.Height; y++ {
						for x := 0; x < layer.Width; x++ {
							neu := layer.Neurons[y][x]

							if rand.Float64() < 0.2 && math.Abs(neu.Value) < 0.05 {
								continue // skip low-signal nodes
							}

							pulse := lr * echo * depthFactor
							neu.Bias += clamp(pulse, -maxUpdate, maxUpdate)

							for k := range neu.Inputs {
								conn := &neu.Inputs[k]
								srcVal := student.Layers[conn.SourceLayer].Neurons[conn.SourceY][conn.SourceX].Value
								wAdj := pulse * srcVal * (1.0 / (1.0 + math.Abs(conn.Weight)))
								conn.Weight += clamp(wAdj, -maxUpdate, maxUpdate)
							}
						}
					}
				}
			}
		}

		// Evaluate student
		var studentExpected, studentPredicted []float64
		for i := range trainSetInputs {
			student.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(student.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			studentExpected = append(studentExpected, trueLabel)
			studentPredicted = append(studentPredicted, pred)
		}
		student.EvaluateModel(studentExpected, studentPredicted)
		studentScore := student.Performance.Score

		// Evaluate teacher
		var teacherExpected, teacherPredicted []float64
		for i := range trainSetInputs {
			teacher.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(teacher.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			teacherExpected = append(teacherExpected, trueLabel)
			teacherPredicted = append(teacherPredicted, pred)
		}
		teacher.EvaluateModel(teacherExpected, teacherPredicted)
		teacherScore := teacher.Performance.Score

		var symbol string
		if studentScore > lastScore {
			symbol = "⬆"
		} else if studentScore < lastScore {
			symbol = "⬇"
		} else {
			symbol = "="
		}
		fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, studentScore, symbol)
		lastScore = studentScore
	}
}

func reverseCausalTraceAlign(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Reverse Causal Trace Align (Behavioral Flow Rewiring)----------")

	student := createStudentNet()
	lr := 0.015
	maxUpdate := 0.35
	damping := 0.4
	lastScore := 0.0

	fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

	for iter := 0; iter < 3; iter++ {
		for i := range trainSetInputs {
			input := trainSetInputs[i]

			// Observe only behavior
			teacher.Forward(input)
			teacherOut := teacher.ExtractOutput()

			student.Forward(input)
			studentOut := student.ExtractOutput()

			for j := range teacherOut {
				err := teacherOut[j] - studentOut[j]
				if math.Abs(err) < 0.01 {
					continue
				}

				// Step 1: output-layer neuron receives error signal
				echo := err * damping

				for layerIndex := student.OutputLayer; layerIndex > 0; layerIndex-- {
					layer := &student.Layers[layerIndex]
					depthDecay := math.Pow(0.6, float64(student.OutputLayer-layerIndex))

					for y := 0; y < layer.Height; y++ {
						for x := 0; x < layer.Width; x++ {
							neuron := layer.Neurons[y][x]

							// Only adjust active neurons
							if math.Abs(neuron.Value) < 0.05 {
								continue
							}

							// Bias shaped by causal echo and current state
							bAdj := lr * echo * neuron.Value * depthDecay
							neuron.Bias += clamp(bAdj, -maxUpdate, maxUpdate)

							for k := range neuron.Inputs {
								inp := &neuron.Inputs[k]
								preAct := student.Layers[inp.SourceLayer].Neurons[inp.SourceY][inp.SourceX].Value

								// This adjustment traces causal link from error to source neuron
								wAdj := lr * echo * preAct * depthDecay
								inp.Weight += clamp(wAdj, -maxUpdate, maxUpdate)
							}
						}
					}

					// Step 2: echo back-propagates through causal links
					echo *= 0.85
				}
			}
		}

		// Evaluate student
		var studentExpected, studentPredicted []float64
		for i := range trainSetInputs {
			student.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(student.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			studentExpected = append(studentExpected, trueLabel)
			studentPredicted = append(studentPredicted, pred)
		}
		student.EvaluateModel(studentExpected, studentPredicted)
		studentScore := student.Performance.Score

		// Evaluate teacher
		var teacherExpected, teacherPredicted []float64
		for i := range trainSetInputs {
			teacher.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(teacher.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			teacherExpected = append(teacherExpected, trueLabel)
			teacherPredicted = append(teacherPredicted, pred)
		}
		teacher.EvaluateModel(teacherExpected, teacherPredicted)
		teacherScore := teacher.Performance.Score

		var symbol string
		if studentScore > lastScore {
			symbol = "⬆"
		} else if studentScore < lastScore {
			symbol = "⬇"
		} else {
			symbol = "="
		}
		fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, studentScore, symbol)
		lastScore = studentScore
	}
}

func errorSculptPropagation(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Error Sculpt Propagation (Activation-Weighted Rewiring)----------")

	student := createStudentNet()
	lr := 10.03
	maxUpdate := 10.4
	lastScore := 0.0

	fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

	for iter := 0; iter < 3; iter++ {
		for i := range trainSetInputs {
			input := trainSetInputs[i]

			teacher.Forward(input)
			tOut := teacher.ExtractOutput()

			student.Forward(input)
			sOut := student.ExtractOutput()

			for j := range tOut {
				err := tOut[j] - sOut[j]
				if math.Abs(err) < 0.01 {
					continue
				}

				// Start from output layer
				echo := math.Tanh(err) // bounded, direction-preserving

				for layerIndex := student.OutputLayer; layerIndex > 0; layerIndex-- {
					layer := &student.Layers[layerIndex]
					decay := math.Pow(0.7, float64(student.OutputLayer-layerIndex))

					for y := 0; y < layer.Height; y++ {
						for x := 0; x < layer.Width; x++ {
							neu := layer.Neurons[y][x]
							if math.Abs(neu.Value) < 0.05 {
								continue
							}

							// Weight contribution scales with activation strength
							weightScale := math.Tanh(neu.Value * echo * decay)
							neu.Bias += clamp(lr*weightScale, -maxUpdate, maxUpdate)

							for k := range neu.Inputs {
								conn := &neu.Inputs[k]
								preAct := student.Layers[conn.SourceLayer].Neurons[conn.SourceY][conn.SourceX].Value

								influence := preAct * weightScale
								conn.Weight += clamp(lr*influence, -maxUpdate, maxUpdate)
							}
						}
					}
				}
			}
		}

		// Evaluate student
		var studentExpected, studentPredicted []float64
		for i := range trainSetInputs {
			student.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(student.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			studentExpected = append(studentExpected, trueLabel)
			studentPredicted = append(studentPredicted, pred)
		}
		student.EvaluateModel(studentExpected, studentPredicted)
		studentScore := student.Performance.Score

		// Evaluate teacher
		var teacherExpected, teacherPredicted []float64
		for i := range trainSetInputs {
			teacher.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(teacher.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			teacherExpected = append(teacherExpected, trueLabel)
			teacherPredicted = append(teacherPredicted, pred)
		}
		teacher.EvaluateModel(teacherExpected, teacherPredicted)
		teacherScore := teacher.Performance.Score

		var symbol string
		if studentScore > lastScore {
			symbol = "⬆"
		} else if studentScore < lastScore {
			symbol = "⬇"
		} else {
			symbol = "="
		}
		fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, studentScore, symbol)
		lastScore = studentScore
	}
}

func eventTraceAlignment(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Event Trace Alignment (Trace-and-Resonate Conditioning)----------")

	student := createStudentNet()
	lr := 0.03
	maxUpdate := 0.4
	lastScore := 0.0

	fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

	for iter := 0; iter < 3; iter++ {
		for i := range trainSetInputs {
			input := trainSetInputs[i]

			// Teacher behavior
			teacher.Forward(input)
			teacherOut := teacher.ExtractOutput()
			targetClass := paragon.ArgMax(teacherOut)

			// Student behavior and trace
			student.Forward(input)

			for layerIndex := 1; layerIndex <= student.OutputLayer; layerIndex++ {
				layer := &student.Layers[layerIndex]
				for y := 0; y < layer.Height; y++ {
					for x := 0; x < layer.Width; x++ {
						neuron := layer.Neurons[y][x]

						boost := neuron.Value
						if boost < 0.05 {
							continue
						}

						classWeight := float64(targetClass) / 10.0

						// Bias nudge
						bAdj := lr * classWeight * boost
						neuron.Bias += clamp(bAdj, -maxUpdate, maxUpdate)

						// Strengthen contributing inputs
						for k := range neuron.Inputs {
							src := &neuron.Inputs[k]
							srcAct := student.Layers[src.SourceLayer].Neurons[src.SourceY][src.SourceX].Value
							wAdj := lr * boost * srcAct
							src.Weight += clamp(wAdj, -maxUpdate, maxUpdate)
						}
					}
				}
			}
		}

		// Evaluate student
		var studentExpected, studentPredicted []float64
		for i := range trainSetInputs {
			student.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(student.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			studentExpected = append(studentExpected, trueLabel)
			studentPredicted = append(studentPredicted, pred)
		}
		student.EvaluateModel(studentExpected, studentPredicted)
		studentScore := student.Performance.Score

		// Evaluate teacher
		var teacherExpected, teacherPredicted []float64
		for i := range trainSetInputs {
			teacher.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(teacher.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			teacherExpected = append(teacherExpected, trueLabel)
			teacherPredicted = append(teacherPredicted, pred)
		}
		teacher.EvaluateModel(teacherExpected, teacherPredicted)
		teacherScore := teacher.Performance.Score

		var symbol string
		if studentScore > lastScore {
			symbol = "⬆"
		} else if studentScore < lastScore {
			symbol = "⬇"
		} else {
			symbol = "="
		}
		fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, studentScore, symbol)
		lastScore = studentScore
	}
}

func eventTraceAlignTopK(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Event Trace Alignment Top-K (Selective Influence Reinforcement)----------")

	student := createStudentNet()
	lr := 0.03
	maxUpdate := 0.4
	topK := 6
	lastScore := 0.0

	fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

	for iter := 0; iter < 3; iter++ {
		for i := range trainSetInputs {
			input := trainSetInputs[i]

			// Teacher behavior
			teacher.Forward(input)
			teacherOut := teacher.ExtractOutput()
			targetClass := paragon.ArgMax(teacherOut)

			// Student behavior and trace
			student.Forward(input)

			for layerIndex := 1; layerIndex <= student.OutputLayer; layerIndex++ {
				layer := &student.Layers[layerIndex]

				type traceNode struct {
					Neuron *paragon.Neuron
					X, Y   int
					Value  float64
				}
				var trace []traceNode

				// Collect all neurons and their activations
				for y := 0; y < layer.Height; y++ {
					for x := 0; x < layer.Width; x++ {
						n := layer.Neurons[y][x]
						if math.Abs(n.Value) > 0.01 {
							trace = append(trace, traceNode{Neuron: n, X: x, Y: y, Value: math.Abs(n.Value)})
						}
					}
				}

				// Sort by absolute activation
				sort.Slice(trace, func(i, j int) bool {
					return trace[i].Value > trace[j].Value
				})

				// Apply reinforcement to top-K
				for k := 0; k < topK && k < len(trace); k++ {
					node := trace[k]
					n := node.Neuron

					classWeight := float64(targetClass) / 10.0
					bAdj := lr * classWeight * n.Value
					n.Bias += clamp(bAdj, -maxUpdate, maxUpdate)

					for w := range n.Inputs {
						src := &n.Inputs[w]
						srcAct := student.Layers[src.SourceLayer].Neurons[src.SourceY][src.SourceX].Value
						adj := lr * srcAct * n.Value
						src.Weight += clamp(adj, -maxUpdate, maxUpdate)
					}
				}
			}
		}

		// Evaluate student
		var studentExpected, studentPredicted []float64
		for i := range trainSetInputs {
			student.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(student.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			studentExpected = append(studentExpected, trueLabel)
			studentPredicted = append(studentPredicted, pred)
		}
		student.EvaluateModel(studentExpected, studentPredicted)
		studentScore := student.Performance.Score

		// Evaluate teacher
		var teacherExpected, teacherPredicted []float64
		for i := range trainSetInputs {
			teacher.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(teacher.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			teacherExpected = append(teacherExpected, trueLabel)
			teacherPredicted = append(teacherPredicted, pred)
		}
		teacher.EvaluateModel(teacherExpected, teacherPredicted)
		teacherScore := teacher.Performance.Score

		var symbol string
		if studentScore > lastScore {
			symbol = "⬆"
		} else if studentScore < lastScore {
			symbol = "⬇"
		} else {
			symbol = "="
		}
		fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, studentScore, symbol)
		lastScore = studentScore
	}
}

func correlationTraceAdjustment(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Correlation Trace Adjustment (Causal Contribution Rewiring)----------")

	student := createStudentNet()
	lr := 0.02
	maxUpdate := 0.3
	lastScore := 0.0

	fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

	for iter := 0; iter < 3; iter++ {
		for i := range trainSetInputs {
			input := trainSetInputs[i]

			teacher.Forward(input)
			teacherOut := teacher.ExtractOutput()

			student.Forward(input)
			studentOut := student.ExtractOutput()

			for j := range teacherOut {
				delta := teacherOut[j] - studentOut[j]

				if math.Abs(delta) < 0.01 {
					continue
				}

				// Create a correlation echo signal
				signal := math.Tanh(delta)

				// Backward trace from output → input
				for layerIndex := student.OutputLayer; layerIndex > 0; layerIndex-- {
					layer := &student.Layers[layerIndex]
					depthScale := math.Pow(0.7, float64(student.OutputLayer-layerIndex))

					for y := 0; y < layer.Height; y++ {
						for x := 0; x < layer.Width; x++ {
							neuron := layer.Neurons[y][x]

							// Skip inactive neurons
							if math.Abs(neuron.Value) < 0.05 {
								continue
							}

							// Adjust bias based on correlation
							bAdj := lr * signal * neuron.Value * depthScale
							neuron.Bias += clamp(bAdj, -maxUpdate, maxUpdate)

							// Adjust weights that carried relevant signal
							for k := range neuron.Inputs {
								inp := &neuron.Inputs[k]
								srcVal := student.Layers[inp.SourceLayer].Neurons[inp.SourceY][inp.SourceX].Value
								corr := signal * srcVal * neuron.Value // three-way correlation

								adj := lr * corr * depthScale
								inp.Weight += clamp(adj, -maxUpdate, maxUpdate)
							}
						}
					}
				}
			}
		}

		// Evaluate student
		var studentExpected, studentPredicted []float64
		for i := range trainSetInputs {
			student.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(student.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			studentExpected = append(studentExpected, trueLabel)
			studentPredicted = append(studentPredicted, pred)
		}
		student.EvaluateModel(studentExpected, studentPredicted)
		studentScore := student.Performance.Score

		// Evaluate teacher
		var teacherExpected, teacherPredicted []float64
		for i := range trainSetInputs {
			teacher.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(teacher.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			teacherExpected = append(teacherExpected, trueLabel)
			teacherPredicted = append(teacherPredicted, pred)
		}
		teacher.EvaluateModel(teacherExpected, teacherPredicted)
		teacherScore := teacher.Performance.Score

		var symbol string
		if studentScore > lastScore {
			symbol = "⬆"
		} else if studentScore < lastScore {
			symbol = "⬇"
		} else {
			symbol = "="
		}
		fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, studentScore, symbol)
		lastScore = studentScore
	}
}

func correlationTraceReinforceV2(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Correlation Trace Reinforce V2 (Memory-Weighted Conditioning)----------")

	student := createStudentNet()
	lr := 0.02
	maxUpdate := 0.3
	lastScore := 0.0

	// 🔁 Per-neuron trust memory (keyed by pointer)
	trustBias := map[*float64]float64{}
	trustWeight := map[*float64]float64{}

	fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

	for iter := 0; iter < 3; iter++ {
		for i := range trainSetInputs {
			input := trainSetInputs[i]

			teacher.Forward(input)
			teacherOut := teacher.ExtractOutput()

			student.Forward(input)
			studentOut := student.ExtractOutput()

			for j := range teacherOut {
				delta := teacherOut[j] - studentOut[j]
				if math.Abs(delta) < 0.01 {
					continue
				}

				// Primary signal: tanh-smoothed directional delta
				signal := math.Tanh(delta)

				// Causal trace from output to input
				for layerIndex := student.OutputLayer; layerIndex > 0; layerIndex-- {
					layer := &student.Layers[layerIndex]
					depthScale := math.Pow(0.7, float64(student.OutputLayer-layerIndex))

					for y := 0; y < layer.Height; y++ {
						for x := 0; x < layer.Width; x++ {
							neuron := layer.Neurons[y][x]
							val := neuron.Value

							if math.Abs(val) < 0.05 {
								continue
							}

							// Bias trust memory
							bKey := &neuron.Bias
							if _, ok := trustBias[bKey]; !ok {
								trustBias[bKey] = 0
							}

							// Trust evolves with alignment
							trustBias[bKey] = 0.95*trustBias[bKey] + 0.05*signal*val
							boost := trustBias[bKey]

							// Bias adjustment
							bAdj := lr * signal * val * boost * depthScale
							neuron.Bias += clamp(bAdj, -maxUpdate, maxUpdate)

							// Per-weight contribution tracking
							for k := range neuron.Inputs {
								conn := &neuron.Inputs[k]
								src := student.Layers[conn.SourceLayer].Neurons[conn.SourceY][conn.SourceX]
								srcVal := src.Value

								wKey := &conn.Weight
								if _, ok := trustWeight[wKey]; !ok {
									trustWeight[wKey] = 0
								}

								// Accumulate trust for aligned correlation
								trustWeight[wKey] = 0.95*trustWeight[wKey] + 0.05*signal*val*srcVal
								boost := trustWeight[wKey]

								// Final adjustment
								adj := lr * signal * srcVal * val * boost * depthScale
								conn.Weight += clamp(adj, -maxUpdate, maxUpdate)
							}
						}
					}
				}
			}
		}

		// Evaluate student
		var studentExpected, studentPredicted []float64
		for i := range trainSetInputs {
			student.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(student.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			studentExpected = append(studentExpected, trueLabel)
			studentPredicted = append(studentPredicted, pred)
		}
		student.EvaluateModel(studentExpected, studentPredicted)
		studentScore := student.Performance.Score

		// Evaluate teacher
		var teacherExpected, teacherPredicted []float64
		for i := range trainSetInputs {
			teacher.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(teacher.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			teacherExpected = append(teacherExpected, trueLabel)
			teacherPredicted = append(teacherPredicted, pred)
		}
		teacher.EvaluateModel(teacherExpected, teacherPredicted)
		teacherScore := teacher.Performance.Score

		var symbol string
		if studentScore > lastScore {
			symbol = "⬆"
		} else if studentScore < lastScore {
			symbol = "⬇"
		} else {
			symbol = "="
		}
		fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, studentScore, symbol)
		lastScore = studentScore
	}
}

func latentSpacePulseInjection(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Latent Space Pulse Injection (Attractor-Oriented Mimicry)----------")

	student := createStudentNet()
	lr := 0.03
	maxUpdate := 0.5
	lastScore := 0.0

	fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

	for iter := 0; iter < 3; iter++ {
		for i := range trainSetInputs {
			input := trainSetInputs[i]

			// Forward teacher and student
			teacher.Forward(input)
			teacherOut := teacher.ExtractOutput()
			teacherConfidence := softmaxSharpness(teacherOut)

			student.Forward(input)
			studentOut := student.ExtractOutput()
			studentConfidence := softmaxSharpness(studentOut)

			// Only pulse if teacher is confident and student is not
			if teacherConfidence < 0.7 || studentConfidence > 0.6 {
				continue
			}

			pulseStrength := (teacherConfidence - studentConfidence) * 1.5

			// Inject into latent (non-output) layers only
			for layerIndex := 1; layerIndex < student.OutputLayer; layerIndex++ {
				layer := &student.Layers[layerIndex]
				depthFactor := math.Pow(0.7, float64(layerIndex))

				for y := 0; y < layer.Height; y++ {
					for x := 0; x < layer.Width; x++ {
						neu := layer.Neurons[y][x]
						if math.Abs(neu.Value) < 0.05 {
							continue
						}

						// Pulse: shaped by current activation and depth
						bAdj := lr * pulseStrength * neu.Value * depthFactor
						neu.Bias += clamp(bAdj, -maxUpdate, maxUpdate)

						for k := range neu.Inputs {
							conn := &neu.Inputs[k]
							srcVal := student.Layers[conn.SourceLayer].Neurons[conn.SourceY][conn.SourceX].Value
							wAdj := lr * pulseStrength * srcVal * neu.Value * depthFactor
							conn.Weight += clamp(wAdj, -maxUpdate, maxUpdate)
						}
					}
				}
			}
		}

		// Evaluate student
		var studentExpected, studentPredicted []float64
		for i := range trainSetInputs {
			student.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(student.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			studentExpected = append(studentExpected, trueLabel)
			studentPredicted = append(studentPredicted, pred)
		}
		student.EvaluateModel(studentExpected, studentPredicted)
		studentScore := student.Performance.Score

		// Evaluate teacher
		var teacherExpected, teacherPredicted []float64
		for i := range trainSetInputs {
			teacher.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(teacher.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			teacherExpected = append(teacherExpected, trueLabel)
			teacherPredicted = append(teacherPredicted, pred)
		}
		teacher.EvaluateModel(teacherExpected, teacherPredicted)
		teacherScore := teacher.Performance.Score

		var symbol string
		if studentScore > lastScore {
			symbol = "⬆"
		} else if studentScore < lastScore {
			symbol = "⬇"
		} else {
			symbol = "="
		}
		fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, studentScore, symbol)
		lastScore = studentScore
	}
}

func softmaxSharpness(vec []float64) float64 {
	// Measures how "peaky" the output distribution is (higher means more confident)
	var maxVal float64 = -math.MaxFloat64
	for _, v := range vec {
		if v > maxVal {
			maxVal = v
		}
	}

	var sumExp float64
	for _, v := range vec {
		sumExp += math.Exp(v - maxVal)
	}

	var peak float64
	for _, v := range vec {
		p := math.Exp(v-maxVal) / sumExp
		if p > peak {
			peak = p
		}
	}
	return peak // closer to 1 = more confident
}

func multiverseInversionAblation(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Multiverse Inversion Ablation (Counterfactual Node Salience)----------")

	student := createStudentNet()
	lr := 0.04
	//maxUpdate := 0.5
	lastScore := 0.0

	fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

	for iter := 0; iter < 3; iter++ {
		for i := range trainSetInputs {
			input := trainSetInputs[i]

			teacher.Forward(input)
			teacherOut := teacher.ExtractOutput()
			teacherClass := paragon.ArgMax(teacherOut)

			student.Forward(input)
			baseOut := student.ExtractOutput()
			baseClass := paragon.ArgMax(baseOut)

			if baseClass == teacherClass {
				continue // already correct
			}

			// --- Lesion test: Try knocking out each neuron ---
			for layerIndex := 1; layerIndex <= student.OutputLayer; layerIndex++ {
				layer := &student.Layers[layerIndex]

				for y := 0; y < layer.Height; y++ {
					for x := 0; x < layer.Width; x++ {
						neuron := layer.Neurons[y][x]
						origBias := neuron.Bias

						// Backup and zero weights
						origInputs := make([]float64, len(neuron.Inputs))
						for j, inp := range neuron.Inputs {
							origInputs[j] = inp.Weight
							inp.Weight = 0
						}
						neuron.Bias = 0

						student.Forward(input)
						newOut := student.ExtractOutput()
						newClass := paragon.ArgMax(newOut)

						// If disabling neuron made student worse, reinforce it
						if newClass != teacherClass && newClass != baseClass {
							neuron.Bias = origBias + lr
							for j := range neuron.Inputs {
								neuron.Inputs[j].Weight = origInputs[j] + lr*0.5
							}
						} else {
							neuron.Bias = origBias
							for j := range neuron.Inputs {
								neuron.Inputs[j].Weight = origInputs[j]
							}
						}
					}
				}
			}
		}

		// --- Evaluation ---
		var studentExpected, studentPredicted []float64
		for i := range trainSetInputs {
			student.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(student.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			studentExpected = append(studentExpected, trueLabel)
			studentPredicted = append(studentPredicted, pred)
		}
		student.EvaluateModel(studentExpected, studentPredicted)
		studentScore := student.Performance.Score

		var teacherExpected, teacherPredicted []float64
		for i := range trainSetInputs {
			teacher.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(teacher.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			teacherExpected = append(teacherExpected, trueLabel)
			teacherPredicted = append(teacherPredicted, pred)
		}
		teacher.EvaluateModel(teacherExpected, teacherPredicted)
		teacherScore := teacher.Performance.Score

		var symbol string
		if studentScore > lastScore {
			symbol = "⬆"
		} else if studentScore < lastScore {
			symbol = "⬇"
		} else {
			symbol = "="
		}
		fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, studentScore, symbol)
		lastScore = studentScore
	}
}
