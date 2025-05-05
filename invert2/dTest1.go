package main

import (
	"fmt"
	"math"
	"math/rand"
	"paragon"
	"sort"
)

func runStudentDistillation(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64, // ✅ FIXED TYPE
	nn *paragon.Network,
) {
	fmt.Println("\n---------Student Network Iterative Distillation on Training Data----------")

	params := []struct {
		MaxUpdate float64
		Damping   float64
	}{
		{0.5, 0.3},
		{0.5, 0.7},
		{0.5, 0.2},
		{0.5, 0.1},
		{0.5, 0.01},
		{0.1, 0.01},
		{5.0, 0.01},
	}

	for _, p := range params {
		fmt.Printf("\n⚙️  maxUpdate=%.2f  damping=%.2f\n", p.MaxUpdate, p.Damping)
		student := createStudentNet()
		lr := 0.01
		lastScore := 0.0
		//tick := 0

		fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

		for iter := 0; iter < 3; iter++ {
			totalError := 0.0

			for i := range trainSetInputs {
				input := trainSetInputs[i]

				nn.Forward(input)
				targetVec := nn.ExtractOutput()

				student.Forward(input)
				predVec := student.ExtractOutput()

				for j := range targetVec {
					err := targetVec[j] - predVec[j]
					if err > 0.1 {
						err = 0.1
					} else if err < -0.1 {
						err = -0.1
					}

					adjustNetworkUpstream(student, input, err, lr, p.MaxUpdate, p.Damping)

					// adjustNetworkUpstream(student, input, err, lr, p.MaxUpdate, p.Damping)
					//adjustNetworkUpstreamSmart(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkBehavioralPulse(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkUpstreamNoSignalAdjsutment(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkUpstreamDepthScaled(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkWaveProp(student, input, err, lr, p.MaxUpdate, p.Damping)
					//adjustNetworkPulseFlow(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkSTDPDirect(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkSparseEcho(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkHebbError(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkMomentum(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkFeatureEcho(student, input, err, lr, p.MaxUpdate, p.Damping)

					/*for k := 0; k < 3; k++ {
						adjustNetworkPhaseTunedContrast(student, input, err, lr, p.MaxUpdate, p.Damping, k)

					}*/

					totalError += err * err
				}
			}

			// Evaluate student
			var trainExpected, trainPredicted []float64
			for i := range trainSetInputs {
				student.Forward(trainSetInputs[i])
				pred := float64(paragon.ArgMax(student.ExtractOutput()))
				trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0])) // ✅ FIXED
				trainExpected = append(trainExpected, trueLabel)
				trainPredicted = append(trainPredicted, pred)
			}
			student.EvaluateModel(trainExpected, trainPredicted)
			currentScore := student.Performance.Score

			// Evaluate teacher
			var teachExpected, teachPredicted []float64
			for i := range trainSetInputs {
				nn.Forward(trainSetInputs[i])
				pred := float64(paragon.ArgMax(nn.ExtractOutput()))
				trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0])) // ✅ FIXED
				teachExpected = append(teachExpected, trueLabel)
				teachPredicted = append(teachPredicted, pred)
			}
			nn.EvaluateModel(teachExpected, teachPredicted)
			teacherScore := nn.Performance.Score

			var symbol string
			if currentScore > lastScore {
				symbol = "⬆"
			} else if currentScore < lastScore {
				symbol = "⬇"
			} else {
				symbol = "="
			}
			fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, currentScore, symbol)
			lastScore = currentScore
		}
	}
}

func runStudentDistillationPermuteErrLR(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	nn *paragon.Network,
) {
	fmt.Println("\n---------Student Network Permuted Error/LR Sweep (🧪 Experimental Upstream Divergence)----------")

	// ✅ Fixed settings
	maxUpdate := 0.10
	damping := 0.01

	// 🎛️ Permutations to try
	errorVariants := []float64{-0.3, -0.1, -0.05, 0.05, 0.1, 0.3}
	learningRates := []float64{0.001, 0.01, 0.05, 0.1, 0.2}

	fmt.Printf("%-12s %-12s %-20s\n", "Error", "LR", "Student ADHD")

	for _, errOffset := range errorVariants {
		for _, lr := range learningRates {
			student := createStudentNet()

			// For all training samples
			for i := range trainSetInputs {
				input := trainSetInputs[i]

				nn.Forward(input)
				targetVec := nn.ExtractOutput()

				student.Forward(input)
				predVec := student.ExtractOutput()

				// Adjust each output element using offset err
				for j := range targetVec {
					rawErr := targetVec[j] - predVec[j]
					err := rawErr + errOffset

					// Clamp err in case offset is too large
					if err > 0.1 {
						err = 0.1
					} else if err < -0.1 {
						err = -0.1
					}

					adjustNetworkUpstream(student, input, err, lr, maxUpdate, damping)
				}
			}

			// 🧠 Evaluate the student after all adjustments
			var expected, predicted []float64
			for i := range trainSetInputs {
				student.Forward(trainSetInputs[i])
				pred := float64(paragon.ArgMax(student.ExtractOutput()))
				trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
				expected = append(expected, trueLabel)
				predicted = append(predicted, pred)
			}
			student.EvaluateModel(expected, predicted)
			fmt.Printf("%-12.2f %-12.3f %-20.2f\n", errOffset, lr, student.Performance.Score)
		}
	}
}

func runStudentDistillationPermuteErrLRExtreme(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	nn *paragon.Network,
) {
	fmt.Println("\n---------Student Network Permuted Error/LR Sweep (🧪 Extreme Range Exploration)----------")

	maxUpdate := 0.10
	damping := 0.01

	errs := []float64{-1.0, -0.5, -0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3, 0.5, 1.0}
	lrs := []float64{0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0}

	fmt.Printf("%-12s %-12s %-20s\n", "Error", "LR", "Student ADHD")

	for _, errVal := range errs {
		for _, lr := range lrs {
			student := createStudentNet()

			for i := range trainSetInputs {
				input := trainSetInputs[i]
				adjustNetworkUpstream(student, input, errVal, lr, maxUpdate, damping)
			}

			var expected, predicted []float64
			for i := range trainSetInputs {
				student.Forward(trainSetInputs[i])
				pred := float64(paragon.ArgMax(student.ExtractOutput()))
				trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
				expected = append(expected, trueLabel)
				predicted = append(predicted, pred)
			}

			student.EvaluateModel(expected, predicted)
			fmt.Printf("%-12.2f %-12.4f %-20.2f\n", errVal, lr, student.Performance.Score)
		}
	}
}

func studentDistillFromHijackedTargetsSweep(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Hijacked Output Permutation Sweep (🧪 Synthetic Student Trials)----------")
	fmt.Printf("%-12s %-12s %-12s\n", "Strategy", "RandomSeed", "Student ADHD")

	seeds := []int{1, 42, 77, 101, 202, 303}
	modes := []string{"random", "flip", "truth_blend", "zeros", "uniform", "offby1"}

	for _, mode := range modes {
		for _, seed := range seeds {
			rand.Seed(int64(seed))
			student := createStudentNet()

			for i := range trainSetInputs {
				input := trainSetInputs[i]
				student.Forward(input)
				studentOut := student.ExtractOutput()

				// 🧪 Create synthetic target
				synth := make([]float64, len(studentOut))
				switch mode {
				case "random":
					synth[rand.Intn(len(synth))] = 1.0
				case "flip":
					pred := paragon.ArgMax(studentOut)
					synth[(pred+1)%len(synth)] = 1.0
				case "truth_blend":
					trueClass := paragon.ArgMax(trainSetTargets[i][0])
					synth[trueClass] = 0.7
					synth[rand.Intn(len(synth))] += 0.3
				case "zeros":
					// all zero (no signal)
				case "uniform":
					for j := range synth {
						synth[j] = 0.1
					}
				case "offby1":
					trueClass := paragon.ArgMax(trainSetTargets[i][0])
					synth[(trueClass+1)%len(synth)] = 1.0
				}

				// Error signal
				for j := range synth {
					err := synth[j] - studentOut[j]
					adjustNetworkUpstream(student, input, err, 0.02, 0.1, 0.01)
				}
			}

			// 📏 Evaluate
			var expected, predicted []float64
			for i := range trainSetInputs {
				student.Forward(trainSetInputs[i])
				pred := float64(paragon.ArgMax(student.ExtractOutput()))
				trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
				expected = append(expected, trueLabel)
				predicted = append(predicted, pred)
			}
			student.EvaluateModel(expected, predicted)
			fmt.Printf("%-12s %-12d %-12.2f\n", mode, seed, student.Performance.Score)
		}
	}
}

func studentDistillFromHijackedTargetsSweepProxyMod(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Hijacked Output + Proxy Sweep (🧪 Signal Injection Combinations)----------")
	fmt.Printf("%-12s %-12s %-12s %-12s\n", "Strategy", "Seed", "ProxyMod", "ADHD")

	seeds := []int{1, 42, 77, 101, 202, 303}
	modes := []string{"random", "flip", "truth_blend", "zeros", "uniform", "offby1"}
	proxyMods := []float64{-1.0, -0.5, 0.0, 0.5, 1.0}

	for _, mode := range modes {
		for _, seed := range seeds {
			for _, proxyMod := range proxyMods {
				rand.Seed(int64(seed))
				student := createStudentNet()

				for i := range trainSetInputs {
					input := trainSetInputs[i]
					student.Forward(input)
					studentOut := student.ExtractOutput()

					// 🧪 Generate synthetic targets
					synth := make([]float64, len(studentOut))
					switch mode {
					case "random":
						synth[rand.Intn(len(synth))] = 1.0
					case "flip":
						pred := paragon.ArgMax(studentOut)
						synth[(pred+1)%len(synth)] = 1.0
					case "truth_blend":
						trueClass := paragon.ArgMax(trainSetTargets[i][0])
						synth[trueClass] = 0.7
						synth[rand.Intn(len(synth))] += 0.3
					case "zeros":
						// leave synth all zero
					case "uniform":
						for j := range synth {
							synth[j] = 0.1
						}
					case "offby1":
						trueClass := paragon.ArgMax(trainSetTargets[i][0])
						synth[(trueClass+1)%len(synth)] = 1.0
					}

					// Apply adjustments using proxyMod variation
					for j := range synth {
						err := synth[j] - studentOut[j]
						adjustNetworkUpstreamModulated(student, input, err, 0.02, 0.1, 0.01, proxyMod)
					}
				}

				// 📏 Evaluate
				var expected, predicted []float64
				for i := range trainSetInputs {
					student.Forward(trainSetInputs[i])
					pred := float64(paragon.ArgMax(student.ExtractOutput()))
					trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
					expected = append(expected, trueLabel)
					predicted = append(predicted, pred)
				}
				student.EvaluateModel(expected, predicted)

				fmt.Printf("%-12s %-12d %-12.2f %-12.2f\n", mode, seed, proxyMod, student.Performance.Score)
			}
		}
	}
}

func experimentalPermutationSweep(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Tri-Axis Experimental Permutation Sweep (🧪 Proxy, Entropy, Reinforce)----------")
	fmt.Printf("%-12s %-12s %-14s %-12s\n", "ProxyMod", "Entropy", "Reinforce", "ADHD")

	proxyMods := []float64{-1.0, 0.0, 1.0}
	entropyModes := []bool{false, true}
	reinforceModes := []bool{false, true}

	for _, proxy := range proxyMods {
		for _, entropy := range entropyModes {
			for _, reinforce := range reinforceModes {
				student := createStudentNet()

				for i := range trainSetInputs {
					input := trainSetInputs[i]
					teacher.Forward(input)
					teacherOut := teacher.ExtractOutput()

					student.Forward(input)
					studentOut := student.ExtractOutput()

					for j := range teacherOut {
						target := teacherOut[j]
						if entropy {
							// flatten or disturb
							target = 1.0 / float64(len(teacherOut))
						}
						err := target - studentOut[j]

						if reinforce {
							err *= rand.Float64()
						}

						adjustNetworkUpstreamModulated(student, input, err, 0.02, 0.1, 0.01, proxy)
					}
				}

				var expected, predicted []float64
				for i := range trainSetInputs {
					student.Forward(trainSetInputs[i])
					pred := float64(paragon.ArgMax(student.ExtractOutput()))
					trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
					expected = append(expected, trueLabel)
					predicted = append(predicted, pred)
				}

				student.EvaluateModel(expected, predicted)
				fmt.Printf("%-12.2f %-12v %-14v %-12.2f\n", proxy, entropy, reinforce, student.Performance.Score)
			}
		}
	}
}

func hybridStudentDistillationSweep(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	teacher *paragon.Network,
) {
	fmt.Println("\n---------Hybrid Distillation Sweep (🧪 Pushing ADHD > 50)----------")
	fmt.Printf("%-12s %-10s %-10s %-6s %-10s\n", "ProxyMod", "Entropy", "Reinforce", "TopK", "ADHD")

	proxyMods := []float64{-0.5, 0.0, 0.5}
	reinforceOptions := []bool{false, true}
	entropyOptions := []bool{false, true}
	topK := 3
	lr := 0.02
	maxUpdate := 0.1
	damping := 0.01

	for _, proxyMod := range proxyMods {
		for _, reinforce := range reinforceOptions {
			for _, entropy := range entropyOptions {
				student := createStudentNet()

				for i := range trainSetInputs {
					input := trainSetInputs[i]

					teacher.Forward(input)
					tOut := teacher.ExtractOutput()

					var targetVec []float64
					if entropy {
						sum := 0.0
						targetVec = make([]float64, len(tOut))
						for j := range tOut {
							targetVec[j] = tOut[j] + 0.01
							sum += targetVec[j]
						}
						for j := range targetVec {
							targetVec[j] /= sum
						}
					} else {
						targetVec = tOut
					}

					student.Forward(input)
					sOut := student.ExtractOutput()

					// Pick top-K deltas
					type idxDelta struct {
						Index int
						Delta float64
					}
					var deltas []idxDelta
					for j := range targetVec {
						deltas = append(deltas, idxDelta{j, math.Abs(targetVec[j] - sOut[j])})
					}
					sort.Slice(deltas, func(i, j int) bool {
						return deltas[i].Delta > deltas[j].Delta
					})
					top := deltas
					if len(top) > topK {
						top = deltas[:topK]
					}

					for _, pair := range top {
						err := targetVec[pair.Index] - sOut[pair.Index]
						if reinforce {
							err *= (0.8 + 0.4*rand.Float64()) // [0.8, 1.2]
						}
						adjustNetworkUpstreamModulated(student, input, err, lr, maxUpdate, damping, proxyMod)
					}
				}

				// Evaluate
				var expected, predicted []float64
				for i := range trainSetInputs {
					student.Forward(trainSetInputs[i])
					pred := float64(paragon.ArgMax(student.ExtractOutput()))
					trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
					expected = append(expected, trueLabel)
					predicted = append(predicted, pred)
				}
				student.EvaluateModel(expected, predicted)
				score := student.Performance.Score
				fmt.Printf("%-12.2f %-10t %-10t %-6d %-10.2f\n", proxyMod, entropy, reinforce, topK, score)
			}
		}
	}
}
