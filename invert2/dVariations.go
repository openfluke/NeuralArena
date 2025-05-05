package main

import (
	"math"
	"math/rand/v2"
	"paragon"
)

// --- Original crude version (commented out) ---
// func adjustNetworkUpstream(net *paragon.Network, input [][]float64, error float64, lr float64, maxUpdate float64, damping float64) { ... }

func adjustNetworkUpstreamSmart(net *paragon.Network, input [][]float64, err float64, lr float64, maxUpdate float64, damping float64) {
	// Compute a normalized signal proxy: mean abs(pixel)
	var proxySignal float64
	var total float64
	var count int
	for _, row := range input {
		for _, v := range row {
			total += math.Abs(v)
			count++
		}
	}
	if count > 0 {
		proxySignal = total / float64(count)
	} else {
		proxySignal = 1.0 // failsafe
	}

	// Work backwards from output to input
	layerCount := net.OutputLayer
	for layerIndex := layerCount; layerIndex > 0; layerIndex-- {
		layer := &net.Layers[layerIndex]

		// Scale this layer's influence based on depth
		layerScale := 1.0 - (float64(layerCount-layerIndex) / float64(layerCount))
		layerDamp := damping * layerScale

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Adaptive adjustment using sigmoid-scaled error
				adj := lr * err * layerDamp
				if adj > maxUpdate {
					adj = maxUpdate
				} else if adj < -maxUpdate {
					adj = -maxUpdate
				}

				// Apply updates
				neuron.Bias += adj
				for i := range neuron.Inputs {
					// Scale weight update by signal proxy
					neuron.Inputs[i].Weight += adj * proxySignal
				}
			}
		}

		// Decay signal proxy as we move inward
		proxySignal *= 0.9
	}
}

func adjustNetworkBehavioralPulse(net *paragon.Network, input [][]float64, err float64, lr float64, maxUpdate float64, damping float64) {
	// Step 1: compute intensity center proxy (more visual-aware)
	var total float64
	var count float64
	for y, row := range input {
		for x, v := range row {
			total += v * float64(x+y)
			count++
		}
	}
	proxySignal := 1.0
	if count > 0 {
		proxySignal = total / count
	}

	// Step 2: compute error energy pulse
	pulse := err * damping
	if pulse > maxUpdate {
		pulse = maxUpdate
	} else if pulse < -maxUpdate {
		pulse = -maxUpdate
	}

	// Step 3: propagate from output to input
	for layerIndex := net.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := &net.Layers[layerIndex]
		layerFactor := 1.0 / float64(net.OutputLayer-layerIndex+1) // diminish with depth

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Adjust bias (non-gradient "push")
				neuron.Bias += pulse * lr * layerFactor

				// Adjust weights based on their size
				for i := range neuron.Inputs {
					w := neuron.Inputs[i].Weight
					sign := 1.0
					if w > 0 {
						sign = 1.0
					} else if w < 0 {
						sign = -1.0
					}

					scale := 1.0 / (1.0 + math.Abs(w)) // bigger weights get smaller changes
					adjust := sign * pulse * lr * proxySignal * scale * layerFactor

					// clamp
					if adjust > maxUpdate {
						adjust = maxUpdate
					} else if adjust < -maxUpdate {
						adjust = -maxUpdate
					}

					neuron.Inputs[i].Weight += adjust
				}
			}
		}

		// decay pulse as it travels upstream
		pulse *= 0.8
	}
}

func adjustNetworkUpstreamNoSignalAdjsutment(net *paragon.Network, input [][]float64, error float64, lr float64, maxUpdate float64, damping float64) {
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

		//proxySignal *= 0.9
	}
}

// adjustNetworkUpstreamDepthScaled applies a depth‑scaled, RMS‑normalised
// update with per‑weight norm clipping and light weight‑decay.
//
// lr        – outer learning‑rate multiplier you pass in the caller
// maxUpdate – absolute ceiling for any individual parameter change
// damping   – extra global dampening term (0‥1 works well)
//
// You can call it exactly where you called the other helpers.
func adjustNetworkUpstreamDepthScaled(
	net *paragon.Network,
	input [][]float64,
	err float64,
	lr float64,
	maxUpdate float64,
	damping float64,
) {
	// ---------------------------------------------
	// 1. RMS proxy of the raw input (more stable than mean).
	// ---------------------------------------------
	var sumSq float64
	var count int
	for _, row := range input {
		for _, v := range row {
			sumSq += v * v
			count++
		}
	}
	rmsSignal := 1.0
	if count > 0 {
		rmsSignal = math.Sqrt(sumSq / float64(count))
	}

	// ---------------------------------------------
	// 2. Walk layers from output → input.
	// ---------------------------------------------
	depth := net.OutputLayer // for scaling factor
	for layerIdx := net.OutputLayer; layerIdx > 0; layerIdx-- {
		layer := &net.Layers[layerIdx]

		// Exponential depth scaling (0.7 ≈ keep 70 % per layer step).
		depthScale := math.Pow(0.7, float64(depth-layerIdx))

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Core adjustment term (same sign as 'err').
				adj := lr * err * damping * depthScale

				// Clamp bias update first.
				if adj > maxUpdate {
					adj = maxUpdate
				} else if adj < -maxUpdate {
					adj = -maxUpdate
				}
				neuron.Bias += adj

				// ---------------------------------
				// 3. Per‑connection updates.
				// ---------------------------------
				for i := range neuron.Inputs {
					w := neuron.Inputs[i].Weight

					// Weight‑norm clipping (inverse‑scaled).
					scale := 1.0 / (1.0 + math.Abs(w))
					wAdj := adj * rmsSignal * scale

					// Final hard cap.
					if wAdj > maxUpdate {
						wAdj = maxUpdate
					} else if wAdj < -maxUpdate {
						wAdj = -maxUpdate
					}

					// Tiny weight‑decay (1 e‑4 of current weight).
					decay := lr * 1e-4 * w

					neuron.Inputs[i].Weight += wAdj - decay
				}
			}
		}

		// Optional: decay the proxy signal as we travel upstream.
		rmsSignal *= 0.9
	}
}

// adjustNetworkWaveProp adjusts the network's weights and biases using a wave-like propagation mechanism.
// It uses a spatially-weighted proxy signal, adaptive damping, and momentum to guide updates.
func adjustNetworkWaveProp(net *paragon.Network, input [][]float64, error float64, lr float64, maxUpdate float64, damping float64) {
	// Step 1: Compute a spatially-weighted proxy signal based on input intensity.
	// For MNIST, emphasize central pixels where digits are more likely to appear.
	var proxySignal float64
	var weightSum float64
	var count int
	rows, cols := len(input), len(input[0])
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// Gaussian weighting: pixels closer to center contribute more.
			dy := float64(y - rows/2)
			dx := float64(x - cols/2)
			weight := math.Exp(-(dx*dx + dy*dy) / (2.0 * float64(rows/4)))
			proxySignal += input[y][x] * weight
			weightSum += weight
			count++
		}
	}
	if count > 0 && weightSum > 0 {
		proxySignal /= weightSum // Normalize by total weight
	} else {
		proxySignal = 1.0 // Fallback
	}

	// Step 2: Initialize wave parameters.
	// The wave carries error energy, modulated by layer depth and position.
	waveEnergy := math.Abs(error) * damping
	if waveEnergy > maxUpdate {
		waveEnergy = maxUpdate
	}
	momentum := 0.0 // Accumulates across layers for smoother updates

	// Step 3: Propagate wave backwards through the network.
	for layerIndex := net.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := &net.Layers[layerIndex]

		// Adaptive damping scales with layer depth and error magnitude.
		depthFactor := 1.0 - float64(net.OutputLayer-layerIndex)/float64(net.OutputLayer+1)
		adaptiveDamp := damping * (0.5 + 0.5*math.Tanh(waveEnergy)) * depthFactor

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Compute adjustment with momentum and non-linear scaling.
				// Use hyperbolic tangent to bound updates smoothly.
				adj := lr * error * adaptiveDamp * (1.0 + 0.2*momentum)
				adj = math.Tanh(adj) * maxUpdate // Smoothly clamp within maxUpdate

				// Update bias with wave energy.
				neuron.Bias += adj

				// Update weights with spatial awareness.
				for i := range neuron.Inputs {
					w := neuron.Inputs[i].Weight
					// Scale update inversely with weight magnitude to stabilize large weights.
					weightScale := 1.0 / (1.0 + math.Abs(w))
					// Incorporate proxy signal for input relevance.
					weightAdj := adj * proxySignal * weightScale

					// Apply momentum to weight updates.
					weightAdj += 0.1 * momentum * sign(w)

					// Clamp weight adjustment.
					if weightAdj > maxUpdate {
						weightAdj = maxUpdate
					} else if weightAdj < -maxUpdate {
						weightAdj = -maxUpdate
					}

					neuron.Inputs[i].Weight += weightAdj
				}

				// Update momentum based on position and error.
				// Neurons closer to the center of the layer contribute more to momentum.
				dy := float64(y - layer.Height/2)
				dx := float64(x - layer.Width/2)
				spatialFactor := math.Exp(-(dx*dx + dy*dy) / (2.0 * float64(layer.Width/4)))
				momentum += waveEnergy * spatialFactor * 0.1
			}
		}

		// Decay wave energy and momentum as it propagates inward.
		waveEnergy *= 0.85
		momentum *= 0.9
	}
}

// sign returns the sign of x: 1.0 if x > 0, -1.0 if x < 0, 0.0 if x == 0.
func sign(x float64) float64 {
	if x > 0 {
		return 1.0
	} else if x < 0 {
		return -1.0
	}
	return 0.0
}

// adjustNetworkSTDPDirect applies a simple STDP‑style rule using the
// source‑neuron activation obtained via (SourceLayer, SourceX, SourceY).
//
// Call it where you invoked your other adjust helpers.
func adjustNetworkSTDPDirect(
	net *paragon.Network,
	input [][]float64,
	err float64,
	lr float64,
	maxUpdate float64,
	damping float64,
) {
	//------------------------------------------------------------
	// 1. RMS proxy of raw input (robust to dark / bright digits)
	//------------------------------------------------------------
	var sumSq float64
	var cnt int
	for _, row := range input {
		for _, v := range row {
			sumSq += v * v
			cnt++
		}
	}
	rmsSignal := 1.0
	if cnt > 0 {
		rmsSignal = math.Sqrt(sumSq / float64(cnt))
	}

	//------------------------------------------------------------
	// 2. Walk layers output → input with depth scaling
	//------------------------------------------------------------
	depth := net.OutputLayer
	for l := net.OutputLayer; l > 0; l-- {
		layer := &net.Layers[l]
		depthScale := math.Pow(0.7, float64(depth-l)) // shallower layers, smaller steps

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neu := layer.Neurons[y][x]

				// ---- Bias update ----
				db := lr * err * damping * depthScale
				if db > maxUpdate {
					db = maxUpdate
				}
				if db < -maxUpdate {
					db = -maxUpdate
				}
				neu.Bias += db

				// ---- Per‑connection weight updates ----
				for i := range neu.Inputs {
					inp := &neu.Inputs[i]

					// Activation feeding THIS weight
					srcAct := net.Layers[inp.SourceLayer].
						Neurons[inp.SourceY][inp.SourceX].Value

					// STDP‑like plasticity term (potentiate if err & srcAct have same sign)
					dw := lr * damping * depthScale *
						err * srcAct / (1.0 + math.Abs(inp.Weight)) // norm‑clipped

					// Hard cap
					if dw > maxUpdate {
						dw = maxUpdate
					}
					if dw < -maxUpdate {
						dw = -maxUpdate
					}

					// Optional tiny L2 decay (improves generalisation)
					decay := lr * 1e-4 * inp.Weight

					inp.Weight += dw - decay
				}
			}
		}
		rmsSignal *= 0.9 // fade proxy upstream (keeps earlier layers calmer)
	}
}

// adjustNetworkPulseFlow adjusts the network's weights and biases using a lightweight pulse-based mechanism.
// It uses a minimal proxy signal, adaptive error scaling, and randomized sparsity for efficiency.
func adjustNetworkPulseFlow(net *paragon.Network, input [][]float64, error float64, lr float64, maxUpdate float64, damping float64) {
	// Step 1: Compute a simple proxy signal (mean input intensity).
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
	} else {
		proxySignal = 1.0 // Fallback
	}

	// Step 2: Scale error logarithmically for sensitivity to small errors.
	absError := math.Abs(error)
	scaledError := math.Log1p(absError*10.0) * damping // Log1p for numerical stability
	if scaledError > maxUpdate {
		scaledError = maxUpdate
	} else if scaledError < -maxUpdate {
		scaledError = -maxUpdate
	}
	if error < 0 {
		scaledError = -scaledError
	}

	// Step 3: Propagate pulse backwards through the network.
	for layerIndex := net.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := &net.Layers[layerIndex]

		// Linear depth modulation: deeper layers get smaller updates.
		depthFactor := float64(layerIndex) / float64(net.OutputLayer+1)

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Compute adjustment based on fan-in (number of inputs).
				fanIn := float64(len(neuron.Inputs))
				adj := lr * scaledError * (1.0 / (1.0 + fanIn)) * depthFactor

				// Apply adjustment to bias.
				neuron.Bias += adj

				// Update weights with randomized sparsity (10% chance to skip).
				for i := range neuron.Inputs {
					if rand.Float64() < 0.1 {
						continue // Skip update to reduce compute
					}
					weightAdj := adj * proxySignal
					if weightAdj > maxUpdate {
						weightAdj = maxUpdate
					} else if weightAdj < -maxUpdate {
						weightAdj = -maxUpdate
					}
					neuron.Inputs[i].Weight += weightAdj
				}
			}
		}

		// Reduce scaled error slightly for deeper layers.
		scaledError *= 0.95
	}
}

// adjustNetworkSparseEcho adjusts the network's weights and biases using a sparse, echo-based mechanism.
// It uses a single-pixel proxy signal, square-root error scaling, and selective weight updates for efficiency.
func adjustNetworkSparseEcho(net *paragon.Network, input [][]float64, error float64, lr float64, maxUpdate float64, damping float64) {
	// Step 1: Compute a minimal proxy signal (max pixel intensity).
	var proxySignal float64
	for _, row := range input {
		for _, v := range row {
			if v > proxySignal {
				proxySignal = v
			}
		}
	}
	if proxySignal == 0 {
		proxySignal = 1.0 // Fallback for blank inputs
	}

	// Step 2: Scale error using square root to amplify small errors.
	absError := math.Abs(error)
	scaledError := math.Sqrt(absError) * damping * 2.0 // Multiply by 2 to boost small errors
	if scaledError > maxUpdate {
		scaledError = maxUpdate
	} else if scaledError < -maxUpdate {
		scaledError = -maxUpdate
	}
	if error < 0 {
		scaledError = -scaledError
	}

	// Step 3: Compute dynamic weight threshold for selective updates.
	// Only update weights with absolute value above the mean weight magnitude.
	var totalWeight, weightCount float64
	for layerIndex := 1; layerIndex <= net.OutputLayer; layerIndex++ {
		layer := net.Layers[layerIndex]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				for _, conn := range layer.Neurons[y][x].Inputs {
					totalWeight += math.Abs(conn.Weight)
					weightCount++
				}
			}
		}
	}
	weightThreshold := totalWeight / weightCount * 0.5 // 50% of mean weight magnitude

	// Step 4: Propagate echo backwards, updating only active neurons and significant weights.
	for layerIndex := net.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := net.Layers[layerIndex]

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Skip low-activation neurons (20% chance if activation near zero).
				if rand.Float64() < 0.2 && math.Abs(neuron.Value) < 0.1 {
					continue
				}

				// Compute adjustment.
				adj := lr * scaledError

				// Update bias.
				neuron.Bias += adj

				// Update only significant weights.
				for i := range neuron.Inputs {
					if math.Abs(neuron.Inputs[i].Weight) < weightThreshold {
						continue // Skip small weights
					}
					weightAdj := adj * proxySignal
					if weightAdj > maxUpdate {
						weightAdj = maxUpdate
					} else if weightAdj < -maxUpdate {
						weightAdj = -maxUpdate
					}
					neuron.Inputs[i].Weight += weightAdj
				}
			}
		}

		// Rapidly decay echo to focus updates near output.
		scaledError *= 0.8
	}
}

// adjustNetworkHebbError updates weights using an error‑weighted Hebbian rule.
//
// Δw  =  lr * damping * err * preAct * postAct / (1 + |w|)
// Δb  =  lr * damping * err * postAct
//
//   - Hebbian coincidence (pre·post) gives direction; the teacher's
//     signed error gates magnitude (positive ⇒ potentiate, negative ⇒ depress).
//   - 1/(1+|w|) keeps big weights from running away (norm‑clip).
//   - A tiny L2 decay is folded in to stop old paths from lingering.
//
// Plug it into your loop exactly where the other adjust‑helpers are called.
func adjustNetworkHebbError(
	net *paragon.Network,
	input [][]float64,
	err float64,
	lr float64,
	maxUpdate float64,
	damping float64,
) {
	depth := net.OutputLayer

	for l := net.OutputLayer; l > 0; l-- {
		layer := &net.Layers[l]

		// Earlier layers move less: exponential depth scaling.
		depthScale := math.Pow(0.7, float64(depth-l))

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				post := layer.Neurons[y][x] // post‑synaptic neuron
				postAct := post.Value       // activation already set by Forward()

				// ---- Bias update (post‑only term) ----
				db := lr * damping * err * postAct * depthScale
				if db > maxUpdate {
					db = maxUpdate
				}
				if db < -maxUpdate {
					db = -maxUpdate
				}
				post.Bias += db

				// ---- Per‑connection Hebbian update ----
				for i := range post.Inputs {
					conn := &post.Inputs[i]

					// Fetch pre‑synaptic activation.
					var preAct float64
					if conn.SourceLayer == 0 {
						preAct = input[conn.SourceY][conn.SourceX] // raw pixel
					} else {
						pre := net.Layers[conn.SourceLayer].
							Neurons[conn.SourceY][conn.SourceX]
						preAct = pre.Value
					}

					// Hebbian‑with‑error term.
					dw := lr * damping * err * preAct * postAct * depthScale
					dw /= (1.0 + math.Abs(conn.Weight)) // norm‑clip

					// Hard cap.
					if dw > maxUpdate {
						dw = maxUpdate
					}
					if dw < -maxUpdate {
						dw = -maxUpdate
					}

					// Light L2 decay.
					decay := lr * 1e-4 * conn.Weight

					conn.Weight += dw - decay
				}
			}
		}
	}
}

// File‑scope maps survive across calls (simple momentum memory).
var (
	velBias   = map[*float64]float64{} // key = address of a Bias
	velWeight = map[*float64]float64{} // key = address of a Weight
)

// adjustNetworkMomentum applies a momentum‑smoothed proxy update.
// Call it where you invoked your other adjust helpers.
func adjustNetworkMomentum(
	net *paragon.Network,
	input [][]float64,
	err float64,
	lr float64,
	maxUpdate float64,
	damping float64,
) {
	beta := 1.0 - damping // higher damping ⇒ faster decay

	for l := net.OutputLayer; l > 0; l-- {
		layer := &net.Layers[l]

		depthScale := math.Pow(0.8, float64(net.OutputLayer-l))

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neu := layer.Neurons[y][x]

				// ---------- Bias momentum ----------
				rawΔb := lr * err * depthScale
				if rawΔb > maxUpdate {
					rawΔb = maxUpdate
				} else if rawΔb < -maxUpdate {
					rawΔb = -maxUpdate
				}

				bKey := &neu.Bias
				vb := velBias[bKey]
				vb = beta*vb + (1-beta)*rawΔb
				velBias[bKey] = vb
				neu.Bias += vb

				// ---------- Weight momentum ----------
				for i := range neu.Inputs {
					conn := &neu.Inputs[i]

					// Pre‑activation proxy
					var preAct float64
					if conn.SourceLayer == 0 {
						preAct = input[conn.SourceY][conn.SourceX]
					} else {
						preAct = net.
							Layers[conn.SourceLayer].
							Neurons[conn.SourceY][conn.SourceX].Value
					}

					rawΔw := lr * err * preAct * depthScale
					rawΔw /= (1.0 + math.Abs(conn.Weight)) // norm‑clip
					if rawΔw > maxUpdate {
						rawΔw = maxUpdate
					} else if rawΔw < -maxUpdate {
						rawΔw = -maxUpdate
					}

					wKey := &conn.Weight
					vw := velWeight[wKey]
					vw = beta*vw + (1-beta)*rawΔw
					velWeight[wKey] = vw

					decay := lr * 1e-4 * conn.Weight
					conn.Weight += vw - decay
				}
			}
		}
	}
}

// getQuadrant returns the quadrant index (0-3) based on the neuron's position in the layer.
func getQuadrant(layer *paragon.Grid, y, x int) int {
	if layer.Height == 1 && layer.Width == 1 {
		return 0
	} else if layer.Height == 1 {
		if x < layer.Width/2 {
			return 0 // left
		} else {
			return 1 // right
		}
	} else if layer.Width == 1 {
		if y < layer.Height/2 {
			return 0 // top
		} else {
			return 2 // bottom
		}
	} else {
		quadY := 0
		if y >= layer.Height/2 {
			quadY = 1
		}
		quadX := 0
		if x >= layer.Width/2 {
			quadX = 1
		}
		return quadY*2 + quadX
	}
}

// adjustNetworkFeatureEcho adjusts the network's weights and biases using feature-based echo propagation.
// It uses multiple proxy signals from input quadrants, sigmoid error scaling, and input-guided weight prioritization.
func adjustNetworkFeatureEcho(net *paragon.Network, input [][]float64, error float64, lr float64, maxUpdate float64, damping float64) {
	// Step 1: Compute feature-based proxy signals from input quadrants.
	rows, cols := len(input), len(input[0])
	quadrantSize := rows / 2 // Assuming square input (28x28 for MNIST)
	if quadrantSize == 0 {
		quadrantSize = 1
	}
	proxies := make([]float64, 4) // Top-left, top-right, bottom-left, bottom-right
	for q := 0; q < 4; q++ {
		startRow := (q / 2) * quadrantSize
		startCol := (q % 2) * quadrantSize
		var sum float64
		for y := startRow; y < startRow+quadrantSize && y < rows; y++ {
			for x := startCol; x < startCol+quadrantSize && x < cols; x++ {
				sum += input[y][x]
			}
		}
		proxies[q] = sum / float64(quadrantSize*quadrantSize)
	}

	// Step 2: Scale error using sigmoid to amplify small errors and saturate large ones.
	absError := math.Abs(error)
	scaledError := (2.0/(1.0+math.Exp(-absError*5.0)) - 1.0) * damping // Sigmoid-like scaling
	if scaledError > maxUpdate {
		scaledError = maxUpdate
	} else if scaledError < -maxUpdate {
		scaledError = -maxUpdate
	}
	if error < 0 {
		scaledError = -scaledError
	}

	// Step 3: Propagate feature-aligned echoes backwards.
	for layerIndex := net.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := &net.Layers[layerIndex]

		// Layer-wise adaptive damping: deeper layers get smaller updates.
		depthFactor := 1.0 - float64(net.OutputLayer-layerIndex)/float64(net.OutputLayer+1)

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Get quadrant for this neuron.
				neuronQuad := getQuadrant(layer, y, x)
				proxy := proxies[neuronQuad]

				// Compute adjustment with feature alignment.
				adj := lr * scaledError * proxy * depthFactor

				// Update bias.
				neuron.Bias += adj

				// Update weights, prioritizing those connected to high-intensity inputs.
				for i := range neuron.Inputs {
					// Get source neuron's quadrant.
					srcY := neuron.Inputs[i].SourceY
					srcX := neuron.Inputs[i].SourceX
					srcLayer := &net.Layers[layerIndex-1]
					srcQuad := getQuadrant(srcLayer, srcY, srcX)
					srcProxy := proxies[srcQuad]

					// Weight update scaled by source proxy.
					weightAdj := adj * srcProxy
					if weightAdj > maxUpdate {
						weightAdj = maxUpdate
					} else if weightAdj < -maxUpdate {
						weightAdj = -maxUpdate
					}
					neuron.Inputs[i].Weight += weightAdj
				}
			}
		}

		// Reduce scaled error for deeper layers.
		scaledError *= 0.9
	}
}

func adjustNetworkPhaseTunedContrast(
	net *paragon.Network,
	input [][]float64,
	err float64,
	lr float64,
	maxUpdate float64,
	damping float64,
	tick int,
) {
	// Phase encoding: oscillates update emphasis over time
	phase := math.Sin(float64(tick) * 0.1)

	// Compute local contrast proxy from input (edge intensity approximation)
	var localContrast float64
	rows, cols := len(input), len(input[0])
	for y := 1; y < rows-1; y++ {
		for x := 1; x < cols-1; x++ {
			c := input[y][x]
			delta := math.Abs(c-input[y+1][x]) +
				math.Abs(c-input[y-1][x]) +
				math.Abs(c-input[y][x+1]) +
				math.Abs(c-input[y][x-1])
			localContrast += delta
		}
	}
	localContrast /= float64((rows - 2) * (cols - 2) * 4)

	// Backward pass through layers
	for layerIndex := net.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := &net.Layers[layerIndex]
		depthFactor := math.Pow(0.6, float64(net.OutputLayer-layerIndex))

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x] // Correct pointer usage

				activation := neuron.Value
				adj := lr * err * phase * activation * damping * localContrast * depthFactor

				// Clamp adjustment
				if adj > maxUpdate {
					adj = maxUpdate
				} else if adj < -maxUpdate {
					adj = -maxUpdate
				}

				// Apply bias adjustment
				neuron.Bias += adj

				// Contrastive input weight update
				for i := range neuron.Inputs {
					src := &neuron.Inputs[i]
					pre := net.Layers[src.SourceLayer].Neurons[src.SourceY][src.SourceX].Value
					contrastiveSignal := pre - activation

					wAdj := adj * contrastiveSignal

					// Clamp weight update
					if wAdj > maxUpdate {
						wAdj = maxUpdate
					} else if wAdj < -maxUpdate {
						wAdj = -maxUpdate
					}

					src.Weight += wAdj
				}
			}
		}
	}
}
