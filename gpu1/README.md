(base) samuel@Steamy:~/git/PARAGON/NeuralArena/gpu1$ go run .
Loaded MNIST: 60000 train / 10000 test samples

Running CPU forward pass benchmark...
[wgpu] [Warn] Detected skylake derivative running on mesa i915. Clears to srgb textures will use manual shader clears.
Running GPU forward pass benchmark...

==== Timing Summary (1000 iterations) ====
CPU total time: 5.010797024s
GPU total time: 839.279538ms
Average CPU Forward Pass: 5.010797ms
Average GPU Forward Pass: 839.279µs
GPU is 5.97x faster than CPU (per forward)
CPU and GPU outputs match within tolerance (1e-5)
