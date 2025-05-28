Loaded MNIST: 10000 test samples

Running CPU forward pass benchmark...
[wgpu] [Warn] Detected skylake derivative running on mesa i915. Clears to srgb textures will use manual shader clears.
Running GPU forward pass benchmark...

==== Timing Summary (1000 iterations) ====
CPU total time: 13.970648325s
GPU total time: 1.464956121s
Average CPU Forward Pass: 13.970648ms
Average GPU Forward Pass: 1.464956ms
GPU is 9.54x faster than CPU (per forward)
CPU and GPU outputs match within tolerance (1e-5)

==== GPU Training Benchmark ====
Training on 100 samples...
Initial average loss: 20.032490
Epoch 1/5: Loss=20.723266, Time=7.567489323s
Epoch 2/5: Loss=19.571973, Time=7.544700092s
Epoch 3/5: Loss=20.493007, Time=7.436104303s
Epoch 4/5: Loss=20.723266, Time=7.406930371s
Epoch 5/5: Loss=21.644300, Time=7.424879878s

==== Training Results ====
Training time: 37.380227132s
Time per epoch: 7.476045426s
Time per sample: 74.760454ms
Initial loss: 20.032490
Final loss: 21.414041
Loss improvement: -1.381551 (-6.90%)
Final accuracy: 7.00% (7/100)

==== Sample Predictions ====
Sample 0: Predicted=5, Actual=7, Confidence=1.0000 ✗
Sample 1: Predicted=5, Actual=2, Confidence=1.0000 ✗
Sample 2: Predicted=5, Actual=1, Confidence=1.0000 ✗
Sample 3: Predicted=5, Actual=0, Confidence=1.0000 ✗
Sample 4: Predicted=5, Actual=4, Confidence=1.0000 ✗
