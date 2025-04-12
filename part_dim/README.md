=== 1) Partitioned Forward/Backward Demo ===
Epoch 10, Tag=0, Loss=0.3222
Epoch 20, Tag=0, Loss=0.3275
Epoch 30, Tag=0, Loss=0.3226
Epoch 40, Tag=0, Loss=0.3191
Epoch 50, Tag=0, Loss=0.3147
Epoch 60, Tag=1, Loss=0.3623
Epoch 70, Tag=1, Loss=0.3592
Epoch 80, Tag=1, Loss=0.3568
Epoch 90, Tag=1, Loss=0.3549
Epoch 100, Tag=1, Loss=0.3534
Final outputs after partitioned training:
Input=[0 0] -> Output=0.4949 (Target=0.0)
Input=[0 1] -> Output=0.6061 (Target=1.0)
Input=[1 0] -> Output=0.4947 (Target=1.0)
Input=[1 1] -> Output=0.4947 (Target=0.0)
Final Partitioned Training Accuracy: 50.00%

=== 2) Standard Gradient Training Demo ===
Epoch 10, Loss=0.2657
Epoch 20, Loss=0.2677
Epoch 30, Loss=0.2692
Epoch 40, Loss=0.2686
Epoch 50, Loss=0.2697
Epoch 60, Loss=0.2708
Epoch 70, Loss=0.2709
Epoch 80, Loss=0.2699
Epoch 90, Loss=0.2694
Epoch 100, Loss=0.2693
Final outputs after standard gradient training:
Input=[0 0] -> Output=0.4176 (Target=0.0)
Input=[0 1] -> Output=0.4176 (Target=1.0)
Input=[1 0] -> Output=0.8222 (Target=1.0)
Input=[1 1] -> Output=0.4176 (Target=0.0)
Final Standard Gradient Accuracy: 75.00%

=== 3) Partitioning + Dimensional Neurons Demo ===
Attached a dimension sub-network to hidden neuron #1 in the mainNet.

Partition-Train Epoch 10, selectedTag=1, avgLoss=0.3449
Partition-Train Epoch 20, selectedTag=1, avgLoss=0.3451
Partition-Train Epoch 30, selectedTag=1, avgLoss=0.3453
Partition-Train Epoch 40, selectedTag=1, avgLoss=0.3455
Partition-Train Epoch 50, selectedTag=1, avgLoss=0.3457

Outputs after partitioning training with a dimension sub-network:
Input=[0 0] -> Output=0.4995 (Target=0.0)
Input=[0 1] -> Output=0.5042 (Target=1.0)
Input=[1 0] -> Output=0.4990 (Target=1.0)
Input=[1 1] -> Output=0.5037 (Target=0.0)
Final Dimension + Partitioning Accuracy: 50.00%

Now let's clone the mainNet to copyNet, then do something on each dimension sub-network...
copyNet loaded from mainNet JSON.

Found sub-network at layer=1, neuron=(0,1). Let's 'train' or 'test' it.
Dimension sub-network out=-0.0776

Finished dimension-partitioning demonstration.

=== 4) Partition + RL-Style Sub-Network Exploration ===
Trying candidate sub-network 1 with shape:
Layer 0: 1x1, Act=relu
Layer 1: 1x1, Act=relu
Candidate 1 final accuracy=50.00%

Trying candidate sub-network 2 with shape:
Layer 0: 1x1, Act=relu
Layer 1: 2x1, Act=tanh
Layer 2: 1x1, Act=relu
Candidate 2 final accuracy=50.00%

Trying candidate sub-network 3 with shape:
Layer 0: 1x1, Act=relu
Layer 1: 1x1, Act=sigmoid
Layer 2: 1x1, Act=relu
Candidate 3 final accuracy=50.00%

Best sub-network found with accuracy=50.00%
Attached best sub-network to mainNet at hidden neuron #1.

Final RL-Style approach accuracy after more training: 75.00%
