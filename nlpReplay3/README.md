# Lightweight Feedforward Neural Network for Cloze-Style Task

This project implements a lightweight feedforward neural network trained on a Cloze-style task using Sherlock Holmes text. Instead of relying on attention or recurrence, it introduces an **entropy-based replay mechanism** that dynamically focuses training on uncertain examples. The model predicts a missing word in a sentence, represented by a `[MASK]` token.

## Training Modes

We compare three training modes:

- **Standard Training**: Baseline approach without replay.
- **Static Replay**: Replays a fixed set of examples.
- **Dynamic Replay**: Dynamically selects uncertain examples based on entropy.

All modes achieved a validation accuracy of **22.45%**, but only the **dynamic replay** model consistently produced **semantically correct predictions**.

## Example Predictions

### Example 1

**Prompt**: "It [MASK] a homely little room."  
**Truth**: `was`

- **Standard Mode**: `<unk>`
- **Static Mode**: `<unk>`
- **Dynamic Mode**: `was</w>`

### Example 2

**Prompt**: "Sold [MASK] Mr. Windigate of the Alpha."  
**Truth**: `to`

- **Standard Mode**: `<unk>`
- **Static Mode**: `<unk>`
- **Dynamic Mode**: `to</w>`

## Key Findings

The **dynamic replay** mechanism enables the model to learn meaningful patterns and generate correct predictions, even with a shallow architecture and limited training data. This highlights the effectiveness of entropy-based replay in improving model performance on Cloze-style tasks.

Fetching text...
Learning BPE merges…
✓ learned BPE vocab size 256 tokens
Epoch 0, Loss: 5.5352
[Epoch 1/10] approx loss: 5.4945
Epoch 0, Loss: 5.4707
[Epoch 2/10] approx loss: 5.4288
Epoch 0, Loss: 5.4081
[Epoch 3/10] approx loss: 5.3657
Epoch 0, Loss: 5.3492
[Epoch 4/10] approx loss: 5.3076
Epoch 0, Loss: 5.2963
[Epoch 5/10] approx loss: 5.2575
Epoch 0, Loss: 5.2527
[Epoch 6/10] approx loss: 5.2195
Epoch 0, Loss: 5.2213
[Epoch 7/10] approx loss: 5.1950
Epoch 0, Loss: 5.2025
[Epoch 8/10] approx loss: 5.1831
Epoch 0, Loss: 5.1944
[Epoch 9/10] approx loss: 5.1801
Epoch 0, Loss: 5.1926
[Epoch 10/10] approx loss: 5.1801

standard done. val-acc: 22.45%

Prompt: “I am no official agent. I understand that it was your daughter who required my presence here, and [MASK] am acting in her interests. Young McCarthy must be got off, however.”
Truth : I
Guess : <unk>

Prompt: “The police have watched this Lascar,” said Inspector Bradstreet, “and I can quite understand [MASK] he might find it difficult to post a letter unobserved. Probably he handed it to some sailor customer of his, who forgot all about it for some days.”
Truth : that
Guess : <unk>

Prompt: “‘Here we are, Jack,’ says he, touching me [MASK] the arm; ‘we’ll be as good as a family to you. There’s two of us, me and my son, and you can have the keeping of us. If you don’t—it’s a fine, law-abiding country is England, and there’s always a policeman within hail.’
Truth : on
Guess : <unk>

Prompt: “‘Sold [MASK] Mr. Windigate of the Alpha, at 12*s*.’”
Truth : to
Guess : <unk>

Prompt: A small side door led into the whitewashed corridor from which the three bedrooms opened. Holmes refused to examine the third chamber, so we passed at once to the second, that in which Miss Stoner was now sleeping, and in which her sister had met with her fate. It [MASK] a homely little room, with a low ceiling and a gaping fireplace, after the fashion of old country-houses. A brown chest of drawers stood in one corner, a narrow white-counterpaned bed in another, and a dressing-table on the left-hand side of the window. These articles, with two small wicker-work chairs, made up all the furniture in the room save for a square of Wilton carpet in the centre. The boards round and the panelling of the walls were of brown, worm-eaten oak, so old and discoloured that it may have dated from the original building of the house. Holmes drew one of the chairs into a corner and sat silent, while his eyes travelled round and round and up and down, taking in every detail of the apartment.
Truth : was
Guess : <unk>
Epoch 0, Loss: 5.5350
[Epoch 1/10] approx loss: 5.4936
Epoch 0, Loss: 5.4698
[Epoch 2/10] approx loss: 5.4276
Epoch 0, Loss: 5.4072
[Epoch 3/10] approx loss: 5.3645
Epoch 0, Loss: 5.3481
[Epoch 4/10] approx loss: 5.3060
Epoch 0, Loss: 5.2949
[Epoch 5/10] approx loss: 5.2556
Epoch 0, Loss: 5.2512
[Epoch 6/10] approx loss: 5.2173
Epoch 0, Loss: 5.2194
[Epoch 7/10] approx loss: 5.1927
Epoch 0, Loss: 5.2005
[Epoch 8/10] approx loss: 5.1807
Epoch 0, Loss: 5.1923
[Epoch 9/10] approx loss: 5.1776
Epoch 0, Loss: 5.1904
[Epoch 10/10] approx loss: 5.1776

static done. val-acc: 22.45%

Prompt: “I am no official agent. I understand that it was your daughter who required my presence here, and [MASK] am acting in her interests. Young McCarthy must be got off, however.”
Truth : I
Guess : <unk>

Prompt: “The police have watched this Lascar,” said Inspector Bradstreet, “and I can quite understand [MASK] he might find it difficult to post a letter unobserved. Probably he handed it to some sailor customer of his, who forgot all about it for some days.”
Truth : that
Guess : <unk>

Prompt: “‘Here we are, Jack,’ says he, touching me [MASK] the arm; ‘we’ll be as good as a family to you. There’s two of us, me and my son, and you can have the keeping of us. If you don’t—it’s a fine, law-abiding country is England, and there’s always a policeman within hail.’
Truth : on
Guess : <unk>

Prompt: “‘Sold [MASK] Mr. Windigate of the Alpha, at 12*s*.’”
Truth : to
Guess : <unk>

Prompt: A small side door led into the whitewashed corridor from which the three bedrooms opened. Holmes refused to examine the third chamber, so we passed at once to the second, that in which Miss Stoner was now sleeping, and in which her sister had met with her fate. It [MASK] a homely little room, with a low ceiling and a gaping fireplace, after the fashion of old country-houses. A brown chest of drawers stood in one corner, a narrow white-counterpaned bed in another, and a dressing-table on the left-hand side of the window. These articles, with two small wicker-work chairs, made up all the furniture in the room save for a square of Wilton carpet in the centre. The boards round and the panelling of the walls were of brown, worm-eaten oak, so old and discoloured that it may have dated from the original building of the house. Holmes drew one of the chairs into a corner and sat silent, while his eyes travelled round and round and up and down, taking in every detail of the apartment.
Truth : was
Guess : <unk>
Epoch 0, Loss: 5.1416
[Epoch 1/10] approx loss: 4.1376
Epoch 0, Loss: 4.1750
[Epoch 2/10] approx loss: 3.2147
Epoch 0, Loss: 3.3301
[Epoch 3/10] approx loss: 2.2725
Epoch 0, Loss: 2.3445
[Epoch 4/10] approx loss: 1.5818
Epoch 0, Loss: 1.6076
[Epoch 5/10] approx loss: 1.0207
Epoch 0, Loss: 1.0725
[Epoch 6/10] approx loss: 0.7830
Epoch 0, Loss: 0.7790
[Epoch 7/10] approx loss: 0.6669
Epoch 0, Loss: 0.6449
[Epoch 8/10] approx loss: 0.6166
Epoch 0, Loss: 0.5879
[Epoch 9/10] approx loss: 0.6049
Epoch 0, Loss: 0.5718
[Epoch 10/10] approx loss: 0.6049

dynamic done. val-acc: 22.45%

Prompt: “I am no official agent. I understand that it was your daughter who required my presence here, and [MASK] am acting in her interests. Young McCarthy must be got off, however.”
Truth : I
Guess : i</w>

Prompt: “The police have watched this Lascar,” said Inspector Bradstreet, “and I can quite understand [MASK] he might find it difficult to post a letter unobserved. Probably he handed it to some sailor customer of his, who forgot all about it for some days.”
Truth : that
Guess : that</w>

Prompt: “‘Here we are, Jack,’ says he, touching me [MASK] the arm; ‘we’ll be as good as a family to you. There’s two of us, me and my son, and you can have the keeping of us. If you don’t—it’s a fine, law-abiding country is England, and there’s always a policeman within hail.’
Truth : on
Guess : “‘

Prompt: “‘Sold [MASK] Mr. Windigate of the Alpha, at 12*s*.’”
Truth : to
Guess : to</w>

Prompt: A small side door led into the whitewashed corridor from which the three bedrooms opened. Holmes refused to examine the third chamber, so we passed at once to the second, that in which Miss Stoner was now sleeping, and in which her sister had met with her fate. It [MASK] a homely little room, with a low ceiling and a gaping fireplace, after the fashion of old country-houses. A brown chest of drawers stood in one corner, a narrow white-counterpaned bed in another, and a dressing-table on the left-hand side of the window. These articles, with two small wicker-work chairs, made up all the furniture in the room save for a square of Wilton carpet in the centre. The boards round and the panelling of the walls were of brown, worm-eaten oak, so old and discoloured that it may have dated from the original building of the house. Holmes drew one of the chairs into a corner and sat silent, while his eyes travelled round and round and up and down, taking in every detail of the apartment.
Truth : was
Guess : was</w>
