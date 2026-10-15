# microGPT

A tiny GPT, trained and sampled in about 300 lines of pure Python with no dependencies. It learns from a list of names, one character at a time, and then generates new, made-up names.

The autograd engine is implemented at the scalar level, so you can see exactly how backpropagation works. It is slow by design: it is meant for learning, not for speed.

Based on Andrej Karpathy's microGPT.

## Why I made this

I wanted to understand how a GPT really works, not just use one. Libraries like PyTorch hide the important parts behind a few function calls, so I worked from a minimal, dependency-free implementation instead: the tokenizer, the autograd engine, the transformer, the optimizer, and the sampling loop, all in one readable file.

I started from Karpathy's microGPT and cleaned it up (functions, named constants, docstrings) so the code is easier to read, study and modify. The behavior is unchanged: it produces the same output as the original for the same seed.

## Requirements

- Python 3.8+
- Internet access on the first run (to download the dataset). After that, `input.txt` is reused.

## Usage

```bash
python microgpt_clean.py
```

The script will:

1. Download `names.txt` from [karpathy/makemore](https://github.com/karpathy/makemore) and save it as `input.txt` (skipped if the file already exists).
2. Train for 1000 steps, printing the loss as it goes.
3. Print 20 newly generated names.

Because the random seed is fixed (`SEED = 42`), runs are reproducible.

To use your own data, replace `input.txt` with a text file containing one document per line.

## How it works

| Part | What it does |
| --- | --- |
| Tokenizer | Each unique character becomes an integer id. One extra `BOS` token marks the start and end of a name. |
| `Value` | A scalar autograd node. It records its inputs and local derivatives, and `backward()` applies the chain rule in reverse. |
| `gpt()` | The model: token and position embeddings, multi-head self-attention, an MLP, and a final linear layer producing logits. |
| `train()` | One name per step. Cross-entropy loss, backward pass, then the Adam optimizer with linear learning-rate decay. |
| `sample_name()` | Starts from `BOS`, samples one character at a time with temperature, and stops when `BOS` is produced. |

The architecture follows GPT-2 with three simplifications: RMSNorm instead of LayerNorm, no biases, and ReLU instead of GeLU.

Attention is causal by construction: tokens are processed one at a time and their keys and values are cached, so each token can only attend to earlier positions.

## Configuration

All settings are constants at the top of `microgpt_clean.py`.

| Constant | Default | Meaning |
| --- | --- | --- |
| `N_LAYER` | 1 | Number of transformer layers |
| `N_EMBD` | 16 | Embedding dimension |
| `N_HEAD` | 4 | Attention heads |
| `BLOCK_SIZE` | 16 | Maximum context length |
| `NUM_STEPS` | 1000 | Training steps |
| `LEARNING_RATE` | 0.01 | Initial Adam learning rate |
| `BETA1`, `BETA2` | 0.85, 0.99 | Adam momentum terms |
| `TEMPERATURE` | 0.5 | Sampling temperature, in (0, 1]. Lower is more conservative. |
| `NUM_SAMPLES` | 20 | Names to generate |

## Notes

- With one name per step and 1000 steps, the model sees only about 1000 names, so output quality is modest. Raise `NUM_STEPS` for better results, at the cost of runtime.
- If you change the dataset, make sure `BLOCK_SIZE` is at least the length of your longest document plus one.

## Credits

This project is based on **microGPT by Andrej Karpathy**. The original design, algorithm and architecture are his work; I only reorganized the code to make it easier to read and study. All credit for the idea goes to him.

The names dataset comes from his [makemore](https://github.com/karpathy/makemore) project.