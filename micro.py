"""microGPT: a tiny GPT trained and sampled in pure, dependency-free Python.

Trains a one-layer transformer on a list of names, character by character,
then samples new names from it. Autograd is implemented at the scalar level.
"""

import math
import os
import random
import urllib.request

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEED = 42

DATA_PATH = "input.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"

N_LAYER = 1                    # number of transformer layers
N_EMBD = 16                    # embedding dimension
BLOCK_SIZE = 16                # max context length (longest name is 15 chars)
N_HEAD = 4                     # attention heads
HEAD_DIM = N_EMBD // N_HEAD    # dimension per head

NUM_STEPS = 1000
LEARNING_RATE = 0.01
BETA1, BETA2, EPS_ADAM = 0.85, 0.99, 1e-8

TEMPERATURE = 0.5              # in (0, 1]; lower = more conservative samples
NUM_SAMPLES = 20


# ----------------------------------------------------------------------------
# Autograd
# ----------------------------------------------------------------------------
class Value:
    """A scalar that records how it was computed, so gradients can flow back."""

    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                  # value from the forward pass
        self.grad = 0                     # d(loss)/d(this node), set in backward
        self._children = children         # nodes this one was computed from
        self._local_grads = local_grads   # d(this node)/d(child) for each child

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        """Backpropagate from this node by applying the chain rule in reverse."""
        topo, visited = [], set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            for child, local_grad in zip(node._children, node._local_grads):
                child.grad += local_grad * node.grad


# ----------------------------------------------------------------------------
# Data and tokenizer
# ----------------------------------------------------------------------------
def load_docs(path=DATA_PATH, url=DATA_URL):
    """Load the dataset (downloading it if needed) as a shuffled list of strings."""
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    with open(path) as f:
        docs = [line.strip() for line in f if line.strip()]
    random.shuffle(docs)
    return docs


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def make_matrix(n_out, n_in, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(n_in)] for _ in range(n_out)]


def init_params(vocab_size):
    """Create all weight matrices. Returns (state_dict, flat list of parameters)."""
    state_dict = {
        "wte": make_matrix(vocab_size, N_EMBD),      # token embeddings
        "wpe": make_matrix(BLOCK_SIZE, N_EMBD),      # position embeddings
        "lm_head": make_matrix(vocab_size, N_EMBD),  # output projection
    }
    for i in range(N_LAYER):
        state_dict[f"layer{i}.attn_wq"] = make_matrix(N_EMBD, N_EMBD)
        state_dict[f"layer{i}.attn_wk"] = make_matrix(N_EMBD, N_EMBD)
        state_dict[f"layer{i}.attn_wv"] = make_matrix(N_EMBD, N_EMBD)
        state_dict[f"layer{i}.attn_wo"] = make_matrix(N_EMBD, N_EMBD)
        state_dict[f"layer{i}.mlp_fc1"] = make_matrix(4 * N_EMBD, N_EMBD)
        state_dict[f"layer{i}.mlp_fc2"] = make_matrix(N_EMBD, 4 * N_EMBD)
    params = [p for mat in state_dict.values() for row in mat for p in row]
    return state_dict, params


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]


def softmax(logits):
    max_val = max(val.data for val in logits)  # subtract max for numerical stability
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    mean_square = sum(xi * xi for xi in x) / len(x)
    scale = (mean_square + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def add_vectors(a, b):
    return [ai + bi for ai, bi in zip(a, b)]


def attention_block(x, layer, state_dict, keys, values):
    """Multi-head self-attention over all cached positions (causal by construction)."""
    q = linear(x, state_dict[f"layer{layer}.attn_wq"])
    k = linear(x, state_dict[f"layer{layer}.attn_wk"])
    v = linear(x, state_dict[f"layer{layer}.attn_wv"])
    keys[layer].append(k)
    values[layer].append(v)

    head_outputs = []
    for h in range(N_HEAD):
        lo, hi = h * HEAD_DIM, (h + 1) * HEAD_DIM
        q_h = q[lo:hi]
        k_h = [ki[lo:hi] for ki in keys[layer]]
        v_h = [vi[lo:hi] for vi in values[layer]]
        scores = [
            sum(q_h[j] * k_t[j] for j in range(HEAD_DIM)) / HEAD_DIM ** 0.5
            for k_t in k_h
        ]
        weights = softmax(scores)
        head_outputs.extend(
            sum(weights[t] * v_h[t][j] for t in range(len(v_h)))
            for j in range(HEAD_DIM)
        )
    return linear(head_outputs, state_dict[f"layer{layer}.attn_wo"])


def mlp_block(x, layer, state_dict):
    x = linear(x, state_dict[f"layer{layer}.mlp_fc1"])
    x = [xi.relu() for xi in x]
    return linear(x, state_dict[f"layer{layer}.mlp_fc2"])


def gpt(token_id, pos_id, keys, values, state_dict):
    """Map one token (at one position) to logits over the next token.

    Follows GPT-2 with minor differences: RMSNorm instead of LayerNorm,
    no biases, and ReLU instead of GeLU.
    """
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = add_vectors(tok_emb, pos_emb)
    x = rmsnorm(x)  # not redundant: affects the backward pass via the residual path

    for layer in range(N_LAYER):
        x = add_vectors(attention_block(rmsnorm(x), layer, state_dict, keys, values), x)
        x = add_vectors(mlp_block(rmsnorm(x), layer, state_dict), x)

    return linear(x, state_dict["lm_head"])


def new_kv_cache():
    return [[] for _ in range(N_LAYER)], [[] for _ in range(N_LAYER)]


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def document_loss(tokens, state_dict):
    """Average cross-entropy loss of the model over one tokenized document."""
    n = min(BLOCK_SIZE, len(tokens) - 1)
    keys, values = new_kv_cache()
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        probs = softmax(gpt(token_id, pos_id, keys, values, state_dict))
        losses.append(-probs[target_id].log())
    return (1 / n) * sum(losses)


def train(docs, char_to_id, bos, state_dict, params):
    m = [0.0] * len(params)  # Adam first-moment buffer
    v = [0.0] * len(params)  # Adam second-moment buffer

    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [bos] + [char_to_id[ch] for ch in doc] + [bos]

        loss = document_loss(tokens, state_dict)
        loss.backward()

        lr = LEARNING_RATE * (1 - step / NUM_STEPS)  # linear decay
        for i, p in enumerate(params):
            m[i] = BETA1 * m[i] + (1 - BETA1) * p.grad
            v[i] = BETA2 * v[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m[i] / (1 - BETA1 ** (step + 1))
            v_hat = v[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0

        print(f"step {step + 1:4d} / {NUM_STEPS:4d} | loss {loss.data:.4f}", end="\r")


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def sample_name(id_to_char, bos, vocab_size, state_dict):
    keys, values = new_kv_cache()
    token_id = bos
    chars = []
    for pos_id in range(BLOCK_SIZE):
        logits = gpt(token_id, pos_id, keys, values, state_dict)
        probs = softmax([logit / TEMPERATURE for logit in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == bos:
            break
        chars.append(id_to_char[token_id])
    return "".join(chars)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    random.seed(SEED)

    docs = load_docs()
    print(f"num docs: {len(docs)}")

    # Tokenizer: each unique character is a token id; one extra id marks BOS.
    uchars = sorted(set("".join(docs)))
    char_to_id = {ch: i for i, ch in enumerate(uchars)}
    id_to_char = dict(enumerate(uchars))
    bos = len(uchars)
    vocab_size = len(uchars) + 1
    print(f"vocab size: {vocab_size}")

    state_dict, params = init_params(vocab_size)
    print(f"num params: {len(params)}")

    train(docs, char_to_id, bos, state_dict, params)

    print("\n--- inference (new, hallucinated names) ---")
    for i in range(NUM_SAMPLES):
        print(f"sample {i + 1:2d}: {sample_name(id_to_char, bos, vocab_size, state_dict)}")


if __name__ == "__main__":
    main()