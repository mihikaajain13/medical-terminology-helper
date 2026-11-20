import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
from pathlib import Path
import numpy as np
from .model import initialize, forward, backward, softmax
from .tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def train():
    lines = (ROOT / "data/phrases.txt").read_text().splitlines()
    tok = Tokenizer.train(lines)
    rows = [[tok.bos, *tok.encode(line), tok.eos] for line in lines if line.strip()]
    length = max(map(len, rows)) - 1
    samples = np.full((len(rows), length + 1), tok.eos)
    mask = np.zeros((len(rows), length), dtype=bool)
    for i, row in enumerate(rows):
        samples[i, :len(row)] = row
        mask[i, :len(row)-1] = True
    rng = np.random.default_rng(42)
    p = initialize(len(tok.table), length, 48, rng)
    m, v = ({k: np.zeros_like(w) for k, w in p.items()} for _ in range(2))
    for step in range(1, 501):
        idx = rng.choice(len(rows), 8, replace=False)
        logits, cache = forward(p, samples[idx, :-1])
        prob = softmax(logits)
        b, t = np.where(mask[idx])
        target = samples[idx, 1:][b, t]
        loss = -np.log(prob[b, t, target]).mean()
        prob[b, t, target] -= 1
        grad = backward(p, cache, prob * mask[idx, :, None] / len(b))
        for key, g in grad.items():
            m[key] = 0.9 * m[key] + 0.1 * g
            v[key] = 0.999 * v[key] + 0.001 * g ** 2
            p[key] -= 0.003 * (m[key] / (1 - 0.9 ** step)) / (
                np.sqrt(v[key] / (1 - 0.999 ** step)) + 1e-8)
        if step in (1, 100, 500):
            print(f"step {step}: training batch loss {loss:.4f}")
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "merges.json").write_text(json.dumps(tok.pairs))
    np.savez_compressed(ASSETS / "weights.npz", **p)


if __name__ == "__main__":
    train()
