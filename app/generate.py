import json
import numpy as np
from .model import forward
from .tokenizer import Tokenizer
from .train import ASSETS


class Helper:
    def __init__(self):
        self.tokenizer = Tokenizer(json.loads((ASSETS / "merges.json").read_text()))
        with np.load(ASSETS / "weights.npz", allow_pickle=False) as archive:
            self.weights = dict(archive)

    def complete(self, prompt, limit=48):
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Enter a partial medical phrase.")
        tok = self.tokenizer
        ids = [tok.bos, *tok.encode(prompt)]
        generated = []
        for _ in range(limit):
            context = ids[-len(self.weights["pos"]):]
            logits, _ = forward(self.weights, np.array([context]))
            token = int(np.argmax(logits[0, -1]))
            if token == tok.eos:
                break
            ids.append(token)
            generated.append(token)
            if tok.table[token] == b".":
                break
        return prompt + tok.decode(generated)
