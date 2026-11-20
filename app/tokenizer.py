import re
from collections import Counter


def parts(text):
    return [list(word.encode()) for word in re.findall(r"\w+|\s+|[^\w\s]", text.lower())]


def merge(tokens, pair, new_id):
    result = []
    i = 0
    while i < len(tokens):
        if tuple(tokens[i:i + 2]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


class Tokenizer:
    bos, eos = 256, 257

    def __init__(self, pairs):
        self.pairs = [tuple(pair) for pair in pairs]
        self.table = [bytes([i]) for i in range(256)] + [b"", b""]
        for left, right in self.pairs:
            self.table.append(self.table[left] + self.table[right])

    @classmethod
    def train(cls, lines, limit=220):
        words = [word for line in lines for word in parts(line)]
        pairs = []
        for i in range(limit):
            counts = Counter(pair for word in words for pair in zip(word, word[1:]))
            if not counts:
                break
            pair = counts.most_common(1)[0][0]
            pairs.append(pair)
            words = [merge(word, pair, 258 + i) for word in words]
        return cls(pairs)

    def encode(self, text):
        words = parts(text)
        for i, pair in enumerate(self.pairs):
            words = [merge(word, pair, 258 + i) for word in words]
        return [token for word in words for token in word]

    def decode(self, tokens):
        return b"".join(self.table[t] for t in tokens).decode(errors="replace")
