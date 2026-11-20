"""One pre-normalized, single-head decoder block with analytic gradients."""
import numpy as np


def initialize(vocab, length, width, rng):
    shapes = dict(embed=(vocab, width), pos=(length, width),
                  q=(width, width), k=(width, width), v=(width, width),
                  proj=(width, width), up=(width, 2 * width),
                  down=(2 * width, width), head=(width, vocab))
    return {key: (rng.normal(size=shape) / np.sqrt(width)).astype("f")
            for key, shape in shapes.items()}


def norm(x):
    centered = x - x.mean(-1, keepdims=True)
    inv = 1 / np.sqrt((centered ** 2).mean(-1, keepdims=True) + 1e-5)
    return centered * inv, inv


def norm_grad(d, normalized, inv):
    return inv * (d - d.mean(-1, keepdims=True)
                  - normalized * (d * normalized).mean(-1, keepdims=True))


def softmax(scores):
    exp = np.exp(scores - scores.max(-1, keepdims=True))
    return exp / exp.sum(-1, keepdims=True)


def forward(p, ids):
    length = ids.shape[1]
    x = p["embed"][ids] + p["pos"][:length]
    n, inv = norm(x)
    q, k, v = [n @ p[key] for key in ("q", "k", "v")]
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.swapaxes(1, 2)) * scale
    scores[:, np.triu_indices(length, 1)[0], np.triu_indices(length, 1)[1]] = -np.inf
    att = softmax(scores)
    mix = att @ v
    residual = x + mix @ p["proj"]
    z, iz = norm(residual)
    hidden = np.maximum(z @ p["up"], 0)
    out = residual + hidden @ p["down"]
    return out @ p["head"], (ids, n, inv, q, k, v, att, mix, z, iz, hidden, out)


def weight_grad(x, d):
    return x.reshape(-1, x.shape[-1]).T @ d.reshape(-1, d.shape[-1])


def backward(p, cache, d):
    ids, n, inv, q, k, v, att, mix, z, iz, hidden, out = cache
    g = {"head": weight_grad(out, d)}
    dr = d @ p["head"].T
    g["down"] = weight_grad(hidden, dr)
    dh = (dr @ p["down"].T) * (hidden > 0)
    g["up"] = weight_grad(z, dh)
    dr = dr + norm_grad(dh @ p["up"].T, z, iz)
    g["proj"] = weight_grad(mix, dr)
    dm = dr @ p["proj"].T
    da = dm @ v.swapaxes(1, 2)
    ds = att * (da - (da * att).sum(-1, keepdims=True)) / np.sqrt(q.shape[-1])
    dn = np.zeros_like(n)
    for name, grad in (("q", ds @ k), ("k", ds.swapaxes(1, 2) @ q),
                       ("v", att.swapaxes(1, 2) @ dm)):
        g[name] = weight_grad(n, grad)
        dn += grad @ p[name].T
    dx = dr + norm_grad(dn, n, inv)
    g["pos"] = np.zeros_like(p["pos"])
    g["pos"][:ids.shape[1]] = dx.sum(0)
    g["embed"] = np.zeros_like(p["embed"])
    np.add.at(g["embed"], ids, dx)
    return g
