import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.model import ZITRAN
from zitran.data import generate, LABELS

def test_tokenizer_roundtrip():
    data = generate(30, seed=1); tok = BPETokenizer(800, 40).train([t for t, _ in data])
    ids = tok.encode("sql injection attack detected")
    assert len(ids) == 40 and ids[0] == tok.token_to_id["<CLS>"]

def test_gradient_check():
    # numerical vs analytical gradient on the classification head — verifies hand-written backprop
    m = ZITRAN(200, d=32, heads=2, ff=64, layers=1, seed=3)
    rng = np.random.default_rng(0); x = rng.integers(3, 200, (4, 12)); y = rng.integers(0, 10, 4); yb = (y != 0).astype(int)
    _, _, grads = m.loss_and_grads(x, y, yb)
    eps = 1e-5; W = m.Wc; num = np.zeros_like(W)
    it = np.nditer(W, flags=["multi_index"])
    checked = 0
    for _ in range(12):  # sample 12 entries
        idx = (rng.integers(0, W.shape[0]), rng.integers(0, W.shape[1])); old = W[idx]
        W[idx] = old + eps; lp, _, _ = m.loss_and_grads(x, y, yb)
        W[idx] = old - eps; lm, _, _ = m.loss_and_grads(x, y, yb)
        W[idx] = old; num[idx] = (lp - lm) / (2 * eps); checked += 1
        assert abs(num[idx] - grads["Wc"][idx]) < 1e-3, (idx, num[idx], grads["Wc"][idx])
    assert checked == 12

def test_training_reduces_loss():
    data = generate(60, seed=2); tok = BPETokenizer(1000, 40).train([t for t, _ in data])
    X = np.array([tok.encode(t) for t, _ in data]); y = np.array([l for _, l in data]); yb = (y != 0).astype(int)
    m = ZITRAN(len(tok), d=64, heads=2, ff=128, layers=2, seed=5)
    l0, _, _ = m.loss_and_grads(X[:32], y[:32], yb[:32]); m.adam_update(3e-3)
    for _ in range(30):
        m.loss_and_grads(X[:32], y[:32], yb[:32]); m.adam_update(3e-3)
    l1, acc, _ = m.loss_and_grads(X[:32], y[:32], yb[:32])
    assert l1 < l0, (l0, l1)
if __name__ == "__main__":
    test_tokenizer_roundtrip(); test_gradient_check(); test_training_reduces_loss(); print("ALL TESTS PASSED")
