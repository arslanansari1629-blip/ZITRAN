"""Temperature calibration: fit a single scalar T on validation logits so
reported confidences mean what they say (minimise NLL over a grid)."""
import numpy as np
from zitran.model import softmax

def collect_logits(model, X, bs=256):
    outs = []
    for i in range(0, len(X), bs):
        logits, _ = model.forward(X[i:i+bs])
        outs.append(logits)
    return np.vstack(outs)

def fit_temperature(logits, y, grid=None):
    grid = grid or np.concatenate([[0.5, 0.7, 0.85], np.linspace(1.0, 3.0, 21)])
    best_t, best_nll = 1.0, 1e18
    for t in grid:
        p = softmax(logits / t)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best_nll: best_t, best_nll = float(t), float(nll)
    return best_t, best_nll

def expected_calibration_error(probs, y, bins=10):
    conf = probs.max(1); pred = probs.argmax(1); ece = 0.0
    for b in range(bins):
        m = (conf > b/bins) & (conf <= (b+1)/bins)
        if m.sum(): ece += abs((pred[m] == y[m]).mean() - conf[m].mean()) * m.mean()
    return float(ece)
