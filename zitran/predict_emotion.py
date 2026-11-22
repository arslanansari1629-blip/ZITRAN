"""Emotion detection inference (GoEmotions-trained NumPy Transformer)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.model import ZITRAN, softmax
from zitran.emotion_data import EMOTION_LABELS

EMOJI = {"joy": "😊", "neutral": "😐", "sadness": "😢", "anger": "😠", "fear": "😨", "surprise": "😲", "disgust": "🤢"}

class EmotionDetector:
    def __init__(self, artifacts="artifacts_emotion"):
        root = os.path.join(os.path.dirname(__file__), "..")
        self.tok = BPETokenizer.load(os.path.join(root, artifacts, "tokenizer.json"))
        self.model = ZITRAN.load(os.path.join(root, artifacts, "emotion_model.npz"))
    def predict(self, text):
        x = np.array([self.tok.encode(text)])
        logits, _ = self.model.forward(x)
        probs = softmax(logits)[0]; i = int(probs.argmax())
        return {"emotion": EMOTION_LABELS[i], "emoji": EMOJI[EMOTION_LABELS[i]],
                "confidence": round(float(probs[i]), 4),
                "top3": [{"emotion": EMOTION_LABELS[j], "prob": round(float(probs[j]), 4)} for j in probs.argsort()[::-1][:3]]}

if __name__ == "__main__":
    d = EmotionDetector()
    for s in sys.argv[1:] or ["I just got promoted, this is the best day ever!",
                              "I'm terrified the hackers deleted everything",
                              "This stupid system locked me out again, so frustrating"]:
        print(d.predict(s))
