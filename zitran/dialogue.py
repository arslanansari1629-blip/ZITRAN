"""Conversational dialogue via retrieval over Cornell Movie-Dialogs pairs.
Given a casual user message, find dialogue contexts sharing the most content
words and return their human responses (filtered). This gives natural,
varied conversation without pretending to be a giant LLM.
"""
import os, re, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.emotion_data import load_dialogue_pairs

STOP = set("the a an is are was were i you he she it we they to of and or in on at for with me my your his her our their this that be am do does did have has had will would can could should what when where who how why not no yes so if but as from by".split())

class RetrievalDialogue:
    def __init__(self):
        self.pairs = load_dialogue_pairs()
        self.index = defaultdict(set)
        for i, p in enumerate(self.pairs):
            for w in set(self._words(p["context"])):
                self.index[w].add(i)

    def _words(self, t):
        return [w for w in re.findall(r"[a-z']{3,}", t.lower()) if w not in STOP]

    def respond(self, text, emotion=None):
        words = set(self._words(text))
        if not words or not self.pairs: return None
        scores = defaultdict(int)
        for w in words:
            for i in self.index.get(w, ()): scores[i] += 1
        if not scores: return None
        best_i, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score < 1: return None
        resp = self.pairs[best_i]["response"]
        if len(resp) < 2 or resp.lower() in ("yes.", "no.", "oh.", "well."): return None
        return {"response": resp, "matched_context": self.pairs[best_i]["context"], "overlap": best_score}
