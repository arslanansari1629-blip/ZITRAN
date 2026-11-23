"""ZITRAN BPE tokenizer — implemented from scratch (no libraries beyond stdlib)."""
import json, re
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=2500, max_len=40):
        self.vocab_size, self.max_len = vocab_size, max_len
        self.token_to_id, self.id_to_token, self.merges = {}, {}, []
        self._cache = {}; self._ranks = None
        self.specials = ["<PAD>", "<UNK>", "<CLS>", "<SEP>"]

    def _words(self, text):
        text = text.lower()
        # keep security-relevant symbols as separate/attached tokens
        text = re.sub(r"([:/.\-_?=&%'\"])", r" \1 ", text)
        return text.split()

    def train(self, texts):
        # initial vocab: characters + specials
        corpus = [self._words(t) for t in texts]
        word_freq = Counter(w for sent in corpus for w in sent)
        # represent word as tuple of chars + end marker
        splits = {w: tuple(list(w) + ["</w>"]) for w in word_freq}
        vocab = set(self.specials)
        for w in splits:
            vocab.update(splits[w])
        merges = []
        while len(vocab) < self.vocab_size:
            pairs = Counter()
            for w, freq in word_freq.items():
                s = splits[w]
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            best, _ = pairs.most_common(1)[0]
            new_token = best[0] + best[1]
            merges.append(best)
            vocab.add(new_token)
            for w in splits:
                s, out, i = splits[w], [], 0
                while i < len(s):
                    if i < len(s) - 1 and (s[i], s[i+1]) == best:
                        out.append(new_token); i += 2
                    else:
                        out.append(s[i]); i += 1
                splits[w] = tuple(out)
        self.merges = merges
        tokens = sorted(vocab - set(self.specials))
        self.token_to_id = {t: i for i, t in enumerate(self.specials + tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        return self

    _cache = {}
    def _encode_word(self, word):
        if word in self._cache:
            return self._cache[word]
        if not hasattr(self, "_ranks") or self._ranks is None:
            self._ranks = {tuple(m): i for i, m in enumerate(self.merges)}
        symbols = list(word) + ["</w>"]
        # rank-order BPE: repeatedly merge the lowest-rank adjacent pair
        # (equivalent to applying merges sequentially, much faster)
        while len(symbols) > 1:
            best_i, best_rank = -1, None
            for i in range(len(symbols) - 1):
                r = self._ranks.get((symbols[i], symbols[i+1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0: break
            pair = (symbols[best_i], symbols[best_i+1]); merged = pair[0] + pair[1]
            out, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
                    out.append(merged); i += 2
                else:
                    out.append(symbols[i]); i += 1
            symbols = out
        self._cache[word] = symbols
        return symbols

    def encode(self, text, add_specials=True):
        ids = [self.token_to_id["<CLS>"]] if add_specials else []
        for w in self._words(text):
            for tok in self._encode_word(w):
                ids.append(self.token_to_id.get(tok, self.token_to_id["<UNK>"]))
        if add_specials:
            ids.append(self.token_to_id["<SEP>"])
        ids = ids[:self.max_len]
        pad = self.token_to_id["<PAD>"]
        return ids + [pad] * (self.max_len - len(ids))

    def decode(self, ids):
        toks = [self.id_to_token.get(int(i), "<UNK>") for i in ids]
        return " ".join(t for t in toks if t not in self.specials).replace("</w>", "").strip()

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size, "max_len": self.max_len,
                       "token_to_id": self.token_to_id, "merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        obj = cls(d["vocab_size"], d["max_len"])
        obj.token_to_id = d["token_to_id"]
        obj.id_to_token = {int(i): t for t, i in d["token_to_id"].items()}
        obj.merges = [tuple(m) for m in d["merges"]]
        return obj

    def __len__(self):
        return len(self.token_to_id)
