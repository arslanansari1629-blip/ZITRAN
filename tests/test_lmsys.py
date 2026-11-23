import sys, os, gzip, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.lmsys_data import PROCESSED
from zitran.sharegpt_data import _safe

def test_lmsys_filtered_pairs():
    assert os.path.exists(PROCESSED), "run: python3 -m zitran.lmsys_data"
    n, docs = 0, []
    with gzip.open(PROCESSED, "rt", encoding="utf-8") as f:
        for line in f:
            n += 1
            if len(docs) < 300: docs.append(json.loads(line)["text"])
    assert n > 6000, n
    for d in docs:
        assert d.startswith("Human: ") and "\nAssistant: " in d, d[:100]
        h, g = d[7:].split("\nAssistant: ", 1)
        assert _safe(h, g), d[:120]
    print(f"lmsys filtered pairs: {n:,}")

if __name__ == "__main__":
    test_lmsys_filtered_pairs(); print("LMSYS TESTS PASSED")
