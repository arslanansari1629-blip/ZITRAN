import sys, os, gzip, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.sharegpt_data import PROCESSED, BLOCK, _safe

def test_processed_sharegpt_exists_and_is_filtered():
    assert os.path.exists(PROCESSED), "run: python3 -m zitran.sharegpt_data"
    docs = []
    with gzip.open(PROCESSED, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 400: break
            docs.append(json.loads(line)["text"])
    assert len(docs) == 400
    for d in docs:
        assert d.startswith("Human: ") and "\nAssistant: " in d, d[:100]
        h, g = d[7:].split("\nAssistant: ", 1)
        assert _safe(h, g), d[:120]  # nothing blocked slipped through

def test_total_filtered_pairs():
    n = 0
    with gzip.open(PROCESSED, "rt", encoding="utf-8") as f:
        for _ in f: n += 1
    assert n > 80_000, n  # 94,145 scanned -> 85,817 kept
    print(f"sharegpt filtered pairs: {n:,}")

if __name__ == "__main__":
    test_processed_sharegpt_exists_and_is_filtered(); test_total_filtered_pairs()
    print("SHAREGPT TESTS PASSED")
