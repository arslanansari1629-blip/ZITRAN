import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.general_kb import answer_general
from zitran.world_data import load_countries, load_elements, build_world_corpus
from zitran.self_learn import generate_self_teaching, quality_filter

def test_real_datasets_loaded():
    assert len(load_countries()) >= 240 and len(load_elements()) >= 118

def test_any_country_capital_currency_language():
    assert "Astana" in answer_general("what is the capital of Kazakhstan?")
    assert "yen" in answer_general("what currency is used in Japan?").lower()
    assert "Portuguese" in answer_general("what language is spoken in Brazil?")

def test_elements_and_encyclopedia():
    assert "Au" in answer_general("what is the chemical symbol for gold?")
    assert "Red Planet" in answer_general("tell me about mars")
    assert "Cristiano Ronaldo" in answer_general("who is ronaldo?")

def test_world_corpus_and_self_teaching():
    docs = build_world_corpus(); assert len(docs) > 1500
    teach = generate_self_teaching(); assert len(teach) > 2500
    assert all(30 <= len(d) <= 1400 for d in teach[:50])

def test_quality_filter_rejects_repetition():
    assert quality_filter(["the the the the the the the the the the the the the the"]) == []


def test_qa_bank_scale_and_retrieval():
    from zitran.world_data import load_qa_bank, answer_qa_bank
    assert len(load_qa_bank()) > 250_000  # trivia 49.7k + jeopardy 214k
    r = answer_qa_bank("For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory")
    assert r and "Copernicus" in r, r
    r2 = answer_qa_bank("of the 6 noble gases on the periodic table it is the lightest")
    assert r2 and "helium" in r2.lower(), r2

if __name__ == "__main__":
    test_real_datasets_loaded(); test_any_country_capital_currency_language()
    test_elements_and_encyclopedia(); test_world_corpus_and_self_teaching()
    test_quality_filter_rejects_repetition(); test_qa_bank_scale_and_retrieval(); print("WORLD + SELF-LEARN TESTS PASSED")

