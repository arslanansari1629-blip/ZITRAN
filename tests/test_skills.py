import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.chat_server import assistant_reply

SID = "skills-test"
def test_who_is_ronaldo():
    r = assistant_reply("who is ronaldo?", "ensemble", SID)
    assert "Cristiano Ronaldo" in r["reply"] and "Portugal" in r["reply"], r["reply"]

def test_capital_and_math_still_work():
    assert "Paris" in assistant_reply("what is the capital of France?", "ensemble", SID)["reply"]
    assert "42" in assistant_reply("what is 6*7?", "ensemble", SID)["reply"]

def test_who_is_phishing_still_works():
    r = assistant_reply("what is phishing?", "ensemble", SID)
    assert "credential" in r["reply"].lower(), r["reply"]

def test_build_html_page():
    r = assistant_reply("build a dark blue login page called SecureBank with contact form", "ensemble", SID)
    assert r.get("generated_url", "").endswith(".html"), r
    path = os.path.join(os.path.dirname(__file__), "..", "web", "generated", r["generated_file"])
    html = open(path).read()
    assert "<html" in html and "SecureBank" in html and "Sign In" in html, html[:500]

def test_unknown_person_honest():
    r = assistant_reply("who is Zqxwv Blorpington?", "ensemble", SID)
    assert "don't have reliable facts" in r["reply"], r["reply"]


def test_currency_question_never_classified_fresh_and_after_incident():
    sid = "bug-regression-session"
    r1 = assistant_reply("ransomware encrypted all files, ransom note demands bitcoin", "ensemble", sid)
    assert r1.get("predicted_class") == "ransomware", r1["reply"][:150]
    r2 = assistant_reply("what currency is used in Japan?", "ensemble", sid)
    assert "yen" in r2["reply"].lower() and "predicted_class" not in r2, r2["reply"][:200]
    r3 = assistant_reply("what language is spoken in Brazil?", "ensemble", "bug-fresh")
    assert "Portuguese" in r3["reply"] and "predicted_class" not in r3, r3["reply"][:200]
    r4 = assistant_reply("who won the battle of waterloo?", "ensemble", "bug-fresh")
    assert "predicted_class" not in r4, r4["reply"][:200]  # question -> never classified

if __name__ == "__main__":
    test_who_is_ronaldo(); test_capital_and_math_still_work(); test_who_is_phishing_still_works()
    test_build_html_page(); test_unknown_person_honest(); test_currency_question_never_classified_fresh_and_after_incident(); print("SKILLS TESTS PASSED")

