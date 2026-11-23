import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.chat_server import available_models, chat_reply

def test_models_available():
    ids = [m["id"] for m in available_models()]
    assert "v1" in ids and "ensemble" in ids, ids

def test_chat_ransomware():
    r = chat_reply("ransom note pay bitcoin files encrypted locked shadow copies deleted", "ensemble")
    assert r["predicted_class"] == "ransomware", r
    assert r["risk_level"] == "CRITICAL" and r["is_threat"], r

def test_chat_benign_and_empty():
    r = chat_reply("user alice logged in successfully via mfa dashboard backup completed", "v4")
    assert "predicted_class" in r and "reply" in r, r
    assert chat_reply("   ", "ensemble") == {"error": "empty message"}

if __name__ == "__main__":
    test_models_available(); test_chat_ransomware(); test_chat_benign_and_empty(); print("CHAT TESTS PASSED")
