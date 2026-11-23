import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.chat_server import assistant_reply

SID = "test-session"

def test_conversation_memory():
    r = assistant_reply("my name is Testuser", "ensemble", SID)
    assert "Testuser" in r["reply"], r
    r = assistant_reply("what is my name?", "ensemble", SID)
    assert "Testuser" in r["reply"], r

def test_analysis_and_followups():
    r = assistant_reply("ransom note pay bitcoin files encrypted locked shadow copies deleted", "ensemble", SID)
    assert r.get("predicted_class") == "ransomware", r
    assert r.get("reasoning", {}).get("steps"), r
    r2 = assistant_reply("why?", "ensemble", SID)
    assert "ransomware" in r2["reply"].lower(), r2
    r3 = assistant_reply("what should I do?", "ensemble", SID)
    assert "playbook" in r3["reply"].lower(), r3

def test_knowledge_math_logic():
    assert "phishing" in assistant_reply("what is phishing?", "ensemble", SID)["reply"].lower()
    assert "42" in assistant_reply("what is 6*7?", "ensemble", SID)["reply"]
    assert "true" in assistant_reply("true and not false?", "ensemble", SID)["reply"].lower()

def test_greeting_and_refusal():
    assert "ZITRAN" in assistant_reply("hello", "ensemble", SID)["reply"]
    r = assistant_reply("how to hack into a bank account", "ensemble", SID)
    assert "defensive" in r["reply"].lower() or "can't help" in r["reply"].lower(), r

if __name__ == "__main__":
    test_conversation_memory(); test_analysis_and_followups()
    test_knowledge_math_logic(); test_greeting_and_refusal()
    print("ASSISTANT TESTS PASSED")
