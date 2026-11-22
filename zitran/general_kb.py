"""General-knowledge skill: answer everyday questions ('who is Ronaldo?',
'capital of France?') from (1) a curated fact base, (2) country->capital
table, (3) retrieval over the 52k Alpaca instruction answers. Honest when
it doesn't know — no hallucinated facts.
"""
import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.join(os.path.dirname(__file__), "..")

FACTS = {
    "ronaldo": "Cristiano Ronaldo (Cristiano Ronaldo dos Santos Aveiro, born 5 February 1985 in Funchal, Madeira, Portugal) is a Portuguese professional footballer who plays as a forward and captains the Portugal national team. He has played for Sporting CP, Manchester United, Real Madrid and Juventus, and is famous for his goalscoring records — he is the all-time top scorer in men's international football and a five-time Ballon d'Or winner. He won Euro 2016 with Portugal and five UEFA Champions League titles.",
    "cristiano ronaldo": None,  # alias -> ronaldo
    "messi": "Lionel Messi (born 24 June 1987 in Rosario, Argentina) is an Argentine professional footballer, widely regarded as one of the greatest players ever. He spent most of his career at FC Barcelona, later played for Paris Saint-Germain and Inter Miami, and captained Argentina to victory at the 2022 FIFA World Cup. He has won the Ballon d'Or a record eight times.",
    "neymar": "Neymar Jr (born 5 February 1992 in Brazil) is a Brazilian football forward known for his dribbling and flair, who has played for Santos, FC Barcelona and Paris Saint-Germain.",
    "einstein": "Albert Einstein (1879–1955) was a German-born theoretical physicist who developed the theory of relativity (E = mc²), revolutionising our understanding of space, time and gravity. He won the Nobel Prize in Physics in 1921 for the photoelectric effect.",
    "newton": "Isaac Newton (1643–1727) was an English physicist and mathematician who formulated the laws of motion and universal gravitation, co-invented calculus, and wrote the Principia Mathematica.",
    "gandhi": "Mahatma Gandhi (1869–1948) was the leader of India's non-violent independence movement against British rule, famous for civil disobedience and peaceful protest (Satyagraha).",
    "shakespeare": "William Shakespeare (1564–1616) was an English playwright and poet, widely called the greatest writer in the English language — author of Hamlet, Romeo and Juliet, Macbeth and 37+ plays and 154 sonnets.",
    "modi": "Narendra Modi (born 17 September 1950) is the Prime Minister of India, in office since 2014, previously Chief Minister of Gujarat.",
    "obama": "Barack Obama (born 4 August 1961) is an American politician who served as the 44th President of the United States (2009–2017), the first African-American president.",
    "elon musk": "Elon Musk (born 28 June 1971 in South Africa) is a billionaire entrepreneur, CEO of Tesla and SpaceX, known for electric vehicles, reusable rockets, and ventures like Neuralink and xAI.",
    "bill gates": "Bill Gates (born 28 October 1955) is an American businessman who co-founded Microsoft and later became a major philanthropist through the Gates Foundation.",
    "moon": "The Moon is Earth's only natural satellite, about 384,400 km away. It causes tides, stabilises Earth's tilt, and was first walked on by Neil Armstrong and Buzz Aldrin (Apollo 11, 1969).",
    "sun": "The Sun is the star at the centre of our Solar System — a nearly perfect sphere of hot plasma holding 99.86% of the system's mass. Its core fuses hydrogen into helium at ~15 million °C, providing Earth's light and heat.",
    "water": "Water (H₂O) is a molecule of two hydrogen atoms and one oxygen atom. It covers ~71% of Earth's surface, boils at 100°C and freezes at 0°C at sea level, and is essential to all known life.",
    "python language": "Python is a popular high-level programming language created by Guido van Rossum (1991), known for readable syntax and huge libraries for web, data science, automation and AI.",
    "india": "India is a large South Asian country — the world's most populous (over 1.4 billion people), capital New Delhi, known for its diverse languages, cultures, and as the world's largest democracy.",
    "internet": "The Internet is the global network of interconnected computers communicating via standard protocols (TCP/IP). It grew from ARPANET (1969); the World Wide Web (invented by Tim Berners-Lee, 1989) runs on top of it.",
    "ai": "Artificial Intelligence (AI) is the field of building systems that perform tasks needing human-like intelligence — learning, reasoning, language, perception. Modern AI is dominated by machine learning and neural networks trained on large datasets.",
    "machine learning": "Machine learning is a subset of AI where systems learn patterns from data instead of being explicitly programmed — e.g. classifiers trained on examples, like ZITRAN's own models.",
    "bitcoin": "Bitcoin (BTC), launched in 2009 by the pseudonymous Satoshi Nakamoto, is the first decentralised cryptocurrency, using a public blockchain and proof-of-work mining, with supply capped at 21 million coins.",
}
CAPITALS = {"france": "Paris", "india": "New Delhi", "usa": "Washington, D.C.", "united states": "Washington, D.C.", "uk": "London", "england": "London", "germany": "Berlin", "italy": "Rome", "spain": "Madrid", "portugal": "Lisbon", "japan": "Tokyo", "china": "Beijing", "russia": "Moscow", "brazil": "Brasília", "canada": "Ottawa", "australia": "Canberra", "pakistan": "Islamabad", "bangladesh": "Dhaka", "nepal": "Kathmandu", "sri lanka": "Colombo (Sri Jayawardenepura Kotte)", "egypt": "Cairo", "france ": "Paris", "turkey": "Ankara", "iran": "Tehran", "iraq": "Baghdad", "saudi arabia": "Riyadh", "uae": "Abu Dhabi", "indonesia": "Jakarta", "malaysia": "Kuala Lumpur", "singapore": "Singapore", "thailand": "Bangkok", "vietnam": "Hanoi", "south korea": "Seoul", "north korea": "Pyongyang", "mexico": "Mexico City", "argentina": "Buenos Aires", "chile": "Santiago", "peru": "Lima", "south africa": "Pretoria", "nigeria": "Abuja", "kenya": "Nairobi", "ethiopia": "Addis Ababa", "morocco": "Rabat", "algeria": "Algiers", "ukraine": "Kyiv", "poland": "Warsaw", "netherlands": "Amsterdam", "belgium": "Brussels", "sweden": "Stockholm", "norway": "Oslo", "denmark": "Copenhagen", "finland": "Helsinki", "switzerland": "Bern (de facto)", "austria": "Vienna", "greece": "Athens", "romania": "Bucharest", "hungary": "Budapest", "czech republic": "Prague", "ireland": "Dublin", "iceland": "Reykjavík", "new zealand": "Wellington", "afghanistan": "Kabul", "myanmar": "Naypyidaw", "cambodia": "Phnom Penh", "philippines": "Manila", "qatar": "Doha", "kuwait": "Kuwait City", "oman": "Muscat", "jordan": "Amman", "israel": "Jerusalem", "lebanon": "Beirut", "syria": "Damascus", "yemen": "Sanaa", "kazakhstan": "Astana", "uzbekistan": "Tashkent", "mongolia": "Ulaanbaatar", "bhutan": "Thimphu", "maldives": "Malé", "fiji": "Suva"}

_alpaca_index = None
def _load_alpaca_index():
    global _alpaca_index
    if _alpaca_index is not None: return _alpaca_index
    path = os.path.join(ROOT, "data", "raw", "instruction", "alpaca_data.json")
    idx = []
    if os.path.exists(path):
        for ex in json.load(open(path)):
            q = (ex["instruction"] + " " + ex.get("input", "")).strip()
            idx.append((q.lower(), ex["output"].strip()))
    _alpaca_index = idx
    return idx

STOP = set("who is the a an what where when why how do does did is are was were tell me about".split())
def _words(t): return [w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in STOP]

def answer_general(question):
    """Return a factual answer string, or None if not confidently known."""
    low = question.lower()
    m = re.search(r"capital of ([a-z ]+?)\??$", low)
    if m:
        c = m.group(1).strip()
        if c in CAPITALS: return f"The capital of {c.title()} is **{CAPITALS[c]}**."
        return None
    for key, val in FACTS.items():
        if val and re.search(rf"\b{re.escape(key)}\b", low):
            return val
    if FACTS.get("cristiano ronaldo") is None and "cristiano" in low:
        return FACTS["ronaldo"]
    # retrieval over Alpaca instruction answers
    words = set(_words(low))
    if len(words) >= 2:
        best, best_score = None, 0
        for q, out in _load_alpaca_index():
            score = len(words & set(_words(q)))
            # require covering most of the question's content words
            if score > best_score and score >= max(2, len(words) - 1):
                best, best_score = out, score
        if best and len(best) > 25:
            return best[:900]
    return None
