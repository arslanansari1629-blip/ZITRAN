"""World knowledge from REAL structured datasets + curated encyclopedia.

Sources (GitHub, free): mledoze/countries (250 countries: capital, currency,
languages, region, population) and Bowserinator/Periodic-Table-JSON (118
elements with summaries) in data/raw/world/. Plus an expanded curated
encyclopedia. Builds Q&A training docs for the generative model and powers
exact factual lookups (no hallucination) for the assistant.
"""
import json, os, random, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD = os.path.join(ROOT, "data", "world")  # committed copy of the GitHub datasets (raw cache in data/raw/world also works)

def load_countries():
    p = os.path.join(WORLD, "countries.json")
    if not os.path.exists(p): return []
    out = []
    for c in json.load(open(p)):
        try:
            out.append({
                "name": c["name"]["common"], "official": c["name"].get("official", c["name"]["common"]),
                "capital": (c.get("capital") or [""])[0],
                "region": c.get("region", ""), "subregion": c.get("subregion", ""),
                "population": c.get("population"),
                "currencies": [v.get("name", "") for v in (c.get("currencies") or {}).values()],
                "languages": list((c.get("languages") or {}).values()),
                "latlng": c.get("latlng", []),
            })
        except Exception:
            continue
    return [c for c in out if c["name"]]

def load_elements():
    p = os.path.join(WORLD, "elements.json")
    if not os.path.exists(p): return []
    return json.load(open(p)).get("elements", [])

# Expanded curated encyclopedia: science, space, history, geography, sports,
# technology, nature, human body. (key -> factual paragraph)
ENCYCLOPEDIA = {
 "sun": "The Sun is the star at the centre of our solar system. It contains 99.86% of the system's mass, is about 4.6 billion years old, and its core temperature reaches about 15 million degrees Celsius. Light from the Sun takes about 8 minutes 20 seconds to reach Earth.",
 "moon": "The Moon is Earth's only natural satellite, about 384,400 km away. It is roughly one quarter of Earth's diameter, causes ocean tides, and was first walked on by Neil Armstrong and Buzz Aldrin on 20 July 1969 during Apollo 11.",
 "earth": "Earth is the third planet from the Sun and the only known planet with life. It is about 4.54 billion years old, 71% of its surface is covered by oceans, and it has one moon. Earth orbits the Sun in about 365.25 days and rotates once every 24 hours.",
 "mars": "Mars is the fourth planet from the Sun, called the Red Planet because of iron oxide (rust) on its surface. It has the tallest volcano in the solar system, Olympus Mons, two small moons (Phobos and Deimos), and a thin carbon-dioxide atmosphere.",
 "jupiter": "Jupiter is the largest planet in our solar system, a gas giant more massive than all other planets combined. Its Great Red Spot is a storm larger than Earth that has raged for centuries. Jupiter has dozens of moons including Ganymede, the largest moon in the solar system.",
 "water": "Water (H2O) is made of two hydrogen atoms and one oxygen atom. It covers about 71% of Earth's surface, boils at 100°C and freezes at 0°C at sea level, and is essential for all known life.",
 "gravity": "Gravity is the force that attracts masses toward each other. Isaac Newton described it mathematically in 1687; Einstein's general relativity (1915) explains it as the curvature of spacetime caused by mass and energy. On Earth it accelerates falling objects at about 9.8 m/s squared.",
 "evolution": "Evolution is the change in inherited traits of populations over generations, driven by mutation, natural selection and genetic drift. Charles Darwin and Alfred Russel Wallace proposed natural selection in the 19th century; modern genetics confirmed the mechanism.",
 "photosynthesis": "Photosynthesis is how green plants, algae and some bacteria convert sunlight, water and carbon dioxide into glucose and oxygen. It produces most of the oxygen in Earth's atmosphere and underpins nearly all food chains.",
 "dna": "DNA (deoxyribonucleic acid) is the molecule that stores genetic instructions in living things. Its double-helix structure was described by Watson, Crick, Franklin and Gosling in 1953. Human DNA has about 3 billion base pairs and roughly 20,000 genes.",
 "atom": "An atom is the smallest unit of ordinary matter: a nucleus of protons and neutrons surrounded by electrons. Atoms combine into molecules; the number of protons (atomic number) defines the chemical element.",
 "electricity": "Electricity is the flow of electric charge, usually electrons through a conductor. Voltage pushes the current, resistance opposes it (Ohm's law: V = I x R). It powers modern civilisation via generators, grids and batteries.",
 "internet": "The Internet is the global network of interconnected computers using TCP/IP protocols. It grew from ARPANET (1969); Tim Berners-Lee invented the World Wide Web in 1989 at CERN. Billions of devices now exchange data over it.",
 "computer": "A computer is a machine that processes data following instructions (programs). Modern computers use transistors, binary logic, memory (RAM), storage and a CPU. The first programmable computer concepts came from Charles Babbage and Ada Lovelace in the 1800s.",
 "artificial intelligence": "Artificial intelligence (AI) is the field of building systems that perform tasks requiring human-like intelligence: learning, reasoning, language and perception. Modern AI is dominated by machine learning and neural networks trained on large datasets.",
 "machine learning": "Machine learning is a branch of AI where systems learn patterns from data instead of being explicitly programmed. Key types: supervised learning (labelled data), unsupervised learning (structure discovery) and reinforcement learning (rewards). Deep learning uses multi-layer neural networks.",
 "python language": "Python is a high-level programming language created by Guido van Rossum, first released in 1991. It emphasises readability and is the most popular language for data science, AI, automation and web backends.",
 "world war 2": "World War II (1939-1945) was a global conflict between the Axis (Germany, Italy, Japan) and the Allies (including the USSR, USA, UK). It began with Germany's invasion of Poland, included the Holocaust, and ended in 1945 after Germany's surrender and atomic bombings of Hiroshima and Nagasaki. Over 70 million people died.",
 "world war 1": "World War I (1914-1918), the Great War, began after the assassination of Archduke Franz Ferdinand. Trench warfare on the Western Front caused millions of deaths; it ended with the Treaty of Versailles (1919) and reshaped Europe's map and politics.",
 "roman empire": "The Roman Empire was founded in 27 BC when Augustus became emperor, succeeding the Roman Republic. At its height it ruled Europe, North Africa and the Near East. The Western Empire fell in AD 476; the Eastern (Byzantine) Empire lasted until 1453.",
 "egypt pyramids": "The Egyptian pyramids, especially the Great Pyramid of Giza (built ~2560 BC for Pharaoh Khufu), are the oldest of the Seven Wonders of the Ancient World and the only one largely surviving. The Great Pyramid was the tallest human structure for over 3,800 years.",
 "india history": "India's history includes the Indus Valley Civilisation (~3300-1300 BC), Vedic age, Maurya and Gupta empires, Mughal Empire (1526-1857), British colonial rule, and independence on 15 August 1947 led by Mahatma Gandhi, Jawaharlal Nehru and others. India is the world's most populous country and largest democracy.",
 "mahatma Gandhi": "Mahatma Gandhi (1869-1948), born Mohandas Karamchand Gandhi, led India's non-violent independence movement against British rule using civil disobedience (satyagraha), including the 1930 Salt March. He was assassinated on 30 January 1948. India honours him as Father of the Nation.",
 "albert einstein": "Albert Einstein (1879-1955) was a German-born theoretical physicist who developed relativity (E = mc^2), revolutionising space, time and gravity. He won the 1921 Nobel Prize in Physics for the photoelectric effect and emigrated to the USA in 1933.",
 "isaac newton": "Isaac Newton (1643-1727) was an English physicist and mathematician who formulated the laws of motion and universal gravitation (Principia, 1687), co-invented calculus, and made key discoveries in optics.",
 "william shakespeare": "William Shakespeare (1564-1616) was an English playwright and poet, widely called the greatest writer in English. He wrote about 39 plays (Hamlet, Macbeth, Romeo and Juliet, Othello) and 154 sonnets.",
 "leonardo da vinci": "Leonardo da Vinci (1452-1519) was an Italian Renaissance polymath: painter of Mona Lisa and The Last Supper, anatomist, engineer and inventor whose notebooks contain designs centuries ahead of their time.",
 "olympics": "The Olympic Games originated in ancient Greece (from 776 BC) and were revived in 1896 in Athens. The Summer and Winter Games alternate every two years; Paris hosted 2024. The symbol is five interlaced rings representing inhabited continents.",
 "football": "Football (soccer) is the world's most popular sport: two teams of 11 try to score in the opponent's goal, mainly with feet. The FIFA World Cup, held every four years since 1930, is the most watched sporting event. Argentina won the 2022 World Cup in Qatar.",
 "cricket": "Cricket is a bat-and-ball sport hugely popular in India, England, Australia, Pakistan and other Commonwealth nations. Formats include Test matches (up to 5 days), One Day Internationals and T20. India won the first T20 World Cup (2007) and the 2024 edition.",
 "cristiano ronaldo": "Cristiano Ronaldo (born 5 February 1985, Funchal, Madeira, Portugal) is a Portuguese professional footballer, forward and captain of Portugal. He played for Sporting CP, Manchester United, Real Madrid and Juventus, is the all-time top scorer in men's international football, a five-time Ballon d'Or winner, Euro 2016 champion and five-time Champions League winner.",
 "lionel messi": "Lionel Messi (born 24 June 1987, Rosario, Argentina) is an Argentine footballer regarded among the greatest ever. He starred at FC Barcelona, then Paris Saint-Germain and Inter Miami, captained Argentina to the 2022 World Cup, and won a record eight Ballon d'Or awards.",
 "human body": "The adult human body has 206 bones, about 600 muscles, and roughly 37 trillion cells. The heart beats about 100,000 times a day; the brain has ~86 billion neurons. Humans are about 60% water.",
 "heart": "The heart is a muscular organ that pumps blood around the body: about 100,000 beats and 7,500 litres per day. It has four chambers; the right side sends blood to the lungs, the left to the body. Heart disease is the leading global cause of death.",
 "brain": "The human brain weighs about 1.4 kg, contains ~86 billion neurons connected by trillions of synapses, and controls thought, memory, emotion and the body. It consumes about 20% of the body's energy despite being 2% of body weight.",
 "ocean": "Earth's oceans cover 71% of the surface and hold 96.5% of its water. The Pacific is the largest and deepest (Mariana Trench ~10,935 m). Oceans regulate climate, produce over half the world's oxygen (via phytoplankton) and host most of Earth's life.",
 "climate change": "Climate change is the long-term warming of Earth driven mainly by human greenhouse-gas emissions (CO2, methane) from fossil fuels and deforestation. Global average temperature has risen ~1.2°C since pre-industrial times, causing sea-level rise and more extreme weather. The Paris Agreement (2015) aims to limit warming to 1.5-2°C.",
 "solar system": "The solar system formed 4.6 billion years ago and has eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune. Mercury is closest to the Sun and fastest; Neptune is farthest and coldest. Pluto is classified a dwarf planet.",
 "galaxy": "A galaxy is a huge system of stars, gas and dark matter bound by gravity. Our galaxy, the Milky Way, contains 100-400 billion stars and is about 100,000 light-years wide. The observable universe holds an estimated 200 billion to 2 trillion galaxies.",
 "black hole": "A black hole is a region of spacetime where gravity is so strong nothing, not even light, can escape past the event horizon. They form from collapsed massive stars; supermassive ones sit at galaxy centres. The first image of a black hole (M87*) was published in 2019.",
 "speed of light": "The speed of light in vacuum is 299,792,458 metres per second (about 300,000 km/s) — the universe's maximum speed. Nothing with mass can reach it, per Einstein's relativity. Light from the Sun takes ~8 minutes to reach Earth.",
 "dinosaur": "Dinosaurs dominated Earth for over 160 million years (from ~240 to 66 million years ago). A 10-km asteroid impact at Chicxulub ended non-bird dinosaurs; birds are their living descendants. Tyrannosaurus rex was among the largest land predators.",
 "tree": "Trees are perennial plants with woody stems; they absorb CO2, release oxygen, store carbon, and shelter most land species. The oldest known tree (a bristlecone pine) is ~5,000 years old; the tallest (coast redwoods) exceed 110 metres.",
 "volcano": "A volcano is an opening in Earth's crust where magma, ash and gases erupt. Most ring the Pacific 'Ring of Fire'. Eruptions can build islands, alter climate and be extremely destructive; about 1,500 volcanoes are potentially active.",
 "earthquake": "Earthquakes are sudden ground shaking from tectonic plate movements along faults. Magnitude is measured on the moment magnitude scale (logarithmic; each step ~32x more energy). The largest recorded was magnitude 9.5 in Chile, 1960.",
 "economy money": "Money is a medium of exchange that solves barter's double coincidence of wants. Modern money is mostly digital, issued under central banks that set interest rates to control inflation. GDP measures a country's total economic output.",
 "bitcoin": "Bitcoin, created in 2009 by the pseudonymous Satoshi Nakamoto, is the first decentralised cryptocurrency. It uses a public blockchain, proof-of-work mining, and has a capped supply of 21 million coins.",
 "stock market": "A stock market is where shares of public companies are traded. Major indices include the S&P 500 (USA), Sensex/Nifty (India) and FTSE 100 (UK). Prices reflect expected future profits; markets can be volatile and are regulated to protect investors.",
 "democracy": "Democracy is government by the people, usually via free elections. India is the largest democracy; Athens in the 5th century BC pioneered direct democracy. Core features: universal suffrage, rule of law, free press and separation of powers.",
 "united nations": "The United Nations, founded in 1945 after WWII, has 193 member states. It works on peace (Security Council), health (WHO), children (UNICEF) and development. Headquarters are in New York; the Secretary-General since 2017 is António Guterres.",
 "elon musk": "Elon Musk (born 1971, Pretoria, South Africa) is an entrepreneur who co-founded PayPal and leads Tesla (electric vehicles) and SpaceX (rockets; first private company to send astronauts to the ISS, 2020). He bought Twitter (now X) in 2022.",
 "bill gates": "Bill Gates (born 1955) co-founded Microsoft in 1975 with Paul Allen, leading the personal-computer revolution with Windows. He later focused on philanthropy through the Gates Foundation, funding global health and vaccines.",
 "Narendra Modi": "Narendra Modi (born 1950) has been Prime Minister of India since 2014, leading the BJP. Earlier he was Chief Minister of Gujarat (2001-2014). His governments promoted Digital India, infrastructure expansion and the GST tax reform.",
 "barack obama": "Barack Obama (born 1961) was the 44th President of the United States (2009-2017), the first African-American president. His signature law was the Affordable Care Act ('Obamacare'); he won the 2009 Nobel Peace Prize.",
 "language hindi": "Hindi is an Indo-Aryan language written in Devanagari script, an official language of India and one of the world's most spoken languages (over 600 million speakers). It is closely related to Urdu in spoken form.",
 "english language": "English is a West Germanic language that became the global lingua franca of business, science and the internet, with ~1.5 billion speakers/learners. It evolved from Old English after 1066's Norman Conquest added French vocabulary.",
 "largest country": "Russia is the largest country by area (17.1 million km2), spanning Eastern Europe and northern Asia across 11 time zones. Canada is second, China third (or USA by some measures of total area).",
 "largest ocean": "The Pacific Ocean is Earth's largest and deepest ocean, covering about 165 million km2 — more than all land combined — and containing the Mariana Trench, the deepest point on Earth.",
 "tallest mountain": "Mount Everest (8,849 m) on the Nepal-China border is Earth's tallest mountain above sea level. Mauna Kea (Hawaii) is tallest from base to peak; Chimborazo's summit is farthest from Earth's centre.",
 "longest river": "The Nile (about 6,650 km, some measurements say 6,850 km) in Africa is generally regarded the longest river, flowing through 11 countries to the Mediterranean. The Amazon carries far more water and some studies measure it longer.",
 "fastest animal": "The peregrine falcon is the fastest animal, diving at over 320 km/h. On land the cheetah reaches ~60-70 km/h in short bursts; in water the sailfish is among the fastest swimmers.",
 "largest animal": "The blue whale is the largest animal known to have existed: up to ~30 m long and ~180 tonnes, with a heart the size of a small car. It feeds almost entirely on tiny krill.",
}

PARAGRAPH_STARTERS = ["Q: What is {t}?\nA: ", "Q: Tell me about {t}.\nA: ", "Q: Explain {t}.\nA: "]

def build_world_corpus(seed=5):
    """Generate factual Q&A/paragraph docs from the real datasets + encyclopedia."""
    rnd = random.Random(seed); docs = []
    for c in load_countries():
        n, cap = c["name"], c["capital"]
        if cap:
            docs.append(f"Q: What is the capital of {n}?\nA: The capital of {n} is {cap}.")
            docs.append(f"Q: Which country's capital is {cap}?\nA: {cap} is the capital of {n}.")
        cur = ", ".join(x for x in c["currencies"] if x)
        if cur: docs.append(f"Q: What currency is used in {n}?\nA: The currency of {n} is the {cur}.")
        lang = ", ".join(c["languages"][:3])
        if lang: docs.append(f"Q: What language is spoken in {n}?\nA: People in {n} mainly speak {lang}.")
        if c["region"]:
            docs.append(f"Q: Where is {n} located?\nA: {n} is a country in {c['subregion'] or c['region']} ({c['region']}). Its capital is {cap or 'unknown'}.")
        if c["population"]:
            docs.append(f"Q: What is the population of {n}?\nA: {n} has a population of about {int(c['population']):,} people.")
        docs.append(f"{n} is a country in {c['subregion'] or c['region']}. Its capital is {cap}, its currency is the {cur or 'local currency'}, and its main language(s) include {lang or 'local languages'}.")
    for e in load_elements():
        try:
            docs.append(f"Q: What is the chemical symbol for {e['name']}?\nA: The chemical symbol for {e['name']} is {e['symbol']}.")
            docs.append(f"Q: What is the atomic number of {e['name']}?\nA: {e['name']} has atomic number {e['number']} and symbol {e['symbol']}.")
            if e.get("summary"): docs.append(f"Q: What is {e['name'].lower()} (element)?\nA: {e['summary'][:400]}")
        except Exception:
            continue
    for topic, text in ENCYCLOPEDIA.items():
        t = topic.replace(" language", "").title()
        for st in PARAGRAPH_STARTERS:
            docs.append(st.format(t=t) + text)
        docs.append(text)
    for st_name, cap in INDIA_STATES.items():
        docs.append(f"Q: What is the capital of {st_name} (Indian state)?\nA: The capital of {st_name} is {cap}.")
        docs.append(f"Q: Which Indian state's capital is {cap}?\nA: {cap} is the capital of {st_name}.")
    for st_name, cap in US_STATES.items():
        docs.append(f"Q: What is the capital of {st_name} (US state)?\nA: The capital of the US state of {st_name} is {cap}.")
    for name, desc in PLANETS:
        docs.append(f"Q: Tell me about the planet {name}.\nA: {name} is {desc}.")
        docs.append(f"{name}: {desc}.")
    for name, desc in CONTINENTS.items():
        docs.append(f"Q: Tell me about {name} (continent).\nA: {name} is the {desc}.")
    for it in load_qa_bank():
        docs.append(f"Q: {it['q']}\nA: {it['a']}")
    rnd.shuffle(docs)
    return docs

def answer_world(q):
    """Exact factual lookup over real datasets; None when unknown."""
    low = q.lower().strip()
    countries = load_countries()
    by_name = {c["name"].lower(): c for c in countries}
    # longest-name match so 'united states' beats 'states'
    m = None
    for name in sorted(by_name, key=len, reverse=True):
        if re.search(r"\b" + re.escape(name) + r"\b", low): m = by_name[name]; break
    elems = {e["name"].lower(): e for e in load_elements()}
    em = None
    for name in sorted(elems, key=len, reverse=True):
        if re.search(r"\b" + re.escape(name) + r"\b", low): em = elems[name]; break
    if m:
        if "capital" in low and m["capital"]: return f"The capital of {m['name']} is **{m['capital']}**."
        if "currency" in low or "money" in low:
            cur = ", ".join(x for x in m["currencies"] if x)
            if cur: return f"The currency of {m['name']} is the **{cur}**."
        if "language" in low or "spoken" in low:
            lang = ", ".join(m["languages"][:3])
            if lang: return f"People in {m['name']} mainly speak **{lang}**."
        if "population" in low and m["population"]: return f"{m['name']} has a population of about **{int(m['population']):,}** people."
        if any(k in low for k in ["where is", "located", "which continent", "tell me about", "who is"]) or low.startswith("what"):
            cur = ", ".join(x for x in m["currencies"] if x)
            return (f"**{m['name']}** (officially {m['official']}) is a country in {m['subregion'] or m['region']}. "
                    f"Capital: {m['capital']}. Currency: {cur}. Languages: {', '.join(m['languages'][:3])}."
                    + (f" Population: about {int(m['population']):,}." if m["population"] else ""))
    if em:
        if "symbol" in low: return f"The chemical symbol for {em['name']} is **{em['symbol']}**."
        if "atomic number" in low or "atomic mass" in low or "weight" in low:
            return f"{em['name']} ({em['symbol']}): atomic number {em['number']}, atomic mass {em.get('atomic_mass')}, category {em.get('category')}."
        if em.get("summary"): return em["summary"][:600]
    sm = None
    for name in sorted(list(INDIA_STATES) + list(US_STATES), key=len, reverse=True):
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", low): sm = name; break
    if sm and "capital" in low:
        table = INDIA_STATES if sm in INDIA_STATES else US_STATES
        region = "Indian state" if sm in INDIA_STATES else "US state"
        return f"The capital of {sm} ({region}) is **{table[sm]}**."
    for name, desc in PLANETS:
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", low) and any(k in low for k in ["planet", "tell", "about", "what", "describe"]):
            return f"**{name}** — {desc}."
    ALIASES = {"red planet": "mars", "father of the nation": "mahatma Gandhi", "city of nawabs": "lucknow",
               "largest planet": "jupiter", "smallest planet": "earth", "morning star": "earth", "evening star": "earth"}
    for phrase, topic in ALIASES.items():
        if phrase in low and topic in ENCYCLOPEDIA: return ENCYCLOPEDIA[topic]
    tv = answer_qa_bank(q)
    if tv: return tv
    for topic, text in ENCYCLOPEDIA.items():
        if re.search(r"\b" + re.escape(topic) + r"\b", low): return text
    return None

INDIA_STATES = {
 "Andhra Pradesh":"Amaravati","Arunachal Pradesh":"Itanagar","Assam":"Dispur","Bihar":"Patna",
 "Chhattisgarh":"Raipur","Goa":"Panaji","Gujarat":"Gandhinagar","Haryana":"Chandigarh",
 "Himachal Pradesh":"Shimla","Jharkhand":"Ranchi","Karnataka":"Bengaluru","Kerala":"Thiruvananthapuram",
 "Madhya Pradesh":"Bhopal","Maharashtra":"Mumbai","Manipur":"Imphal","Meghalaya":"Shillong",
 "Mizoram":"Aizawl","Nagaland":"Kohima","Odisha":"Bhubaneswar","Punjab":"Chandigarh",
 "Rajasthan":"Jaipur","Sikkim":"Gangtok","Tamil Nadu":"Chennai","Telangana":"Hyderabad",
 "Tripura":"Agartala","Uttar Pradesh":"Lucknow","Uttarakhand":"Dehradun","West Bengal":"Kolkata",
 "Delhi":"New Delhi","Jammu and Kashmir":"Srinagar","Ladakh":"Leh","Puducherry":"Puducherry",
 "Chandigarh":"Chandigarh","Andaman and Nicobar":"Port Blair","Dadra and Nagar Haveli and Daman and Diu":"Daman",
 "Lakshadweep":"Kavaratti"}
US_STATES = {
 "Alabama":"Montgomery","Alaska":"Juneau","Arizona":"Phoenix","Arkansas":"Little Rock","California":"Sacramento",
 "Colorado":"Denver","Connecticut":"Hartford","Delaware":"Dover","Florida":"Tallahassee","Georgia":"Atlanta",
 "Hawaii":"Honolulu","Idaho":"Boise","Illinois":"Springfield","Indiana":"Indianapolis","Iowa":"Des Moines",
 "Kansas":"Topeka","Kentucky":"Frankfort","Louisiana":"Baton Rouge","Maine":"Augusta","Maryland":"Annapolis",
 "Massachusetts":"Boston","Michigan":"Lansing","Minnesota":"Saint Paul","Mississippi":"Jackson","Missouri":"Jefferson City",
 "Montana":"Helena","Nebraska":"Lincoln","Nevada":"Carson City","New Hampshire":"Concord","New Jersey":"Trenton",
 "New Mexico":"Santa Fe","New York":"Albany","North Carolina":"Raleigh","North Dakota":"Bismarck","Ohio":"Columbus",
 "Oklahoma":"Oklahoma City","Oregon":"Salem","Pennsylvania":"Harrisburg","Rhode Island":"Providence",
 "South Carolina":"Columbia","South Dakota":"Pierre","Tennessee":"Nashville","Texas":"Austin","Utah":"Salt Lake City",
 "Vermont":"Montpelier","Virginia":"Richmond","Washington":"Olympia","West Virginia":"Charleston",
 "Wisconsin":"Madison","Wyoming":"Cheyenne"}
PLANETS = [
 ("Mercury","closest planet to the Sun, smallest planet, orbits in 88 days, heavily cratered, extreme temperature swings"),
 ("Venus","hottest planet (about 465°C) due to thick CO2 atmosphere and runaway greenhouse effect, rotates backwards, similar size to Earth"),
 ("Earth","our home, only known life-bearing planet, 71% ocean-covered, one moon, 4.54 billion years old"),
 ("Mars","the Red Planet, home to Olympus Mons the tallest volcano, two small moons Phobos and Deimos, target of rover missions"),
 ("Jupiter","largest planet, gas giant, Great Red Spot storm bigger than Earth, moon Ganymede is the solar system's largest"),
 ("Saturn","famous for bright rings of ice and rock, lowest density planet (would float in water), moon Titan has liquid methane lakes"),
 ("Uranus","ice giant tilted on its side (98 degrees), coldest atmosphere, pale blue-green from methane"),
 ("Neptune","farthest planet from the Sun, fastest winds in the solar system (up to 2,100 km/h), deep blue colour")]
CONTINENTS = {
 "Asia":"largest continent by area and population (about 4.7 billion people), home to India, China, Japan and Mount Everest",
 "Africa":"second-largest continent, 54 countries, Sahara the largest hot desert, Nile river, origin of humankind",
 "Europe":"second-smallest continent, about 44 countries, birthplace of the Industrial Revolution and the Renaissance",
 "North America":"third-largest continent, includes USA, Canada, Mexico; Grand Canyon and Rockies",
 "South America":"fourth-largest continent, Amazon rainforest and river, Andes the longest mountain range",
 "Australia":"smallest continent and a single country, unique wildlife like kangaroos and koalas, Great Barrier Reef",
 "Antarctica":"coldest, driest, windiest continent, covered in ice, no permanent population, research stations only"}

_trivia_cache = None
def load_trivia():
    global _trivia_cache
    if _trivia_cache is not None: return _trivia_cache
    p = os.path.join(WORLD, "trivia.json")
    out = []
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            for cat, items in d.items():
                for it in items:
                    q, a = str(it.get("question","")).strip(), str(it.get("answer","")).strip()
                    if len(q) > 8 and 0 < len(a) < 200: out.append({"q": q, "a": a, "cat": cat})
        except Exception: pass
    _trivia_cache = out
    return out

_STOP = set("what is the of a an in on for to and or which who whom whose when where why how do does did is are was were".split())
def _kw(text): return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2]

def answer_trivia(q):
    """Retrieval over 49,716 real trivia Q&A; answer only on strong overlap."""
    words = set(_kw(q))
    if len(words) < 2: return None
    best, score = None, 0
    for it in load_trivia():
        qs = set(_kw(it["q"]))
        if not qs: continue
        ov = len(words & qs) / max(1, len(qs))
        if ov > score: best, score = it, ov
    if best and score >= 0.72:
        return f"**{best['a']}** — {best['q'].rstrip('?')}? The answer is {best['a']}."
    return None

ENCYCLOPEDIA.update({
 "periodic table": "The periodic table organises 118 chemical elements by atomic number into rows (periods) and columns (groups) with similar properties. It was pioneered by Dmitri Mendeleev in 1869, who predicted undiscovered elements from gaps in his table.",
 "oxygen": "Oxygen (O, atomic number 8) is a colourless gas making up 21% of Earth's atmosphere, produced by photosynthesis and essential for respiration and combustion. Liquid oxygen is pale blue and magnetic.",
 "gold": "Gold (Au, atomic number 79) is a dense, unreactive yellow metal prized since antiquity for coins, jewellery and electronics. It is measured in karats; the largest gold nuggets came from Australia.",
 "iron": "Iron (Fe, atomic number 26) is the most common metal on Earth by mass, core of the planet, and key ingredient of steel. Humans have used iron for over 3,000 years (Iron Age).",
 "hydrogen": "Hydrogen (H, atomic number 1) is the lightest and most abundant element in the universe (~75% of normal matter), fuel of stars via fusion, and combines with oxygen to make water.",
 "carbon": "Carbon (C, atomic number 6) is the basis of all known life, forming more compounds than any other element. Its forms include diamond, graphite and graphene.",
 "sun vs earth size": "The Sun's diameter is about 109 times Earth's; about 1.3 million Earths could fit inside it. Yet the Sun is an average-sized star (a G-type yellow dwarf).",
 "great wall of china": "The Great Wall of China, built over centuries to repel northern invasions, stretches about 21,196 km including all branches. The Ming dynasty built the best-known sections; it is a UNESCO World Heritage site.",
 "taj mahal": "The Taj Mahal in Agra, India, is a white-marble mausoleum built by Mughal emperor Shah Jahan (1632-1653) for his wife Mumtaz Mahal. It is a UNESCO site and one of the New7Wonders of the World. Lucknow, the capital of Uttar Pradesh, is about 335 km away.",
 "india states": "India has 28 states and 8 Union Territories. Uttar Pradesh is the most populous state; Rajasthan the largest by area; Goa the smallest state. State capitals include Lucknow (UP), Mumbai (Maharashtra), Chennai (Tamil Nadu), Bengaluru (Karnataka) and Kolkata (West Bengal).",
 "lucknow": "Lucknow is the capital and largest city of Uttar Pradesh, India, known as the City of Nawabs, famous for Awadhi culture, tehzeeb (etiquette), chikankari embroidery, kebabs, and monuments like Bara Imambara and Rumi Darwaza.",
 "himalayas": "The Himalayas are Asia's great mountain system, stretching ~2,400 km across five countries, formed by the collision of the Indian and Eurasian plates (still rising ~5 mm/year). They include Everest and feed rivers like the Ganga and Brahmaputra.",
 "ganga river": "The Ganga (Ganges) is India's most sacred river, flowing 2,525 km from the Himalayas (Gangotri) to the Bay of Bengal. It supports hundreds of millions of people; Varanasi and Prayagraj (Kumbh Mela) are major pilgrimage cities on its banks.",
 "sahara": "The Sahara in North Africa is the world's largest hot desert, about 9.2 million km2 (nearly the size of the USA). Temperatures can exceed 50°C, yet it also has mountains, oases and rock art from wetter eras.",
 "amazon rainforest": "The Amazon rainforest covers ~7 million km2 in South America (60% in Brazil), produces vast oxygen and rainfall, and holds ~10% of known species. It is called the 'lungs of the Earth' though ocean phytoplankton produce more oxygen.",
 "cooking rice": "To cook basmati rice: rinse 1 cup rice, soak 20-30 minutes, boil in 1.5 cups water with salt, cover on low heat 10-12 minutes, rest 5 minutes, then fluff. The water ratio and resting give separate, fluffy grains.",
 "covid": "COVID-19 is a disease caused by the SARS-CoV-2 coronavirus, declared a pandemic in March 2020. Vaccines (mRNA, viral-vector and others) greatly reduced severe disease; over 7 million deaths were confirmed worldwide.",
 "vaccine": "Vaccines train the immune system using harmless forms or parts of a pathogen (or its genetic instructions), creating memory cells so real infection is fought faster. Vaccination eradicated smallpox and nearly polio.",
 "antibiotics": "Antibiotics are medicines that kill or stop bacteria (not viruses). Penicillin, discovered by Alexander Fleming in 1928, was the first widely used antibiotic. Misuse drives antibiotic resistance, a major global health threat.",
 "human evolution": "Humans (Homo sapiens) evolved in Africa about 300,000 years ago. Close relatives include Neanderthals and Denisovans, with whom early humans interbred. Agriculture began ~12,000 years ago; writing ~5,400 years ago.",
 "french revolution": "The French Revolution (1789-1799) overthrew the monarchy, proclaimed the Declaration of the Rights of Man, descended into the Reign of Terror, and ended with Napoleon Bonaparte's coup — spreading ideals of liberty, equality and fraternity across Europe.",
 "printing press": "Johannes Gutenberg's movable-type printing press (around 1440, Germany) made books cheap and plentiful, fuelling the Renaissance, Reformation and Scientific Revolution. It is ranked among history's most influential inventions.",
 "telephone": "The telephone, patented by Alexander Graham Bell in 1876, transmits voice electrically. It evolved from switchboards to mobile phones to today's smartphones, which are pocket computers with cameras, GPS and internet.",
 "airplane": "The Wright brothers achieved the first powered, controlled airplane flight on 17 December 1903 at Kitty Hawk, USA (12 seconds, 36.5 m). Aviation then shrank the world within decades, from propeller planes to jets and wide-body airliners.",
 "moon landing": "Apollo 11 landed on the Moon on 20 July 1969; Neil Armstrong and Buzz Aldrin walked on the surface while Michael Collins orbited. In total 12 astronauts walked on the Moon (1969-1972); no human has returned since Apollo 17.",
 "mars missions": "Robotic missions to Mars include NASA's rovers (Sojourner 1997, Spirit/Opportunity 2004, Curiosity 2012, Perseverance 2021) and India's Mangalyaan (2014, first Asian Mars orbiter, succeeded on its first attempt). Human missions are planned for the 2030s+.",
 "isro": "ISRO (Indian Space Research Organisation, founded 1969) is India's space agency, known for cost-effective missions: Chandrayaan lunar orbiters/landers (Chandrayaan-3 landed near the lunar south pole in 2023, a world first) and Mangalyaan to Mars (2014).",
 "stock market india": "India's main stock exchanges are the BSE (Bombay Stock Exchange, est. 1875, Asia's oldest) and NSE (National Stock Exchange). Benchmark indices are the Sensex (30 BSE stocks) and Nifty 50 (50 NSE stocks); regulator is SEBI.",
 "income tax india": "In India, income tax is levied by the central government under the Income-tax Act, with annual slabs for individuals; GST (Goods and Services Tax, launched 1 July 2017) unified indirect taxes across states. The financial year runs April to March.",
 "school exams india": "India's major school boards include CBSE and ICSE plus state boards. Class 10 and 12 board exams are key milestones; competitive exams include JEE (engineering), NEET (medical) and UPSC civil services among the world's toughest by applicants.",
 "yoga": "Yoga is an ancient Indian system of physical, mental and spiritual practice, systematised in Patanjali's Yoga Sutras (~2,000 years ago). Modern yoga emphasises asanas (postures), breathing (pranayama) and meditation; the UN marks International Yoga Day on 21 June.",
 "chess": "Chess is a two-player strategy board game evolved from Indian chaturanga (~6th century). The world championship dates to 1886; India's Viswanathan Anand was a five-time world champion, and India won double gold at the 2024 Chess Olympiad.",
 "tennis": "Tennis is a racket sport played on grass, clay or hard courts. The four Grand Slams are the Australian Open, French Open (Roland-Garros), Wimbledon (oldest, 1877) and US Open. Novak Djokovic holds the men's record of 24 Slam singles titles; Serena Williams won 23.",
 "badminton": "Badminton is a fast racket sport using a shuttlecock; top nations include China, Indonesia, India, Denmark and Japan. Indian stars include P.V. Sindhu (two Olympic medals) and Saina Nehwal. The BWF World Championships are the top annual event after the Olympics.",
 "hockey india": "Field hockey is historically India's most successful Olympic team sport: 8 gold medals (1928-1980). The men's team won bronze at Tokyo 2020 and Paris 2024, ending a long medal drought. Major Dhyan Chand, the 'Wizard', is its greatest legend.",
})

import gzip as _gzip
_qa_bank_cache = None
_qa_index_cache = None

def load_qa_bank():
    """Unified fact bank: OpenTriviaQA (49,716) + Jeopardy (214,236) = ~264k Q&A."""
    global _qa_bank_cache
    if _qa_bank_cache is not None: return _qa_bank_cache
    bank = list(load_trivia())
    gz = os.path.join(WORLD, "jeopardy.jsonl.gz")
    if os.path.exists(gz):
        try:
            with _gzip.open(gz, "rt", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("q") and d.get("a"): bank.append(d)
        except Exception:
            pass
    _qa_bank_cache = bank
    return bank

def _qa_index():
    global _qa_index_cache
    if _qa_index_cache is not None: return _qa_index_cache
    bank = load_qa_bank(); idx = {}; doc_kw = []
    for i, it in enumerate(bank):
        kw = set(_kw(it["q"]))
        doc_kw.append(kw)
        for w in kw: idx.setdefault(w, []).append(i)
    _qa_index_cache = (idx, doc_kw)
    return _qa_index_cache

def answer_qa_bank(q):
    """Fast inverted-index retrieval over ~264k real Q&A facts; strong matches only."""
    words = set(_kw(q))
    if len(words) < 2: return None
    idx, doc_kw = _qa_index()
    from collections import Counter
    cand = Counter()
    for w in words:
        for i in idx.get(w, ()): cand[i] += 1
    if not cand: return None
    bank = load_qa_bank(); best, bs = None, 0.0
    for i, inter in cand.most_common(60):
        if inter < 3: continue
        score = inter / max(1, len(doc_kw[i]))
        if score > bs: best, bs = bank[i], score
    if best and bs >= 0.55:
        return f"**{best['a']}** — {best['q'].rstrip('?')}? The answer is {best['a']}."
    return None

if __name__ == "__main__":
    docs = build_world_corpus()
    print(f"world corpus: {len(docs)} docs from {len(load_countries())} countries + {len(load_elements())} elements + {len(ENCYCLOPEDIA)} encyclopedia topics")
    print(answer_world("what is the capital of Brazil?"))
    print(answer_world("what currency is used in Japan?"))
    print(answer_world("what is the chemical symbol for gold?"))
