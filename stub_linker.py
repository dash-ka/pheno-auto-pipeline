"""A stand-in for BeLink in the smoke test: links a mention by the words it shares with an HPO name."""
NAMES = {"HP:0000121": "nephrocalcinosis", "HP:0000083": "renal insufficiency",
         "HP:0008341": "distal renal tubular acidosis", "HP:0002150": "hypercalciuria", "HP:0000103": "polyuria"}


def link(mentions):
    out = []
    for m in mentions:
        text = f"{m.get('text', '')} {m.get('llm_term', '')}".lower()
        best = max(NAMES, key=lambda i: sum(w in text for w in NAMES[i].split()))
        hit = sum(w in text for w in NAMES[best].split())
        out.append({"id": best, "score": 0.99 if hit else 0.2, "label": NAMES[best].capitalize()})
    return out
