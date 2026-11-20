"""Reference policies to sanity-check the grader and expose a common failure mode."""
import json
import re
from evaluate import ROOT, read


def words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def retrieve(case):
    query = words(case["question"]) - words(case["company"]) - {"what", "is", "the", "of", "for", "in", "on"}
    sources = case["sources"]
    best = max(sources, key=lambda sid: len(query & words(sources[sid])))
    text = sources[best]
    amounts = re.findall(r"\b(?:\d+|six|nine)\b", text)
    names = re.findall(r"\b[A-Z][a-z]+ Fund\b|\b(?:India|Nepal)\b", text)
    candidates = names or amounts
    if not candidates:
        return dict(status="insufficient_evidence", value=None, sources=[])
    value = candidates[-1]
    return dict(status="answered", value={"six": "6", "nine": "9"}.get(value, value), sources=[best])


if __name__ == "__main__":
    cases = read(ROOT / "data/cases.json")
    policies = {"lexical": retrieve, "always_abstain": lambda case: dict(
        status="insufficient_evidence", value=None, sources=[])}
    for name, predict in policies.items():
        result = dict(policy=name, outputs={case["id"]: predict(case) for case in cases})
        (ROOT / "runs" / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
