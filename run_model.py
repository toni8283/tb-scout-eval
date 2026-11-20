"""Evaluate a local model through Ollama or llama.cpp without sending the answer key."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from evaluate import ROOT, read

BASIC = (
    "Use only the two equally authoritative source records. Return one JSON object with "
    "status (answered, insufficient_evidence, or conflicting_evidence), value (bare number "
    "in the requested units or a name for an answer, otherwise null), and sources (an array "
    "of supporting source IDs, empty for missing evidence)."
)
PROMPTS = {
    "basic": BASIC,
    "evidence": BASIC + " Match the exact metric, date and population. A related metric "
    "cannot establish the requested value. If both sources disagree about the exact "
    "requested fact, return conflicting_evidence with a null value and cite both sources.",
}
SETTINGS = {"temperature": 0, "seed": 42, "max_tokens": 128}


def request_json(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def messages_for(prompt, case):
    evidence = {field: case[field] for field in ("company", "question", "sources")}
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
    ]


def ask(backend, base_url, model, messages):
    payload = {"model": model, "messages": messages, "stream": False}
    if backend == "ollama":
        payload["format"] = "json"
        payload["options"] = {
            "temperature": SETTINGS["temperature"], "seed": SETTINGS["seed"],
            "num_predict": SETTINGS["max_tokens"],
        }
        result = request_json(base_url + "/api/chat", payload)
        return result["message"]["content"]
    payload.update(SETTINGS)
    payload["response_format"] = {"type": "json_object"}
    result = request_json(base_url + "/v1/chat/completions", payload)
    return result["choices"][0]["message"]["content"]


def parse_answer(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def write_snapshot(path, run):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Ollama model tag or llama.cpp server model alias")
    parser.add_argument("--backend", choices=["ollama", "llama.cpp"], default="ollama")
    parser.add_argument("--url", help="Base URL of the local inference server")
    parser.add_argument("--runtime-version", help="Inference server build or version")
    parser.add_argument("--model-digest", help="Model artifact hash for run provenance")
    parser.add_argument("--output", required=True, type=Path, help="New directory for the run")
    args = parser.parse_args()
    base_url = (args.url or ("http://localhost:11434" if args.backend == "ollama"
                            else "http://localhost:8080")).rstrip("/")
    cases_path = ROOT / "data/cases.json"
    cases = read(cases_path)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model": args.model, "backend": args.backend,
        "runtime_version": args.runtime_version, "model_digest": args.model_digest,
        "settings": SETTINGS, "dataset_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    for name, prompt in PROMPTS.items():
        run = {**metadata, "condition": name, "prompt": prompt, "raw": {}, "outputs": {}}
        path = args.output / f"{name}.json"
        for case in cases:
            qid = case["id"]
            try:
                raw = ask(args.backend, base_url, args.model, messages_for(prompt, case))
            except (URLError, TimeoutError, ValueError, KeyError, TypeError, IndexError) as exc:
                parser.exit(1, f"{name}/{qid}: model request failed ({exc}). "
                            f"Any completed responses are in {args.output}.\n")
            run["raw"][qid] = raw
            run["outputs"][qid] = parse_answer(raw)
            write_snapshot(path, run)
            print(f"{name}: {qid} saved", flush=True)


if __name__ == "__main__":
    main()
