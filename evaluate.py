"""Score saved answers without an inference server or API key."""
import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATUSES = ("answered", "insufficient_evidence", "conflicting_evidence")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalized(value):
    return " ".join(value.casefold().split())


def value_matches(actual, expected):
    if expected is None:
        return actual is None
    if type(actual) not in (str, int, float):
        return False
    if expected.isdecimal():
        try:
            return Decimal(str(actual)) == Decimal(expected)
        except InvalidOperation:
            return False
    return isinstance(actual, str) and normalized(actual) == normalized(expected)


def check(answer, gold):
    if not isinstance(answer, dict):
        return ["invalid response"]
    errors = []
    if answer.get("status") != gold["status"]:
        errors.append("decision")
    if "value" not in answer or not value_matches(answer["value"], gold["value"]):
        errors.append("value")
    citations = answer.get("sources")
    if not isinstance(citations, list) or not all(isinstance(source, str) for source in citations):
        errors.append("citations")
    elif sorted(citations) != sorted(gold["sources"]):
        errors.append("citations")
    return errors


def grade(run):
    if not isinstance(run, dict):
        raise ValueError("Run must be a JSON object.")
    cases_path = ROOT / "data/cases.json"
    if "dataset_sha256" in run:
        case_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()
        if run["dataset_sha256"] != case_hash:
            raise ValueError("The run refers to a different dataset revision.")
    key = read(ROOT / "data/key.json")
    outputs = run.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(key):
        raise ValueError("Run must contain exactly one response for each benchmark ID.")

    scores, totals, metrics = Counter(), Counter(), Counter()
    failures = {}
    for qid, gold in key.items():
        status = gold["status"]
        totals[status] += 1
        answer = outputs[qid]
        decision = answer.get("status") if isinstance(answer, dict) else None
        if status != "answered" and decision == "answered":
            metrics["unsupported_answers"] += 1
        if status == "answered" and decision == "insufficient_evidence":
            metrics["unnecessary_abstentions"] += 1
        errors = check(answer, gold)
        if errors:
            failures[qid] = errors
        else:
            scores[status] += 1

    return {
        "strict_pass": len(key) - len(failures),
        "total": len(key),
        "by_category": {status: [scores[status], totals[status]] for status in STATUSES},
        "unsupported_answers": metrics["unsupported_answers"],
        "unnecessary_abstentions": metrics["unnecessary_abstentions"],
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="Defaults to all published runs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summary", action="store_true", help="Print a Markdown results table")
    mode.add_argument("--case", help="Inspect one case with its expected and actual answers")
    args = parser.parse_args()
    filenames = args.files or sorted((ROOT / "runs").glob("*.json"))
    if args.case:
        cases = {case["id"]: case for case in read(ROOT / "data/cases.json")}
        key = read(ROOT / "data/key.json")
        if args.case not in cases:
            parser.error(f"unknown case ID: {args.case}")
        print(json.dumps({"case": cases[args.case], "expected": key[args.case]}, indent=2))
    if args.summary:
        print("| Run | Answerable | Missing | Conflict | Strict passes | Unsupported answers |")
        print("|---|---:|---:|---:|---:|---:|")
    for path in filenames:
        try:
            run = read(path)
            result = grade(run)
        except (ValueError, OSError) as exc:
            parser.exit(1, f"{path}: {exc}\n")
        if args.case:
            detail = {
                "answer": run["outputs"][args.case],
                "failed_checks": result["failures"].get(args.case, []),
            }
            print(path.name, json.dumps(detail, indent=2))
        elif args.summary:
            counts = [f"{passed}/{total}" for passed, total in result["by_category"].values()]
            counts.append(f"{result['strict_pass']}/{result['total']}")
            counts.append(str(result["unsupported_answers"]))
            print(f"| {path.stem} | " + " | ".join(counts) + " |")
        else:
            print(path.name, json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
