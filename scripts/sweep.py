# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Varies one ranking weight at a time and runs every eval with each value, so a weight can be checked against all of
them at once. Weights are read from RUMMAGE_<NAME> (search::tuned) by the search service when it starts, so the
services are stopped before each run.

    uv run scripts/sweep.py --evals <folder> WORD_WEIGHT=1,2,3 HUB_PENALTY=0.5,1,1.5 [--zgate 1.5,2,2.5]

--evals is a folder holding DigUp's evals (digup/evals/*.json), its testbed (testbed/) and the indexes built from it
(idx-code for code.json, idx-docs for the rest, idx-mixed, the testbed plus testbed/own, for our own queries in
own/*.json: see own_evals.py).
"""

import argparse
import os
import pathlib
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--evals", default="~/.cache/rummage-eval")
parser.add_argument("--zgate", help="z gates to try, comma separated")
parser.add_argument("weights", nargs="*", help="NAME=v1,v2,…")
args = parser.parse_args()

root = pathlib.Path(args.evals).expanduser()
here = pathlib.Path(__file__).resolve().parent
rummage = here.parent / "target/release/rummage"
sets = [
    ("code", root / "digup/evals/code.json", root / "testbed/code", root / "idx-code"),
    ("synthetic", root / "digup/evals/synthetic.json", root / "testbed", root / "idx-docs"),
    ("multilingual", root / "digup/evals/multilingual.json", root / "testbed", root / "idx-docs"),
]
# Our own queries, when there are any.
for extra in sorted((root / "own").glob("*.json")) if (root / "own").exists() else []:
    sets.append((extra.stem, extra, root / "testbed", root / "idx-mixed"))


def run(env, extra=()):
    for _, _, _, index in sets:
        subprocess.run([rummage, "--index", index, "stop"], capture_output=True)
    scores = []
    for name, queries, base, index in sets:
        out = subprocess.run(
            ["uv", "run", "-q", here / "eval.py", queries, "--base", base, "--index", index, "--", *extra],
            capture_output=True, text=True, env={**os.environ, **env},
        )
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr.strip()
        # "40 queries   R@1 0.88   R@5 1.00   MRR 0.94"
        fields = line.split()
        scores.append((name, fields[3], fields[5], fields[7]) if len(fields) == 8 else (name, "?", "?", line))
    for _, _, _, index in sets:
        subprocess.run([rummage, "--index", index, "stop"], capture_output=True)
    return scores


def show(label, scores):
    cells = "   ".join(f"{name} {r1}/{r5}/{mrr}" for name, r1, r5, mrr in scores)
    mean = sum(float(s[3]) for s in scores if s[3] != "?") / len(scores)
    print(f"{label:<24} {cells}   mean MRR {mean:.3f}", flush=True)


show("defaults", run({}))
for spec in args.weights:
    name, _, values = spec.partition("=")
    for value in values.split(","):
        show(f"{name}={value}", run({f"RUMMAGE_{name}": value}))
if args.zgate:
    for value in args.zgate.split(","):
        show(f"zgate={value}", run({}, ["--zgate", value]))
