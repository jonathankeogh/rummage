# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Recall of known answers: runs each query of a DigUp-style eval file through `rummage search --json` and reports
R@1, R@5 and MRR, counting a hit when a result is one of the query's expected files.

    uv run scripts/eval.py <queries.json> --base <folder> --index <index> [--only txt,md,pdf,docx] [-v] [-- extra search args]

Eval files are DigUp's (evals/*.json in its repo): {"base": "...", "queries": [{"q": ..., "expect": [path[@place], ...]}]}.
Queries whose expected files rummage can't read yet (images, audio, rtf…) are skipped: --only lists the file types
that count.
"""

import argparse
import json
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("queries")
parser.add_argument("--base", required=True, help="the folder the expected paths are relative to")
parser.add_argument("--index", required=True)
parser.add_argument("--only", default="txt,md,markdown,pdf,docx,odt,html,htm,py,ts,tsx,swift,ipynb,js,rs,go,sh")
parser.add_argument("--rummage", default=str(pathlib.Path(__file__).resolve().parent.parent / "target/release/rummage"))
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("extra", nargs="*", help="more arguments for rummage search (after --)")
args = parser.parse_args()

base = pathlib.Path(args.base).expanduser().resolve()
kinds = set(args.only.split(","))
queries = json.loads(pathlib.Path(args.queries).read_text())["queries"]

def expected_files(query):
    files = {e.split("@")[0] for e in query["expect"]}
    return {f for f in files if f.rsplit(".", 1)[-1].lower() in kinds and (base / f).exists()}

counted, r1, r5, mrr = 0, 0, 0, 0.0
for query in queries:
    expected = expected_files(query)
    if not expected:
        continue
    out = subprocess.run(
        [args.rummage, "--index", args.index, "search", query["q"], "-n", "10", "--json", *args.extra],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit(f"rummage failed on {query['q']!r}: {out.stderr.strip()}")
    hits = [str(pathlib.Path(h["path"]).resolve().relative_to(base)) if h["path"].startswith(str(base)) else h["path"]
            for h in json.loads(out.stdout)]
    rank = next((i + 1 for i, h in enumerate(hits) if h in expected), None)
    counted += 1
    r1 += rank == 1
    r5 += rank is not None and rank <= 5
    mrr += 1 / rank if rank else 0
    if args.verbose and rank != 1:
        print(f"  #{rank or '-'}  {query['q']!r}  want {sorted(expected)[0]}  got {hits[:3]}")

if counted == 0:
    sys.exit("no query has an expected file of the counted kinds under --base")
print(f"{counted} queries   R@1 {r1 / counted:.2f}   R@5 {r5 / counted:.2f}   MRR {mrr / counted:.2f}")
