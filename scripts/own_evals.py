# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Writes rummage's own eval set: queries DigUp's evals don't cover (exact lookups, names, words with other endings,
common words where recency decides, a log only grep reads, near-duplicates told apart by one code). The files go in
<folder>/testbed/own and the queries in <folder>/own/words.json, which scripts/sweep.py runs against an index of the
whole testbed (idx-mixed), so DigUp's files are the distractors:

    uv run scripts/own_evals.py ~/.cache/rummage-eval
    rummage --index ~/.cache/rummage-eval/idx-mixed index ~/.cache/rummage-eval/testbed/{synthetic,traps,own}
"""

import json
import os
import pathlib
import random
import sys
import time

root = pathlib.Path(sys.argv[1]).expanduser()
own = root / "testbed/own"
own.mkdir(parents=True, exist_ok=True)
now = time.time()
DAY = 86400

files = {
    "invoices/northwind-october.txt": (10, """Northwind Supplies
Invoice INV-20931
Date: 2 October 2026
Bill to: Harbour Studio

Item                         Qty   Price
Recycled A4 paper, 5 reams    4    €38.00
Toner cartridge, black        2    €91.50

Total due: €335.00 by 2 November 2026. Bank transfer only.
"""),
    "invoices/northwind-september.txt": (40, """Northwind Supplies
Invoice INV-20932
Date: 3 September 2026
Bill to: Harbour Studio

Item                         Qty   Price
Desk lamp, LED                1    €64.00
Whiteboard markers, 12 pack   3    €11.20

Total due: €97.60 by 3 October 2026. Bank transfer only.
"""),
    "invoices/brightline-hosting.txt": (25, """Brightline Hosting — receipt
Order 7Q4-KX81
Plan: Small VPS, 2 vCPU, 4 GB RAM, monthly
Amount charged: $24.00 to the card on file
Next renewal: 14 November 2026
"""),
    "notes/rust-build-errors.md": (15, """# Build errors I keep hitting

## E0425
error[E0425]: cannot find value `cfg` in this scope. Usually a missing `let` or the variable was moved into a closure
defined below. Fix: declare it before the closure.

## E0308
error[E0308]: mismatched types, expected `&str`, found `String`. Borrow it with `&` or call `.as_str()`.
"""),
    "notes/parcel.txt": (5, """Parcel from the camera shop
Carrier: UPS, tracking number 1Z999AA10123456784
Expected Thursday. Leave with the neighbour at number 12 if nobody is in.
"""),
    "notes/garden-plan.md": (70, """# Garden plan for spring

Raised bed one: tomatoes along the back fence where they get afternoon sun, basil between them.
Raised bed two: carrots and radishes, sown in rows a hand apart. Net them against the pigeons.
Order seed potatoes in February; chit them on the windowsill for six weeks before planting.
"""),
    "notes/sourdough.md": (120, """# Sourdough, the way that finally worked

Feed the starter the night before. Mix 500 g flour with 350 g water and leave it for an hour, then add 100 g of
starter and 10 g of salt. Fold every half hour for two hours, then leave it alone until it has grown by half, around
five hours in a warm kitchen. Shape, rest in the fridge overnight, bake at 250 °C in a covered pot.
"""),
    "notes/lease-email.txt": (200, """From the letting agent, about moving out:

The bond will be returned within ten working days of the final inspection, minus the cost of repainting the hallway
if the scuff marks are still there. Please leave the keys with the concierge and send us a forwarding address.
"""),
    "notes/team-contacts.md": (30, """# Who to ask

- Priya Raman: payroll, holiday requests, anything about contracts.
- Tomasz Wolny: the office network, laptops, the printer that jams.
- Aiko Mori: the design system and brand colours.
"""),
    "standups/standup-2026-10-17.md": (2, """# Standup notes

Shipped the export button. Next: fix the flaky login test. Blocked on design for the empty state.
"""),
    "standups/standup-2026-08-20.md": (60, """# Standup notes

Started the export feature. Next: wire up the CSV writer. Nothing blocking.
"""),
    "standups/standup-2025-09-10.md": (404, """# Standup notes

Planning the quarter. Next: hire a second designer. Waiting on budget sign-off.
"""),
    "trips/lisbon-2026.md": (20, """# Lisbon, May 2026

Flights booked: out Friday 8 May, back Tuesday 12 May. Staying in Alfama, near the castle.
Must do: tram 28 early before the crowds, pastéis de nata in Belém, the tile museum.
"""),
    "trips/lisbon-2019.md": (2500, """# Lisbon, 2019

We stayed in Bairro Alto. Too loud at night. Tram 28 was packed by nine. Ate grilled sardines every day.
"""),
}

for name, (age, text) in files.items():
    path = own / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (now - age * DAY, now - age * DAY))

# A server log only grep reads (.log isn't indexed): one unusual line deep inside 40,000 ordinary ones.
random.seed(7)
log = own / "logs/api-gateway.log"
log.parent.mkdir(exist_ok=True)
lines = []
for i in range(40000):
    path = random.choice(["/v1/items", "/v1/users/me", "/v1/orders", "/healthz"])
    lines.append(f"2026-10-18T09:{i // 600 % 60:02d}:{i % 60:02d}Z INFO GET {path} 200 {random.randint(3, 90)}ms")
lines[31337] = "2026-10-18T10:12:44Z ERROR upstream payments-gw ECONNRESET after 30012ms, retrying (attempt 3 of 5)"
log.write_text("\n".join(lines) + "\n")
os.utime(log, (now - 1 * DAY, now - 1 * DAY))

queries = [
    # Exact lookups: codes and numbers.
    ("INV-20931", ["own/invoices/northwind-october.txt"]),
    ("invoice INV-20932", ["own/invoices/northwind-september.txt"]),
    ("7Q4-KX81", ["own/invoices/brightline-hosting.txt"]),
    ("E0425", ["own/notes/rust-build-errors.md"]),
    ("1Z999AA10123456784", ["own/notes/parcel.txt"]),
    ("payments-gw ECONNRESET", ["own/logs/api-gateway.log"]),
    # Names and rare words.
    ("Priya", ["own/notes/team-contacts.md"]),
    ("Tomasz", ["own/notes/team-contacts.md"]),
    ("Brightline", ["own/invoices/brightline-hosting.txt"]),
    ("Alfama", ["own/trips/lisbon-2026.md"]),
    # Common words: several files hold them, the newest should come first.
    ("standup notes", ["own/standups/standup-2026-10-17.md"]),
    ("Lisbon tram 28", ["own/trips/lisbon-2026.md", "own/trips/lisbon-2019.md"]),
    ("Northwind invoice", ["own/invoices/northwind-october.txt"]),
    # Descriptions with few of the file's words.
    ("when do I get my deposit back after moving out", ["own/notes/lease-email.txt"]),
    ("how long should the bread dough rise", ["own/notes/sourdough.md"]),
    ("who fixes the printer", ["own/notes/team-contacts.md"]),
    ("where to plant the tomatoes", ["own/notes/garden-plan.md"]),
    ("when is the package arriving", ["own/notes/parcel.txt"]),
    ("how much is the server costing me each month", ["own/invoices/brightline-hosting.txt"]),
    ("rust compiler says the types don't match", ["own/notes/rust-build-errors.md"]),
    ("which day do we fly home from Portugal", ["own/trips/lisbon-2026.md"]),
    # Other endings than the file's.
    ("Northwind invoices", ["own/invoices/northwind-october.txt"]),
    ("chitting the seed potato", ["own/notes/garden-plan.md"]),
    ("Brightline renewals", ["own/invoices/brightline-hosting.txt"]),
    ("jammed printers", ["own/notes/team-contacts.md"]),
]
evals = root / "own"
evals.mkdir(exist_ok=True)
(evals / "words.json").write_text(
    json.dumps({"base": "testbed", "queries": [{"q": q, "expect": e} for q, e in queries]}, indent=1)
)
print(f"{len(files) + 1} files, {len(queries)} queries")
