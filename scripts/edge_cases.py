# /// script
# requires-python = ">=3.11"
# dependencies = ["pypdf[crypto]==5.1.0"]
# ///
"""Builds and checks the indexing edge cases (the "indexing edge cases plan"): each case is a folder of files that
once made indexing slow, stuck or wrong, indexed into its own index with the release build, then checked.

    cargo build --release
    uv run scripts/edge_cases.py              # every case
    uv run scripts/edge_cases.py limits retries
    uv run scripts/edge_cases.py icloud       # only when named: it uses your iCloud Drive (a test folder, deleted)

The folders go in ~/.cache/rummage-eval/edge/<case>; each run starts it afresh.
"""

import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
RUMMAGE = REPO / "target/release/rummage"
EDGE = pathlib.Path("~/.cache/rummage-eval/edge").expanduser()
CASES = {}
failures = []


def case(fn):
    CASES[fn.__name__] = fn
    return fn


def check(ok: bool, what: str):
    print(f"  {'pass' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)


def fresh(name: str) -> pathlib.Path:
    folder = EDGE / name
    if folder.exists():
        subprocess.run(["chmod", "-R", "u+rwx", str(folder)], check=True)  # a case may have locked a folder
        shutil.rmtree(folder)
    (folder / "files").mkdir(parents=True)
    return folder


def index(folder: pathlib.Path, env: dict | None = None, timeout: float = 900, files=None) -> tuple[str, float]:
    """Indexes folder/files (or `files`) into folder/index; returns what it said and how long it took."""
    started = time.time()
    result = subprocess.run(
        [str(RUMMAGE), "--index", str(folder / "index"), "index", str(files or folder / "files")],
        env={**os.environ, **(env or {})},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    said = result.stderr.replace("\r", "\n")
    if result.returncode != 0:
        print(said)
        raise SystemExit(f"rummage index failed ({result.returncode})")
    return said, time.time() - started


def db(folder: pathlib.Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{folder / 'index/index.sqlite'}?mode=ro", uri=True)


def file_row(folder: pathlib.Path, name: str):
    """(state, error, segments) of the file called `name`."""
    with db(folder) as c:
        row = c.execute("SELECT id, state, error FROM files WHERE path LIKE ?", (f"%/{name}",)).fetchone()
        if row is None:
            return None
        segments = c.execute("SELECT COUNT(*) FROM segments WHERE file_id = ?", (row[0],)).fetchone()[0]
        return row[1], row[2], segments


def changed(said: str) -> int:
    """How many files a run called new or changed."""
    for line in said.splitlines():
        if "new or changed" in line:
            return int(line.split(":")[1].split("new or changed")[0])
    raise ValueError(said)


def write_pdf(path: pathlib.Path, pages: list[str]):
    """A small PDF with one line of text a page (as the Rust tests write them)."""
    n = len(pages)
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join(f"{4 + 2 * i} 0 R" for i in range(n)), n),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {5 + 2 * i} 0 R >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out = "%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects):
        offsets.append(len(out))
        out += f"{i + 1} 0 obj\n{obj}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_text(out)


def page_texts(n: int, topic: str) -> list[str]:
    return [f"Page {i + 1} of the {topic} report talks about harbour cranes and tide tables." for i in range(n)]


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def video(path: pathlib.Path, seconds: float, faststart: bool = False):
    """A test video: moving colour bars and a tone."""
    ffmpeg(
        "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=25:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}",
        "-c:v", "libx264", "-crf", "30", "-c:a", "aac", "-shortest",
        *(["-movflags", "+faststart"] if faststart else []),
        str(path),
    )  # fmt: skip


@case
def limits():
    """Fix 3: a tool that hangs is killed; a file that fails anywhere is failed whole, with nothing of it left."""
    folder = fresh("limits")
    files = folder / "files"
    # A pdftotext that hangs on files named "hang", and on any later step (from page 31) of a long PDF.
    bin_dir = folder / "bin"
    bin_dir.mkdir()
    real = shutil.which("pdftotext")
    fake = bin_dir / "pdftotext"
    fake.write_text(f"""#!/bin/sh
case "$1" in -v) exec "{real}" -v ;; esac
first=1
prev=""
for a in "$@"; do
  [ "$prev" = "-f" ] && first="$a"
  prev="$a"
  case "$a" in *hang*) exec sleep 1000 ;; esac
done
[ "$first" -gt 30 ] && exec sleep 1000
exec "{real}" "$@"
""")
    fake.chmod(0o755)
    write_pdf(files / "hangs on reading.pdf", page_texts(3, "hanging"))
    write_pdf(files / "long report.pdf", page_texts(80, "long"))
    write_pdf(files / "fine.pdf", page_texts(3, "fine"))
    # A video cut off halfway (its index is at the end, so it's gone).
    whole = folder / "whole.mp4"
    video(whole, 20)
    data = whole.read_bytes()
    (files / "cut off.mp4").write_bytes(data[: len(data) // 2])
    (files / "note.md").write_text("A note about the harbour: cranes, tides, and the ferry timetable.")

    said, took = index(folder, {"PATH": f"{bin_dir}:{os.environ['PATH']}", "RUMMAGE_TOOL_SECONDS": "3"})
    check(took < 120, f"indexing finishes despite the hangs ({took:.0f} s)")
    hang = file_row(folder, "hangs on reading.pdf")
    check(hang is not None and hang[0] == "failed" and "took over" in (hang[1] or ""), f"hanging PDF failed: {hang}")
    long = file_row(folder, "long report.pdf")
    check(long is not None and long[0] == "failed" and long[2] == 0, f"long PDF failed in a later pass, whole: {long}")
    cut = file_row(folder, "cut off.mp4")
    check(cut is not None and cut[0] == "failed" and cut[2] == 0, f"cut-off video failed whole: {cut}")
    for name in ["fine.pdf", "note.md"]:
        row = file_row(folder, name)
        check(row is not None and row[0] == "done" and row[2] > 0, f"{name} indexed: {row}")


@case
def retries():
    """Fix 4: a failed file is tried again when it changes, or by a newer rummage, not at every sweep."""
    folder = fresh("retries")
    files = folder / "files"
    from pypdf import PdfReader, PdfWriter

    plain = folder / "plain.pdf"
    write_pdf(plain, page_texts(2, "locked"))
    writer = PdfWriter(clone_from=PdfReader(plain))
    writer.encrypt(user_password="secret", owner_password="owner", algorithm="AES-128")
    with open(files / "locked.pdf", "wb") as out:
        writer.write(out)
    (files / "broken.pdf").write_text("%PDF-1.4\nthis is not a PDF at all\n")
    (files / "note.md").write_text("A note about bread: flour, water, salt and a long rise.")

    said, _ = index(folder)
    for name in ["locked.pdf", "broken.pdf"]:
        row = file_row(folder, name)
        check(row is not None and row[0] == "failed", f"{name} failed first time: {row}")
    retried = 0
    for i in range(10):
        (files / f"new note {i}.md").write_text(f"Note {i}: a list of things to bring to the market on Saturday.")
        said, _ = index(folder)
        retried += changed(said) - 1
    check(retried == 0, f"10 changes elsewhere: {retried} retries of the failed files")
    os.utime(files / "locked.pdf")
    said, _ = index(folder)
    check(changed(said) == 1, f"touching locked.pdf: {changed(said)} file read again")
    # A newer rummage tries them again.
    with sqlite3.connect(folder / "index/index.sqlite") as c:
        c.execute("UPDATE files SET failed_by = '0.0.0' WHERE state = 'failed'")
    said, _ = index(folder)
    check(changed(said) == 2, f"after an update: {changed(said)} failed files read again")


@case
def photos():
    """Fix 2: inside a Photos library only the originals are read; Lightroom previews are skipped; a library macOS
    won't open gives the Full Disk Access message."""
    folder = fresh("photos")
    photos = sorted((REPO / "demo/photos").glob("*.jpg"))[:3]
    library = folder / "files/Pictures/Photos Library.photoslibrary"
    codes = ["4F2A9C1E-0B7D-4E5A-9C3B-2D1E0F4A5B6C", "8B3D2E7F-1C4A-4B9E-8D2F-3E4A5B6C7D8E", "C1D2E3F4-A5B6-4C7D-8E9F-0A1B2C3D4E5F"]
    for code, photo in zip(codes, photos):
        original = library / "originals" / code[0] / f"{code}.jpeg"
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(photo, original)
        for copy in [f"resources/derivatives/{code[0]}/{code}_1_105_c.jpeg", f"resources/renders/{code[0]}/{code}_1_201_a.jpeg"]:
            (library / copy).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(photo, library / copy)
    (library / "database").mkdir()
    (library / "database/Photos.sqlite").write_bytes(b"SQLite format 3\0")
    previews = folder / "files/Pictures/Lightroom/Catalog Previews.lrdata/0/0A1B"
    previews.mkdir(parents=True)
    shutil.copy(photos[0], previews / "preview.jpg")
    shutil.copy(photos[1], folder / "files/Pictures/beach.jpg")
    locked = folder / "files/Pictures/Old Library.photoslibrary"
    (locked / "originals/0").mkdir(parents=True)
    shutil.copy(photos[2], locked / "originals/0/0A0B0C0D.jpeg")
    locked.chmod(0)

    said, _ = index(folder)
    with db(folder) as c:
        paths = [r[0] for r in c.execute("SELECT path FROM files WHERE state = 'done'")]
    in_library = [p for p in paths if ".photoslibrary/" in p]
    check(len(in_library) == 3 and all("/originals/" in p for p in in_library), f"only the originals: {in_library}")
    check(not any(".lrdata" in p for p in paths), "Lightroom previews skipped")
    check(any(p.endswith("beach.jpg") for p in paths), "an ordinary photo next to them is read")
    check("Full Disk Access" in said and "Old Library.photoslibrary" in said, "a locked library gives the message")
    shown = subprocess.run(
        [str(RUMMAGE), "--index", str(folder / "index"), "search", "--no-serve", "--zgate=-9", "picture"],
        capture_output=True, text=True,
    ).stdout  # fmt: skip
    check("Photos Library › " in shown, "results name them Photos Library › …")


@case
def busy():
    """Fix 5: a file written every 2 s isn't read again and again while it changes, but once it's still; a new note
    meanwhile is read in seconds. (The service watches; "still" is cut from 5 minutes to 45 s.)"""
    folder = fresh("busy")
    files = folder / "files"
    (files / "old note.md").write_text("A note from before: the car goes in for its service next week.")
    index(folder)
    BUSY = 45  # over the watcher's 30 s sweep of a folder that never goes quiet
    service = subprocess.Popen(
        [str(RUMMAGE), "--index", str(folder / "index"), "watch"],
        env={**os.environ, "RUMMAGE_BUSY_SECONDS": str(BUSY)},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=open(folder / "service.log", "w"),
    )  # fmt: skip

    def stored(name):
        with db(folder) as c:
            return c.execute("SELECT state, mtime FROM files WHERE path LIKE ?", (f"%/{name}",)).fetchone()

    try:
        time.sleep(5)
        log = files / "readings.txt"
        reads = set()  # the dates it was read at
        note_written = note_read = None
        started = time.time()
        while time.time() - started < 60:
            with open(log, "a") as out:
                out.write(f"{time.strftime('%H:%M:%S')} the pump ran, pressure steady at the north tank\n")
            if note_written is None and time.time() - started > 10:
                (files / "new note.md").write_text("A new note: ask the neighbours about the hedge on Sunday.")
                note_written = time.time()
            row = stored("new note.md")
            if note_written and note_read is None and row and row[0] == "done":
                note_read = time.time()
            row = stored("readings.txt")
            if row and row[0] == "done":
                reads.add(row[1])
            time.sleep(2)
        last_write = int(os.stat(log).st_mtime)
        while time.time() - last_write < BUSY + 45:
            row = stored("readings.txt")
            if row and row[0] == "done":
                reads.add(row[1])
                if row[1] == last_write:
                    break
            time.sleep(1)
        final_at = time.time()
    finally:
        service.terminate()
        service.wait()
    check(len(reads) <= 2, f"read {len(reads)} times over 60 s of writes (first sight, then once still)")
    check(last_write in reads, "read once still, as last written")
    check(final_at - last_write >= BUSY - 2, f"…after it was still ({final_at - last_write:.0f} s after the last write)")
    took = (note_read or 1e9) - (note_written or 0)
    check(took < 45, f"a new note meanwhile read in {took:.0f} s")


SF_DATALESS = 0x40000000


def dataless(path: pathlib.Path) -> bool:
    return bool(os.stat(path).st_flags & SF_DATALESS)


@case
def icloud():
    """Fix 1: a file in iCloud only (evicted) is left out and counted, never downloaded. Touches iCloud Drive: one
    small test folder, deleted after. Run only when named."""
    drive = pathlib.Path("~/Library/Mobile Documents/com~apple~CloudDocs").expanduser()
    if not drive.is_dir():
        print("  skipped: no iCloud Drive here")
        return
    test = drive / f"rummage-edge-test-{os.getpid()}"
    folder = fresh("icloud")
    try:
        test.mkdir()
        (test / "kept.md").write_text("A note that stays on this Mac: the plumber comes on Thursday.")
        evicted = test / "evicted.md"
        evicted.write_text("A note only in iCloud: the boiler service is booked for March.")
        # Uploaded first: only an uploaded file can be evicted.
        for _ in range(120):
            if subprocess.run(["brctl", "evict", str(evicted)], capture_output=True).returncode == 0 and dataless(evicted):
                break
            time.sleep(2)
        check(dataless(evicted), "the test file is evicted (in iCloud only)")
        said, _ = index(folder, files=test)
        check("1 file in iCloud" in said, f"counted: {[l for l in said.splitlines() if 'iCloud' in l]}")
        check(file_row(folder, "evicted.md") is None, "not indexed")
        kept = file_row(folder, "kept.md")
        check(kept is not None and kept[0] == "done", f"the downloaded one is: {kept}")
        check(dataless(evicted), "still in iCloud only after indexing (not downloaded)")
    finally:
        shutil.rmtree(test, ignore_errors=True)


OPT_IN = {"icloud"}

if __name__ == "__main__":
    if not RUMMAGE.exists():
        sys.exit("build first: cargo build --release")
    names = sys.argv[1:] or [n for n in CASES if n not in OPT_IN]
    for name in names:
        print(f"{name}: {CASES[name].__doc__.splitlines()[0]}")
        CASES[name]()
    print(f"\n{'all passed' if not failures else f'{len(failures)} failed'}")
    sys.exit(1 if failures else 0)
