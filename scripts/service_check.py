# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Checks the background service the way someone uses it: installed as a login item (`rummage watch --install`:
launchd on macOS, a systemd user service on Linux), it picks up a new file, a changed one, a renamed one and a
deleted one within seconds, comes back after it crashes, and goes with `rummage watch --uninstall`.

    cargo build --release
    uv run scripts/service_check.py              # installs and uninstalls the login item (as CI does)
    uv run scripts/service_check.py --foreground # runs `rummage watch` itself instead: nothing installed

The index and its folder go in a temporary folder, removed after.
"""

import json
import os
import pathlib
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
RUMMAGE = str(REPO / "target/release/rummage")
FOREGROUND = "--foreground" in sys.argv
MACOS = platform.system() == "Darwin"
WAIT = 180  # seconds for a change to show; the service runs at background priority
failures = []


def check(ok: bool, what: str):
    print(f"  {'pass' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        failures.append(what)


def rummage(*args, check_ok=True) -> subprocess.CompletedProcess:
    result = subprocess.run([RUMMAGE, "--index", str(index), *args], capture_output=True, text=True,
                            stdin=subprocess.DEVNULL)  # fmt: skip
    if check_ok and result.returncode != 0:
        print(result.stdout, result.stderr)
        raise SystemExit(f"rummage {' '.join(args)} failed ({result.returncode})")
    return result


def service_pid() -> int | None:
    """The login item's process, as launchd or systemd sees it (the child's in --foreground)."""
    if FOREGROUND:
        return child.pid if child and child.poll() is None else None
    if MACOS:
        out = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/io.github.rummage.watch"],
                             capture_output=True, text=True).stdout  # fmt: skip
        pid = next((line.split("=")[1].strip() for line in out.splitlines() if line.strip().startswith("pid =")), None)
    else:
        out = subprocess.run(["systemctl", "--user", "show", "rummage-watch.service", "-p", "MainPID", "--value"],
                             capture_output=True, text=True).stdout.strip()  # fmt: skip
        pid = out if out not in ("", "0") else None
    return int(pid) if pid else None


def found(word: str) -> list[str]:
    """Files the index (not grep, which reads the disk itself) finds for `word`."""
    out = rummage("search", "--json", "--no-grep", "-n", "5", word).stdout
    return [pathlib.Path(hit["path"]).name for hit in json.loads(out)]


def until(what: str, condition, timeout: float = WAIT):
    started = time.time()
    while time.time() - started < timeout:
        try:
            if condition():
                check(True, f"{what} ({time.time() - started:.0f} s)")
                return
        except (SystemExit, json.JSONDecodeError):
            pass
        time.sleep(1)
    check(False, f"{what}: not within {timeout:.0f} s")


base = pathlib.Path(tempfile.mkdtemp(prefix="rummage-service-"))
files, index = base / "files", base / "index"
files.mkdir()
# An hour old: editing it now is one change, read at once. (A file changed twice within 5 minutes waits until it's
# still: edge_cases.py checks that.)
(files / "garden.md").write_text("Notes on the garden: beans by the fence, zanzibarite in the far bed.")
os.utime(files / "garden.md", (time.time() - 3600, time.time() - 3600))
child = None
installed = False
try:
    rummage("index", str(files))
    print(f"service: {'rummage watch, run here' if FOREGROUND else 'launchd' if MACOS else 'systemd'}")
    if FOREGROUND:
        child = subprocess.Popen([RUMMAGE, "--index", str(index), "watch"], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=open(base / "watch.log", "w"))  # fmt: skip
    else:
        print(rummage("watch", "--install").stdout.strip())
        installed = True
    until("the service is running", lambda: service_pid() is not None and "running" in rummage("status").stdout)
    first_pid = service_pid()

    (files / "trip.md").write_text("Packing for the trip: the wallabyite charger, sunscreen and two maps.")
    until("a new file is found", lambda: "trip.md" in found("wallabyite"))
    (files / "garden.md").write_text("Notes on the garden: beans by the fence, quokkaberry in the far bed.")
    until("a changed file is found by its new words, not its old",
          lambda: "garden.md" in found("quokkaberry") and "garden.md" not in found("zanzibarite"))  # fmt: skip
    (files / "garden.md").rename(files / "allotment.md")
    until("a renamed file is found under its new name", lambda: found("quokkaberry")[:1] == ["allotment.md"])
    (files / "allotment.md").unlink()
    until("a deleted file is gone", lambda: "allotment.md" not in found("quokkaberry"))

    if not FOREGROUND:
        os.kill(first_pid, signal.SIGKILL)
        until("after a crash it's started again", lambda: service_pid() not in (None, first_pid), timeout=90)
        (files / "after.md").write_text("After the restart: the wombatine lamp needs a new bulb.")
        until("and keeps the index up to date", lambda: "after.md" in found("wombatine"))
        print(rummage("watch", "--uninstall").stdout.strip())
        installed = False
        until("uninstalled, it's stopped", lambda: service_pid() is None, timeout=30)
        item = (pathlib.Path.home() / "Library/LaunchAgents/io.github.rummage.watch.plist" if MACOS
                else pathlib.Path.home() / ".config/systemd/user/rummage-watch.service")  # fmt: skip
        check(not item.exists(), f"and its login item is gone ({item.name})")
finally:
    if installed:
        rummage("watch", "--uninstall", check_ok=False)
    if child:
        child.terminate()
        child.wait()
    rummage("stop", check_ok=False)
    if failures:
        for log in [base / "watch.log", index / "serve.log"]:
            if log.exists():
                print(f"--- {log.name}\n{log.read_text()[-3000:]}")
    shutil.rmtree(base, ignore_errors=True)

print(f"\n{'all passed' if not failures else f'{len(failures)} failed'}")
sys.exit(1 if failures else 0)
