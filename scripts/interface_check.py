# /// script
# requires-python = ">=3.11"
# dependencies = ["pyte==0.8.2"]
# ///
"""Checks the interface the way a person meets it: started in a terminal (a pseudo-terminal read by a terminal
emulator, pyte, which answers like a plain xterm and not to picture queries), and again inside tmux when tmux is
installed. It must draw within seconds (not hang asking the terminal about pictures), find a note as the query is
typed, hand the file to the desktop's opener on ⏎ (a stand-in that writes down what it got, so nothing opens), say so
when there's no opener (Linux without xdg-open), and leave the terminal as it found it on esc.

    cargo build --release
    uv run scripts/interface_check.py

Needs the model files (the index is built here).
"""

import os
import pathlib
import platform
import pty
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
import fcntl

import pyte

REPO = pathlib.Path(__file__).resolve().parent.parent
RUMMAGE = str(REPO / "target/release/rummage")
COLUMNS, ROWS = 120, 36
failures = []


def check(ok: bool, what: str):
    print(f"  {'pass' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        failures.append(what)


base = pathlib.Path(tempfile.mkdtemp(prefix="rummage-interface-"))
notes, index, bin_dir = base / "notes", base / "index", base / "bin"
notes.mkdir()
bin_dir.mkdir()
(notes / "boiler.md").write_text("The boiler service is booked for March; the engineer brings the parts.\n")
(notes / "bread.md").write_text("Sourdough: feed the starter the night before, let the dough rise for five hours.\n")
shutil.copy(sorted((REPO / "demo/photos").glob("*.jpg"))[0], notes / "beach.jpg")
# The desktop's opener, standing in: it writes down what it was asked to open.
opened = base / "opened.txt"
opener = "open" if platform.system() == "Darwin" else "xdg-open"
(bin_dir / opener).write_text(f'#!/bin/sh\necho "$@" >> "{opened}"\n')
(bin_dir / opener).chmod(0o755)
subprocess.run([RUMMAGE, "--index", str(index), "index", str(notes)], check=True, capture_output=True,
               stdin=subprocess.DEVNULL)  # fmt: skip
env = {**os.environ, "TERM": "xterm-256color", "PATH": f"{bin_dir}:{os.environ['PATH']}"}
env.pop("RUMMAGE_PICTURES", None)


class Terminal:
    """rummage in a pseudo-terminal, its output drawn by pyte."""

    def __init__(self, env, args=None):
        self.screen = pyte.Screen(COLUMNS, ROWS)
        self.stream = pyte.ByteStream(self.screen)
        self.raw = b""
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execve(RUMMAGE, [RUMMAGE, *(args or ["--index", str(index)])], env)
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLUMNS, 0, 0))
        os.kill(self.pid, signal.SIGWINCH)
        # Answer what a plain xterm answers (device attributes, cursor position); picture queries go unanswered.
        self.screen.write_process_input = lambda data: os.write(self.fd, data.encode())

    def pump(self, seconds=0.1):
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([self.fd], [], [], 0.05)
            if ready:
                try:
                    data = os.read(self.fd, 65536)
                except OSError:
                    return
                self.raw += data
                self.stream.feed(data)

    def text(self):
        return "\n".join(self.screen.display)

    def until(self, condition, timeout):
        started = time.time()
        while time.time() - started < timeout:
            self.pump()
            if condition(self.text()):
                return time.time() - started
        return None

    def type(self, keys: str):
        """Types `keys` a key at a time; an escape sequence (an arrow key) goes whole, as a terminal sends it."""
        i = 0
        while i < len(keys):
            n = 3 if keys.startswith("\x1b[", i) else 1
            os.write(self.fd, keys[i : i + n].encode())
            self.pump(0.05)
            i += n

    def exited(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            self.pump()
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                return os.waitstatus_to_exitcode(status)
        return None


def plain_terminal():
    print("in a terminal:")
    term = Terminal(env)
    took = term.until(lambda t: "esc quit" in t, 15)
    check(took is not None, f"draws within seconds ({took:.1f} s)" if took else "draws: not within 15 s")
    term.type("when is the boiler serviced")
    took = term.until(lambda t: "boiler.md" in t, 20)
    check(took is not None, "finds the note as the query is typed")
    check(term.until(lambda t: "engineer" in t, 5) is not None, "previews it")
    opened.unlink(missing_ok=True)
    term.type("\r")
    term.until(lambda t: opened.exists(), 5)
    check(opened.exists() and "boiler.md" in opened.read_text(), f"⏎ hands it to {opener}")
    term.type("\x1b")
    code = term.exited(10)
    check(code == 0, f"esc quits (exit {code})")
    check(b"\x1b[?1049l" in term.raw, "and gives the terminal back (leaves the alternate screen)")


def no_opener():
    print("with no opener (Linux without xdg-open):")
    path = ":".join(p for p in os.environ["PATH"].split(":") if not (pathlib.Path(p) / "xdg-open").exists())
    term = Terminal({**env, "PATH": path})
    term.until(lambda t: "esc quit" in t, 15)
    term.type("boiler")
    term.until(lambda t: "boiler.md" in t, 20)
    term.type("\r")
    check(term.until(lambda t: "can't open" in t, 3) is not None, "⏎ says it can't open, and goes on")
    term.type("\x1b")
    check(term.exited(10) == 0, "esc still quits")


def first_run():
    """The setup on a first run, in a made-up home: home folder, one folder unticked, no music, no login item."""
    print("first run (setup):")
    home = base / "home"
    for name, file in [("Documents", "lease.md"), ("Downloads", "receipt.md"), ("Music", "song.mp3")]:
        (home / name).mkdir(parents=True)
        (home / name / file).write_text(f"The {name.lower()} file: something to find about the lease and the rent.")
    setup_index = base / "setup-index"
    models = {k: v for k, v in os.environ.items() if k in ("HF_HUB_CACHE", "HF_HOME", "XDG_DATA_HOME", "RUMMAGE_MODELS")}
    # The model stays where it is (rummage looks under HOME otherwise).
    real_home = pathlib.Path.home()
    first_env = {**env, **models, "HOME": str(home), "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", str(real_home / ".cache/huggingface/hub"))}
    if "XDG_DATA_HOME" not in first_env and (real_home / ".local/share/rummage/models").exists():
        first_env["RUMMAGE_MODELS"] = str(next((real_home / ".local/share/rummage/models").iterdir()))
    term = Terminal(first_env, ["--index", str(setup_index), "index"])
    check(term.until(lambda t: "Index your home folder" in t, 10) is not None, "asks about the home folder")
    term.type("y\r")
    check(term.until(lambda t: "[x] Documents" in t and "[x] Music" in t, 10) is not None, "lists its folders, all ticked")
    term.type("\x1b[B ")  # down to Downloads, untick it
    check(term.until(lambda t: "[ ] Downloads" in t, 5) is not None, "space unticks one")
    term.type("\r")
    check(term.until(lambda t: "Include your music" in t, 10) is not None, "asks about music (Music is ticked)")
    term.type("n\r")
    check(term.until(lambda t: "background" in t, 10) is not None, "asks about the background service")
    term.type("n\r")
    code = term.exited(300)
    check(code == 0, f"indexes what was chosen (exit {code})")
    import sqlite3
    with sqlite3.connect(setup_index / "index.sqlite") as c:
        roots = [r[0] for r in c.execute("SELECT path FROM roots")]
        files = sorted(pathlib.Path(r[0]).name for r in c.execute("SELECT path FROM files"))
    check(roots == [str(home.resolve())] and files == ["lease.md"],
          f"the home folder, without Downloads or music: {files}")  # fmt: skip
    # Naming folders instead: a typo is explained, a real folder taken.
    term = Terminal(first_env, ["--index", str(base / "setup-index-2"), "index"])
    term.until(lambda t: "Index your home folder" in t, 10)
    term.type("n\r")
    term.until(lambda t: "Which folder?" in t, 5)
    term.type("~/Documnets\r")
    check(term.until(lambda t: "no such folder" in t, 5) is not None, "a mistyped folder is explained")
    term.type("~/Documents\r")
    check(term.until(lambda t: "added ~/Documents" in t, 5) is not None, "a real one is added")
    term.type("\r")
    term.until(lambda t: "background" in t, 5)
    term.type("n\r")
    check(term.exited(300) == 0, "and indexed")


def in_tmux():
    print("inside tmux:")
    socket = f"rummage-check-{os.getpid()}"
    tmux = ["tmux", "-L", socket]
    command = f"PATH={bin_dir}:$PATH TERM=tmux-256color {RUMMAGE} --index {index}"
    subprocess.run([*tmux, "new-session", "-d", "-x", str(COLUMNS), "-y", str(ROWS), "-s", "check", command],
                   check=True, env={k: v for k, v in env.items() if k != "TMUX"})  # fmt: skip

    def pane():
        return subprocess.run([*tmux, "capture-pane", "-p", "-t", "check"], capture_output=True, text=True).stdout

    def until(condition, timeout):
        started = time.time()
        while time.time() - started < timeout:
            if condition(pane()):
                return time.time() - started
            time.sleep(0.2)
        return None

    try:
        took = until(lambda t: "esc quit" in t, 15)
        check(took is not None, f"draws within seconds ({took:.1f} s)" if took else "draws: not within 15 s")
        subprocess.run([*tmux, "send-keys", "-t", "check", "-l", "how long should the dough rise"])
        check(until(lambda t: "bread.md" in t, 20) is not None, "finds the note as the query is typed")
        subprocess.run([*tmux, "send-keys", "-t", "check", "Escape"])
        gone = until(lambda t: subprocess.run([*tmux, "has-session", "-t", "check"], capture_output=True).returncode
                     != 0, 10)  # fmt: skip
        check(gone is not None, "esc quits")
    finally:
        subprocess.run([*tmux, "kill-server"], capture_output=True)


try:
    plain_terminal()
    first_run()
    if platform.system() != "Darwin" and not shutil.which("xdg-open"):
        no_opener()
    if shutil.which("tmux"):
        in_tmux()
    else:
        print("inside tmux: skipped (no tmux)")
finally:
    subprocess.run([RUMMAGE, "--index", str(index), "stop"], capture_output=True)
    shutil.rmtree(base, ignore_errors=True)

print(f"\n{'all passed' if not failures else f'{len(failures)} failed'}")
sys.exit(1 if failures else 0)
