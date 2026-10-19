"""Runs rummage in this terminal and types the demo's queries like a person would; record.sh films it. The terminal's
own input still reaches rummage, so it can answer rummage's question about pictures (kitty, iTerm2 or sixel).

    python3 demo/drive.py <rummage> [args…]     # the environment says which index and home to use
"""

import os
import pty
import random
import select
import signal
import sys
import termios
import time
import tty

# (text to type, or a key; seconds to wait after it)
CTRL_U, CTRL_V, CTRL_S, RIGHT = "\x15", "\x16", "\x13", "\x1b[C"
STEPS = [
    (None, 1.2),
    ("dog playing at the beach", 3.0),
    (CTRL_U, 0.3),
    ("late fee if the rent is paid late", 2.4),
    (CTRL_V, 2.2),
    (RIGHT, 1.6),
    (CTRL_U, 0.3),
    ("how long does caffeine stay in your body", 2.0),
    (CTRL_S, 3.6),
    (CTRL_U, 0.3),
    ("card declined", 3.0),
    (CTRL_U, 0.3),
    ("snowy cabin in the woods", 3.2),
    (CTRL_U, 0.3),
    ("un chat qui dort", 3.0),
    (CTRL_U, 0.3),
    ("how long should the dough rise", 3.4),
]


def duration() -> float:
    """Seconds the whole script takes (for the recorder)."""
    return sum(wait + (len(text) * 0.075 if text and len(text) > 1 and text[0] != "\x1b" else 0) for text, wait in STEPS)


if len(sys.argv) > 1 and sys.argv[1] == "--duration":
    print(f"{duration():.1f}")
    sys.exit()

pid, master = pty.fork()
if pid == 0:
    os.execvp(sys.argv[1], sys.argv[1:])

# The child's terminal is as big as this one, also when the window settles on its size after starting.
def resize(*_):
    termios.tcsetwinsize(master, termios.tcgetwinsize(sys.stdout.fileno()))
    os.kill(pid, signal.SIGWINCH)


resize()
signal.signal(signal.SIGWINCH, resize)
saved = termios.tcgetattr(sys.stdin.fileno())
tty.setraw(sys.stdin.fileno())


def pump(seconds: float):
    """Shows rummage's output and passes the terminal's input on, for `seconds`."""
    end = time.time() + seconds
    while True:
        left = end - time.time()
        if left <= 0:
            return
        ready, _, _ = select.select([master, sys.stdin], [], [], min(left, 0.02))
        if master in ready:
            try:
                data = os.read(master, 1 << 16)
            except OSError:
                raise SystemExit
            os.write(sys.stdout.fileno(), data)
        if sys.stdin in ready:
            os.write(master, os.read(sys.stdin.fileno(), 1024))


random.seed(4)
try:
    # Wait for the recorder to start (it creates $DEMO_GO), showing rummage meanwhile.
    go = os.environ.get("DEMO_GO")
    started = time.time()
    while go and not os.path.exists(go) and time.time() - started < 60:
        pump(0.05)
    for text, wait in STEPS:
        if text and len(text) > 1 and not text.startswith("\x1b"):
            for c in text:
                os.write(master, c.encode())
                pump(random.uniform(0.045, 0.105))
        elif text:
            os.write(master, text.encode())
        pump(wait)
    # The last result stays on screen until the recorder closes the window.
    while True:
        pump(1.0)
finally:
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)
