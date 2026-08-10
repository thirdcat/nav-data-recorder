#!/usr/bin/env python3
"""Receive sessions the phone uploads over Wi-Fi.

The app's uploader is a background `URLSession` that `PUT`s one file at a time
to `<baseURL>/<sessionID>/<relativePath>`, with an optional
`Authorization: Bearer <token>`, and treats any 2xx as success. That is the
whole protocol, so the server is the whole protocol too — no framework, no
dependencies, standard library only like the rest of `tools/`.

    python3 tools/upload_server.py                    # serve into ~/nav_data
    python3 tools/upload_server.py --root /data --port 8080
    python3 tools/upload_server.py --token s3cret     # require a bearer token

It prints the URL to type into Settings → Upload on the phone. Both devices
have to be on the same network; a phone on cellular cannot reach a laptop.

**Why a background upload rather than the Files app.** A 30 s capture is a
couple of hundred megabytes and a session directory is hundreds of files.
Dragging that out of the Files app is a per-session chore that gets skipped,
and data that is annoying to collect does not get collected. The uploader
retries across app launches and network drops; this side just has to be
boring and correct.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Files arrive one at a time but several sessions can be in flight, and the
# background session opens more than one connection. Reading in chunks keeps a
# 50 MB depth.bin from being held in memory twice.
CHUNK = 1 << 20


IDLE_DONE = 6.0          # seconds of quiet before a session is called finished

_print_lock = threading.Lock()


def note(line: str):
    """Print above the repainting status line without tearing it."""
    with _print_lock:
        sys.stdout.write("\r\033[K" + line + "\n")
        sys.stdout.flush()


def expected_files(manifest_path: str) -> int | None:
    """How many files this session will send, read from its own manifest.

    The uploader walks the session directory and skips hidden files, so the
    payload is the nine index/stream files, one or two binaries, and one JPEG
    per still. `counts.frames` is in the manifest, which arrives like any other
    file — so once it lands the console can show a denominator instead of a
    number that only goes up.

    Returns None rather than guessing when the manifest is not what we expect;
    a wrong denominator is worse than none.
    """
    try:
        with open(manifest_path) as f:
            m = json.load(f)
        counts = m["counts"]
        config = m.get("config", {})
    except (OSError, ValueError, KeyError):
        return None

    # manifest, pose, motion, location, heading, planes, events, depth index,
    # and — in stills mode — the frames index.
    streams = 8
    if config.get("captureMode", "stills") == "stills":
        streams += 1                                     # frames.jsonl
        images = int(counts.get("frames", 0))
    else:
        images = 0                                       # one video.mov instead
        streams += 1
    binaries = 1                                         # depth.bin
    if config.get("recordConfidence"):
        binaries += 1
    return streams + binaries + images


class State:
    """What the console needs to answer 'how far along is it, and is it done'.

    Completed-file lines alone cannot answer either question: a 50 MB depth.bin
    prints nothing for its whole transfer, and silence at the end looks the
    same as a stall. So this also tracks bytes in flight per active transfer,
    and how long a session has been quiet.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.sessions: dict[str, dict] = {}
        self.active: dict[int, tuple[str, str, int, int]] = {}   # thread -> …
        self.done_announced: set[str] = set()

    def _session(self, name: str) -> dict:
        return self.sessions.setdefault(
            name, {"files": 0, "bytes": 0, "expected": None,
                   "first": time.time(), "last": time.time()})

    def begin(self, session: str, rel: str, total: int):
        with self.lock:
            self._session(session)
            self.active[threading.get_ident()] = (session, rel, 0, total)

    def advance(self, sent: int):
        with self.lock:
            key = threading.get_ident()
            if key in self.active:
                s, r, _, t = self.active[key]
                self.active[key] = (s, r, sent, t)

    def finish(self, session: str, rel: str, nbytes: int, root: str):
        with self.lock:
            self.active.pop(threading.get_ident(), None)
            s = self._session(session)
            s["files"] += 1
            s["bytes"] += nbytes
            s["last"] = time.time()
            self.done_announced.discard(session)
            if rel == "manifest.json":
                s["expected"] = expected_files(
                    os.path.join(root, session, "manifest.json"))
            return s["files"], s["bytes"]


def safe_parts(path: str) -> list[str] | None:
    """Split a request path into components, or None if it is not safe to use.

    The session id and relative path come off the network. What the app
    actually sends is `<yyyyMMdd-HHmmss-xxxxxx>/<name>` or
    `.../frames/000123.jpg`, so this allows exactly that shape and refuses the
    rest rather than trying to sanitise it.

    `%` is rejected because nothing the app sends contains one, and
    `BaseHTTPRequestHandler` does not percent-decode — so without this a
    `%2e%2e` walks in as a literal directory. It cannot escape the root, but
    junk directories in a data tree are their own kind of bug.
    """
    parts = [p for p in path.split("/") if p != ""]
    if len(parts) < 2:
        return None
    for p in parts:
        # The completion marker is the only dotfile in the payload.
        if p.startswith(".") and p != ".complete":
            return None
        if "%" in p or "\0" in p:
            return None
        if os.sep in p or (os.altsep and os.altsep in p):
            return None
    return parts


def inside(root: str, dest: str) -> bool:
    """Belt and braces: the resolved destination really is under the root.

    `safe_parts` should make this impossible to fail, which is the reason to
    check it — a path guard that is never verified is a guard nobody knows is
    broken. Resolving also catches a symlink planted in the tree.
    """
    try:
        return os.path.commonpath([os.path.realpath(root),
                                   os.path.realpath(dest)]) == os.path.realpath(root)
    except ValueError:                       # different drives, on Windows
        return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "nav-data-recorder-upload/1"

    root: str
    token: str
    state: State

    def log_message(self, fmt, *args):
        """The default logs every request; this logs only what went wrong.

        Suppressing it entirely was a mistake worth naming: a phone can be
        sending a hundred requests a minute and getting a hundred 401s, and the
        console will sit there looking idle. Silence has to mean *nothing
        arrived*, or it tells you nothing at all.
        """
        pass

    def log_error(self, fmt, *args):
        note(f"  ! {self.address_string()}  {fmt % args}")

    def handle_one_request(self):
        # A client that opens a connection and closes it without sending
        # anything — a keep-alive going idle, a cancelled task — raises out of
        # the base class as an unhandled traceback. It is normal, and a page of
        # stack trace per occurrence buries anything that is not.
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            self.close_connection = True

    def _reject(self, code: int, why: str):
        """Refuse a request and hang up.

        Hanging up is the whole point. A rejected PUT still has its body in the
        socket — tens of megabytes of it — and this connection is keep-alive, so
        whatever is left gets read as the *next* request line and parses as
        garbage. One 401 then poisons every request that follows on the same
        connection, which looks like a network fault rather than a bad token.
        Draining the body instead would mean accepting the upload we just
        refused, so the answer is `Connection: close`.
        """
        note(f"  ! {self.address_string()}  {self.command} {self.path}"
             f"  -> {code} {why}")
        body = (why + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _authorised(self) -> bool:
        if not self.token:
            return True
        got = self.headers.get("Authorization", "")
        return hmac.compare_digest(got, f"Bearer {self.token}")

    def do_PUT(self):
        if not self._authorised():
            self._reject(401, "bad or missing bearer token")
            return

        parts = safe_parts(self.path.split("?", 1)[0])
        if parts is None:
            self._reject(400, "path must be <sessionID>/<file> and stay inside it")
            return

        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            # The uploader always sets it. Without it there is no way to know
            # where the body ends on a keep-alive connection.
            self._reject(411, "Content-Length required")
            return

        dest = os.path.join(self.root, *parts)
        if not inside(self.root, dest):
            self._reject(400, "destination escapes the root")
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        # Write beside the target and rename. An interrupted upload then leaves
        # a .part file rather than a truncated one that reads as a good capture
        # — the same failure the session packer exists to avoid.
        relative = "/".join(parts[1:])
        tmp = dest + ".part"
        received = 0
        self.state.begin(parts[0], relative, length)
        try:
            with open(tmp, "wb") as f:
                while received < length:
                    chunk = self.rfile.read(min(CHUNK, length - received))
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)
                    self.state.advance(received)
            if received != length:
                os.unlink(tmp)
                self.state.active.pop(threading.get_ident(), None)
                self._reject(400, f"short body: {received} of {length}")
                return
            os.replace(tmp, dest)
        except OSError as exc:
            self.state.active.pop(threading.get_ident(), None)
            if os.path.exists(tmp):
                os.unlink(tmp)
            self._reject(500, f"write failed: {exc}")
            return

        self.state.finish(parts[0], relative, received, self.root)

        self.send_response(201)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        """A browser check, so 'is it running' does not need the phone."""
        if not self._authorised():
            self._reject(401, "bad or missing bearer token")
            return
        with self.state.lock:
            lines = [
                f"{sid}: {s['files']}"
                + (f"/{s['expected']}" if s.get("expected") else "")
                + f" files, {s['bytes'] / 1e6:.0f} MB"
                + ("  (receiving)" if time.time() - s["last"] < IDLE_DONE
                   else "  (idle)")
                for sid, s in sorted(self.state.sessions.items())]
        body = ("nav-data-recorder upload server\nroot: " + self.root + "\n\n"
                + ("\n".join(lines) if lines else "nothing received yet")
                + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def bar(fraction: float, width: int = 22) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "▓" * filled + "░" * (width - filled)


def console(state: State, stop: threading.Event):
    """Repaint one status line, and announce a session when it goes quiet.

    A repainting line rather than a line per file: the interesting quantities —
    which file is moving, how fast, how much of the session is left — all
    change continuously, and a scrolling log of finished files shows none of
    them while a 50 MB binary is in flight.
    """
    last_bytes, last_time, rate = 0, time.time(), 0.0
    painted = False
    while not stop.wait(0.25):
        now = time.time()
        with state.lock:
            active = list(state.active.values())
            sessions = {k: dict(v) for k, v in state.sessions.items()}
            newly_done = [
                name for name, s in state.sessions.items()
                if s["files"] and now - s["last"] > IDLE_DONE
                and name not in state.done_announced]
            for name in newly_done:
                state.done_announced.add(name)

        total_bytes = sum(s["bytes"] for s in sessions.values())
        if now - last_time >= 1.0:
            rate = (total_bytes - last_bytes) / (now - last_time)
            last_bytes, last_time = total_bytes, now

        for name in newly_done:
            s = sessions[name]
            painted = False
            exp = s["expected"]
            if exp is None:
                verdict = "manifest not received, cannot check completeness"
            elif s["files"] >= exp:
                verdict = f"complete, {s['files']}/{exp} files"
            else:
                verdict = f"INCOMPLETE — {s['files']} of {exp} files"
            note(f"  {name}  {s['bytes'] / 1e6:.0f} MB in "
                 f"{s['last'] - s['first']:.0f}s   {verdict}")

        if not active:
            if painted:
                with _print_lock:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                painted = False
            continue

        name, rel, sent, total = active[0]
        s = sessions.get(name, {})
        exp, done = s.get("expected"), s.get("files", 0)
        head = f"{done}/{exp} files" if exp else f"{done} files"
        frac = (done / exp) if exp else 0.0
        extra = f" +{len(active) - 1}" if len(active) > 1 else ""
        line = (f"  {name}  {bar(frac)} {head}  "
                f"{s.get('bytes', 0) / 1e6:6.0f} MB  {rate / 1e6:5.1f} MB/s  "
                f"▸ {rel[-24:]} {100 * sent / max(total, 1):3.0f}%{extra}")
        with _print_lock:
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()
        painted = True


def lan_addresses() -> list[str]:
    """Addresses the phone could plausibly reach us on.

    `gethostbyname(gethostname())` returns 127.0.1.1 on many Linux setups,
    which the phone cannot use, so the routable address is found by asking the
    kernel which source it would pick for an outbound route. No packet is sent.
    """
    found = []
    for probe in ("8.8.8.8", "192.168.1.1"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 80))
            addr = s.getsockname()[0]
            if addr not in found and not addr.startswith("127."):
                found.append(addr)
        except OSError:
            pass
        finally:
            s.close()
    return found


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.expanduser("~/nav_data"),
                    help="where sessions land (default: ~/nav_data)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="0.0.0.0",
                    help="0.0.0.0 so the phone can reach it; 127.0.0.1 to hide")
    ap.add_argument("--token", default="",
                    help="require Authorization: Bearer <token>")
    args = ap.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(args.root))
    os.makedirs(root, exist_ok=True)

    Handler.root = root
    Handler.token = args.token
    Handler.state = State()

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True

    stop = threading.Event()
    painter = threading.Thread(target=console, args=(Handler.state, stop),
                               daemon=True)

    print(f"serving into {root}")
    addrs = lan_addresses() if args.bind == "0.0.0.0" else [args.bind]
    print("\nSettings -> Upload -> Base URL, on the phone:")
    for a in addrs or ["<this machine's LAN address>"]:
        print(f"    http://{a}:{args.port}")
    if args.token:
        print(f"  and Auth token: {args.token}")
    else:
        print("  (no token — anyone on this network can PUT here; use --token"
              " if that is not what you want)")
    print("\nThe phone uploads only sessions it has finished writing, retries"
          "\nacross launches, and keeps its local copy unless you turned that"
          "\noff. Ctrl-C to stop.\n")

    painter.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop.set()
        painter.join(timeout=1.0)
        print("\r\033[K\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
