"""
Web UI for the emotional companion (companion_v5).

    python app.py                                   # offline rules, no key needed
    python app.py --backend groq                    # needs GROQ_API_KEY
    python app.py --backend ollama --model llama3.2
    python app.py --port 8080 --no-browser

Opens http://127.0.0.1:8000 in your browser: chat on the left, the companion's
inner state on the right (emotions, mood, trust, emotions over time, memories),
and a "why" panel that explains any reply you click.

Uses only the Python standard library for the server - nothing extra to install.
Same save file and log as companion_v5.py, so the CLI and the UI share one companion.
"""
import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from companion_v5 import Companion, SAVE_FILE, LOG_FILE
from llm_backends import LLM, PROVIDERS, key_status

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "ui", "index.html")
MAX_TIMELINE = 300


def read_timeline(path, limit=MAX_TIMELINE):
    """Turn records from the log since the last reset, so the chart survives a restart."""
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "reset":
                rows = []
            elif "turn" in rec:
                rows.append(rec)
    return rows[-limit:]


class App:
    def __init__(self, args):
        self.args = args
        _, _, dflt, dflt_fast = PROVIDERS[args.backend]
        self.main_model = args.model or dflt
        self.fast_model = args.fast_model or (self.main_model if args.backend == "ollama" else dflt_fast)
        self.log_file = None if args.no_log else args.log
        self.lock = threading.Lock()
        self.companion = self.make()
        self.timeline = read_timeline(self.log_file)

    def make(self):
        return Companion(LLM(self.args.backend, self.main_model),
                         llm_fast=LLM(self.args.backend, self.fast_model),
                         state_file=self.args.state, log_file=self.log_file)

    def info(self):
        c = self.companion
        return {"backend": self.args.backend,
                "model": None if self.args.backend == "offline" else self.main_model,
                "snapshot": c.snapshot(),
                "history": c.history[-40:],
                "timeline": self.timeline}

    def chat(self, text):
        with self.lock:
            reply = self.companion.say(text)
            turn = self.companion.last_turn
            self.timeline = (self.timeline + [turn])[-MAX_TIMELINE:]
            return {"reply": reply, "turn": turn, "snapshot": self.companion.snapshot()}

    def reset(self):
        with self.lock:
            if os.path.exists(self.args.state):
                os.remove(self.args.state)
            if self.log_file:
                self.companion.write_log({"event": "reset"})
            self.companion = self.make()
            self.timeline = []
            return self.info()


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, data, code=200):
            body = json.dumps(data, ensure_ascii=False, default=float).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                try:
                    with open(PAGE, "rb") as f:
                        body = f.read()
                except OSError:
                    self.send_error(500, "ui/index.html is missing - keep the ui folder next to app.py")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self.send_json(app.info())
            else:
                self.send_error(404)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self.send_json({"error": "bad request"}, 400)
                return
            if self.path == "/api/chat":
                text = str(data.get("message", "")).strip()[:2000]
                if not text:
                    self.send_json({"error": "empty message"}, 400)
                    return
                try:
                    self.send_json(app.chat(text))
                except Exception as e:                      # show the problem in the page, keep serving
                    self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)
            elif self.path == "/api/reset":
                self.send_json(app.reset())
            else:
                self.send_error(404)

        def log_message(self, fmt, *args):                  # keep the terminal quiet
            pass

    return Handler


def start_server(app, port):
    """Try the requested port, then a few others. Windows often blocks or reserves ports
    (WinError 10013 / 10048), e.g. when Hyper-V, WSL or Docker is installed."""
    tries = [port] + [p for p in (5050, 7860, 8501, 8765, 9000, 0) if p != port]
    last = None
    for p in tries:                                   # 0 = let the system pick any free port
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), make_handler(app))
            if p != port:
                print(f"(port {port} is blocked or busy on this computer - using another port instead)")
            return server
        except OSError as e:
            last = e
    raise SystemExit(f"Could not open any port to run the UI: {last}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="offline", choices=list(PROVIDERS))
    p.add_argument("--model", default=None, help="main model writing the replies")
    p.add_argument("--fast-model", default=None, help="cheap model that rates each message")
    p.add_argument("--state", default=SAVE_FILE, help="where feelings and memories are saved")
    p.add_argument("--log", default=LOG_FILE, help="where every message is logged")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    a = p.parse_args()

    status = key_status(a.backend)
    if status:
        print(status)
    app = App(a)
    server = start_server(app, a.port)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    print(f"Companion UI running at {url}   [{a.backend}"
          + (f": {app.main_model}" if a.backend != "offline" else "") + "]")
    print("Press Ctrl+C to stop.")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
