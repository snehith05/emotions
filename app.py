"""
Web UI for the emotional companion (companion_v7).

    python app.py                                   # offline rules, no key needed
    python app.py --backend groq                    # needs GROQ_API_KEY
    python app.py --backend groq --persona gandhi --intensity 6 --voice genz
    python app.py --backend ollama --model llama3.2
    python app.py --port 8080 --no-browser

Opens http://127.0.0.1:8000 in your browser: chat on the left, the companion's
inner state on the right (emotions, mood, trust, emotions over time, memories),
and a "why" panel that explains any reply you click. At the top you choose WHO it is
(persona), HOW EMOTIONAL it is (0 = off ... 10) and HOW it texts (voice); per-emotion
limits are under "Emotion limits".

Uses only the Python standard library for the server - nothing extra to install.
Same save files and log as companion_v7.py, so the CLI and the UI share each persona.
"""
import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from companion_v7 import Companion7, LOG_FILE, VERSION, EMOTIONS, state_path
from llm_backends import LLM, PROVIDERS, key_status
from persona import load_persona, list_personas, DEFAULT_ID
from voices import load_voice, list_voices

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "ui", "index.html")
MAX_TIMELINE = 300


def read_timeline(path, persona=DEFAULT_ID, limit=MAX_TIMELINE):
    """Turn records from the log since the last reset, so the chart survives a restart.
    Only this persona's records from this version count."""
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "reset" and rec.get("persona", DEFAULT_ID) == persona:
                rows = []
            elif "turn" in rec and rec.get("version") == VERSION and rec.get("persona", DEFAULT_ID) == persona:
                rows.append(rec)
    return rows[-limit:]


class App:
    def __init__(self, args):
        self.args = args
        _, _, dflt, dflt_fast = PROVIDERS[args.backend]
        self.main_model = args.model or dflt
        self.fast_model = args.fast_model or (self.main_model if args.backend == "ollama" else dflt_fast)
        self.log_file = None if args.no_log else args.log
        status = key_status(args.backend) or ""
        self.warning = (f"No {PROVIDERS[args.backend][1]} found, so every reply comes from the fixed offline "
                        f"rules, not the model. Put {PROVIDERS[args.backend][1]}=your-key in a file named .env "
                        f"in the project folder and restart app.py.") if status.startswith("WARNING") else None
        self.lock = threading.Lock()
        self.personas = {}                                  # loaded personas (their writings are indexed once)
        self.persona_id = args.persona
        self.companion = self.make(intensity=args.intensity,
                                   voice="none" if args.voice == "none" else load_voice(args.voice) if args.voice else None)
        self.timeline = read_timeline(self.log_file, self.persona_id)

    def persona(self, pid):
        if pid not in self.personas:
            self.personas[pid] = load_persona(pid)
        return self.personas[pid]

    def state_file(self):
        return self.args.state if self.args.state and self.persona_id == self.args.persona else state_path(self.persona_id)

    def make(self, intensity=None, voice=None):
        return Companion7(LLM(self.args.backend, self.main_model),
                          llm_fast=LLM(self.args.backend, self.fast_model), persona=self.persona(self.persona_id),
                          state_file=self.state_file(), log_file=self.log_file, intensity=intensity, voice=voice)

    def info(self):
        c = self.companion
        return {"backend": self.args.backend,
                "model": None if self.args.backend == "offline" else self.main_model,
                "warning": self.warning,
                "snapshot": c.snapshot(),
                "history": c.history[-40:],
                "timeline": self.timeline,
                "personas": list_personas(),
                "voices": list_voices(),
                "emotion_names": EMOTIONS}

    def chat(self, text):
        with self.lock:
            reply = self.companion.say(text)
            turn = self.companion.last_turn
            self.timeline = (self.timeline + [turn])[-MAX_TIMELINE:]
            return {"reply": reply, "turn": turn, "snapshot": self.companion.snapshot()}

    def reset(self):
        with self.lock:
            keep = self.companion.emo.intensity, self.companion.voice or "none"
            if os.path.exists(self.companion.state_file):
                os.remove(self.companion.state_file)
            if self.log_file:
                self.companion.write_log({"event": "reset", "persona": self.persona_id})
            self.companion = self.make(*keep)
            self.timeline = []
            return self.info()

    def settings(self, data):
        """{"intensity": 0-10, "voice": "genz"|"none", "limits": {"anger": 3, "joy": null}}"""
        with self.lock:
            c = self.companion
            if "intensity" in data:
                c.set_intensity(int(data["intensity"]))
            if "voice" in data:
                c.set_voice(data["voice"] or "none")
            for emo, n in (data.get("limits") or {}).items():
                c.set_limit(emo, None if n is None else int(n))
            if data.get("clear_limits"):
                for emo in list(c.emo.user_caps):
                    c.set_limit(emo, None)
            return {"snapshot": c.snapshot()}

    def switch(self, pid):
        """Talk to someone else. Each persona keeps its own feelings, memories and settings."""
        with self.lock:
            if pid not in {p["id"] for p in list_personas()}:
                raise ValueError(f"no persona '{pid}'")
            self.persona_id = pid
            self.companion = self.make()
            self.timeline = read_timeline(self.log_file, pid)
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
            elif self.path in ("/api/settings", "/api/persona"):
                try:
                    out = app.settings(data) if self.path == "/api/settings" else app.switch(str(data.get("id", "")))
                    self.send_json(out)
                except (ValueError, TypeError, SystemExit) as e:
                    self.send_json({"error": str(e)}, 400)
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
    p.add_argument("--persona", default=DEFAULT_ID, help="who it is: companion, gandhi, or a folder in personas/")
    p.add_argument("--intensity", type=int, default=None, help="0 = emotions off, 1-10 = how emotional")
    p.add_argument("--voice", default=None, help="how it texts: genz, millennial, desi, filmy, none")
    p.add_argument("--state", default=None, help="where feelings and memories are saved")
    p.add_argument("--log", default=LOG_FILE, help="where every message is logged")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    a = p.parse_args()

    status = key_status(a.backend)
    if status and status.startswith("WARNING"):
        # don't quietly run on keyword rules while the page says "groq" - that only looks like a dumb AI
        raise SystemExit(status.replace("Until then every reply uses the offline rules.",
                                        "Stopping here. To try without a model, run: python app.py"))
    if status:
        print(status)
    app = App(a)
    server = start_server(app, a.port)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    print(f"Companion UI running at {url}   [{a.backend}"
          + (f": {app.main_model}" if a.backend != "offline" else "") + f"]   talking to: {app.companion.persona.name}")
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
