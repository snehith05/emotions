"""
PERSONAS: give the emotion engine a specific person to be.

A persona is a folder:

    personas/<id>/
        data/                  everything you have about the person: .txt / .md files, or .jsonl with
                               {"text": ...} per line (writings, letters, speeches, interviews, chats)
        persona.json           who they are and how they feel - written by hand (optional)
        persona.generated.json the same, worked out from data/ by `python persona.py build <id>`

persona.json wins over persona.generated.json, so hand edits survive a rebuild.

What the persona changes:
    knowledge    their writings are cut into passages; the ones related to each message are found
                 (BM25 search) and given to the model, so it answers from what they actually said (RAG)
    personality  Big Five traits -> how hard good and bad things hit, mood, memory
    emotions     resting levels (baseline), how easily each emotion is stirred (sensitivity),
                 hard limits (caps), how much of each feeling they let show (display)
    triggers     things that move THIS person: "violence or cruelty -> grief 6, sadness 5"
    expression   how they show each emotion ("anger -> calm, firm resolve, never hatred")
    voice        how they speak, era, values, example replies

Commands:
    python persona.py list
    python persona.py new <id> --name "Full Name"      make the folder; then put files in data/
    python persona.py build <id> --backend groq        work out the profile from data/ with a model
    python persona.py search <id> "what is swaraj"     test what the knowledge search finds
"""
import argparse
import json
import os
import random
import re
import time

from retrieval import BM25, tokens, chunk_text, best_sentences

HERE = os.path.dirname(os.path.abspath(__file__))
PERSONA_DIR = os.path.join(HERE, "personas")
DEFAULT_ID = "companion"

# filled in from companion_v6 when first needed (avoids a circular import)
_EMO = {}


def _emotions():
    if not _EMO:
        from companion_v6 import EMOTIONS, canon
        _EMO["list"], _EMO["canon"] = EMOTIONS, canon
    return _EMO["list"], _EMO["canon"]


# ============================================================ KNOWLEDGE (RAG)
class KnowledgeBase:
    """A person's writings, cut into passages and indexed for search."""

    def __init__(self, folder):
        self.passages = []                       # dicts: text, source, heading
        self.index = BM25()
        self.files = []
        if not os.path.isdir(folder):
            return
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].lower()
            try:
                if ext in (".txt", ".md"):
                    with open(path, encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    self._add_text(text, self._title(text, name))
                elif ext == ".jsonl":
                    with open(path, encoding="utf-8", errors="replace") as f:
                        for line in f:
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(rec, dict) and str(rec.get("text", "")).strip():
                                self._add_text(str(rec["text"]), str(rec.get("source") or name))
                else:
                    continue
                self.files.append(name)
            except OSError as e:
                print(f"  (could not read {path}: {e})")

    @staticmethod
    def _title(text, filename):
        first = text.lstrip().split("\n", 1)[0].strip()
        if first.startswith("#"):
            return re.split(r" \(| - ", first.lstrip("# "))[0].strip()[:90]
        return os.path.splitext(filename)[0].replace("_", " ")

    def _add_text(self, text, source):
        for heading, passage in chunk_text(text):
            if heading.startswith(source[:20]):  # the file's own title line is not a chapter
                heading = ""
            self.passages.append({"text": passage, "source": source, "heading": heading})
            self.index.add(tokens(passage + " " + heading))

    def __len__(self):
        return len(self.passages)

    def words(self):
        return sum(len(p["text"].split()) for p in self.passages)

    def search(self, query, k=3):
        """The k passages that match the query best - at most one per chapter, so the model
        gets several angles instead of three slices of the same page."""
        out, seen = [], set()
        for i, score in self.index.search(query, k=k * 4):
            p = self.passages[i]
            key = (p["source"], p["heading"])
            if key in seen and p["heading"]:
                continue
            seen.add(key)
            out.append(dict(p, score=round(score, 2), id=i))
            if len(out) == k:
                break
        return out


# ============================================================ THE PERSONA
DEFAULT_PROFILE = {
    "name": "Companion", "short": "Companion",
    "summary": "", "era": "", "voice": "", "values": [],
    "traits": {"N": 0.4, "O": 0.8, "C": 0.6, "E": 0.6},
    "default_intensity": 5,
    "emojis": True,
    "baseline": {}, "sensitivity": {}, "caps": {}, "display": {},
    "expression": {}, "triggers": [], "examples": [], "greeting": "",
}


class Persona:
    def __init__(self, pid, folder=None, profile=None):
        self.id, self.folder = pid, folder
        self.profile = dict(DEFAULT_PROFILE, **(profile or {}))
        self.kb = KnowledgeBase(os.path.join(folder, "data")) if folder else None
        self._clean()

    @property
    def is_default(self):
        return self.id == DEFAULT_ID

    @property
    def name(self):
        return self.profile["name"]

    def _clean(self):
        """Keep only valid emotion names and numbers in range, whatever the JSON said."""
        emotions, canon = _emotions()
        p = self.profile

        def emo_map(d, lo, hi):
            out = {}
            for k, v in (d or {}).items():
                e = canon(k)
                try:
                    if e:
                        out[e] = min(hi, max(lo, float(v)))
                except (TypeError, ValueError):
                    pass
            return out

        p["baseline"] = emo_map(p.get("baseline"), 0, 10)          # levels 0-10, like everywhere else
        p["sensitivity"] = emo_map(p.get("sensitivity"), 0, 3)      # 1 = normal
        p["caps"] = emo_map(p.get("caps"), 0, 10)
        p["display"] = emo_map(p.get("display"), 0, 1.5)            # 1 = shows what it feels
        p["expression"] = {canon(k): str(v) for k, v in (p.get("expression") or {}).items() if canon(k)}
        t = p.get("traits") or {}
        p["traits"] = {k: min(1.0, max(0.0, float(t.get(k, DEFAULT_PROFILE["traits"][k]))))
                       for k in ("N", "O", "C", "E")}
        try:
            p["default_intensity"] = int(min(10, max(0, round(float(p.get("default_intensity", 5))))))
        except (TypeError, ValueError):
            p["default_intensity"] = 5
        trigs = []
        for i, tr in enumerate(p.get("triggers") or []):
            if not isinstance(tr, dict) or not tr.get("when"):
                continue
            feel = {canon(k): int(min(10, max(1, round(float(v)))))
                    for k, v in (tr.get("feel") or {}).items() if canon(k) and _isnum(v)}
            if not feel:
                continue
            trigs.append({"id": str(tr.get("id") or f"t{i + 1}"), "when": str(tr["when"])[:160],
                          "keywords": [str(w).lower() for w in (tr.get("keywords") or [])][:25],
                          "feel": feel, "show": str(tr.get("show") or "")[:200]})
        p["triggers"] = trigs

    def passages(self, query, k=3):
        if not self.kb or not len(self.kb):
            return []
        return self.kb.search(query, k)

    def offline_triggers(self, text, ap=None):
        """Without a model: a trigger fires when the message uses one of its keywords."""
        low = " " + text.lower() + " "
        toks = set(tokens(text))
        fired = []
        for tr in self.profile["triggers"]:
            for kw in tr["keywords"]:
                if (" " in kw and kw in low) or (set(tokens(kw)) and set(tokens(kw)) <= toks):
                    fired.append(tr)
                    break
        return fired

    def describe(self):
        kb = self.kb
        return {"id": self.id, "name": self.name, "short": self.profile.get("short") or self.name,
                "summary": self.profile.get("summary", ""), "era": self.profile.get("era", ""),
                "default_intensity": self.profile["default_intensity"],
                "passages": len(kb) if kb else 0, "words": kb.words() if kb else 0,
                "files": kb.files if kb else [], "greeting": self.profile.get("greeting", ""),
                "suggestions": [str(x) for x in (self.profile.get("suggestions") or [])][:8]}


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def load_persona(pid=None):
    """'gandhi' -> the persona in personas/gandhi. None or 'companion' -> the original companion."""
    if not pid or pid == DEFAULT_ID:
        return Persona(DEFAULT_ID)
    folder = pid if os.path.isdir(pid) else os.path.join(PERSONA_DIR, pid)
    if not os.path.isdir(folder):
        raise SystemExit(f"No persona '{pid}'. Make one with: python persona.py new {pid} --name \"Their Name\"")
    profile = _read_json(os.path.join(folder, "persona.generated.json"))
    profile.update(_read_json(os.path.join(folder, "persona.json")))
    profile.setdefault("name", os.path.basename(folder.rstrip("/\\")).title())
    return Persona(os.path.basename(os.path.normpath(folder)), folder, profile)


def list_personas():
    out = [{"id": DEFAULT_ID, "name": "Companion (default)"}]
    if os.path.isdir(PERSONA_DIR):
        for pid in sorted(os.listdir(PERSONA_DIR)):
            folder = os.path.join(PERSONA_DIR, pid)
            if os.path.isdir(folder):
                prof = _read_json(os.path.join(folder, "persona.generated.json"))
                prof.update(_read_json(os.path.join(folder, "persona.json")))
                out.append({"id": pid, "name": prof.get("name") or pid.title()})
    return out


# ============================================================ BUILDING A PROFILE FROM DATA
NOTES_PROMPT = """You are studying a real person from samples of their own words, so that an AI can
later simulate how they think, FEEL and react. Read the samples and reply with ONLY a JSON object:
{"values": ["what they stand for, 3-8 items"],
 "beliefs": ["firm opinions they hold, 3-10 short items"],
 "reactions": [{"when": "kind of situation or remark", "feel": {"<emotion>": 1-10},
                "show": "how they express it", "keywords": ["words that signal it"]}],
 "temperament": "how emotional they are, what moves them, how they handle anger, grief, praise, insult",
 "voice": "how they write and speak: sentence length, tone, typical moves, address forms",
 "phrases": ["short phrases they actually use, copied exactly from the samples"]}
Use emotions ONLY from this list: EMOTION_LIST
Base everything on the samples. No other text."""

PROFILE_PROMPT = """You turn research notes about a real person into an emotional profile that drives
an emotion simulation. Reply with ONLY a JSON object:
{"name": "full name", "short": "short name",
 "summary": "2-3 sentences: who they are",
 "era": "when they lived and the last events they knew about",
 "values": ["5-8 short items"],
 "voice": "2-4 sentences: how they speak in conversation",
 "traits": {"N": 0-1, "O": 0-1, "C": 0-1, "E": 0-1},
 "default_intensity": 1-10,
 "baseline": {"<emotion>": 0-10},
 "sensitivity": {"<emotion>": 0-3},
 "caps": {"<emotion>": 0-10},
 "display": {"<emotion>": 0-1.5},
 "expression": {"<emotion>": "how THIS person shows it, one sentence"},
 "triggers": [{"id": "short_id", "when": "situation", "keywords": ["5-15 words"],
               "feel": {"<emotion>": 1-10}, "show": "how they respond"}],
 "examples": [{"them": "a message someone might send", "you": "how they would answer, in their voice"}],
 "greeting": "how they would greet a visitor, one sentence"}
traits: Big Five - N neuroticism (worry, emotional swings), O openness, C conscientiousness, E extraversion.
default_intensity: how emotional they are overall on 1-10 (5 = typical person).
baseline: resting emotions they carry most days (0-10, only the noticeable ones, 1-4, never above 5).
sensitivity: how easily each emotion is stirred compared with a typical person (1 = typical, 0.2 = hardly,
             2 = very easily). Only list emotions that clearly differ.
caps: emotions they never let go above a level (e.g. "hatred": 1 for someone who refuses to hate).
display: how much of a felt emotion they let show (1 = shows it all, 0.3 = keeps it mostly inside).
expression: give one for each emotion that matters for them (at least 10). This is a TEXT conversation:
            describe it in words - tone, sentence style, what they would say - never body language.
triggers: 6-12 situations that move them strongly, with the keywords that signal each. "feel" holds
          passing emotions only (not love or trust).
examples: 4-6 short exchanges.
Use emotions ONLY from this list: EMOTION_LIST
Stay faithful to the notes. No other text."""


def _ask_json(llm, system, body, max_tokens):
    raw = llm(system, [{"role": "user", "content": body}], max_tokens=max_tokens, json_mode=True)
    if not raw:
        raw = llm(system, [{"role": "user", "content": body}], max_tokens=max_tokens)
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        return None


def build_profile(pid, llm, batches=3, words_per_batch=2200, name=None, seed=0):
    """Read samples spread across the person's data, take notes on how they think and feel,
    then turn the notes into a profile. Writes personas/<id>/persona.generated.json."""
    emotions, _ = _emotions()
    persona = load_persona(pid)
    kb = persona.kb
    if not kb or not len(kb):
        raise SystemExit(f"personas/{pid}/data has no .txt/.md/.jsonl files to learn from.")
    name = name or persona.profile.get("name")
    rng = random.Random(seed)
    order = list(range(len(kb.passages)))
    rng.shuffle(order)
    notes, used = [], 0
    for b in range(batches):
        sample, n = [], 0
        while used < len(order) and n < words_per_batch:
            p = kb.passages[order[used]]
            used += 1
            sample.append(f"[{p['source']}] {p['text']}")
            n += len(p["text"].split())
        if not sample:
            break
        print(f"  reading sample {b + 1}/{batches} ({n} words)...")
        got = _ask_json(llm, NOTES_PROMPT.replace("EMOTION_LIST", ", ".join(emotions)),
                        f"Person: {name}\n\nSamples of their own words:\n\n" + "\n\n".join(sample), 1500)
        if got:
            notes.append(got)
        else:
            print("  (no usable notes from this sample - skipping it)")
    if not notes:
        raise SystemExit("The model gave no usable notes - check the backend/key and try again.")
    print("  writing the profile...")
    prof = _ask_json(llm, PROFILE_PROMPT.replace("EMOTION_LIST", ", ".join(emotions)),
                     f"Person: {name}\n\nNotes from {len(notes)} samples of their writing:\n"
                     + json.dumps(notes, ensure_ascii=False)[:14000], 3000)
    if not prof:
        raise SystemExit("The model did not return a usable profile - try again.")
    prof["name"] = name or prof.get("name")
    prof["built"] = {"time": time.strftime("%Y-%m-%d %H:%M"), "samples": len(notes),
                     "files": kb.files, "model": getattr(llm, "model", "?")}
    path = os.path.join(PERSONA_DIR, pid, "persona.generated.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=1, ensure_ascii=False)
    return path, Persona(pid, os.path.join(PERSONA_DIR, pid), prof)


def main():
    from llm_backends import LLM, PROVIDERS, key_status
    ap = argparse.ArgumentParser(description="Make, build and test personas.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    n = sub.add_parser("new")
    n.add_argument("id")
    n.add_argument("--name", required=True)
    b = sub.add_parser("build")
    b.add_argument("id")
    b.add_argument("--backend", default="groq", choices=[p for p in PROVIDERS if p != "offline"])
    b.add_argument("--model", default=None)
    b.add_argument("--batches", type=int, default=3, help="how many samples of the data to read")
    s = sub.add_parser("search")
    s.add_argument("id")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=3)
    a = ap.parse_args()

    if a.cmd == "list":
        for p in list_personas():
            info = ""
            if p["id"] != DEFAULT_ID:
                d = load_persona(p["id"]).describe()
                info = f"  {d['passages']} passages, {d['words']:,} words from {len(d['files'])} files"
            print(f"  {p['id']:<14} {p['name']}{info}")
    elif a.cmd == "new":
        folder = os.path.join(PERSONA_DIR, a.id)
        os.makedirs(os.path.join(folder, "data"), exist_ok=True)
        path = os.path.join(folder, "persona.json")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"name": a.name}, f, indent=1)
        print(f"Made {folder}\n  1. put their writings in {os.path.join(folder, 'data')} (.txt, .md or .jsonl)\n"
              f"  2. python persona.py build {a.id} --backend groq\n"
              f"  3. python app.py --backend groq --persona {a.id}")
    elif a.cmd == "build":
        status = key_status(a.backend)
        if status:
            print(status)
        model = a.model or PROVIDERS[a.backend][2]
        path, p = build_profile(a.id, LLM(a.backend, model), batches=a.batches)
        prof = p.profile
        print(f"\nSaved {path}")
        print(f"  {prof['name']}: intensity {prof['default_intensity']}/10, traits {prof['traits']}")
        print(f"  {len(prof['triggers'])} triggers: " + ", ".join(t['when'][:40] for t in prof['triggers']))
        print("  Edit persona.json (same keys) to override anything - it wins over the generated file.")
    elif a.cmd == "search":
        p = load_persona(a.id)
        t = time.time()
        hits = p.passages(a.query, a.k)
        print(f"{len(p.kb)} passages searched in {1000 * (time.time() - t):.0f} ms\n")
        for h in hits:
            where = h["source"] + (f" / {h['heading']}" if h["heading"] else "")
            print(f"[{h['score']:.1f}] {where}\n  {best_sentences(h['text'], a.query, 3)}\n")


if __name__ == "__main__":
    main()
