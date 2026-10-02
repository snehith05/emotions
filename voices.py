"""
VOICES: HOW it texts, separate from WHO it is and WHAT it feels.

    persona  = who (Gandhi, the companion, anyone you have data on)
    emotions = what it feels and how strongly (the dial)
    voice    = how those feelings come out in words: Gen Z, millennial, desi Hinglish, filmy...

So "Gandhi, intensity 7, Gen Z voice" is Gandhi's views and feelings, texted the way a Gen Z kid texts.

A voice is a folder:

    voices/<id>/
        voice.json      name, description, rules, emoji habits, intensity words for low/mid/high feelings
        data/*.jsonl    example messages, one per line:
                          {"emotion": "joy", "level": 8, "text": "I'M SCREAMING 😭 no way you got it"}
                        "emotion" and "level" are optional - untagged lines are still used as general style

Every reply, the voice picks the example messages closest to what it feels right now (same emotion, nearest
level) and shows them to the model as style - "this is how you text sadness at 7/10" - never to copy.

Add your own data: any chat or dialogue text you have the right to use (your own chats with consent,
datasets licensed for this, lines you write). More examples per emotion = a more faithful voice.

Commands:
    python voices.py list
    python voices.py show genz sadness 7                    examples it would use
    python voices.py generate genz --n 1000 --backend groq  write more original examples with a model
"""
import argparse
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(HERE, "voices")


class Voice:
    def __init__(self, vid, folder):
        self.id, self.folder = vid, folder
        with open(os.path.join(folder, "voice.json"), encoding="utf-8-sig") as f:
            self.cfg = json.load(f)
        self.examples = []                         # dicts: text, emotion (or None), level (or None)
        data = os.path.join(folder, "data")
        if os.path.isdir(data):
            for name in sorted(os.listdir(data)):
                if name.endswith(".jsonl"):
                    self._load(os.path.join(data, name))

    def _load(self, path):
        from companion_v6 import canon
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(r.get("text") or r.get("reply") or "").strip()
                if not text:
                    continue
                emo = canon(r.get("emotion")) if r.get("emotion") else None
                try:
                    lv = int(r["level"]) if r.get("level") is not None else None
                except (TypeError, ValueError):
                    lv = None
                self.examples.append({"text": text[:300], "emotion": emo, "level": lv})

    @property
    def name(self):
        return self.cfg.get("name", self.id)

    def pick(self, feelings, k=4, seed=None):
        """Examples for what it feels now. feelings = [(emotion, level 0-10), ...] strongest first.
        Same emotion and nearest level first; a little randomness so replies don't converge."""
        rng = random.Random(seed)
        out = []
        for i, (emo, lv) in enumerate(feelings[:2]):
            pool = [x for x in self.examples if x["emotion"] == emo]
            rng.shuffle(pool)
            pool.sort(key=lambda x: abs((x["level"] if x["level"] is not None else 5) - lv))
            out += pool[: (3 if i == 0 else 1)]
        if len(out) < k:                           # calm, or an emotion with no examples: general style
            rest = [x for x in self.examples if x not in out and x["emotion"] in (None, "contentment", "joy")]
            rng.shuffle(rest)
            out += rest[: k - len(out)]
        return out[:k]

    def words_for(self, lv):
        w = self.cfg.get("intensity_words") or {}
        key = "low" if lv <= 3 else "mid" if lv <= 6 else "high"
        return w.get(key) or []

    def prompt(self, feelings):
        """The block that goes into the model's instructions."""
        c = self.cfg
        lines = [f"[your voice: {self.name}] {c.get('description', '')}",
                 "Text in this voice while staying true to who you are, what you believe and what you feel:"]
        lines += [f"- {r}" for r in c.get("rules", [])]
        if feelings:
            emo, lv = feelings[0]
            words = self.words_for(lv)
            if words:
                lines.append(f"- For a feeling at {lv}/10 this voice reaches for words like: {', '.join(words)}")
        ex = self.pick(feelings)
        if ex:
            lines.append("How this voice texts feelings like yours right now (for STYLE only - never copy them, "
                         "don't reuse their topics):")
            lines += [f'  ({x["emotion"] or "general"}{" " + str(x["level"]) + "/10" if x["level"] else ""}) '
                      f'"{x["text"]}"' for x in ex]
        return "\n".join(lines)

    def describe(self):
        tagged = {}
        for x in self.examples:
            if x["emotion"]:
                tagged[x["emotion"]] = tagged.get(x["emotion"], 0) + 1
        return {"id": self.id, "name": self.name, "description": self.cfg.get("description", ""),
                "examples": len(self.examples), "emotions": len(tagged)}


def load_voice(vid):
    if not vid or vid == "none":
        return None
    folder = vid if os.path.isdir(vid) else os.path.join(VOICE_DIR, vid)
    if not os.path.exists(os.path.join(folder, "voice.json")):
        raise SystemExit(f"No voice '{vid}'. Available: " + ", ".join(v["id"] for v in list_voices()))
    return Voice(os.path.basename(os.path.normpath(folder)), folder)


def list_voices():
    out = []
    if os.path.isdir(VOICE_DIR):
        for vid in sorted(os.listdir(VOICE_DIR)):
            path = os.path.join(VOICE_DIR, vid, "voice.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8-sig") as f:
                    out.append({"id": vid, "name": json.load(f).get("name", vid)})
    return out


# ============================================================ MAKING MORE EXAMPLES
GENERATE = """You write ORIGINAL example text messages for a style guide of how people text their feelings.
Voice: {name} - {description}
Rules of the voice:
{rules}
Write {n} different messages from someone feeling {emotion} at {level}/10 ({word}), texting a close friend
about varied everyday situations (study, work, family, friends, love, money, food, travel, games, news).
Make the intensity match {level}/10 exactly: {scale}
Each message 1-2 sentences, natural, varied in situation and wording. Original lines only - do not quote
films, songs or real people. Reply with ONLY a JSON object: {{"messages": ["...", "..."]}}"""

SCALE = "1-2 barely there, 3-4 a bit, 5-6 clearly, 7-8 strongly, 9-10 overwhelming"


def generate(vid, llm, n=1000, per_call=10, seed=0):
    """Fill voices/<id>/data/generated.jsonl with original examples across emotions and levels."""
    from companion_v6 import EMOTIONS, intensity_word
    voice = load_voice(vid)
    path = os.path.join(voice.folder, "data", "generated.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    have = {x["text"].lower() for x in voice.examples}
    grid = [(e, lv) for e in EMOTIONS if e not in ("trust",) for lv in (2, 5, 8)]
    random.Random(seed).shuffle(grid)
    made, i, misses = 0, 0, 0
    while made < n and misses < 8:
        emo, lv = grid[i % len(grid)]
        i += 1
        system = GENERATE.format(name=voice.name, description=voice.cfg.get("description", ""),
                                 rules="\n".join("- " + r for r in voice.cfg.get("rules", [])),
                                 n=per_call, emotion=emo, level=lv, word=intensity_word(lv), scale=SCALE)
        raw = llm(system, [{"role": "user", "content": f"{per_call} messages, {emo} {lv}/10."}],
                  max_tokens=60 * per_call, json_mode=True)
        try:
            msgs = json.loads(re.search(r"\{.*\}", raw or "", re.S).group()).get("messages", [])
        except Exception:
            misses += 1
            continue
        misses = 0
        new = []
        for m in msgs:
            m = " ".join(str(m).split())
            if 3 <= len(m) <= 300 and m.lower() not in have:
                have.add(m.lower())
                new.append({"emotion": emo, "level": lv, "text": m})
        with open(path, "a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        made += len(new)
        print(f"  {made}/{n}  ({emo} {lv}/10: +{len(new)})")
    return made, path


def main():
    from llm_backends import LLM, PROVIDERS, key_status
    ap = argparse.ArgumentParser(description="List, inspect and grow texting voices.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("id")
    s.add_argument("emotion")
    s.add_argument("level", type=int)
    g = sub.add_parser("generate")
    g.add_argument("id")
    g.add_argument("--n", type=int, default=1000)
    g.add_argument("--backend", default="groq", choices=[p for p in PROVIDERS if p != "offline"])
    g.add_argument("--model", default=None)
    a = ap.parse_args()
    if a.cmd == "list":
        for v in list_voices():
            d = load_voice(v["id"]).describe()
            print(f"  {d['id']:<12} {d['name']:<22} {d['examples']} examples over {d['emotions']} emotions")
    elif a.cmd == "show":
        print(load_voice(a.id).prompt([(a.emotion, a.level)]))
    elif a.cmd == "generate":
        status = key_status(a.backend)
        if status:
            print(status)
        made, path = generate(a.id, LLM(a.backend, a.model or PROVIDERS[a.backend][2]), a.n)
        print(f"\nAdded {made} examples to {path}")


if __name__ == "__main__":
    main()
