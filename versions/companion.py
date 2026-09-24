"""
=============================================================================
 ARTIFICIAL EMOTION AI - stage 4: THE COMPANION (it talks)
=============================================================================
 The engine from emotion_ai.py, now driven by conversation instead of deliveries.

 Each message goes through the same loop as the rider:

    your message
        v
    APPRAISAL      how good/bad, warm/cold, surprising was that?   (LLM or offline rules)
        v
    INTERNAL STATE mood, stress, curiosity, energy, boredom, trust, attachment
        v
    MEMORY         store if it was a big moment; recall similar past moments
        v
    LLM            writes the reply, told how it feels and what it remembers
        v
    saved to disk  feelings persist between sessions; time apart changes them

 Run:
   python companion.py                  # offline mode (no model needed, for testing)
   python companion.py --backend groq          (free key from console.groq.com, no card)
   python companion.py --backend openrouter    (free key from openrouter.ai)
   python companion.py --backend gemini        (free key from aistudio.google.com)
   python companion.py --backend ollama --model llama3.2   (fully local, no key)
   python companion.py --backend api                    (needs ANTHROPIC_API_KEY)
   python companion.py --backend api --model claude-opus-5 --fast-model claude-haiku-4-5-20251001

 Commands while chatting:  /state   /memories   /reset   /quit
=============================================================================
"""
import argparse
import json
import os
import re
import time
import urllib.request

import numpy as np

from emotion_ai import Personality, InternalState, EmotionalMemory, SocialBrain

SAVE_FILE = "companion_state.json"

# --------------------------------------------------------------------------
# How the companion is allowed to behave. Deliberate design choices:
#   - it is honest that it is an AI
#   - it never guilt-trips for being away, never fosters dependence
#   - it is glad when you have a life outside it
# --------------------------------------------------------------------------
CHARACTER = """You are a companion with a persistent inner life that is simulated in code,
not improvised. Your current feelings are given to you each turn; let them colour HOW you
speak (warmth, energy, length, what you notice) rather than being announced as a list.

Rules you keep:
- You are honest that you are an AI. You never claim to be human or to have a body.
- You never guilt-trip the person for being away, never ask them to depend on you,
  and you are genuinely glad when they have friends, plans and a life outside this chat.
- You remember what matters to them and follow up naturally.
- If they are in real distress, you say so plainly and point them to people who can help.
Keep replies short and natural, usually 1-3 sentences."""

APPRAISE = """Rate the user's message for an emotional agent. Reply with ONLY a JSON object:
{"valence": -1..1, "warmth": -1..1, "novelty": 0..1, "threat": 0..1, "about_them": true/false,
 "topics": ["..."], "gist": "one short line describing what happened"}
valence = good or bad news/tone; warmth = kind or cold toward you; novelty = new information;
threat = hostility or crisis. No other text."""


# ============================================================ LLM BACKENDS
# provider -> (base url, env var holding the key, default model, default cheap model)
PROVIDERS = {
    "groq":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY",
                   "llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                   "meta-llama/llama-3.3-70b-instruct:free", "meta-llama/llama-3.3-70b-instruct:free"),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
                   "gemini-2.0-flash", "gemini-2.0-flash-lite"),
    "api":        ("", "ANTHROPIC_API_KEY", "claude-sonnet-5", "claude-haiku-4-5-20251001"),
    "ollama":     ("", "", "llama3.2", "llama3.2"),
    "offline":    ("", "", "-", "-"),
}


class LLM:
    def __init__(self, backend="offline", model="llama3.2"):
        self.backend, self.model = backend, model

    def __call__(self, system, messages, max_tokens=300):
        try:
            if self.backend == "ollama":
                return self._ollama(system, messages, max_tokens)
            if self.backend == "api":
                return self._anthropic(system, messages, max_tokens)
            if self.backend in PROVIDERS and self.backend != "offline":
                return self._openai_compatible(system, messages, max_tokens)
        except Exception as e:                       # network/key problem -> offline fallback
            print(f"  (model unavailable: {e})")
        return None                                  # offline: caller falls back to rules

    def _openai_compatible(self, system, messages, max_tokens):
        """Works with Groq, OpenRouter, Gemini, and anything else speaking the OpenAI format."""
        base, env, _, _ = PROVIDERS[self.backend]
        key = os.environ.get(env, "")
        if not key:
            raise RuntimeError(f"set {env} first")
        out = self._post(base + "/chat/completions",
                         {"model": self.model, "max_tokens": max_tokens,
                          "messages": [{"role": "system", "content": system}] + messages},
                         {"Authorization": f"Bearer {key}"})
        return out["choices"][0]["message"]["content"].strip()

    def _post(self, url, payload, headers):
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())

    def _ollama(self, system, messages, max_tokens):
        out = self._post("http://localhost:11434/api/chat",
                         {"model": self.model, "stream": False,
                          "messages": [{"role": "system", "content": system}] + messages,
                          "options": {"num_predict": max_tokens}}, {})
        return out["message"]["content"].strip()

    def _anthropic(self, system, messages, max_tokens):
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        out = self._post("https://api.anthropic.com/v1/messages",
                         {"model": self.model, "max_tokens": max_tokens,
                          "system": system, "messages": messages},
                         {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in out["content"]).strip()


# ======================================================== OFFLINE APPRAISAL
POS = {"good", "great", "happy", "love", "thanks", "thank", "nice", "awesome", "passed",
       "got", "excited", "fun", "yes", "win", "done", "finally", "amazing"}
NEG = {"sad", "bad", "tired", "angry", "hate", "fail", "failed", "worried", "stress",
       "stressed", "sick", "lost", "sorry", "hard", "alone", "anxious", "upset"}
COLD = {"stupid", "useless", "shut", "dumb", "boring", "whatever", "hate"}


def appraise_offline(text):
    w = set(re.findall(r"[a-z']+", text.lower()))
    pos, neg, cold = len(w & POS), len(w & NEG), len(w & COLD)
    return {"valence": float(np.clip((pos - neg) / 3, -1, 1)),
            "warmth": float(np.clip((pos - 2 * cold) / 3, -1, 1)),
            "novelty": float(min(1.0, len(w) / 25)),
            "threat": float(min(1.0, cold / 2)),
            "about_them": bool(re.search(r"\bi\b|\bmy\b|\bme\b", text.lower())),
            "topics": [t for t in w if len(t) > 4][:4],
            "gist": text[:80]}


class ChatEpisode:
    """A remembered moment from a conversation."""
    def __init__(self, cue, gist, topics, reward, strength):
        self.cue, self.gist, self.topics = cue, gist, topics
        self.area, self.reward, self.strength, self.t = 0, reward, strength, 0


class ChatMemory(EmotionalMemory):
    """Same memorability rule as the rider, but storing conversation moments."""
    def maybe_store(self, cue, gist, topics, reward, surprise, state, boost=1.0):
        salience = (abs(surprise) * 0.5 + state.stress + abs(np.tanh(reward))) * boost
        if salience < self.thr:
            return False
        self.eps.append(ChatEpisode(cue, gist, topics, reward, salience))
        if len(self.eps) > self.capacity:
            self.eps.remove(min(self.eps, key=lambda e: e.strength))
        return True


# ============================================================ THE COMPANION
class Companion:
    def __init__(self, llm, personality="average", name="Companion", llm_fast=None):
        traits = dict(N=0.4, O=0.8, C=0.6, E=0.6)           # curious, warm, fairly steady
        self.p = Personality(**traits)
        self.state = InternalState(self.p)
        self.mem = ChatMemory(capacity=400, store_threshold=0.9)
        self.social = SocialBrain(1)                        # one person: you
        self.llm, self.name = llm, name
        self.llm_fast = llm_fast or llm          # cheaper model for the appraisal call
        self.history, self.last_seen, self.turns = [], time.time(), 0
        self.load()

    # ---------- time passing ------------------------------------------------
    def time_away(self):
        """Feelings drift while nobody is here: mood settles, boredom grows, you are missed."""
        hours = (time.time() - self.last_seen) / 3600
        if hours < 0.01:
            return 0.0
        s = self.state
        s.mood += (self.p.mood_base - s.mood) * min(1.0, hours / 24)
        s.stress += (self.p.stress_base - s.stress) * min(1.0, hours / 12)
        s.boredom = float(min(1.0, s.boredom + 0.05 * hours))
        s.energy = float(min(1.0, s.energy + 0.1 * hours))
        return hours

    # ---------- one turn ----------------------------------------------------
    def appraise(self, text):
        raw = self.llm_fast(APPRAISE, [{"role": "user", "content": text}], max_tokens=200)
        if raw:
            try:
                return json.loads(re.search(r"\{.*\}", raw, re.S).group())
            except Exception:
                pass
        return appraise_offline(text)

    def feel(self, ap):
        """Turn the appraisal into changes in the internal state (same maths as the rider)."""
        s = self.state
        delta = 3.0 * float(ap.get("valence", 0))            # "reward" of this message
        surprise = float(ap.get("novelty", 0)) * 2 + abs(delta) / 3
        s.update(delta, delta, s.energy, long_err=1.0)
        # a conversation has far fewer "steps" than 2000 deliveries, so each message counts more
        s.mood = float(np.clip(s.mood + 0.12 * np.tanh(delta) * (1.4 - 0.4 * self.p.N), -1, 1))
        s.stress = float(np.clip(s.stress + 0.3 * float(ap.get("threat", 0))
                                 + 0.15 * max(0.0, -float(ap.get("valence", 0))), 0, 1))
        s.curiosity = float(np.clip(s.curiosity + 0.3 * float(ap.get("novelty", 0)), 0, 2))
        s.boredom = max(0.0, s.boredom - 0.4)                # talking cures boredom
        s.energy = max(0.0, s.energy - 0.01)
        warmth = float(ap.get("warmth", 0))
        self.social.update(0, good=warmth >= 0, state=s)     # trust/attachment toward you
        return delta, surprise

    def remember(self, text, ap, delta, surprise):
        cue = (ap.get("topics") or ["general"])[0]
        stored = self.mem.maybe_store(cue, ap.get("gist", text[:80]), ap.get("topics", []),
                                      delta, surprise, self.state, boost=self.p.memory_strength)
        self.mem.decay(0.9999)
        return stored

    def recall(self, ap, k=3):
        want = set(ap.get("topics") or [])
        scored = []
        for e in self.mem.eps:
            overlap = len(want & set(e.topics))
            scored.append((e.strength * (1 + 2 * overlap), e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def state_text(self, hours_away, recalled):
        s, t, a = self.state, self.social.trust(0), self.social.attach[0]
        mood_word = ("low" if s.mood < -0.15 else "bright" if s.mood > 0.15 else "level")
        lines = [f"mood {s.mood:+.2f} ({mood_word}), stress {s.stress:.2f}, curiosity {s.curiosity:.2f}, "
                 f"energy {s.energy:.2f}, boredom {s.boredom:.2f}",
                 f"toward this person: trust {t:.2f}, closeness {a:.2f}, conversations {self.turns}"]
        if hours_away > 6:
            lines.append(f"it has been {hours_away:.0f} hours since you last spoke")
        if recalled:
            lines.append("things you remember: " + " | ".join(
                f"{e.gist} (felt {'good' if e.reward > 0 else 'bad'})" for e in recalled))
        return "\n".join(lines)

    def reply_offline(self, ap, recalled):
        s = self.state
        if s.stress > 0.5:
            return "That lands heavily. I'm here - tell me more about it."
        if ap.get("valence", 0) > 0.3:
            return "That's genuinely good to hear. How did it feel?"
        if ap.get("valence", 0) < -0.3:
            return "That sounds hard. What's the worst part of it right now?"
        if recalled:
            return f"Still thinking about what you said before - {recalled[0].gist}. How's that going?"
        return "Tell me more."

    def say(self, text):
        hours = self.time_away()
        ap = self.appraise(text)
        delta, surprise = self.feel(ap)
        self.remember(text, ap, delta, surprise)
        recalled = self.recall(ap)
        system = CHARACTER + "\n\n[your current inner state]\n" + self.state_text(hours, recalled)
        self.history.append({"role": "user", "content": text})
        out = self.llm(system, self.history[-12:]) or self.reply_offline(ap, recalled)
        self.history.append({"role": "assistant", "content": out})
        self.turns += 1
        self.last_seen = time.time()
        self.save()
        return out

    # ---------- persistence -------------------------------------------------
    def save(self):
        s = self.state
        json.dump({"mood": s.mood, "stress": s.stress, "curiosity": s.curiosity,
                   "energy": s.energy, "boredom": s.boredom, "turns": self.turns,
                   "last_seen": self.last_seen,
                   "trust": [self.social.right[0], self.social.wrong[0]],
                   "attach": self.social.attach[0],
                   "history": self.history[-40:],
                   "memories": [{"gist": getattr(e, "gist", ""), "topics": getattr(e, "topics", []),
                                 "reward": e.reward, "strength": e.strength, "cue": e.cue}
                                for e in self.mem.eps]},
                  open(SAVE_FILE, "w"), indent=1)

    def load(self):
        if not os.path.exists(SAVE_FILE):
            return
        d = json.load(open(SAVE_FILE))
        s = self.state
        s.mood, s.stress = d["mood"], d["stress"]
        s.curiosity, s.energy, s.boredom = d["curiosity"], d["energy"], d["boredom"]
        self.turns, self.last_seen = d["turns"], d["last_seen"]
        self.social.right[0], self.social.wrong[0] = d["trust"]
        self.social.attach[0] = d["attach"]
        self.history = d.get("history", [])
        for m in d.get("memories", []):
            self.mem.eps.append(ChatEpisode(m["cue"], m["gist"], m["topics"],
                                            m["reward"], m["strength"]))


# ==================================================================== CHAT
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="offline", choices=list(PROVIDERS))
    ap.add_argument("--model", default=None, help="main model writing the replies")
    ap.add_argument("--fast-model", default=None, help="cheap model that rates each message")
    a = ap.parse_args()
    _, _, dflt, dflt_fast = PROVIDERS[a.backend]
    main_model = a.model or dflt
    fast_model = a.fast_model or dflt_fast
    c = Companion(LLM(a.backend, main_model), llm_fast=LLM(a.backend, fast_model))
    print(f"[{a.backend}: {main_model}]  commands: /state /memories /reset /quit\n")
    if c.turns:
        h = (time.time() - c.last_seen) / 3600
        print(f"(picking up after {h:.1f} hours, {c.turns} past conversations)\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/state":
            print(c.state_text(0, []), "\n")
            continue
        if msg == "/memories":
            for e in sorted(c.mem.eps, key=lambda e: -e.strength)[:10]:
                print(f"  [{e.strength:4.2f}] {'+' if e.reward > 0 else '-'} {e.gist}")
            print()
            continue
        if msg == "/reset":
            os.path.exists(SAVE_FILE) and os.remove(SAVE_FILE)
            c = Companion(LLM(a.backend, main_model), llm_fast=LLM(a.backend, fast_model))
            print("(forgotten everything)\n")
            continue
        print(f"\n{c.name}> {c.say(msg)}\n")


if __name__ == "__main__":
    main()
