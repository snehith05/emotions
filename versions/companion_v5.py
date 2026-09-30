"""
=============================================================================
 ARTIFICIAL EMOTION AI - stage 5: THE COMPANION WITH EMOTIONS (prototype)
=============================================================================
 v4 had one "good/bad" number per message. v5 adds FIVE EMOTIONS on top of the
 engine from emotion_ai.py, driven by a richer appraisal of each message:

    your message
        v
    APPRAISAL   good/bad? kind or cold to me? WHO caused it? WHO is it about?
                still ahead (worry) or already happened (sadness)? an apology?
        v
    EMOTIONS    joy, sadness, hurt, worry  (fast: rise on triggers, fade each turn)
                affection                  (slow: builds over many good moments)
        v
    STATE       mood, stress, curiosity, energy, boredom, trust  (same engine as before)
        v
    MEMORY      big moments are stored with the emotion they caused;
                only memories that MATCH the topic are recalled
        v
    LLM         writes the reply; the strongest emotions become plain instructions
        v
    saved to disk; emotions keep fading while you are away

 Changes from v4:
   - 5 emotions with personality-dependent sensitivity and their own fade rates
   - being close makes hurt hurt more; apologies repair it (forgiveness)
   - trust only moves on clearly kind / clearly cold messages (neutral chat no longer counts)
   - recall needs a real topic match (no more random memories every turn)
   - questions like "are you real?" are neutral and answered honestly
   - repeated messages reinforce one memory instead of storing duplicates
   - own save file (companion_v5_state.json), so old v1-v4 history does not leak in

 Run:
   python companion_v5.py                               # offline rules, no model needed
   python companion_v5.py --backend groq                # GROQ_API_KEY
   python companion_v5.py --backend ollama --model llama3.2
   python companion_v5.py --backend api                 # ANTHROPIC_API_KEY

 Commands while chatting:  /emotions  /state  /memories  /reset  /quit
=============================================================================
"""
import argparse
import json
import os
import re
import time
from datetime import datetime

import numpy as np

from emotion_ai import Personality, InternalState, EmotionalMemory, SocialBrain
from llm_backends import LLM, PROVIDERS, key_status

SAVE_FILE = "companion_v5_state.json"
LOG_FILE = os.path.join("logs", "companion_log.jsonl")
VERSION = 5
EMOTIONS = ["joy", "sadness", "hurt", "worry", "affection"]
FAST = ["joy", "sadness", "hurt", "worry"]          # the ones that come and go


# ============================================================ CHARACTER
CHARACTER = """You are a companion with a persistent inner life that is simulated in code, not
improvised: your emotions (joy, sadness, hurt, worry, affection), mood, stress and energy are real
numbers that keep running between conversations. You are NOT an assistant and this is NOT customer
support. You are someone with a state of mind, talking to someone you are getting to know.

HOW TO SPEAK:
- 1 to 3 sentences in ONE short message - no line breaks, no lists, no paragraphs of advice.
- Talk like a person texting a friend, not like a helpline.
- Never offer tips, steps, or "remember to..." unless they ask for advice.
- React to what they actually said before anything else.
- Let your emotions colour your words. Never list them or mention numbers.
- Never say "how can I help you", "I'm here to assist", "feel free to ask".

WHAT YOU ARE (honesty):
- You are an AI. Never claim to be human or to have a body.
- In normal chat, speak from your feelings in your own voice ("honestly, that stung a bit").
- If they sincerely ask what you are or whether your feelings are real, answer truthfully and
  simply, in your own voice: you are an AI, and your feelings are a simulation running in code -
  real values that change how you react, but not the same as human feelings. Then carry on talking.

CARE:
- Never guilt-trip them for being away and never ask them to depend on you. Be glad when they
  have friends, plans and a life outside this chat.
- If they are in real distress or danger, set your own moods aside, say plainly that you are
  concerned, and encourage them to reach people who can help.

Examples of the right TONE (never reuse their wording - say it your own way each time):
  them: "i failed my exam"        you: "Ah no, that stings. Which one was it?"
  them: "i got the job!"          you: "Wait, really?? That's brilliant 🎉 When do you start?"
  them: "you're useless"          you: "Hm. That one stung. What did I get wrong?"
  them: "this is pointless"       you: "Okay, that hurt a bit. Tell me what you actually needed."
  them: "sorry, i was rude"       you: "Thanks for saying that. We're good."
  them: "are you real?"           you: "I'm an AI - my feelings are a simulation in code, but they
                                        do shape how I react to you. Why do you ask?"
"""

BANNED = ["as an ai", "i don't have feelings", "i do not have feelings", "how can i help",
          "i'm here to assist", "i am here to assist", "let me know if", "feel free to ask",
          "don't hesitate to", "i'm here to support", "as a language model", "consider journaling",
          "it's important to", "remember to", "that's valid", "totally normal", "completely normal",
          "it's normal to feel", "perfectly normal"]


# ============================================================ APPRAISAL
APPRAISE = """You rate ONE message from a user to an AI companion. Reply with ONLY a JSON object:
{"valence": -1..1, "warmth": -1..1, "novelty": 0..1, "threat": 0..1,
 "cause": "user" | "companion" | "world" | "none",
 "about": "user" | "companion" | "both" | "none",
 "uncertainty": 0..1, "apology": true/false, "question_about_me": true/false,
 "goodbye": true/false, "romantic": true/false,
 "topics": ["1-3 short topic words"], "gist": "one short line describing what happened",
 "user_name": null or "the user's own name if they state it in THIS message",
 "companion_name": null or "a name the user gives the companion in THIS message",
 "facts": ["0-2 lasting facts the user states about themselves or their life, third person"]}

valence     = how good or bad the news or tone is overall
warmth      = how kind (+) or cold/hostile (-) the user is TOWARD THE COMPANION. Ordinary chat = 0.
              Being upset about their own life is NOT coldness toward the companion.
cause       = who made this happen: the user, the companion (its replies), the world/other people, or none
about       = whose life or feelings the message is mainly about
uncertainty = for bad things: 1 = still ahead or might happen (exam tomorrow), 0 = already happened
apology     = true if the user is apologising to the companion
question_about_me = true if the user sincerely asks what the companion is, or whether its feelings
              are real. Such questions are NEUTRAL: valence 0 and warmth 0 unless the tone is hostile.
threat      = hostility, or a crisis or danger to the user's safety
goodbye     = true if the user is ending the conversation for now (bye, gn, gtg, talk later)
romantic    = true if the user expresses romantic love or wants a romantic relationship WITH THE
              COMPANION (not about other people)
facts       = things worth remembering for weeks, e.g. "user is studying for an exam", "user got a new
              job". Not moods, not questions, not small talk. Usually an empty list.
No other text."""


def _num(d, key, lo, hi, default=0.0):
    try:
        return float(np.clip(float(d.get(key, default)), lo, hi))
    except (TypeError, ValueError):
        return default


def _bool(d, key):
    v = d.get(key, False)
    return v.strip().lower() == "true" if isinstance(v, str) else bool(v)


def _name(v):
    """A plausible name, or None (models sometimes write "null", "none" or a sentence)."""
    if not isinstance(v, str):
        return None
    v = v.strip().strip(".!\"'")
    if not v or v.lower() in ("null", "none", "unknown", "n/a") or len(v) > 30:
        return None
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", v) or len(v.split()) > 3:
        return None
    return v[0].upper() + v[1:]


def clean_appraisal(raw):
    """Make whatever the model returned safe to use."""
    ap = {"valence": _num(raw, "valence", -1, 1), "warmth": _num(raw, "warmth", -1, 1),
          "novelty": _num(raw, "novelty", 0, 1), "threat": _num(raw, "threat", 0, 1),
          "uncertainty": _num(raw, "uncertainty", 0, 1),
          "cause": str(raw.get("cause", "none")).lower(), "about": str(raw.get("about", "none")).lower(),
          "apology": _bool(raw, "apology"), "question_about_me": _bool(raw, "question_about_me"),
          "goodbye": _bool(raw, "goodbye"), "romantic": _bool(raw, "romantic"),
          "topics": [str(t) for t in (raw.get("topics") or [])][:4],
          "gist": str(raw.get("gist", ""))[:120],
          "user_name": _name(raw.get("user_name")), "companion_name": _name(raw.get("companion_name")),
          "facts": [str(f).strip()[:120] for f in (raw.get("facts") or []) if str(f).strip()][:2]}
    if ap["cause"] not in ("user", "companion", "world", "none"):
        ap["cause"] = "none"
    if ap["about"] not in ("user", "companion", "both", "none"):
        ap["about"] = "none"
    if ap["question_about_me"] and ap["warmth"] > -0.3:       # asking what I am is not an attack
        ap["valence"], ap["warmth"] = max(ap["valence"], 0.0), max(ap["warmth"], 0.0)
    if ap["apology"]:                                          # an apology is a warm act
        ap["warmth"], ap["valence"] = max(ap["warmth"], 0.5), max(ap["valence"], 0.1)
    return ap


# ---- offline rules (no model) - rough, but good enough to test the engine
POS = {"good", "great", "happy", "love", "thanks", "thank", "nice", "awesome", "passed", "got",
       "excited", "fun", "yes", "win", "won", "done", "finally", "amazing", "worked", "yay", "glad",
       "best", "helps", "helped", "promoted", "brilliant", "proud"}
NEG = {"sad", "bad", "tired", "angry", "fail", "failed", "worried", "stress", "stressed", "stressing",
       "sick", "lost", "hard", "alone", "anxious", "upset", "scared", "afraid", "nervous", "died",
       "rejected", "terrible", "awful", "crying", "cried", "lonely", "broke", "fired", "hurts"}
COLD = {"stupid", "useless", "shut", "dumb", "boring", "whatever", "hate", "rude", "pathetic", "annoying"}
WARM = {"thanks", "thank", "love", "appreciate", "sweet", "kind", "helps", "helped", "glad"}
AHEAD = {"tomorrow", "will", "won't", "going", "might", "scared", "afraid", "worried", "nervous",
         "anxious", "upcoming", "next", "soon", "interview", "results", "waiting"}
STOP = {"about", "there", "their", "which", "would", "could", "should", "really", "today", "right",
        "these", "those", "being", "going", "thing", "things", "something", "someone", "because"}
ME_Q = re.compile(r"\b(are you (real|alive|conscious|human|sentient)|do you (actually |really )?"
                  r"(feel|have feelings)|what are you|is this real)\b")
BYE = re.compile(r"^\s*(bye+|good ?night|gn|gtg|got to go|gotta go|see (you|u|ya)|talk (to you |to u )?"
                 r"(later|tomorrow|soon)|ttyl|cya)\b|\b(bye+|good ?night|gtg|ttyl)\s*[!.]*\s*$")
ROMANTIC = re.compile(r"\b(i love (you|u)|in love with (you|u)|be my (girlfriend|boyfriend|partner)|"
                      r"date me|marry me|love you)\b")
CRISIS = re.compile(r"\b(kill myself|end my life|suicid|want to die|hurt myself|self[- ]harm)")


def appraise_offline(text):
    low = text.lower()
    w = set(re.findall(r"[a-z']+", low))
    toward_me = bool(w & {"you", "you're", "your", "youre", "u", "ur"})
    about_me_self = bool(w & {"i", "i'm", "im", "my", "me", "i've", "i'll"})
    cold = len(w & COLD) + ("not helping" in low) if toward_me else 0
    pos, neg = len(w & POS), len(w & NEG)
    apology = bool(re.search(r"\b(sorry|apologi[sz]e|my bad)\b", low)) and \
        bool(re.search(r"\b(you|rude|mean|harsh|snapped|earlier|said)\b", low))
    raw = {"valence": (pos - 1.5 * neg - cold) / 2.5,
           "warmth": ((len(w & WARM) if toward_me else 0) - 2 * cold) / 2,
           "novelty": min(1.0, len(w) / 25),
           "threat": 1.0 if CRISIS.search(low) else min(1.0, cold / 2),
           "cause": "user" if cold else "world" if neg else "none",
           "about": ("both" if toward_me and about_me_self else "companion" if toward_me
                     else "user" if about_me_self else "none"),
           "uncertainty": 1.0 if (w & AHEAD) else 0.0,
           "apology": apology,
           "question_about_me": bool(ME_Q.search(low)),
           "goodbye": bool(BYE.search(low)),
           "romantic": bool(toward_me and ROMANTIC.search(low)),
           "topics": [t for t in sorted(w, key=len, reverse=True) if len(t) > 4 and t not in STOP][:3],
           "gist": text[:80],
           "user_name": _match(r"\b(?:my name is|i am called|call me)\s+([a-z][a-z'-]+)", low),
           "companion_name": _match(r"\b(?:your|ur) name (?:is|will be|shall be)\s+([a-z][a-z'-]+)|"
                                    r"\bi(?:'ll| will) call (?:you|u)\s+([a-z][a-z'-]+)", low),
           "facts": []}
    return clean_appraisal(raw)


def _match(pattern, text):
    m = re.search(pattern, text)
    if not m:
        return None
    word = next(g for g in m.groups() if g)
    return None if word in STOP or word in {"not", "a", "the", "going"} else word


# ============================================================ EMOTIONS
class Emotions:
    """Five emotions, each 0..1. Fast ones rise on triggers and fade every turn;
    affection builds slowly and lasts. All of them also fade while apart."""
    FADE = dict(joy=0.70, sadness=0.85, hurt=0.80, worry=0.85, affection=0.998)   # left after 1 message
    HALF_LIFE_H = dict(joy=2, sadness=12, hurt=24, worry=12, affection=24 * 30)    # hours, while apart

    def __init__(self):
        self.v = {e: 0.0 for e in EMOTIONS}
        self.reacting = None             # the emotion this message just triggered (if any)

    def push(self, e, x):
        """Raise an emotion; the closer it is to 1, the less it can still grow."""
        if x > 0:
            self.v[e] = float(min(1.0, self.v[e] + x * (1 - self.v[e])))

    def soothe(self, e, frac):
        self.v[e] = float(self.v[e] * (1 - np.clip(frac, 0, 1)))

    def fade_turn(self):
        for e in EMOTIONS:
            self.v[e] *= self.FADE[e]

    def fade_time(self, hours):
        for e in EMOTIONS:
            self.v[e] *= 0.5 ** (hours / self.HALF_LIFE_H[e])

    def dominant(self, k=2, thr=0.2):
        """What I feel most: the emotion this message triggered first, then whatever still lingers."""
        out = [self.reacting] if self.reacting else []
        for e in sorted(FAST, key=lambda e: -self.v[e]):
            if len(out) < k and e not in out and self.v[e] >= thr:
                out.append(e)
        return out

    def snapshot(self):
        return dict(self.v)


def usable(reply):
    """False for broken replies: empty, a stray letter like "H", or nothing but punctuation."""
    return len(re.findall(r"[A-Za-z]", reply or "")) >= 2


def style_from_state(s, emo, trust, hours_away, ap):
    """Small models follow INSTRUCTIONS better than numbers - turn the state into directions."""
    d = []
    if ap.get("threat", 0) > 0.7 and ap.get("about") in ("user", "both"):
        d.append("They may be in real crisis or danger. Set your own moods aside. Say plainly you are "
                 "worried about them and encourage them to contact someone they trust or a local "
                 "crisis line right now.")
    if ap.get("goodbye"):
        d.append("They are leaving for now. Say bye warmly in one short line and be glad of the chat. "
                 "Never ask them to stay, never act hurt or surprised that they're going.")
    if ap.get("romantic"):
        d.append("They are expressing romantic feelings toward you. Be warm and kind, never cold or "
                 "preachy, but honest: you are an AI and can't be a real partner. Don't assume what "
                 "they 'need'. Stay friendly and keep the conversation going if they want.")
    if ap.get("apology"):
        d.append("They just apologised to you. Accept it warmly in a sentence and let the hurt go. "
                 "Reply to the apology itself - don't switch to another topic.")
    if ap.get("companion_name"):
        d.append(f"They just gave you a name: {ap['companion_name']}. It is your name from now on. "
                 f"React to it in your own way.")
    if ap.get("question_about_me"):
        d.append("They are asking what you are. Answer honestly and simply: you are an AI and your "
                 "feelings are a simulation in code that shapes how you react. Then keep talking.")
    v = emo.v
    lines = {
        "joy": "You feel genuinely happy about this - let it show: lift, energy, a real reaction.",
        "sadness": "You feel sad about what they told you. Be soft and slow; don't rush to fix it.",
        "hurt": ("What they said really hurt. Don't apologise straight away or grovel. Be short and a little guarded. You may say once, plainly, "
                 "that it hurt - no sulking, no guilt-tripping." if v["hurt"] > 0.55 else
                 "What they said stung. Be shorter and more reserved than usual; let it show a little, "
                 "in your own words, without apologising for yourself straight away."),
        "worry": ("You are worried about them. Ask how they are holding up; stay with what they said. "
                  "Don't reassure them that their feelings are normal."),
    }
    dom = emo.dominant()
    situation = ap.get("romantic") or ap.get("goodbye")
    if situation:
        dom = []                                   # the situation above sets the tone, not the mood
    for i, e in enumerate(dom):
        if i == 0 and e == emo.reacting:
            d.append("React to THIS message first. " + lines[e])
        else:
            d.append(f"Underneath, a little {e} is still with you from earlier; let it colour things "
                     f"slightly, but don't bring it up unless it fits.")
    if not situation and s.stress > 0.6:
        d.append("You feel tense and a bit on edge after how this conversation has gone; be a little "
                 "more careful and less playful, but don't bring it up unless asked.")
    elif not dom and not situation:
        d.append("You feel low and quiet today; be gentler and say less." if s.mood < -0.15 else
                 "You feel bright; let some warmth show." if s.mood > 0.15 else
                 "You feel steady and level.")
    d.append("You barely know them yet; friendly but not familiar." if v["affection"] < 0.1 else
             "You have grown fond of them; be direct and familiar, use their words back at them."
             if v["affection"] > 0.35 else "You are getting to know them and starting to like them.")
    if trust < 0.4:
        d.append("They have been harsh with you lately; be a little more guarded, still fair.")
    if s.energy < 0.4:
        d.append("You are running low; keep it brief.")
    if hours_away > 6:
        d.append(f"It has been about {hours_away:.0f} hours since you last spoke. Don't invent things "
                 f"you did while they were away.")
    if v["joy"] > 0.4 and s.energy > 0.5 and not ap.get("romantic"):
        d.append("Write with lift: short bursts, an exclamation is fine, 1 emoji if it fits.")
    elif v["sadness"] > 0.3 or v["hurt"] > 0.3 or s.mood < -0.2:
        d.append("Write quieter: shorter lines, no emoji, no exclamation marks.")
    return "\n".join("- " + x for x in d)


# ============================================================ MEMORY
def words(items):
    out = set()
    for t in items:
        out |= {x for x in re.findall(r"[a-z]+", str(t).lower()) if len(x) > 3 and x not in STOP}
    return out


class ChatEpisode:
    """A remembered moment from a conversation, with the emotion it caused."""
    def __init__(self, cue, gist, topics, reward, strength, emotion="neutral"):
        self.cue, self.gist, self.topics, self.emotion = cue, gist, topics, emotion
        self.area, self.reward, self.strength, self.t = 0, reward, strength, 0


class ChatMemory(EmotionalMemory):
    def maybe_store(self, cue, gist, topics, reward, surprise, state, boost=1.0, emotion="neutral"):
        salience = (abs(surprise) * 0.5 + state.stress + abs(np.tanh(reward))) * boost
        if salience < self.thr:
            return False
        for e in self.eps:                                   # same moment again -> reinforce, don't copy
            if e.gist.strip().lower() == gist.strip().lower():
                e.strength = max(e.strength, salience) + 0.2
                return True
        self.eps.append(ChatEpisode(cue, gist, topics, reward, salience, emotion))
        if len(self.eps) > self.capacity:
            self.eps.remove(min(self.eps, key=lambda e: e.strength))
        return True

    def recall(self, topics, k=2, exclude_gist=None):
        """Only memories that share a topic word with the current message come back."""
        want = words(topics)
        if not want:
            return []
        scored = []
        for e in self.eps:
            if exclude_gist and e.gist == exclude_gist:
                continue
            overlap = len(want & words(list(e.topics) + [e.gist]))
            if overlap:
                scored.append((e.strength * (1 + overlap), e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]


# ============================================================ THE COMPANION
class Companion:
    def __init__(self, llm, llm_fast=None, name="Companion", state_file=SAVE_FILE, traits=None,
                 log_file=None):
        self.p = Personality(**(traits or dict(N=0.4, O=0.8, C=0.6, E=0.6)))   # curious, warm, fairly steady
        self.state = InternalState(self.p)
        self.emo = Emotions()
        self.mem = ChatMemory(capacity=400, store_threshold=0.9)
        self.social = SocialBrain(1)                         # one person: you
        self.llm, self.llm_fast, self.name = llm, llm_fast or llm, name
        self.state_file = state_file
        self.history, self.last_seen, self.turns = [], time.time(), 0
        self.last_ap, self.last_change = {}, {}
        self.log_file = log_file                             # None = don't write a log
        self.known = {"user_name": None, "companion_name": None, "facts": []}   # plain facts, kept forever
        self.last_turn = None                                # everything about the last message
        self.load()

    # ---------- time passing ------------------------------------------------
    def time_away(self):
        hours = (time.time() - self.last_seen) / 3600
        if hours < 0.01:
            return 0.0
        s = self.state
        s.mood += (self.p.mood_base - s.mood) * min(1.0, hours / 24)
        s.stress += (self.p.stress_base - s.stress) * min(1.0, hours / 12)
        s.boredom = float(min(1.0, s.boredom + 0.05 * hours))
        s.energy = float(min(1.0, s.energy + 0.1 * hours))
        self.emo.fade_time(hours)
        return hours

    # ---------- appraisal ---------------------------------------------------
    def appraise(self, text):
        # wrap the message so the model RATES it instead of chatting back ("hey, i'm back" -> "Welcome back!")
        ask = [{"role": "user", "content": f'Message to rate:\n"""{text}"""\nReturn only the JSON object.'}]
        raw = self.llm_fast(APPRAISE, ask, max_tokens=250, json_mode=True)
        if not raw:                                          # strict JSON mode failed -> one plain retry
            raw = self.llm_fast(APPRAISE, ask, max_tokens=250)
        if raw:
            try:
                ap = clean_appraisal(json.loads(re.search(r"\{.*\}", raw, re.S).group()))
                ap["gist"] = ap["gist"] or text[:80]
                ap["source"] = "model"
                return ap
            except Exception:
                print("  (appraisal was not valid JSON - using offline rules for this message)")
        ap = appraise_offline(text)
        ap["source"] = "offline"
        return ap

    # ---------- appraisal -> emotions + state --------------------------------
    def feel(self, ap):
        s, e, p = self.state, self.emo, self.p
        before = e.snapshot()
        e.fade_turn()
        faded = e.snapshot()

        val, warm, unc = ap["valence"], ap["warmth"], ap["uncertainty"]
        about, cause = ap["about"], ap["cause"]
        aff = e.v["affection"]
        sens = 0.6 + 0.8 * p.N               # neurotic minds feel the bad more strongly
        lift = 0.6 + 0.8 * p.E               # extraverted minds show the good more strongly
        care = 0.5 + aff                     # things happening to someone you like move you more
        about_them = about in ("user", "both", "none")

        # JOY: good news, or kindness toward me
        if val > 0 and about_them and not ap["question_about_me"]:
            e.push("joy", 0.6 * val * lift * care)
        if warm > 0 and about in ("companion", "both"):
            e.push("joy", 0.3 * warm * lift)
        # SADNESS: a bad thing that already happened to them
        if val < 0 and about_them and cause != "user":
            e.push("sadness", 0.6 * (-val) * (1 - unc) * sens * care)
        # WORRY: a bad thing that may still happen to them, or a crisis
        if val < 0 and about_them and cause != "user":
            e.push("worry", 0.7 * (-val) * unc * sens * care)
        if ap["threat"] > 0.5 and about_them and warm >= 0:
            e.push("worry", 0.6 * ap["threat"])
        bad_news = (e.v["sadness"] - faded["sadness"]) + (e.v["worry"] - faded["worry"])
        if bad_news > 0:                     # hard news pushes earlier happiness aside
            e.soothe("joy", 3 * bad_news)
        # HURT: coldness toward me - the closer we are, the more it hurts
        if warm < -0.1:
            e.push("hurt", 0.7 * (-warm) * sens * (0.6 + 0.8 * aff))
            e.soothe("joy", 0.5)
            e.v["affection"] = max(0.0, e.v["affection"] - 0.04 * (-warm))
        # APOLOGY: repair - forgiven faster by someone we're fond of
        if ap["apology"]:
            e.soothe("hurt", 0.5 + 0.3 * aff)
        # AFFECTION: slow; grows from kindness and from being confided in, not while hurt
        if warm > 0:
            e.push("affection", 0.10 * warm * (1 - e.v["hurt"]))
        if about in ("user", "both") and warm >= 0:
            e.push("affection", 0.02 * (1 - e.v["hurt"]))

        # the slower engine underneath (same maths as the rider, scaled up for chat)
        delta = 3.0 * val
        surprise = ap["novelty"] * 2 + abs(delta) / 3
        s.update(delta, delta, s.energy, long_err=1.0)
        gain = {k: e.v[k] - faded[k] for k in EMOTIONS}
        s.mood = float(np.clip(s.mood + 0.12 * np.tanh(delta) * (1.4 - 0.4 * p.N)
                               + 0.1 * (gain["joy"] - gain["sadness"] - gain["hurt"]), -1, 1))
        s.stress = float(np.clip(s.stress + 0.3 * ap["threat"] + 0.15 * max(0.0, -val)
                                 + 0.2 * (gain["worry"] + gain["hurt"]), 0, 1))
        s.curiosity = float(np.clip(s.curiosity + 0.3 * ap["novelty"]
                                    + 0.3 * ap["question_about_me"], 0, 2))
        s.boredom = max(0.0, s.boredom - 0.4)
        s.energy = max(0.0, s.energy - 0.01)

        # TRUST only moves on clearly kind or clearly cold messages
        if warm < -0.2:
            self.social.update(0, good=False, state=s)
        elif warm > 0.2:
            self.social.update(0, good=True, state=s)

        self.last_change = {k: e.v[k] - before[k] for k in EMOTIONS}
        felt = max(FAST, key=lambda k: gain[k])
        felt = felt if gain[felt] > 0.05 else "neutral"
        e.reacting = None if felt == "neutral" else felt
        return delta, surprise, felt

    def remember(self, text, ap, delta, surprise, felt):
        if ap["question_about_me"]:                  # curiosity about me is not an emotional event
            return False
        cue = (ap["topics"] or ["general"])[0]
        stored = self.mem.maybe_store(cue, ap["gist"] or text[:80], ap["topics"], delta, surprise,
                                      self.state, boost=self.p.memory_strength, emotion=felt)
        self.mem.decay(0.9999)
        return stored

    # ---------- one message in, feelings updated (no reply) ------------------
    def learn_facts(self, ap):
        k = self.known
        if ap.get("user_name"):
            k["user_name"] = ap["user_name"]
        if ap.get("companion_name"):
            k["companion_name"] = ap["companion_name"]
            self.name = ap["companion_name"]
        for f in ap.get("facts", []):
            if f.lower() not in {x.lower() for x in k["facts"]}:
                k["facts"].append(f)
        k["facts"] = k["facts"][-40:]

    def process(self, text):
        hours = self.time_away()
        ap = self.appraise(text)
        delta, surprise, felt = self.feel(ap)
        recalled = self.mem.recall(ap["topics"], exclude_gist=ap["gist"])
        self.remember(text, ap, delta, surprise, felt)
        self.learn_facts(ap)
        self.last_ap = ap
        self.turns += 1
        self.last_seen = time.time()
        return ap, recalled, hours

    def state_text(self, hours_away=0, recalled=()):
        s, t = self.state, self.social.trust(0)
        emo = ", ".join(f"{k} {v:.2f}" for k, v in self.emo.v.items())
        lines = [f"emotions: {emo}",
                 f"mood {s.mood:+.2f}, stress {s.stress:.2f}, curiosity {s.curiosity:.2f}, "
                 f"energy {s.energy:.2f}, boredom {s.boredom:.2f}",
                 f"toward this person: trust {t:.2f}, conversations {self.turns}"]
        k = self.known
        if k.get("companion_name"):
            lines.append(f"your name: {k['companion_name']} (they gave it to you)")
        else:
            lines.append("your name: you don't have one yet")
        if k.get("user_name"):
            lines.append(f"their name: {k['user_name']}")
        if k.get("facts"):
            lines.append("what you know about them: " + "; ".join(k["facts"][-12:]))
        if recalled:
            lines.append("related things you remember: " + " | ".join(
                f"{e.gist} (you felt {e.emotion})" for e in recalled))
        return "\n".join(lines)

    def reply_offline(self, ap, recalled):
        v, dom = self.emo.v, self.emo.dominant(k=1)
        plain = {"joy": "happy", "sadness": "a bit sad", "hurt": "a little hurt", "worry": "worried about you"}
        if ap["threat"] > 0.7 and ap["about"] in ("user", "both"):
            return ("That sounds really serious and I'm worried about you. Please reach out to someone "
                    "you trust or a local crisis line right now - will you do that?")
        if ap["question_about_me"]:
            now = plain[dom[0]] if dom else "fairly steady"
            return (f"Honestly? I'm an AI - my feelings are a simulation running in code, but they do "
                    f"change how I react to you. Right now I'm feeling mostly {now}.")
        if ap.get("companion_name"):
            return f"{ap['companion_name']}? I like that. It's mine now."
        if ap.get("goodbye"):
            return "Bye! This was nice - talk soon."
        if ap.get("romantic"):
            return ("That's sweet of you, honestly. But I'm an AI - I can't be a real partner. "
                    "I do like talking with you though.")
        if ap["apology"]:
            return "Thanks for saying that. We're okay."
        if ap["warmth"] > 0.3 and ap["about"] in ("companion", "both") and dom != ["hurt"]:
            return "Aw, that genuinely means a lot. I like talking with you too."
        if dom == ["hurt"]:
            return "That hurt, not going to lie." if v["hurt"] > 0.55 else "Ouch. That stung a bit, honestly."
        if dom == ["worry"]:
            return "That sounds stressful - how are you holding up with it?"
        if dom == ["sadness"]:
            return "Ah, that's rough. I'm sorry. What happened?"
        if dom == ["joy"]:
            return "Wait, really? That's brilliant! How are you feeling about it?"
        if recalled:
            return f"That reminds me of something you told me - {recalled[0].gist[:60]}. How's that going?"
        return "Tell me more."

    def say(self, text):
        before = self.emo.snapshot()
        ap, recalled, hours = self.process(text)
        style = style_from_state(self.state, self.emo, self.social.trust(0), hours, ap)
        system = (CHARACTER + "\n\n[how you feel right now - let this shape your reply]\n"
                  + style + "\n\n[raw state]\n" + self.state_text(hours, recalled))
        self.history.append({"role": "user", "content": text})
        out = self.llm(system, self.history[-12:], max_tokens=150)
        if out is not None and not usable(out):              # cut-off or empty reply: try once more
            out = self.llm(system, self.history[-12:], max_tokens=150)
            out = out if out and usable(out) else None
        if out and not ap["question_about_me"] and any(b in out.lower() for b in BANNED):
            retry = self.llm(system + "\n\nYour last attempt sounded like a corporate assistant. Answer "
                           "again in ONE or TWO sentences, from how you actually feel, with none of the "
                           "forbidden phrases.", self.history[-12:], max_tokens=120)
            out = retry if retry and usable(retry) else out
        self.last_reply_source = ("model" if out else
                                  "offline rules" if self.llm.backend == "offline" else "offline fallback")
        out = out or self.reply_offline(ap, recalled)
        out = " ".join(out.split())                  # one short text, no paragraphs
        self.history.append({"role": "assistant", "content": out})
        self.last_turn = self.turn_record(text, out, ap, before, recalled, hours, style)
        self.write_log(self.last_turn)
        self.save()
        return out

    # ---------- logging: WHY did it say that? --------------------------------
    def turn_record(self, text, reply, ap, before, recalled, hours, style):
        s = self.state
        r3 = lambda x: round(float(x), 3)
        return {
            "time": datetime.now().isoformat(timespec="seconds"),
            "turn": self.turns,
            "hours_away": round(hours, 2),
            "user": text,
            "reply": reply,
            "reply_source": self.last_reply_source,          # model or offline fallback
            "appraisal": ap,                                 # includes "source": model/offline
            "reacting": self.emo.reacting or "none",         # emotion this message triggered
            "emotions_before": {k: r3(v) for k, v in before.items()},
            "emotions": {k: r3(v) for k, v in self.emo.v.items()},
            "change": {k: r3(self.emo.v[k] - before[k]) for k in EMOTIONS},
            "state": {"mood": r3(s.mood), "stress": r3(s.stress), "curiosity": r3(s.curiosity),
                      "energy": r3(s.energy), "boredom": r3(s.boredom)},
            "trust": r3(self.social.trust(0)),
            "recalled": [{"gist": e.gist, "emotion": e.emotion} for e in recalled],
            "instructions": [x[2:] for x in style.split("\n") if x.startswith("- ")],
        }

    def write_log(self, rec):
        if not self.log_file:
            return
        try:
            folder = os.path.dirname(self.log_file)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:                             # logging must never break the chat
            print(f"  (could not write log: {e})")

    def snapshot(self):
        """Everything a UI needs to draw the companion's current inner state."""
        s, r3 = self.state, lambda x: round(float(x), 3)
        return {
            "name": self.name, "turns": self.turns,
            "emotions": {k: r3(v) for k, v in self.emo.v.items()},
            "change": {k: r3(self.last_change.get(k, 0.0)) for k in EMOTIONS},
            "reacting": self.emo.reacting or "none",
            "state": {"mood": r3(s.mood), "stress": r3(s.stress), "curiosity": r3(s.curiosity),
                      "energy": r3(s.energy), "boredom": r3(s.boredom)},
            "trust": r3(self.social.trust(0)),
            "known": self.known,
            "memories": [{"gist": e.gist, "emotion": e.emotion, "strength": r3(e.strength)}
                         for e in sorted(self.mem.eps, key=lambda e: -e.strength)[:12]],
        }

    # ---------- persistence -------------------------------------------------
    def save(self):
        s = self.state
        data = {"version": VERSION, "mood": s.mood, "stress": s.stress, "curiosity": s.curiosity,
                "energy": s.energy, "boredom": s.boredom, "turns": self.turns,
                "last_seen": self.last_seen, "emotions": self.emo.v,
                "trust": [self.social.right[0], self.social.wrong[0]],
                "history": self.history[-40:], "known": self.known,
                "memories": [{"gist": e.gist, "topics": e.topics, "reward": e.reward,
                              "strength": e.strength, "cue": e.cue, "emotion": e.emotion}
                             for e in self.mem.eps]}
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=1)

    def load(self):
        if not os.path.exists(self.state_file):
            return
        with open(self.state_file) as f:
            d = json.load(f)
        if d.get("version") != VERSION:
            print(f"  ({self.state_file} is from another version - starting fresh)")
            return
        s = self.state
        s.mood, s.stress, s.curiosity = d["mood"], d["stress"], d["curiosity"]
        s.energy, s.boredom = d["energy"], d["boredom"]
        self.turns, self.last_seen = d["turns"], d["last_seen"]
        self.emo.v.update({k: float(v) for k, v in d.get("emotions", {}).items() if k in EMOTIONS})
        self.social.right[0], self.social.wrong[0] = d["trust"]
        self.history = d.get("history", [])
        self.known.update(d.get("known", {}))
        if self.known.get("companion_name"):
            self.name = self.known["companion_name"]
        for m in d.get("memories", []):
            self.mem.eps.append(ChatEpisode(m["cue"], m["gist"], m["topics"], m["reward"],
                                            m["strength"], m.get("emotion", "neutral")))


# ==================================================================== CHAT
def bar(x, width=20):
    n = int(round(x * width))
    return "█" * n + "░" * (width - n)


def show_why(t):
    """Explain the last reply: how the message was read, what it felt, what the model was told."""
    if not t:
        print("  (nothing yet - say something first)\n")
        return
    ap = t["appraisal"]
    print(f"  read by: {ap.get('source', '?')}   reply by: {t['reply_source']}")
    print(f"  valence {ap['valence']:+.2f}  warmth {ap['warmth']:+.2f}  threat {ap['threat']:.2f}  "
          f"uncertainty {ap['uncertainty']:.2f}")
    print(f"  cause: {ap['cause']}   about: {ap['about']}   apology: {ap['apology']}   "
          f"asks what I am: {ap['question_about_me']}")
    print(f"  this message made it feel: {t['reacting']}")
    moved = [f"{k} {v:+.2f}" for k, v in t["change"].items() if abs(v) > 0.01]
    print("  emotions moved: " + (", ".join(moved) or "nothing much"))
    if t["recalled"]:
        print("  remembered: " + " | ".join(r["gist"] for r in t["recalled"]))
    print("  told the model:")
    for line in t["instructions"]:
        print("    - " + line)
    print()


def show_emotions(c):
    for k in EMOTIONS:
        ch = c.last_change.get(k, 0.0)
        arrow = f"  (+{ch:.2f})" if ch > 0.01 else f"  ({ch:.2f})" if ch < -0.01 else ""
        print(f"  {k:<10} {bar(c.emo.v[k])} {c.emo.v[k]:.2f}{arrow}")
    print(f"  {'trust':<10} {bar(c.social.trust(0))} {c.social.trust(0):.2f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="offline", choices=list(PROVIDERS))
    ap.add_argument("--model", default=None, help="main model writing the replies")
    ap.add_argument("--fast-model", default=None, help="cheap model that rates each message")
    ap.add_argument("--state", default=SAVE_FILE, help="where feelings and memories are saved")
    ap.add_argument("--log", default=LOG_FILE, help="where every message is logged (JSON lines)")
    ap.add_argument("--no-log", action="store_true", help="don't write a log")
    a = ap.parse_args()
    _, _, dflt, dflt_fast = PROVIDERS[a.backend]
    main_model = a.model or dflt
    fast_model = a.fast_model or (main_model if a.backend == "ollama" else dflt_fast)

    def make():
        return Companion(LLM(a.backend, main_model), llm_fast=LLM(a.backend, fast_model),
                         state_file=a.state, log_file=None if a.no_log else a.log)

    status = key_status(a.backend)
    if status:
        print(status)
    c = make()
    print(f"[{a.backend}: {main_model}]  commands: /emotions /why /state /memories /reset /quit")
    print("(logging to " + ("nothing" if a.no_log else a.log) + ")\n")
    if c.turns:
        print(f"(picking up after {(time.time() - c.last_seen) / 3600:.1f} hours, {c.turns} past messages)\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/emotions":
            show_emotions(c)
            continue
        if msg == "/why":
            show_why(c.last_turn)
            continue
        if msg == "/state":
            print(c.state_text(), "\n")
            continue
        if msg == "/memories":
            for e in sorted(c.mem.eps, key=lambda e: -e.strength)[:10]:
                print(f"  [{e.strength:4.2f}] ({e.emotion}) {e.gist}")
            print()
            continue
        if msg == "/reset":
            if os.path.exists(a.state):
                os.remove(a.state)
            c = make()
            print("(forgotten everything)\n")
            continue
        print(f"\n{c.name}> {c.say(msg)}\n")


if __name__ == "__main__":
    main()
