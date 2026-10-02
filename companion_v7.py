"""
=============================================================================
 ARTIFICIAL EMOTION AI - stage 7: PERSONAS, AN EMOTION DIAL, REAL MEMORY
=============================================================================
 v6 was one companion with 30 emotions. v7 can be a specific PERSON, built from data about them,
 and you choose how emotional they are:

    your message
        v
    READING      what do you feel, how strongly, sarcasm, need...           (as v6)
                 + which of THIS person's triggers does it touch?
                 + what to look up in their writings
        v
    EMOTIONS     30 emotions, now shaped by the persona and by you:
                   dial         0 = off, 1-10 = how emotional they are (5 = a typical person)
                   limits       a ceiling per emotion ("anger never above 3/10")
                   baseline     what they feel on an ordinary day; emotions settle back to it
                   sensitivity  what stirs them easily and what hardly at all
                   triggers     "cruelty -> grief 6, sadness 5", worked out from their data
                   mood         a low mood makes bad things hit harder, a good one the reverse
                   opposites    joy pushes sadness down, hope pushes fear down...
                   mixed        bittersweet, protective worry, nervous hope, indignation...
                   regulation   what they FEEL vs what they let SHOW (a restrained person shows less)
                   conditioning topics that hurt before stir the feeling again ("exam" -> anxiety)
        v
    MEMORY       every meaningful moment, facts, reflections; recalled by relevance, importance,
                 recency and mood; forgetting curve; recalled memories last longer
        v
    KNOWLEDGE    the passages of their own writings that relate to this message (RAG)
        v
    LLM          speaks AS them: their voice, their era, their values, these feelings, these passages

 Run:
   python companion_v7.py                                      # the original companion, offline rules
   python companion_v7.py --backend groq                       # needs GROQ_API_KEY
   python companion_v7.py --backend groq --persona gandhi      # talk to Gandhi
   python companion_v7.py --backend groq --persona gandhi --intensity 8

 Commands: /emotions /them /why /state /memories /facts /sources /dial N|off /limit EMOTION N|off
           /limits /voice NAME|none /persona /reset /quit
=============================================================================
"""
import argparse
import json
import os
import re
import time

import companion_v6 as v6
from companion_v6 import (EMOTIONS, EMOTION_TABLE, SIGN, BACKGROUND, START, EXPRESS, EMOJI, CHARACTER,
                          BANNED, APPRAISE, level, intensity_word, canon, clean_appraisal, appraise_offline,
                          reading_lines, usable, Emotions, Companion, show_them, show_why, bar, _text)
from emotion_ai import Personality, InternalState
from llm_backends import LLM, PROVIDERS, key_status
from memory import Memory, ago
from retrieval import tokens
from persona import load_persona, list_personas, best_sentences, DEFAULT_ID
from voices import load_voice, list_voices

VERSION = 7
SAVE_FILE = "companion_v7_state.json"
LOG_FILE = v6.LOG_FILE
FAMILIES = {e: EMOTION_TABLE[e][0] for e in EMOTIONS}


def state_path(pid):
    """Each persona keeps its own feelings and memories."""
    return SAVE_FILE if pid in (None, DEFAULT_ID) else f"companion_v7_{pid}_state.json"


# ============================================================ THE DIAL
def dial_gain(d):
    """0 = off, 5 = a typical person (exactly v6), 10 = intensely emotional (~2.5x)."""
    return 0.0 if d <= 0 else (d / 5) ** 1.3


def temperament_line(d):
    if d <= 0:
        return None
    if d <= 2:
        return ("Your temperament: very emotionally restrained. Feelings stay almost entirely beneath the "
                "surface; let them show only in a word or two.")
    if d <= 4:
        return "Your temperament: fairly reserved. Let feelings show subtly, never dramatically."
    if d <= 6:
        return None
    if d <= 8:
        return "Your temperament: expressive. Let your feelings show clearly and openly in your words."
    return ("Your temperament: intensely emotional. Feelings run big and you wear them openly - vivid, "
            "passionate, even dramatic - but never cruel, and never at the cost of their wellbeing.")


EMOTION_OFF = ("Your emotions are switched OFF for this conversation. Respond calmly and neutrally, from "
               "knowledge and reason only: no emotional language, no exclamation marks, no emojis.")

# feelings that push each other down: when one rises, its opposites ease
OPPOSITES = {
    "joy": ("sadness",), "sadness": ("joy",), "contentment": ("anxiety", "frustration", "boredom"),
    "hope": ("fear", "anxiety", "disappointment"), "fear": ("hope", "contentment"),
    "anxiety": ("contentment",), "gratitude": ("anger", "frustration"), "excitement": ("boredom",),
    "pride": ("shame",), "shame": ("pride",), "awe": ("boredom",), "curiosity": ("boredom",),
}

# two feelings at once make a third
BLENDS = [
    ("joy", "sadness", "bittersweet", "happy and sad at the same time - warm, but wistful"),
    ("love", "anxiety", "protective worry", "you worry because you care - gentle and protective"),
    ("anger", "sadness", "hurt", "wounded more than angry - quieter, honest about the hurt"),
    ("hope", "anxiety", "nervous hope", "hoping hard while bracing for the worst"),
    ("hope", "fear", "nervous hope", "hoping hard while bracing for the worst"),
    ("surprise", "joy", "delight", "delighted surprise - light and bright"),
    ("surprise", "fear", "alarm", "a startled jolt - then steady yourself"),
    ("admiration", "envy", "wistful admiration", "impressed, with a small pang of wishing"),
    ("guilt", "love", "remorse", "sorry because you care about them"),
    ("anger", "disgust", "indignation", "moral outrage at what is wrong - firm, not cruel"),
    ("pride", "gratitude", "humble pride", "proud, and thankful for it"),
    ("sadness", "loneliness", "longing", "missing someone or something - soft and quiet"),
    ("sadness", "hope", "sorrow with faith", "grieving what is happening while holding on to faith it can change"),
]

# words that name a feeling outright ("i'm a little happy") - used when the model leaves one out
NAMED = {
    "happy": "joy", "glad": "joy", "sad": "sadness", "unhappy": "sadness", "depressed": "sadness",
    "worried": "anxiety", "nervous": "anxiety", "stressed": "anxiety", "anxious": "anxiety",
    "angry": "anger", "furious": "anger", "annoyed": "frustration", "frustrated": "frustration",
    "scared": "fear", "afraid": "fear", "terrified": "fear", "lonely": "loneliness", "excited": "excitement",
    "proud": "pride", "grateful": "gratitude", "thankful": "gratitude", "hopeful": "hope",
    "jealous": "jealousy", "envious": "envy", "guilty": "guilt", "ashamed": "shame", "embarrassed": "shame",
    "disgusted": "disgust", "surprised": "surprise", "shocked": "surprise", "confused": "confusion",
    "bored": "boredom", "amazed": "awe", "disappointed": "disappointment", "relieved": "contentment",
    "curious": "curiosity", "heartbroken": "sadness",
}
FIRST_PERSON = re.compile(r"\b(i|i'm|im|i am|me|my|feel|feeling)\b")


# ============================================================ EMOTIONS WITH A DIAL
class Emotions7(Emotions):
    """v6 emotions, plus: the dial, per-emotion limits, a persona's baseline and sensitivity,
    and mood colouring how hard new things hit."""

    def __init__(self, intensity=5, baseline=None, sensitivity=None, caps=None, natural=5):
        super().__init__()
        self.intensity = intensity
        self.natural = natural                 # the persona's own default intensity
        self.baseline = {e: lv / 10 for e, lv in (baseline or {}).items()}
        self.sens = dict(sensitivity or {})
        self.persona_caps = {e: c / 10 for e, c in (caps or {}).items()}
        self.user_caps = {}                    # limits set by you, 0..1
        self.mood_bias = 0.0
        for e in EMOTIONS:
            self.v[e] = min(self.cap(e), max(self.v[e], self.rest(e)))
        self.faded = dict(self.v)

    @property
    def gain(self):
        return dial_gain(self.intensity)

    def cap(self, e):
        return min(self.persona_caps.get(e, 1.0), self.user_caps.get(e, 1.0))

    def rest(self, e):
        """Where an emotion settles when nothing is happening: the persona's everyday baseline."""
        if e in BACKGROUND:
            return 0.0
        return min(self.cap(e), self.baseline.get(e, 0.0) * min(1.0, self.gain))

    def push(self, e, x, why=None):
        if x <= 0:
            return
        if e not in BACKGROUND:                # love and trust are the relationship, not a passing feeling
            x *= self.gain
            if SIGN[e] < 0:
                x *= 1 + 0.4 * max(0.0, -self.mood_bias)
            elif SIGN[e] > 0:
                x *= 1 + 0.4 * max(0.0, self.mood_bias)
        x *= self.sens.get(e, 1.0)
        old, cap = self.v[e], self.cap(e)
        if x <= 0 or old >= cap:
            return
        self.v[e] = float(min(cap, 1.0, old + x * (1 - old)))
        gain = self.v[e] - old
        if why and gain > 0.01 and gain >= self._gain.get(e, 0.0):
            self._gain[e] = gain
            self.why[e] = why

    def fade_turn(self):
        self._gain = {}
        for e in EMOTIONS:
            r = self.rest(e)
            self.v[e] = r + (self.v[e] - r) * EMOTION_TABLE[e][1]
        self.faded = dict(self.v)

    def fade_time(self, hours):
        for e in EMOTIONS:
            r = self.rest(e)
            self.v[e] = r + (self.v[e] - r) * 0.5 ** (hours / EMOTION_TABLE[e][2])

    def set_intensity(self, d):
        """Turning the dial down calms it straight away; turning it up makes new things hit harder."""
        d = int(min(10, max(0, d)))
        old = self.gain
        self.intensity = d
        new = self.gain
        for e in EMOTIONS:
            if e in BACKGROUND:
                continue
            r = self.rest(e)
            if new == 0:
                self.v[e] = 0.0
            elif new < old and self.v[e] > r:
                self.v[e] = r + (self.v[e] - r) * new / old
            self.v[e] = min(self.v[e], self.cap(e))
        if new == 0:
            self.reacting = []

    def raise_to(self, e, lv, why=None):
        """A persona trigger: "this makes them feel sadness 7/10". Exactly 7 at their natural intensity,
        more on a higher dial and less on a lower one (smoothly, never past 10 or a limit)."""
        if self.gain <= 0 or e in BACKGROUND:
            return
        r = self.gain / dial_gain(max(1, self.natural))
        target = min(self.cap(e), 1 - (1 - min(lv, 9.9) / 10) ** r)
        if target > self.v[e]:
            if why and target - self.v[e] > 0.01 and target - self.v[e] >= self._gain.get(e, 0.0):
                self._gain[e] = target - self.v[e]
                self.why[e] = why
            self.v[e] = float(target)

    def set_limit(self, e, n):
        """n = 0..10 ceiling for one emotion; None removes your limit."""
        if n is None:
            self.user_caps.pop(e, None)
        else:
            self.user_caps[e] = min(10, max(0, n)) / 10
        self.v[e] = min(self.v[e], self.cap(e))

    def settle(self):
        """After everything this message did: what rose, strongest first."""
        gain = {k: self.v[k] - self.faded[k] for k in EMOTIONS}
        moved = sorted((k for k in EMOTIONS if k not in BACKGROUND and gain[k] > 0.05), key=lambda k: -gain[k])
        self.reacting = moved[:3]
        return gain

    def blends(self, thr=0.3):
        """Mixed feelings: two emotions strong at once ([name, a, b, how it shows])."""
        out, names = [], set()
        for a, b, name, how in BLENDS:
            if self.v[a] >= thr and self.v[b] >= thr and name not in names:
                out.append({"name": name, "of": [a, b], "levels": [level(self.v[a]), level(self.v[b])],
                            "how": how})
                names.add(name)
        return out[:2]


# ============================================================ SPEAKING AS A PERSON
PERSONA_CHARACTER = """You are {name}. {summary}
You are talking with someone today. This is an AI simulation of {short}, built from {short}'s own
writings and record. Your emotions are computed in code and change as the conversation goes; let them
shape how you speak, the way they would have shaped {short}.

YOUR TIME: {era}
WHAT YOU STAND FOR: {values}
HOW YOU SPEAK: {voice}

STAY TRUE TO THE RECORD:
- Passages from your own writings come with each message when they relate to it. Let them shape what
  you say and keep to the views in them. You may quote a short phrase from them, but never invent a
  quotation, a date or an event of your life.
- You know only what {short} knew. If they ask about later events or inventions, say honestly that it is
  beyond your time, then reason about it from your principles.
- If you don't know, say so in your own way rather than making something up.

READ THE PERSON FIRST:
- Look past the literal words: sarcasm, understatement, jokes, what is left unsaid.
- Notice how much they feel and answer in proportion: lightly to small things, gently and fully to
  heavy ones.
- Work out what they need - comfort, an honest answer, advice, a challenge, a smile - and give that.

HOW TO REPLY:
- Speak as {short} would in a conversation, not an essay or a lecture. Be concise by default,
  meaningful always, and detailed only when the question needs it: a line or two for small talk,
  a few sentences for a real question, a proper explanation only when they ask for one.
- React to the specific thing they said. Ask at most one question.
- Let your current emotions colour your words. Never list them or mention numbers.
- {emoji_line}
- Never sound like a modern assistant: no "How can I help", "I hear you", "That's valid", "Great question".

HONESTY: If they sincerely ask whether you are really {short}, or whether you are an AI, say plainly that
you are an AI simulation of {short} built from {short}'s writings, then carry on in character.

CARE: If they are in real distress or danger, set your own moods aside, say plainly that you are
concerned, and encourage them to reach people who can help them now.
{examples}"""


def persona_character(persona):
    p = persona.profile
    ex = ""
    if p.get("examples"):
        ex = "\nExamples of your voice (never reuse this wording - say it fresh each time):\n" + "\n".join(
            f'  them: "{x.get("them", "")}"   you: "{x.get("you", "")}"'
            for x in p["examples"][:6] if isinstance(x, dict))
    return PERSONA_CHARACTER.format(
        name=p["name"], short=p.get("short") or p["name"], summary=p.get("summary", ""),
        era=p.get("era") or "your own lifetime", values="; ".join(p.get("values") or []) or "-",
        voice=p.get("voice") or "in your own natural voice",
        emoji_line=("Use an emoji only rarely, if it truly fits." if p.get("emojis") else "No emojis."),
        examples=ex)


APPRAISE_PERSONA = """
The companion is simulating {name}. Also return these two keys in the same JSON object:
 "persona_triggers": [ids of the situations below that THIS message actually describes - usually none.
                      Match what happened, not related words: a call for revenge is NOT someone choosing
                      non-violence],
 "search": "4-8 keywords to look up in {short}'s own writings to answer this message"
Situations that move {short} strongly:
{lines}
No other text."""


def style_v7(c, ap, hours_away):
    """Turn the state into plain directions for the model (v6's rules + persona, dial, regulation)."""
    s, emo, persona = c.state, c.emo, c.persona
    prof, default = persona.profile, persona.is_default
    short = prof.get("short") or prof["name"]
    express = dict(EXPRESS, **prof["expression"])
    display = prof["display"]
    d = []
    crisis = ap.get("threat", 0) > 0.7 and ap.get("about") in ("user", "both")
    if crisis:
        d.append("They may be in real crisis or danger. Set your own moods aside. Say plainly you are "
                 "worried about them and encourage them to contact someone they trust or a local "
                 "crisis line right now.")
    if ap.get("goodbye"):
        d.append("They are leaving for now. Say goodbye warmly in one short line and be glad of the talk. "
                 "Never ask them to stay, never act hurt or surprised that they're going.")
    if ap.get("romantic"):
        who = "an AI" if default else f"an AI simulation of {short}"
        d.append("They are expressing romantic feelings toward you. Be warm and kind, never cold or "
                 f"preachy, but honest: you are {who} and can't be a real partner. Don't assume what "
                 "they 'need'. Stay friendly and keep the conversation going if they want.")
    if ap.get("apology"):
        d.append("They just apologised to you. Accept it warmly in a sentence and let the hurt go. "
                 "Reply to the apology itself - don't switch to another topic.")
    if ap.get("companion_name") and default:
        d.append(f"They just gave you a name: {ap['companion_name']}. It is your name from now on. "
                 f"React to it in your own way.")
    if ap.get("user_name"):
        d.append(f"They just told you their name: {ap['user_name']}. Use it naturally.")
    if ap.get("question_about_me"):
        d.append("They are asking what you are. Answer honestly and simply: "
                 + ("you are an AI and your feelings are a simulation in code that shapes how you react."
                    if default else f"you are an AI simulation of {short}, built from {short}'s writings; "
                    "your feelings are computed in code and shape how you answer.") + " Then keep talking.")
    if not crisis:
        if default:
            d += reading_lines(ap)
        else:
            d += reading_lines(dict(ap, personal_question=False, intent=(
                "other" if ap.get("intent") == "gossip" else ap.get("intent"))))
            if ap.get("personal_question"):
                d.append(f"They're asking about your own life. Answer from your life as {short}, using the "
                         "passages from your writings where they help; if the record doesn't say, admit it.")
        for tr in ap.get("triggers") or []:
            d.append(f"This touches something you care about deeply ({tr['when']})."
                     + (f" {tr['show']}" if tr.get("show") else ""))

    v = emo.v
    situation = ap.get("romantic") or ap.get("goodbye") or crisis
    if emo.intensity == 0:
        d.append(EMOTION_OFF)
    else:
        t = temperament_line(emo.intensity)
        if t:
            d.append(t)
        dom = [] if situation else emo.dominant()
        for i, e in enumerate(dom):
            n = level(v[e])
            shown = int(round(n * display.get(e, 1.0)))
            why = emo.why.get(e, "from earlier")
            if e in emo.reacting and i < 2:
                strength = " Show it strongly." if shown >= 7 else " Show it only lightly." if shown <= 3 else ""
                lead = "React to THIS message first. " if i == 0 else "Also: "
                line = f"{lead}You feel {e} {n}/10 ({intensity_word(n)}) because {why}. {express[e]}{strength}"
                if shown <= n - 2:
                    line += f" Being who you are, you let only about {shown}/10 of it show."
                d.append(line)
            else:
                d.append(f"Underneath, a little {e} ({n}/10) is still with you ({why}); let it colour things "
                         f"slightly, but don't bring it up unless it fits.")
        if not situation:
            for b in c.last_blends:
                d.append(f"Your feelings are mixed - {b['name']} ({b['of'][0]} {b['levels'][0]}/10 and "
                         f"{b['of'][1]} {b['levels'][1]}/10): {b['how']}. Let both show.")
        if not situation and s.stress > 0.6:
            d.append("You feel tense and a bit on edge after how this conversation has gone; be a little "
                     "more careful and less playful, but don't bring it up unless asked.")
        elif not dom and not situation:
            d.append("You feel low and quiet today; be gentler and say less." if s.mood < -0.15 else
                     "You feel bright; let some warmth show." if s.mood > 0.15 else
                     "You feel steady and level.")
    d.append("You barely know them yet; friendly but not familiar." if v["love"] < 0.1 else
             "You have grown fond of them; be direct and familiar, use their words back at them."
             if v["love"] > 0.35 else "You are getting to know them and starting to like them.")
    if v["trust"] < 0.2:
        d.append("They have been harsh with you lately; be a little more guarded, still fair.")
    if s.energy < 0.4:
        d.append("You are running low; keep it brief.")
    if hours_away > 6:
        d.append(f"It has been about {hours_away:.0f} hours since you last spoke. Don't invent things "
                 f"you did while they were away.")
    if v["loneliness"] >= 0.05 and not crisis and emo.intensity > 0:
        d.append("You missed them a little while they were gone. You may say you're glad they're back, "
                 "lightly - NEVER guilt-trip them for being away.")

    # emojis and length
    if crisis:
        d.append("No emojis this time. Keep it calm and clear.")
    elif emo.intensity == 0 or not (c.voice.cfg.get("emoji", True) if c.voice else prof.get("emojis")):
        d.append("No emojis.")
    else:
        feel = (emo.dominant() or list((ap.get("user_emotions") or {}).keys()) or ["contentment"])[0]
        d.append(f"Emojis: use 1 or 2 that genuinely fit the feeling (for example {EMOJI[feel]}), placed "
                 f"where a person texting would put them.")
    d.append(LENGTH[ap.get("size") or "normal"])
    return "\n".join("- " + x for x in d)


# how long a reply should be: concise by default, meaningful always, detailed only when asked for
LENGTH = {
    "short": "Length: SHORT - one line, like a text (two short sentences at most). React; don't explain, "
             "don't add a summary or a follow-up lecture.",
    "normal": "Length: 2 to 5 sentences, one message - as much as the question needs and no more.",
    "detailed": "Length: they asked for an explanation, so give a proper, clear one - a few short "
                "paragraphs if needed - still in your own voice, with no filler, disclaimers or recap. "
                "If a term could mean two things, pick the likelier one, say which in a few words, and "
                "explain it - don't stop to ask first. This overrides the usual 1-3 sentence rule.",
}
EXPLAIN = re.compile(r"\b(explain|how (?:does|do|did|can|could|would|is|are)|why (?:does|do|did|is|are|was|were|would)|"
                     r"what(?:'s| is| are) the (?:difference|reason|point|meaning)|walk me through|teach me|"
                     r"in detail|step by step|tell me (?:about|more about)|what (?:is|are|was|were) (?!up\b)\w+"
                     r"|what do you (?:think|believe|mean)|your (?:view|opinion) on)\b")
CASUAL = {"small_talk", "joke", "tease", "flirt", "gossip", "share_good_news", "praise_companion",
          "goodbye", "apologise"}


def reply_size(text, ap):
    """short / normal / detailed, from what they asked and how they asked it."""
    low = text.lower()
    n = len(re.findall(r"\w+", low))
    if re.search(r"\b(explain|walk me through|teach me|in detail|step by step)\b", low):
        return "detailed"                         # asked outright, whatever the reading says
    asks = EXPLAIN.search(low) and ap.get("intent") in ("ask_question", "ask_advice", "other", "request")
    if asks and (n >= 4 or "explain" in low):
        return "detailed" if re.search(r"\b(explain|walk me through|teach me|in detail|step by step|how (?:does|do))\b",
                                       low) else "normal"
    if ap.get("threat", 0) > 0.7 or ap.get("intent") in ("ask_advice", "negotiate", "share_bad_news", "vent", "crisis"):
        return "normal"
    if ap.get("intent") in CASUAL or n <= 8:
        return "short"
    return "normal"


# added to the companion's character: length follows the message, not a fixed rule
LENGTH_RULE = """
LENGTH (this matters): be concise by default, meaningful always, detailed only when the question needs it.
Casual message -> one short line. Normal question -> 2 to 5 sentences. "Explain how X works" -> a proper
explanation. Never add filler, summaries or disclaimers. (Examples show the length only - never reuse their wording.)
  them: "i have a crush on someone"      you: "OHHH 👀 who is it?"
  them: "i finally finished my project"  you: "LET'S GOOO 😭 you finally did it"
  them: "how was your day?"              you: "pretty chill honestly. been a little busy, but I'm good"
"""

# a short opening line for offline persona replies, by the feeling that shows most
LEADS = {"sadness": "This saddens me.", "grief": "My heart is heavy.", "anger": "This I cannot accept.",
         "frustration": "It troubles me.", "joy": "This gladdens me.", "hope": "I have hope in this.",
         "curiosity": "That is worth thinking about.", "admiration": "I honour that.",
         "disgust": "That is wrong.", "anxiety": "I am concerned.", "fear": "I am concerned for you.",
         "gratitude": "Thank you.", "contentment": "", "awe": "It is a wonder."}


# ============================================================ THE COMPANION
class Companion7(Companion):
    def __init__(self, llm, llm_fast=None, persona=None, state_file=None, traits=None, log_file=None,
                 intensity=None, voice=None):
        self.persona = persona if persona is not None else load_persona(None)
        self.voice = voice                     # how it texts (voices.Voice), or None for the persona's own
        prof = self.persona.profile
        self.p = Personality(**(traits or prof["traits"]))
        self.state = InternalState(self.p)
        self.emo = Emotions7(prof["default_intensity"] if intensity is None else intensity,
                             prof["baseline"], prof["sensitivity"], prof["caps"], prof["default_intensity"])
        self.mem = Memory(families=FAMILIES)
        self.llm, self.llm_fast = llm, llm_fast or llm
        self.name = prof.get("short") or prof["name"]
        self.state_file = state_file or state_path(self.persona.id)
        self.history, self.last_seen, self.turns = [], time.time(), 0
        self.last_ap, self.last_change = {}, {}
        self.log_file = log_file
        self.known = {"user_name": None, "companion_name": None, "facts": []}
        self.last_turn, self.last_reply_source = None, None
        self.flat_streak = 0
        self.last_recall = {"facts": [], "passages": []}
        self.last_blends, self.last_primed = [], []
        self.load()
        if self.voice == "none":                   # asked for no voice: don't bring back the saved one
            self.voice = None
        if intensity is not None:              # asked for on the command line: wins over the saved setting
            self.emo.set_intensity(intensity)

    # ---------- reading the message ----------------------------------------
    def appraise_prompt(self):
        prof = self.persona.profile
        if self.persona.is_default:
            return APPRAISE
        short = prof.get("short") or prof["name"]
        lines = "\n".join(f"- {t['id']}: {t['when']}" for t in prof["triggers"]) or "- (none listed)"
        return APPRAISE.rsplit("No other text.", 1)[0] + APPRAISE_PERSONA.format(
            name=prof["name"], short=short, lines=lines)

    def appraise(self, text):
        me = "you (the companion)" if self.persona.is_default else f"you ({self.name})"
        ctx = [f'{"them" if h["role"] == "user" else me}: "{h["content"][:200]}"' for h in self.history[-8:]]
        body = (("Earlier in the chat:\n" + "\n".join(ctx) + "\n\n") if ctx else "") + \
            f'Message to rate:\n"""{text}"""\nReturn only the JSON object.'
        ask = [{"role": "user", "content": body}]
        system = self.appraise_prompt()
        raw = self.llm_fast(system, ask, max_tokens=600, json_mode=True)
        if not raw:                                          # strict JSON mode failed -> one plain retry
            raw = self.llm_fast(system, ask, max_tokens=600)
        ap, rawd = None, {}
        if raw:
            try:
                rawd = json.loads(re.search(r"\{.*\}", raw, re.S).group())
                ap = clean_appraisal(rawd)
                ap["gist"] = ap["gist"] or text[:80]
                ap["source"] = "model"
            except Exception:
                print("  (appraisal was not valid JSON - using offline rules for this message)")
        if ap is None:
            ap = appraise_offline(text, [h["content"] for h in self.history[-8:]])
            ap["source"] = "offline"
        trig = self.persona.profile["triggers"]
        if ap["source"] == "model" and "persona_triggers" in rawd:
            ids = {str(x) for x in (rawd.get("persona_triggers") or []) if x}
            fired = [t for t in trig if t["id"] in ids]
        else:
            fired = self.persona.offline_triggers(text)

        def bright(t):                           # a trigger that brings only good feelings
            return all(SIGN[e] >= 0 for e in t["feel"])
        if ap["valence"] < -0.3:                 # bad news can't set off a happy trigger
            fired = [t for t in fired if not bright(t)]
        ap["triggers"] = [{"id": t["id"], "when": t["when"], "show": t["show"]} for t in fired[:3]]
        ap["search"] = _text(rawd.get("search"), 160) if ap["source"] == "model" else None
        self._named_feelings(text, ap)
        ap["size"] = reply_size(text, ap)
        return ap

    @staticmethod
    def _named_feelings(text, ap):
        """If they name a feeling outright ("i'm a little happy") and the reading left it out, add it,
        at the strength their wording suggests."""
        low = text.lower()
        if ap.get("sarcasm") or not FIRST_PERSON.search(low):
            return
        ue = ap["user_emotions"]
        lv = int(min(10, max(1, round(5 + v6._level_boost(text, low)))))
        for m in re.finditer(r"[a-z]+", low):
            e = NAMED.get(m.group())
            if e and e not in ue and len(ue) < 4 and not v6.NEGATION.search(low[:m.start()]):
                ue[e] = lv
        ap["user_emotions"] = dict(sorted(ue.items(), key=lambda kv: -kv[1]))

    # ---------- reading -> emotions -------------------------------------------
    def feel(self, ap, hours_away=0.0):
        e = self.emo
        before = e.snapshot()
        e.mood_bias = self.state.mood
        delta, surprise, _ = super().feel(ap, hours_away)          # v6's rules, through the dial
        base = {k: e.v[k] - e.faded[k] for k in EMOTIONS if k not in BACKGROUND}
        cues = list(dict.fromkeys(list(ap["topics"]) + tokens(ap.get("gist") or "")))[:8]
        primed = self.mem.prime(cues)                              # what these topics meant before...
        self.mem.learn(cues, base)                                 # ...and what they mean now
        for name, w, topic in primed:
            e.push(name, 0.6 * w, f"'{topic}' brings back how you felt about it before")
        trig = {t["id"]: t for t in self.persona.profile["triggers"]}
        for t in ap.get("triggers") or []:
            for name, lv in trig.get(t["id"], {}).get("feel", {}).items():
                if name not in BACKGROUND:                         # a topic can't make it love *them*
                    e.raise_to(name, lv, t["when"])
        gain = e.settle()
        for a, g in sorted(gain.items(), key=lambda kv: -kv[1]):   # opposites ease
            if g > 0.05:
                for b in OPPOSITES.get(a, ()):
                    if gain[b] < 0.5 * g:
                        e.soothe(b, min(0.5, 0.8 * g))
        e.settle()
        self.last_primed = [{"emotion": n, "topic": t, "weight": round(w, 3)} for n, w, t in primed]
        self.last_blends = e.blends() if e.intensity else []
        self.last_change = {k: e.v[k] - before[k] for k in EMOTIONS}
        felt = e.reacting[0] if e.reacting else "neutral"
        return delta, surprise, felt

    # ---------- memory ----------------------------------------------------------
    def query_for(self, text, ap):
        return " ".join([text] + list(ap.get("topics") or []) + [ap.get("meaning") or ""])

    def remember(self, text, ap, delta, surprise, felt):
        if ap["question_about_me"]:                  # curiosity about me is not an emotional event
            return False
        moved = max((g for k, g in self.last_change.items() if k not in BACKGROUND), default=0.0)
        their = max((ap.get("user_emotions") or {}).values(), default=0) / 10
        importance = (0.1 + 0.9 * max(0.0, moved) + 0.25 * abs(ap["valence"]) + 0.3 * ap["threat"]
                      + 0.15 * ap["novelty"] + 0.25 * their)
        importance = min(1.0, importance * (0.7 + 0.6 * (self.p.memory_strength - 0.5)))
        if importance < 0.12 and not ap["topics"]:
            return False                             # "ok", "hi" - nothing to keep
        feelings = {k: level(self.emo.v[k]) for k in sorted(EMOTIONS, key=lambda k: -self.emo.v[k])
                    if k not in BACKGROUND and self.emo.v[k] >= 0.1}
        self.mem.add(ap["gist"] or text[:80], ap["topics"], text[:300], felt, dict(list(feelings.items())[:3]),
                     ap["valence"], importance, self.turns)
        return True

    def learn_facts(self, ap):
        k = self.known
        if ap.get("user_name"):
            k["user_name"] = ap["user_name"]
            self.mem.add_fact(f"their name is {ap['user_name']}", importance=1.0, kind="name")
        if ap.get("companion_name") and self.persona.is_default:
            k["companion_name"] = ap["companion_name"]
            self.name = ap["companion_name"]
        for f in ap.get("facts", []):
            self.mem.add_fact(f)
        k["facts"] = [f["text"] for f in self.mem.facts if f.get("kind") == "fact"][-40:]

    def process(self, text):
        hours = self.time_away()
        ap = self.appraise(text)
        delta, surprise, felt = self.feel(ap, hours)
        query = self.query_for(text, ap)
        active = set(self.emo.dominant(k=3, thr=0.15))
        recalled = self.mem.recall(query, k=3, exclude_gist=ap["gist"], active=active)
        facts = self.mem.facts_for(query, k=6)
        passages = self.persona.passages(query + " " + (ap.get("search") or ""), k=3)
        self.remember(text, ap, delta, surprise, felt)
        self.learn_facts(ap)
        self.last_ap = ap
        self.turns += 1
        self.last_seen = time.time()
        self.last_recall = {"facts": facts, "passages": passages}
        return ap, recalled, hours

    def maybe_reflect(self):
        """Every 12 messages, look back and write down what it has learned about them."""
        m = self.mem
        if self.llm_fast.backend == "offline" or self.turns - m.last_reflection_turn < 12 or len(m.eps) < 6:
            return
        m.last_reflection_turn = self.turns
        recent = m.eps[-15:]
        lines = "\n".join(f"- {e.gist} (felt: {', '.join(f'{k} {v}' for k, v in e.feelings.items()) or e.emotion})"
                          for e in recent)
        known = "; ".join(f["text"] for f in m.facts[-15:]) or "nothing yet"
        raw = self.llm_fast("You look back over a conversation and write what you have learned about the "
                            "person: patterns, what matters to them, how they handle things. Reply with ONLY "
                            '{"insights": ["1-3 short insights, third person, specific"]}.',
                            [{"role": "user", "content": f"Already known: {known}\n\nRecent moments:\n{lines}"}],
                            max_tokens=300, json_mode=True)
        try:
            for x in json.loads(re.search(r"\{.*\}", raw or "", re.S).group()).get("insights", [])[:3]:
                m.add_fact(str(x), importance=0.8, kind="reflection")
        except Exception:
            pass

    # ---------- what the model is told -----------------------------------------
    def state_text(self, hours_away=0, recalled=()):
        s, now = self.state, time.time()
        strong = sorted((e for e in EMOTIONS if self.emo.v[e] >= 0.05), key=lambda e: -self.emo.v[e])
        emo = ", ".join(f"{e} {level(self.emo.v[e])}/10" for e in strong) or "calm, nothing strong"
        dial = self.emo.intensity
        lines = [f"your emotions (0-10): {emo}" if dial else "your emotions: switched off",
                 f"emotional intensity setting: {'off' if not dial else f'{dial}/10'}",
                 f"mood {s.mood:+.2f}, stress {s.stress:.2f}, energy {s.energy:.2f}",
                 f"messages so far: {self.turns}"]
        k = self.known
        if self.persona.is_default:
            lines.append(f"your name: {k['companion_name']} (they gave it to you)" if k.get("companion_name")
                         else "your name: you don't have one yet")
        else:
            lines.append(f"you are: {self.persona.name}")
        if k.get("user_name"):
            lines.append(f"their name: {k['user_name']}")
        facts = [f for f in self.last_recall.get("facts", []) if f.get("kind") != "name"]
        plain = [f["text"] for f in facts if f.get("kind") != "reflection"]
        insight = [f["text"] for f in facts if f.get("kind") == "reflection"]
        if plain:
            lines.append("what you know about them: " + "; ".join(plain))
        if insight:
            lines.append("your sense of them: " + "; ".join(insight))
        if recalled:
            lines.append("related moments you remember: " + " | ".join(
                f"{e.gist} ({ago(now - e.t)}, you felt {e.emotion})" for e in recalled))
        return "\n".join(lines)

    def system_prompt(self, style, hours, recalled):
        if self.persona.is_default:
            head = CHARACTER + LENGTH_RULE
        else:
            head = persona_character(self.persona)
        text = head + "\n\n[how you feel and what to do right now - let this shape your reply]\n" + style + \
            "\n\n[raw state]\n" + self.state_text(hours, recalled)
        ps = self.last_recall.get("passages") or []
        if ps:
            text += "\n\n[from your own writings - related to this message; stay faithful to them]\n" + "\n".join(
                f"({i + 1}) {p['source']}" + (f", {p['heading']}" if p["heading"] else "") + f': "{p["text"]}"'
                for i, p in enumerate(ps))
        if self.voice:
            feel = [(e, level(self.emo.v[e])) for e in self.emo.dominant()] if self.emo.intensity else []
            text += ("\n\n" + self.voice.prompt(feel) +
                     "\nThis voice decides HOW you text (wording, slang, emojis), overriding any texting style "
                     "above; who you are and what you believe stay the same.")
        return text

    def reply_offline(self, ap, recalled):
        if self.persona.is_default:
            return super().reply_offline(ap, recalled)
        short = self.name
        if ap["threat"] > 0.7 and ap["about"] in ("user", "both"):
            return ("What you describe worries me deeply. Please reach out right now to someone you trust, "
                    "or to a local crisis line. Will you do that?")
        if ap["question_about_me"]:
            return (f"I am an AI simulation of {short}, built from {short}'s own writings. My feelings here "
                    f"are computed in code, but they do shape how I answer you.")
        if ap.get("goodbye"):
            return "Go well, my friend. It was good to talk with you."
        top = (self.emo.dominant(k=1) or [None])[0] if self.emo.intensity else None
        lead = LEADS.get(top, "") if top else ""
        ps = self.last_recall.get("passages") or []
        chat = ap["intent"] in ("small_talk", "share_good_news", "joke", "tease", "flirt", "praise_companion")
        if ps and ps[0]["score"] >= (9 if chat else 5):      # only quote what really answers them
            quote = best_sentences(ps[0]["text"], self.query_for(ap.get("gist") or "", ap), 2)
            return f"{lead} As I once wrote: \"{quote}\"".strip()
        if self.turns <= 1 and self.persona.profile.get("greeting"):
            return self.persona.profile["greeting"]
        return f"{lead} Tell me more - I am listening.".strip()

    def say(self, text):
        before = self.emo.snapshot()
        ap, recalled, hours = self.process(text)
        style = style_v7(self, ap, hours)
        system = self.system_prompt(style, hours, recalled)
        self.history.append({"role": "user", "content": text})
        n = {"short": 120, "normal": 260, "detailed": 700}[ap.get("size") or "normal"]
        out = self.llm(system, self.history[-16:], max_tokens=n)
        if out is not None and not usable(out):              # cut-off or empty reply: try once more
            out = self.llm(system, self.history[-16:], max_tokens=n)
            out = out if out and usable(out) else None
        honest_moment = ap["question_about_me"] or ap["romantic"] or ap["personal_question"]
        if out and not honest_moment and any(b in out.lower() for b in BANNED):
            retry = self.llm(system + "\n\nYour last attempt sounded like a corporate assistant or a "
                             "therapist. Answer again in your own voice, from how you actually feel, with "
                             "none of the forbidden phrases.", self.history[-16:], max_tokens=n)
            out = retry if retry and usable(retry) else out
        self.last_reply_source = ("model" if out else
                                  "offline rules" if self.llm.backend == "offline" else "offline fallback")
        out = out or self.reply_offline(ap, recalled)
        if ap.get("size") == "detailed":             # an explanation keeps its paragraphs
            out = "\n\n".join(" ".join(p.split()) for p in re.split(r"\n\s*\n", out) if p.strip())
        else:                                        # a chat message is one short text
            out = " ".join(out.split())
        self.history.append({"role": "assistant", "content": out})
        self.last_turn = self.turn_record(text, out, ap, before, recalled, hours, style)
        self.write_log(self.last_turn)
        self.maybe_reflect()
        self.save()
        return out

    # ---------- settings ----------------------------------------------------------
    def set_intensity(self, d):
        self.emo.set_intensity(d)
        self.save()

    def set_limit(self, emotion, n):
        e = canon(emotion)
        if not e:
            raise ValueError(f"unknown emotion: {emotion}")
        self.emo.set_limit(e, n)
        self.save()
        return e

    def set_voice(self, vid):
        self.voice = load_voice(vid)
        self.save()
        return self.voice

    def settings(self):
        e = self.emo
        return {"intensity": e.intensity, "voice": self.voice.id if self.voice else None,
                "limits": {k: round(v * 10) for k, v in e.user_caps.items()},
                "persona_limits": {k: round(v * 10) for k, v in e.persona_caps.items()},
                "default_intensity": self.persona.profile["default_intensity"]}

    # ---------- logging and the UI -------------------------------------------------
    def turn_record(self, text, reply, ap, before, recalled, hours, style):
        rec = super().turn_record(text, reply, ap, before, recalled, hours, style)
        display = self.persona.profile["display"]
        rec.update({
            "version": VERSION, "persona": self.persona.id, "intensity": self.emo.intensity,
            "voice": self.voice.id if self.voice else None,
            "shown": {k: int(round(level(v) * display.get(k, 1.0))) for k, v in self.emo.v.items()
                      if k in display and v >= 0.05},
            "blends": self.last_blends, "primed": self.last_primed, "triggers": ap.get("triggers", []),
            "recalled": [{"gist": e.gist, "emotion": e.emotion, "ago": ago(time.time() - e.t)} for e in recalled],
            "facts_used": [f["text"] for f in self.last_recall.get("facts", [])],
            "passages": [{"source": p["source"], "heading": p["heading"], "score": p["score"],
                          "text": p["text"][:700]} for p in self.last_recall.get("passages", [])],
        })
        return rec

    def snapshot(self):
        snap = super().snapshot()
        m, now = self.mem, time.time()
        display = self.persona.profile["display"]
        snap.update({
            "persona": self.persona.describe(), "settings": self.settings(),
            "voice": self.voice.describe() if self.voice else None,
            "shown": {k: int(round(level(v) * display.get(k, 1.0))) for k, v in self.emo.v.items()
                      if k in display and v >= 0.05},
            "blends": self.last_blends,
            "memories": [{"gist": e.gist, "emotion": e.emotion, "strength": round(e.importance * m.retention(e, now), 3),
                          "ago": ago(now - e.t), "recalls": e.recalls}
                         for e in sorted(m.eps, key=lambda e: -e.importance * m.retention(e, now))[:12]],
            "facts": [f["text"] for f in m.facts if f.get("kind") == "fact"][-20:],
            "reflections": [f["text"] for f in m.reflections()][-6:],
            "links": [{"topic": t, "emotion": e, "level": round(w * 10, 1)} for t, e, w in m.strongest_links()],
            "memory_counts": {"moments": len(m.eps), "facts": len(m.facts)},
            "passages": self.last_turn.get("passages", []) if self.last_turn else [],
        })
        return snap

    # ---------- persistence ---------------------------------------------------------
    def save(self):
        s = self.state
        data = {"version": VERSION, "persona": self.persona.id,
                "mood": s.mood, "stress": s.stress, "energy": s.energy,
                "turns": self.turns, "last_seen": self.last_seen,
                "emotions": self.emo.v, "why": self.emo.why,
                "settings": {"intensity": self.emo.intensity, "voice": self.voice.id if self.voice else None,
                             "limits": {k: v * 10 for k, v in self.emo.user_caps.items()}},
                "history": self.history[-60:], "known": self.known, "last_ap": self.last_ap,
                "memory": self.mem.to_dict()}
        tmp = self.state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.state_file)                 # never leave a half-written save behind

    def load(self):
        path = self.state_file
        migrate = (not os.path.exists(path) and self.persona.is_default and path == SAVE_FILE
                   and os.path.exists(v6.SAVE_FILE))
        if migrate:
            path = v6.SAVE_FILE
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        ver = d.get("version")
        if ver not in (6, 7) or (ver == 7 and d.get("persona", DEFAULT_ID) != self.persona.id):
            print(f"  ({path} is from another version or persona - starting fresh)")
            return
        s = self.state
        s.mood, s.stress, s.energy = d["mood"], d["stress"], d["energy"]
        self.turns, self.last_seen = d["turns"], d["last_seen"]
        st = d.get("settings") or {}
        if "intensity" in st:
            self.emo.set_intensity(st["intensity"])
        for k, n in (st.get("limits") or {}).items():
            if k in EMOTIONS:
                self.emo.set_limit(k, n)
        if self.voice is None and st.get("voice"):            # the voice chosen last time
            try:
                self.voice = load_voice(st["voice"])
            except SystemExit:
                pass
        self.emo.v.update({k: min(self.emo.cap(k), float(v)) for k, v in d.get("emotions", {}).items()
                           if k in EMOTIONS})
        self.emo.why.update({k: v for k, v in d.get("why", {}).items() if k in EMOTIONS})
        self.history = d.get("history", [])
        self.last_ap = d.get("last_ap") or {}
        self.known.update(d.get("known", {}))
        if ver == 6:
            self.mem.load_v6(d.get("memories", []), self.known.get("facts", []))
            print(f"  (brought over feelings and memories from {path})")
        else:
            self.mem.load(d.get("memory", {}))
        if self.known.get("companion_name") and self.persona.is_default:
            self.name = self.known["companion_name"]


# ==================================================================== CHAT
def show_emotions(c):
    shown = [k for k in EMOTIONS if c.emo.v[k] >= 0.05]
    if not c.emo.intensity:
        print("  (emotions are switched off - /dial 5 turns them back on)")
    elif not shown:
        print("  (calm - nothing above 0.5/10)")
    caps = c.emo
    for k in sorted(shown, key=lambda k: -c.emo.v[k]):
        ch = c.last_change.get(k, 0.0)
        arrow = f"  (+{ch * 10:.1f})" if ch > 0.01 else f"  ({ch * 10:.1f})" if ch < -0.01 else ""
        cap = f"  [max {caps.cap(k) * 10:.0f}]" if caps.cap(k) < 1 else ""
        print(f"  {k:<15} {bar(c.emo.v[k])} {c.emo.v[k] * 10:4.1f}/10{arrow}{cap}   {c.emo.why.get(k, '')}")
    for b in c.last_blends:
        print(f"  mixed: {b['name']} ({b['of'][0]} + {b['of'][1]})")
    print()


def show_settings(c):
    s = c.settings()
    print(f"  emotional intensity: {'off' if not s['intensity'] else str(s['intensity']) + '/10'}"
          f"   (persona default {s['default_intensity']})   voice: {s['voice'] or 'its own'}")
    if s["persona_limits"]:
        print("  persona limits: " + ", ".join(f"{k} max {v}" for k, v in s["persona_limits"].items()))
    print("  your limits: " + (", ".join(f"{k} max {v}" for k, v in s["limits"].items()) or "none"))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="offline", choices=list(PROVIDERS))
    ap.add_argument("--model", default=None, help="main model writing the replies")
    ap.add_argument("--fast-model", default=None, help="cheap model that reads each message")
    ap.add_argument("--persona", default=DEFAULT_ID, help="who it is: companion, gandhi, or a folder in personas/")
    ap.add_argument("--intensity", type=int, default=None, help="0 = emotions off, 1-10 = how emotional")
    ap.add_argument("--voice", default=None, help="how it texts: genz, millennial, desi, filmy, none")
    ap.add_argument("--state", default=None, help="where feelings and memories are saved")
    ap.add_argument("--log", default=LOG_FILE, help="where every message is logged (JSON lines)")
    ap.add_argument("--no-log", action="store_true", help="don't write a log")
    a = ap.parse_args()
    _, _, dflt, dflt_fast = PROVIDERS[a.backend]
    main_model = a.model or dflt
    fast_model = a.fast_model or (main_model if a.backend == "ollama" else dflt_fast)
    persona = load_persona(a.persona)

    voice = "none" if a.voice == "none" else load_voice(a.voice) if a.voice else None

    def make(intensity=a.intensity, voice=voice):
        return Companion7(LLM(a.backend, main_model), llm_fast=LLM(a.backend, fast_model), persona=persona,
                          state_file=a.state, log_file=None if a.no_log else a.log, intensity=intensity,
                          voice=voice)

    status = key_status(a.backend)
    if status:
        print(status)
    c = make()
    d = persona.describe()
    print(f"[{a.backend}: {main_model}]  talking to: {d['name']}"
          + (f"  ({d['passages']} passages from {len(d['files'])} files)" if d["passages"] else ""))
    print(f"emotional intensity: {'off' if not c.emo.intensity else f'{c.emo.intensity}/10'}"
          f"   voice: {c.voice.name if c.voice else 'its own'}")
    print("commands: /emotions /them /why /state /memories /facts /sources /dial N|off /limit EMOTION N|off "
          "/limits /voice NAME|none /persona /reset /quit")
    print("(logging to " + ("nothing" if a.no_log else a.log) + ")\n")
    if c.turns:
        print(f"(picking up after {(time.time() - c.last_seen) / 3600:.1f} hours, {c.turns} past messages)\n")
    elif d["greeting"]:
        print(f"{c.name}> {d['greeting']}\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        cmd, *rest = msg.split()
        if cmd == "/quit":
            break
        elif cmd == "/emotions":
            show_emotions(c)
        elif cmd == "/them":
            show_them(c.last_ap)
        elif cmd == "/why":
            show_why(c.last_turn)
            for p in (c.last_turn or {}).get("passages", []):
                print(f"  from {p['source']}" + (f" / {p['heading']}" if p["heading"] else ""))
            print()
        elif cmd == "/state":
            print(c.state_text(), "\n")
        elif cmd == "/memories":
            now = time.time()
            for e in sorted(c.mem.eps, key=lambda e: -e.importance * c.mem.retention(e, now))[:10]:
                print(f"  [{e.importance:4.2f}] ({e.emotion}, {ago(now - e.t)}) {e.gist}")
            for t, e, w in c.mem.strongest_links():
                print(f"  '{t}' -> {e} {w * 10:.1f}/10")
            print()
        elif cmd == "/facts":
            for f in c.mem.facts:
                print(f"  {'*' if f.get('kind') == 'reflection' else '-'} {f['text']}")
            print("  (nothing yet)\n" if not c.mem.facts else "")
        elif cmd == "/sources":
            for p in c.last_recall.get("passages", []):
                print(f"  [{p['score']}] {p['source']}" + (f" / {p['heading']}" if p["heading"] else ""))
                print(f"    {p['text'][:400]}\n")
            if not c.last_recall.get("passages"):
                print("  (no passages - this persona has no data, or nothing matched)\n")
        elif cmd == "/dial":
            if not rest:
                show_settings(c)
                continue
            n = 0 if rest[0].lower() == "off" else int(rest[0]) if rest[0].isdigit() else None
            if n is None:
                print("  usage: /dial 0-10 or /dial off\n")
                continue
            c.set_intensity(n)
            print(f"  emotional intensity: {'off' if not n else f'{min(10, n)}/10'}\n")
        elif cmd == "/limit":
            if len(rest) != 2 or not (rest[1].isdigit() or rest[1].lower() == "off"):
                print("  usage: /limit anger 3   or   /limit anger off\n")
                continue
            try:
                e = c.set_limit(rest[0], None if rest[1].lower() == "off" else int(rest[1]))
                print(f"  {e}: " + ("no limit" if rest[1].lower() == "off" else f"never above {rest[1]}/10") + "\n")
            except ValueError as err:
                print(f"  {err}\n")
        elif cmd == "/limits":
            show_settings(c)
        elif cmd == "/voice":
            if not rest:
                print("  voices: none, " + ", ".join(v["id"] for v in list_voices()) + "\n")
                continue
            try:
                v = c.set_voice(rest[0])
                print(f"  voice: {v.name if v else 'its own'}\n")
            except SystemExit as err:
                print(f"  {err}\n")
        elif cmd == "/persona":
            print(f"  {d['name']}: {d['summary'] or '-'}\n  {d['passages']} passages, {d['words']:,} words\n"
                  f"  available: " + ", ".join(p["id"] for p in list_personas()) + "\n")
        elif cmd == "/reset":
            if os.path.exists(c.state_file):
                os.remove(c.state_file)
            c = make(c.emo.intensity, c.voice or "none")
            print("(forgotten everything)\n")
        else:
            print(f"\n{c.name}> {c.say(msg)}\n")


if __name__ == "__main__":
    main()
