"""
MEMORY (v7) - how the companion remembers, modelled on how people do.

    episodes      every meaningful moment: what was said, what it felt (top emotions with levels),
                  how good/bad it was, how important. Stored every turn, not only the dramatic ones.
    facts         lasting things about the person ("studying for a big exam"), merged when repeated
    reflections   every so often the model looks back over recent moments and writes 1-3 insights
                  ("they downplay bad news with jokes") - kept as important facts
    associations  emotional conditioning: if "exam" came with anxiety before, the word alone now
                  stirs a little anxiety again (and the link fades if the topic stops hurting)

Recall ranks memories like the human mind does:
    relevance       BM25 search on the message, its topics and its meaning (only related things return)
  + importance      strong, emotional, surprising moments are kept and found more easily
  + recency         recent moments come back more easily
  + mood match      when sad, sad memories surface more easily (mood-congruent recall)
Forgetting follows a curve: weak memories fade within days, important ones last for months, and every
time a memory is recalled it fades more slowly (the spacing effect). The weakest go first when full.
"""
import math
import time

from retrieval import BM25, tokens


class Episode:
    def __init__(self, gist, topics, user="", emotion="neutral", feelings=None, valence=0.0,
                 importance=0.3, t=None, turn=0, recalls=0, last_recalled=None, cue=None):
        self.gist, self.topics, self.user = gist, list(topics or []), user
        self.emotion = emotion                      # the main emotion this moment caused
        self.feelings = dict(feelings or {})        # {emotion: level 0-10} at the time
        self.valence, self.importance = float(valence), float(importance)
        self.t = t or time.time()
        self.turn, self.recalls, self.last_recalled = turn, recalls, last_recalled
        self.cue = cue or (self.topics[0] if self.topics else "general")
        self.tok = tokens(" ".join([gist, " ".join(self.topics), user]))

    # names used by v6 code and tests
    @property
    def reward(self):
        return self.valence

    @property
    def strength(self):
        return self.importance

    def to_dict(self):
        return {"gist": self.gist, "topics": self.topics, "user": self.user, "emotion": self.emotion,
                "feelings": self.feelings, "valence": round(self.valence, 3),
                "importance": round(self.importance, 3), "t": self.t, "turn": self.turn,
                "recalls": self.recalls, "last_recalled": self.last_recalled, "cue": self.cue}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("gist", ""), d.get("topics", []), d.get("user", ""), d.get("emotion", "neutral"),
                   d.get("feelings"), d.get("valence", 0.0), d.get("importance", 0.3), d.get("t"),
                   d.get("turn", 0), d.get("recalls", 0), d.get("last_recalled"), d.get("cue"))


def ago(seconds):
    """3 minutes ago, 5 hours ago, 2 days ago - for the model and the UI."""
    m = seconds / 60
    if m < 2:
        return "just now"
    if m < 60:
        return f"{m:.0f} minutes ago"
    if m < 60 * 36:
        return f"{m / 60:.0f} hours ago"
    return f"{m / 1440:.0f} days ago"


class Memory:
    def __init__(self, families=None, capacity=1500, max_facts=200):
        self.eps, self.facts, self.assoc = [], [], {}
        self.families = families or {}               # emotion -> family, for mood-congruent recall
        self.capacity, self.max_facts = capacity, max_facts
        self.last_reflection_turn = 0

    # ---------- episodes ------------------------------------------------------
    def retention(self, ep, now=None):
        """The forgetting curve. Half-life: ~1 week for a dull moment, ~3 weeks for an important one,
        longer still each time it's recalled."""
        days = ((now or time.time()) - ep.t) / 86400
        half = 7 * (1 + 2 * ep.importance) * (1 + ep.recalls)
        return 0.5 ** (days / half)

    def add(self, gist, topics, user="", emotion="neutral", feelings=None, valence=0.0, importance=0.3,
            turn=0, now=None):
        now = now or time.time()
        g = gist.strip().lower()
        for ep in self.eps[-30:]:                     # the same moment again -> it matters more
            if ep.gist.strip().lower() == g:
                ep.importance = min(1.0, max(ep.importance, importance) + 0.1)
                ep.t, ep.turn = now, turn
                return ep
        ep = Episode(gist, topics, user, emotion, feelings, valence, importance, now, turn)
        self.eps.append(ep)
        if len(self.eps) > self.capacity:             # forget what has faded most
            self.eps.remove(min(self.eps, key=lambda e: e.importance * self.retention(e, now)))
        return ep

    def recall(self, query, k=3, exclude_gist=None, active=(), now=None, rehearse=True):
        """Memories related to `query` (text or a list of topic words), best first."""
        now = now or time.time()
        q = tokens(query if isinstance(query, str) else " ".join(map(str, query or [])))
        pool = [e for e in self.eps if not (exclude_gist and e.gist == exclude_gist)]
        if not q or not pool:
            return []
        index = BM25()
        for e in pool:
            index.add(e.tok)
        hits = index.search(q, k=50)
        if not hits:
            return []
        top = hits[0][1]
        active = set(active or ())
        moods = {self.families.get(a) for a in active}
        scored = []
        for i, s in hits:
            rel = s / top
            if rel < 0.25:                             # a faint, accidental match
                continue
            e = pool[i]
            recency = 0.5 ** ((now - e.t) / 3600 / 72)
            congruent = 1.0 if e.emotion in active else 0.5 if self.families.get(e.emotion) in moods else 0.0
            score = rel + 0.6 * e.importance * self.retention(e, now) + 0.3 * recency + 0.25 * congruent
            scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        out = [e for _, e in scored[:k]]
        if rehearse:                                   # remembering something makes it last
            for e in out:
                e.recalls += 1
                e.last_recalled = now
        return out

    # ---------- facts and reflections ----------------------------------------
    def add_fact(self, text, importance=0.6, kind="fact", now=None):
        text = " ".join(str(text).split())[:200]
        if not text:
            return None
        new = set(tokens(text))
        for f in self.facts:
            old = set(tokens(f["text"]))
            if new and old and len(new & old) / len(new | old) >= 0.6:    # same fact again: keep the newer wording
                f.update(text=text, t=now or time.time(), mentions=f.get("mentions", 1) + 1,
                         importance=max(f["importance"], importance))
                return f
        f = {"text": text, "importance": importance, "kind": kind, "t": now or time.time(), "mentions": 1}
        self.facts.append(f)
        if len(self.facts) > self.max_facts:
            self.facts.remove(min(self.facts, key=lambda f: (f["importance"], f["t"])))
        return f

    def facts_for(self, query, k=6):
        """Facts related to the message, topped up with the most important ones (always worth knowing)."""
        if not self.facts:
            return []
        index = BM25()
        for f in self.facts:
            index.add(tokens(f["text"]))
        q = tokens(query if isinstance(query, str) else " ".join(map(str, query or [])))
        out = [self.facts[i] for i, _ in index.search(q, k=max(1, k - 2))] if q else []
        for f in sorted(self.facts, key=lambda f: (-f["importance"], -f["t"])):
            if len(out) >= k:
                break
            if f not in out:
                out.append(f)
        return out

    def reflections(self):
        return [f for f in self.facts if f.get("kind") == "reflection"]

    # ---------- emotional associations (conditioning) ------------------------
    def prime(self, topics, limit=2, threshold=0.05):
        """Emotions these topics have come with before: [(emotion, weight, topic word)]."""
        out = {}
        for topic in topics or []:
            for t in tokens(topic):
                for emo, w in self.assoc.get(t, {}).get("emo", {}).items():
                    if w >= threshold and w > out.get(emo, (0,))[0]:
                        out[emo] = (w, topic)
        ranked = sorted(out.items(), key=lambda kv: -kv[1][0])[:limit]
        return [(emo, w, topic) for emo, (w, topic) in ranked]

    def learn(self, topics, gains):
        """After a message: each topic's emotional link moves toward what was felt this time.
        A topic that keeps coming up without the feeling slowly loses it (extinction)."""
        felt = {e: g for e, g in gains.items() if g > 0.05}
        for t in {t for topic in (topics or [])[:8] for t in tokens(topic)}:
            a = self.assoc.setdefault(t, {"emo": {}, "n": 0})
            a["n"] += 1
            emo = a["emo"]
            for e in set(emo) | set(felt):
                emo[e] = 0.7 * emo.get(e, 0.0) + 0.3 * felt.get(e, 0.0)
            a["emo"] = dict(sorted(((e, round(w, 4)) for e, w in emo.items() if w >= 0.005),
                                   key=lambda kv: -kv[1])[:5])
        if len(self.assoc) > 600:                      # keep the topics that matter most
            keep = sorted(self.assoc.items(), key=lambda kv: -sum(kv[1]["emo"].values()))[:500]
            self.assoc = dict(keep)

    def strongest_links(self, n=6):
        rows = [(t, e, w) for t, a in self.assoc.items() for e, w in a["emo"].items() if w >= 0.04]
        return sorted(rows, key=lambda r: -r[2])[:n]

    # ---------- persistence ---------------------------------------------------
    def to_dict(self):
        return {"episodes": [e.to_dict() for e in self.eps], "facts": self.facts, "assoc": self.assoc,
                "last_reflection_turn": self.last_reflection_turn}

    def load(self, d):
        self.eps = [Episode.from_dict(x) for x in d.get("episodes", [])]
        self.facts = list(d.get("facts", []))
        self.assoc = dict(d.get("assoc", {}))
        self.last_reflection_turn = d.get("last_reflection_turn", 0)

    def load_v6(self, memories, facts, now=None):
        """Bring over a v6 save: its strong moments and plain facts."""
        now = now or time.time()
        for m in memories or []:
            self.eps.append(Episode(m.get("gist", ""), m.get("topics", []), "", m.get("emotion", "neutral"),
                                    None, math.tanh(m.get("reward", 0.0)), min(1.0, m.get("strength", 1.0) / 3),
                                    now - 3600))
        for f in facts or []:
            self.add_fact(f, now=now)
