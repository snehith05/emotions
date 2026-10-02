"""
Search over text with nothing to install: BM25 ranking (the method behind classic search engines)
with light stemming and a few synonyms, so "exams" finds "exam" and "mom" finds "mother".

Used for two things:
  - a persona's knowledge: their writings, cut into short passages (RAG)
  - the companion's own memories: moments and facts from past conversations
"""
import math
import re
from collections import Counter, defaultdict

STOP = set("""a about above after again against all am an and any are aren't as at be because been before
being below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me
more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same
shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those through to too under until up very
was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
u ur im ive ill dont cant wont didnt doesnt isnt wasnt lol yeah yes ok okay oh hey hi also just like really
get got gonna wanna will shall may might must one upon thus hath unto thee thy""".split())

# words people use for the same thing - a query for one also finds the others
RELATED = [
    ("mom", "mother", "mum", "amma"), ("dad", "father", "papa", "appa"), ("exam", "test"),
    ("job", "work", "career", "office"), ("friend", "buddy", "pal"), ("sad", "sorrow", "grief", "unhappy"),
    ("happy", "joy", "glad"), ("angry", "anger", "rage", "wrath"), ("fear", "afraid", "scared", "fright"),
    ("violence", "violent", "force", "killing", "bloodshed"), ("nonviolence", "ahimsa", "non-violence", "nonviolent", "non-violent"),
    ("truth", "satya", "honesty"), ("freedom", "swaraj", "independence", "liberty", "self-rule"),
    ("british", "english", "englishmen", "empire"), ("god", "divine", "religion", "faith"),
    ("love", "affection"), ("money", "wealth", "riches"), ("machine", "machinery", "industry"),
    ("hindu", "hindus"), ("muslim", "muslims", "mussalman", "mussalmans", "mahomedan", "mahomedans"),
    ("untouchability", "untouchable", "untouchables", "pariah"), ("boycott", "non-cooperation"),
    ("phone", "mobile"), ("movie", "film"), ("girlfriend", "gf"), ("boyfriend", "bf"),
]


def stem(w):
    """A tiny stemmer: love/loves/loved/loving -> lov, family/families -> famili, exams -> exam."""
    if len(w) <= 3 or not w.isalpha():
        return w
    for suf, rep in (("ational", "ate"), ("fulness", "ful"), ("iveness", "ive"), ("ousness", "ous"),
                     ("ization", "ize"), ("ements", ""), ("ments", ""), ("ement", ""), ("ment", ""),
                     ("nesses", ""), ("ness", ""), ("ingly", ""), ("edly", ""), ("ings", ""), ("ing", ""),
                     ("ies", "y"), ("ied", "y"), ("ed", ""), ("es", "e"), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not (suf == "s" and w.endswith("ss")):
            w = w[: len(w) - len(suf)] + rep
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":       # stopp -> stop, runn -> run
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w.endswith("y"):
        w = w[:-1] + "i"
    return w


_ALIAS = {}
for _group in RELATED:
    _key = stem(_group[0].replace("-", ""))
    for _w in _group:
        _ALIAS[stem(_w.replace("-", ""))] = _key


def tokens(text):
    """Words worth searching on: lowercase, no stop words, stemmed, synonyms folded together."""
    out = []
    for w in re.findall(r"[a-z0-9][a-z0-9'-]*", str(text).lower()):
        if w in STOP:
            continue
        w = w.replace("'", "").replace("-", "")
        if len(w) < 2 or w in STOP:
            continue
        s = stem(w)
        out.append(_ALIAS.get(s, s))
    return out


class BM25:
    """Ranks documents by how well they match a query. Rare words count more than common ones,
    and long documents don't win just by being long."""

    def __init__(self, k1=1.4, b=0.75):
        self.k1, self.b = k1, b
        self.tf, self.lens = [], []
        self.df = Counter()
        self.post = defaultdict(list)
        self.total = 0

    def add(self, toks):
        """Add one document (already split by tokens()); returns its index."""
        i = len(self.tf)
        tf = Counter(toks)
        self.tf.append(tf)
        self.lens.append(len(toks))
        self.total += len(toks)
        for t in tf:
            self.df[t] += 1
            self.post[t].append(i)
        return i

    def __len__(self):
        return len(self.tf)

    def scores(self, query_toks):
        n = len(self.tf)
        if not n:
            return {}
        avg = self.total / n or 1.0
        out = defaultdict(float)
        for t in set(query_toks):
            df = self.df.get(t)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i in self.post[t]:
                f = self.tf[i][t]
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / avg))
        return out

    def search(self, query, k=5):
        """[(index, score), ...] best first. Documents sharing no word with the query never come back."""
        q = tokens(query) if isinstance(query, str) else list(query)
        return sorted(self.scores(q).items(), key=lambda kv: -kv[1])[:k]


# ============================================================ CUTTING TEXT INTO PASSAGES
def _clean(p):
    p = " ".join(p.split())
    p = re.sub(r"\[\d+\]", "", p)                    # footnote marks
    return re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", p)  # _italics_


def _is_heading(p):
    words = p.split()
    return 0 < len(words) <= 12 and (p.isupper() or re.match(r"(?i)^(chapter|part|book|section)\b", p)
                                     or (p.startswith("#")))


def chunk_text(text, size=150, max_size=230):
    """Split a long text into passages of about `size` words, on paragraph and sentence boundaries,
    each labelled with the heading it falls under. Returns [(heading, passage), ...]."""
    heading = ""
    out, buf = [], []

    def flush():
        if buf:
            out.append((heading, " ".join(buf)))
            buf.clear()

    for raw in re.split(r"\n\s*\n", text):
        p = _clean(raw)
        if not p:
            continue
        if _is_heading(p):
            flush()
            heading = p.lstrip("# ").strip()[:90]
            continue
        n = len(p.split())
        if n > max_size:                              # long paragraph -> sentence groups
            flush()
            part = []
            for s in re.split(r"(?<=[.!?;])\s+", p):
                part.append(s)
                if len(" ".join(part).split()) >= size:
                    out.append((heading, " ".join(part)))
                    part = []
            if part:
                buf.append(" ".join(part))
            continue
        if buf and len(" ".join(buf).split()) + n > max_size:
            flush()
        buf.append(p)
        if len(" ".join(buf).split()) >= size:
            flush()
    flush()
    return [(h, p) for h, p in out if len(p.split()) >= 12]   # drop scraps (page numbers, captions)


def best_sentences(passage, query, n=2):
    """The one or two sentences of a passage that match the query best (for short quotes)."""
    q = set(tokens(query))
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if len(s.split()) >= 5]
    if not sents:
        return passage[:300]
    ranked = sorted(range(len(sents)), key=lambda i: -len(q & set(tokens(sents[i]))))[:n]
    return " ".join(sents[i] for i in sorted(ranked))
