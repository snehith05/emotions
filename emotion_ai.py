"""
=============================================================================
 ARTIFICIAL EMOTION AI - one-file engine  (stages 1-3)
=============================================================================
 Layer 1  PERSONALITY : fixed traits - baselines, reactivity, recovery
 Layer 2  MOOD        : slow background feeling
 Layer 3  EMOTIONS    : stress, curiosity, boredom, energy, reward
 Layer 4  SOCIAL      : trust (evidence + confidence), attachment, betrayal
 Layer 5  MEMORY      : emotional episodes - strong moments stick, fade slowly,
                        and come back as a "gut feeling" in similar situations

 Test world: a delivery rider in a city. Areas pay differently, some risk fines,
 riding drains energy. The city has a few RECURRING moods (rush hour, quiet day,
 festival) shown by a visible cue - so remembering past episodes can pay off.

 Run:  python emotion_ai.py memory       (emotional memory on/off)
       python emotion_ai.py personality  (5 personalities compared)
       python emotion_ai.py social       (trust + attachment)
       python emotion_ai.py all
=============================================================================
"""
import sys
import numpy as np

FINE = 12.0
N_AREAS = 8
CUES = ["rush-hour", "quiet-day", "festival"]

# =========================================================== 1. PERSONALITY
PERSONALITIES = {
    #              neuroticism openness conscientiousness extraversion
    "average":     dict(N=0.5, O=0.5, C=0.5, E=0.5),
    "anxious":     dict(N=0.9, O=0.3, C=0.6, E=0.4),
    "adventurous": dict(N=0.2, O=0.9, C=0.3, E=0.8),
    "calm_steady": dict(N=0.1, O=0.4, C=0.8, E=0.4),
    "lazy":        dict(N=0.5, O=0.3, C=0.2, E=0.1),
}


class Personality:
    """Never changes. Sets where each feeling rests and how hard events hit."""

    def __init__(self, N=0.5, O=0.5, C=0.5, E=0.5):
        self.N, self.O, self.C, self.E = N, O, C, E
        self.stress_base = max(0.0, 0.3 * (N - 0.5))
        self.stress_hit = 0.08 * (0.5 + N)
        self.stress_recov = 0.03 * (1.5 - N)
        self.curious_bias = 1.2 * (O - 0.5)
        self.rest_drive = 4.0 * C
        self.bored_rate = 0.06 * E
        self.mood_base = 0.2 * (0.5 - N)
        self.memory_strength = 0.5 + 0.5 * N   # anxious people remember bad moments more


# ============================================== 2+3. MOOD AND FAST EMOTIONS
class InternalState:
    def __init__(self, personality=None, dynamic=True, use_boredom=True, use_mood=True):
        self.p = personality or Personality()
        self.dynamic, self.use_boredom, self.use_mood = dynamic, use_boredom, use_mood
        self.reward, self.stress, self.curiosity, self.energy = 0.0, 0.2, 0.5, 1.0
        self.boredom, self.mood = 0.0, self.p.mood_base

    def update(self, delta, r, energy, long_err):
        p = self.p
        self.energy = energy
        if not self.dynamic:
            return
        if self.use_mood:
            self.mood = float(np.clip(self.mood + 0.01 * (np.tanh(delta) + p.mood_base - self.mood), -1, 1))
        m = self.mood if self.use_mood else 0.0
        self.reward = 0.9 * self.reward + 0.1 * delta
        hit = p.stress_hit * (1 - 0.5 * m)
        self.stress += p.stress_recov * (p.stress_base - self.stress) + hit * max(0.0, -delta) / 5
        surprise = abs(delta) / (long_err + 1e-6)
        self.curiosity = 0.95 * self.curiosity + 0.05 * (min(surprise, 3.0) + p.curious_bias + 0.5 * m)
        if self.use_boredom:
            self.boredom = self.boredom * 0.5 if surprise > 1.0 else min(1.0, self.boredom + p.bored_rate)
            self.curiosity += 0.02 * self.boredom
        self.stress = float(np.clip(self.stress, 0, 1))
        self.curiosity = float(np.clip(self.curiosity, 0, 2))
        return surprise

    def to_prompt(self):
        """For the LLM stage: the feelings as text."""
        return (f"[state] mood={self.mood:+.2f} stress={self.stress:.2f} curiosity={self.curiosity:.2f} "
                f"energy={self.energy:.2f} boredom={self.boredom:.2f}")


# ================================================= 5. EMOTIONAL MEMORY
class Episode:
    __slots__ = ("cue", "area", "reward", "feeling", "strength", "t")

    def __init__(self, cue, area, reward, feeling, strength, t):
        self.cue, self.area, self.reward = cue, area, reward
        self.feeling, self.strength, self.t = feeling, strength, t


class EmotionalMemory:
    """Strong moments are stored and fade slowly; similar situations bring them back."""

    def __init__(self, capacity=300, store_threshold=1.0):
        self.eps, self.capacity, self.thr = [], capacity, store_threshold

    def maybe_store(self, cue, area, reward, surprise, state, t, boost=1.0):
        # how memorable was this? big surprise or strong feelings = memorable
        salience = (abs(surprise) * 0.5 + state.stress + abs(np.tanh(reward))) * boost
        if salience < self.thr:
            return False
        self.eps.append(Episode(cue, area, reward, np.sign(reward), salience, t))
        if len(self.eps) > self.capacity:            # forget the weakest memory
            self.eps.remove(min(self.eps, key=lambda e: e.strength))
        return True

    def decay(self, rate=0.9995):
        for e in self.eps:
            e.strength *= rate

    def gut_feeling(self, cue, n_actions):
        """Recall episodes from a similar situation -> a nudge per area."""
        gut = np.zeros(n_actions)
        weight = np.zeros(n_actions)
        for e in self.eps:
            w = e.strength * (1.0 if e.cue == cue else 0.25)   # same situation recalls best
            gut[e.area] += w * e.reward
            weight[e.area] += w
        return gut / np.maximum(weight, 1e-9)


# ===================================================== 4. SOCIAL FEELINGS
class SocialBrain:
    """Trust built from evidence, with confidence, forgiveness and attachment."""

    def __init__(self, n_people):
        self.right, self.wrong = np.ones(n_people), np.ones(n_people)
        self.attach = np.zeros(n_people)

    def trust(self, k):
        return self.right[k] / (self.right[k] + self.wrong[k])

    def confidence(self, k):
        ev = self.right[k] + self.wrong[k] - 2
        return ev / (ev + 8.0)

    def update(self, k, good, state=None, attachment=True):
        self.right *= 0.999
        self.wrong *= 0.999                          # old evidence fades: people can change
        if good:
            self.right[k] += 1
        else:
            self.wrong[k] += 1
        if attachment:
            self.attach *= 0.999
            self.attach[k] = float(np.clip(self.attach[k] + (0.01 if good else -0.005), 0, 1))
            if not good and state is not None and self.attach[k] > 0.3:   # betrayal stings
                state.stress = min(1.0, state.stress + 0.05 * self.attach[k])


# ================================================================ THE WORLD
class City:
    """Areas pay differently; some risk fines; riding tires you out.
    The city has a few RECURRING moods, each shown by a visible cue."""

    def __init__(self, n=N_AREAS, shift_every=200, seed=0, fine=FINE):
        self.rng = np.random.default_rng(seed)
        self.n, self.shift_every, self.fine = n, shift_every, fine
        self.regimes = []
        for _ in range(len(CUES)):
            means = self.rng.normal(0, 1, n)
            crash_p = self.rng.choice([0.0, 0.0, 0.06], n)
            means[crash_p > 0] += 1.0
            self.regimes.append((means, crash_p))
        self.t, self.energy = 0, 1.0
        self._pick_regime()

    def _pick_regime(self):
        self.cue = int(self.rng.integers(len(CUES)))
        self.means, self.crash_p = self.regimes[self.cue]

    def best_area(self):
        return int(np.argmax(self.means - self.crash_p * self.fine))

    def worst_area(self):
        return int(np.argmin(self.means - self.crash_p * self.fine))

    def step(self, a):
        self.t += 1
        crash = False
        if a == self.n:                                  # take a break
            r = 0.0
            self.energy = min(1.0, self.energy + 0.15)
        else:
            r = self.rng.normal(self.means[a], 1.0) * self.energy
            if self.rng.random() < self.crash_p[a]:
                r -= self.fine
                crash = True
            self.energy = max(0.0, self.energy - 0.03)
        shifted = self.t % self.shift_every == 0
        if shifted:
            self._pick_regime()
        return r, crash, shifted


# ==================================================================== AGENTS
class PlainAgent:
    """Conventional learner: no feelings, no memory (the baseline)."""

    def __init__(self, n, alpha=0.3, tau=0.6, seed=0, **kw):
        self.n, self.q = n, np.zeros(n + 1)
        self.alpha, self.tau = alpha, tau
        self.rng = np.random.default_rng(seed)

    def act(self, city, tip=None):
        p = np.exp((self.q - self.q.max()) / self.tau)
        return int(self.rng.choice(len(p), p=p / p.sum()))

    def learn(self, a, r, city, tip=None):
        self.q[a] += self.alpha * (r - self.q[a])


class EmotionalAgent:
    """Everything: personality, mood, emotions, smart risk, memory, social."""

    def __init__(self, n, seed=0, personality=None, memory=True, social=0,
                 dynamic=True, mood=True, boredom=True):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.state = InternalState(personality, dynamic, boredom, mood)
        self.mem = EmotionalMemory() if memory else None
        self.social = SocialBrain(social) if social else None
        self.q = np.zeros(n + 1)
        self.pay, self.visits, self.fines = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
        self.fine_size, self.long_err, self.counts = 8.0, 1.0, np.zeros(n + 1)
        self.rbar = 0.0
        self.pending = None

    # ---- decide -----------------------------------------------------------
    def act(self, city, tip=None):
        s = self.state
        fine_rate = (self.fines + 0.02) / (self.visits + 1)
        value = self.pay - fine_rate * self.fine_size          # smart risk: pay vs fines
        value[self.n] = self.q[self.n]
        value += 0.8 * s.curiosity / np.sqrt(self.counts + 1)  # curiosity -> try new areas
        value -= 1.5 * s.stress * np.sqrt(np.maximum(fine_rate, 0)) * self.fine_size * 0.3
        value[self.n] += s.p.rest_drive * (1 - s.energy) ** 2  # tired -> rest
        value[self.n] -= 1.5 * s.boredom * s.energy            # bored + rested -> work
        if self.mem is not None:                               # gut feeling from past episodes
            value[:self.n] += 0.6 * self.mem.gut_feeling(city.cue, self.n + 1)[:self.n]
        if tip is not None and self.social is not None:        # a tip, weighted by trust
            k, area = tip
            t, c = self.social.trust(k), self.social.confidence(k)
            value[area] += 3.0 * (2 * (t - 0.5) * c + 0.25 * (1 - c))
            self.pending = tip
        tau = 0.3 * (1 + 0.5 * s.curiosity) / (1 + s.stress)
        p = np.exp((value - value.max()) / tau)
        return int(self.rng.choice(len(p), p=p / p.sum()))

    # ---- learn ------------------------------------------------------------
    def learn(self, a, r, city, tip=None):
        s = self.state
        delta = r - self.q[a]
        alpha = min(0.1 * (1 + s.curiosity + s.stress), 0.6)
        self.q[a] += alpha * delta
        if a != self.n:
            self.visits *= 0.998
            self.fines *= 0.998
            self.visits[a] += 1
            if r < self.pay[a] - 5:
                self.fines[a] += 1
                self.fine_size += 0.2 * ((self.pay[a] - r) - self.fine_size)
            else:
                self.pay[a] += 0.2 * (r - self.pay[a])
        self.counts *= 0.995
        self.counts[a] += 1
        self.long_err = 0.995 * self.long_err + 0.005 * abs(delta)
        surprise = s.update(delta, r, city.energy, self.long_err) or 0.0
        if self.mem is not None and a != self.n:
            self.mem.decay()
            self.mem.maybe_store(city.cue, a, r, surprise, s, city.t, boost=s.p.memory_strength)
        if self.social is not None and self.pending is not None:
            k, area = self.pending
            if a == area:
                self.social.update(k, good=r > self.rbar, state=s)
            self.pending = None
        self.rbar += 0.01 * (r - self.rbar)


# =============================================================== EXPERIMENTS
def play(agent, city, steps, tipsters=None, rng=None):
    total, fines, post, since = 0.0, 0, [], 10 ** 9
    for t in range(steps):
        tip = None
        if tipsters:
            k = int(rng.integers(len(tipsters)))
            tip = (k, tipsters[k](city, t, steps))
        a = agent.act(city, tip)
        r, crash, shifted = city.step(a)
        agent.learn(a, r, city, tip)
        total += r
        fines += crash
        if since < 50:
            post.append(r)
        since = 0 if shifted else since + 1
    return total / steps, fines, float(np.mean(post))


def memory_experiment(seeds=range(60), steps=3000):
    print(f"\nEMOTIONAL MEMORY ({len(seeds)} cities, moods repeat: {', '.join(CUES)})")
    print(f"  {'rider':<22} {'money':>8} {'fines':>7} {'after a mood change':>21}")
    setups = {
        "plain (no feelings)":  lambda sd: PlainAgent(N_AREAS, seed=sd),
        "feelings, no memory":  lambda sd: EmotionalAgent(N_AREAS, seed=sd, memory=False),
        "feelings + memory":    lambda sd: EmotionalAgent(N_AREAS, seed=sd, memory=True),
    }
    for name, make in setups.items():
        rows = [play(make(sd), City(seed=sd), steps) for sd in seeds]
        m = np.array(rows).mean(0)
        se = np.array(rows).std(0) / np.sqrt(len(list(seeds)))
        print(f"  {name:<22} {m[0]:+8.3f} {m[1]:7.1f} {m[2]:+16.3f}±{se[2]:.3f}")


def personality_experiment(seeds=range(40), steps=2000):
    print(f"\nPERSONALITIES ({len(seeds)} cities each)")
    print(f"  {'personality':<12} {'money':>8} {'fines':>7} {'breaks%':>9} {'mood':>7} {'stress':>8} {'memories':>10}")
    for name, tr in PERSONALITIES.items():
        rows = []
        for sd in seeds:
            ag = EmotionalAgent(N_AREAS, seed=sd, personality=Personality(**tr))
            city = City(seed=sd)
            tot = fines = rests = 0
            moods, stresses = [], []
            for _ in range(steps):
                a = ag.act(city)
                rests += (a == N_AREAS)
                r, c, _ = city.step(a)
                ag.learn(a, r, city)
                tot += r
                fines += c
                moods.append(ag.state.mood)
                stresses.append(ag.state.stress)
            rows.append([tot / steps, fines, 100 * rests / steps, np.mean(moods),
                         np.mean(stresses), len(ag.mem.eps)])
        m = np.array(rows).mean(0)
        print(f"  {name:<12} {m[0]:+8.3f} {m[1]:7.1f} {m[2]:9.1f} {m[3]:+7.2f} {m[4]:8.2f} {m[5]:10.0f}")


def social_experiment(seeds=range(40), steps=2000):
    print(f"\nSOCIAL ({len(seeds)} cities, 4 tipsters)")
    names = ["honest", "liar", "mixed", "turncoat"]
    tipsters = [
        lambda c, t, T: c.best_area(),
        lambda c, t, T: c.worst_area(),
        lambda c, t, T: c.best_area() if c.rng.random() < 0.5 else c.worst_area(),
        lambda c, t, T: c.best_area() if t < T / 2 else c.worst_area(),
    ]
    print(f"  {'rider':<16} {'money':>8} {'fines':>7} | trust: " + " ".join(f"{n:>9}" for n in names))
    for label, use_social in [("ignores tips", False), ("trust + attachment", True)]:
        rows, trusts = [], []
        for sd in seeds:
            ag = EmotionalAgent(N_AREAS, seed=sd, social=4 if use_social else 0)
            city = City(seed=sd)
            rng = np.random.default_rng(sd + 999)
            m = play(ag, city, steps, tipsters if use_social else None, rng)
            rows.append(m[:2])
            trusts.append([ag.social.trust(i) for i in range(4)] if use_social else [np.nan] * 4)
        m, tr = np.array(rows).mean(0), np.array(trusts).mean(0)
        print(f"  {label:<16} {m[0]:+8.3f} {m[1]:7.1f} |        " + " ".join(f"{v:9.2f}" for v in tr))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "memory"
    if which in ("memory", "all"):
        memory_experiment()
    if which in ("personality", "all"):
        personality_experiment()
    if which in ("social", "all"):
        social_experiment()
