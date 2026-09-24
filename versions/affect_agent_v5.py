"""
Artificial Emotion AI - prototype v5 (+ PERSONALITY and MOOD)
Tests whether persistent internal states (reward, stress, curiosity, energy)
produce more adaptive behavior than a conventional learner.

Environment: a non-stationary "multi-armed bandit" with
  - regime shifts (the best option changes over time)
  - risky options (tempting average, rare catastrophic loss)
  - fatigue (acting drains energy; low energy scales rewards down; a REST action restores it)

Agents:
  A        - conventional Q-learner (cannot sense energy)
  A+energy - conventional Q-learner that CAN sense energy (fair baseline)
  (both A agents use fixed parameters, tuned by grid search on separate seeds)
  B        - same learner + internal-state system that modulates learning/exploration/risk
  B+boredom - B plus a boredom drive (gets restless when nothing happens)
  B+fear    - B plus fear memory (remembers WHICH areas gave big fines)
  B+both    - boredom + fear memory: restless, but avoids places that hurt it
  B+smart   - boredom + calculated risk: "does this area's pay beat its fines?"
  B_static - ablation: B's machinery, but internal states frozen (tests the DYNAMICS, not just extra terms)
"""
import numpy as np

# ---------------------------------------------------------------- environment
class Env:
    def __init__(self, n_arms=6, shift_every=250, seed=0, fine=12.0):
        self.fine = fine
        self.rng = np.random.default_rng(seed)
        self.n = n_arms            # action n == REST
        self.shift_every = shift_every
        self.t, self.energy = 0, 1.0
        self._new_regime()

    def _new_regime(self):
        self.means = self.rng.normal(0, 1, self.n)
        self.crash_p = self.rng.choice([0.0, 0.0, 0.06], self.n)
        self.means[self.crash_p > 0] += 1.0      # risky arms look attractive

    def step(self, a):
        self.t += 1
        crash = False
        if a == self.n:                           # REST
            r = 0.0
            self.energy = min(1.0, self.energy + 0.15)
        else:
            r = self.rng.normal(self.means[a], 1.0) * self.energy
            if self.rng.random() < self.crash_p[a]:
                r -= self.fine; crash = True
            self.energy = max(0.0, self.energy - 0.03)
        shifted = self.t % self.shift_every == 0
        if shifted: self._new_regime()
        return r, crash, shifted

# ---------------------------------------------------------------- internal state
PERSONALITIES = {
    # traits 0..1.  0.5 everywhere = the "average" rider used in v0-v4
    #            neuroticism  openness  conscientiousness  extraversion
    "average":     dict(N=0.5, O=0.5, C=0.5, E=0.5),
    "anxious":     dict(N=0.9, O=0.3, C=0.6, E=0.4),   # worries a lot, plays safe
    "adventurous": dict(N=0.2, O=0.9, C=0.3, E=0.8),   # calm, curious, restless, careless with rest
    "calm_steady": dict(N=0.1, O=0.4, C=0.8, E=0.4),   # unbothered, disciplined
    "lazy":        dict(N=0.5, O=0.3, C=0.2, E=0.1),   # rarely bored, doesn't plan rest
}

class Personality:
    """Layer 1 - never changes. Sets each feeling's baseline, how hard events hit, how fast it recovers."""
    def __init__(self, N=0.5, O=0.5, C=0.5, E=0.5):
        self.N, self.O, self.C, self.E = N, O, C, E
        self.stress_base   = max(0.0, 0.3 * (N - 0.5))    # anxious people are never fully relaxed
        self.stress_hit    = 0.08 * (0.5 + N)             # ...react more strongly
        self.stress_recov  = 0.03 * (1.5 - N)             # ...and recover more slowly
        self.curious_bias  = 1.2 * (O - 0.5)              # open people are curious by default
        self.rest_drive    = 4.0 * C                      # conscientious people rest BEFORE exhaustion
        self.bored_rate    = 0.06 * E                     # extraverts get restless quickly
        self.mood_base     = 0.2 * (0.5 - N)              # a slightly sunnier or gloomier default mood

class InternalState:
    """Layer 3 (emotions, fast) + Layer 2 (mood, slow), shaped by Layer 1 (personality)."""
    def __init__(self, dynamic=True, use_boredom=False, personality=None, use_mood=False):
        self.dynamic, self.use_boredom, self.use_mood = dynamic, use_boredom, use_mood
        self.p = personality or Personality()
        self.reward, self.stress, self.curiosity, self.energy = 0.0, 0.2, 0.5, 1.0
        self.boredom = 0.0
        self.mood = self.p.mood_base                     # -1 = low, 0 = neutral, +1 = great

    def update(self, delta, r, energy, long_err):
        p = self.p
        self.energy = energy                       # interoception: sense the body
        if not self.dynamic: return
        # MOOD (slow): drifts with how life has been going lately, pulled back to personality default
        if self.use_mood:
            self.mood += 0.01 * (np.tanh(delta) + p.mood_base - self.mood)
            self.mood = float(np.clip(self.mood, -1, 1))
        m = self.mood if self.use_mood else 0.0
        # reward / dopamine-like: running average of prediction errors
        self.reward = 0.9 * self.reward + 0.1 * delta
        # stress: spikes on bad surprises (softened by good mood), recovers toward personality baseline
        hit = p.stress_hit * (1 - 0.5 * m)
        self.stress += p.stress_recov * (p.stress_base - self.stress) + hit * max(0.0, -delta) / 5
        # curiosity: rises when the world is less predictable than usual; openness + good mood raise it
        surprise = abs(delta) / (long_err + 1e-6)
        self.curiosity = 0.95 * self.curiosity + 0.05 * (min(surprise, 3.0) + p.curious_bias + 0.5 * m)
        # boredom: grows while nothing surprising happens (faster for extraverts)
        if self.use_boredom:
            if surprise > 1.0: self.boredom *= 0.5
            else:              self.boredom = min(1.0, self.boredom + p.bored_rate)
            self.curiosity += 0.02 * self.boredom      # boredom feeds curiosity
        self.stress = float(np.clip(self.stress, 0, 1))
        self.curiosity = float(np.clip(self.curiosity, 0, 2))

    def to_prompt(self):
        """Later: this text goes into the LLM's context."""
        return (f"[internal state] mood={self.mood:+.2f} reward={self.reward:+.2f} stress={self.stress:.2f} "
                f"curiosity={self.curiosity:.2f} energy={self.energy:.2f} boredom={self.boredom:.2f}")

# ---------------------------------------------------------------- agents
class AgentA:
    def __init__(self, n, alpha=0.1, tau=0.3, seed=0):
        self.q = np.zeros(n + 1); self.alpha, self.tau = alpha, tau
        self.rng = np.random.default_rng(seed)
    def act(self, energy):
        p = np.exp((self.q - self.q.max()) / self.tau); p /= p.sum()
        return self.rng.choice(len(p), p=p)
    def learn(self, a, r, energy):
        self.q[a] += self.alpha * (r - self.q[a])

class AgentA_E:
    """Conventional learner that CAN sense energy (fair baseline).
    Keeps separate scores for each energy level (5 bins): no feelings, same information as B."""
    def __init__(self, n, alpha=0.1, tau=0.3, seed=0, bins=5):
        self.q = np.zeros((bins, n + 1)); self.alpha, self.tau, self.bins = alpha, tau, bins
        self.rng = np.random.default_rng(seed); self.b = bins - 1
    def _bin(self, energy): return min(int(energy * self.bins), self.bins - 1)
    def act(self, energy):
        self.b = self._bin(energy); q = self.q[self.b]
        p = np.exp((q - q.max()) / self.tau); p /= p.sum()
        return self.rng.choice(len(p), p=p)
    def learn(self, a, r, energy):
        self.q[self.b, a] += self.alpha * (r - self.q[self.b, a])

class AgentB:
    def __init__(self, n, dynamic=True, seed=0, boredom=False, fear_memory=False, smart=False,
                 personality=None, mood=False):
        self.n, self.fear_memory, self.smart = n, fear_memory, smart
        # smart risk: keep "normal pay" and "how often fined" SEPARATELY, like a human would
        self.pay = np.zeros(n + 1)          # usual pay in each area (fines excluded)
        self.visits = np.zeros(n + 1)       # recent visits (fade -> can relearn after city changes)
        self.fines = np.zeros(n + 1)        # recent fines per area (fade too)
        self.fine_size = 8.0                # guess of how bad a fine is; learned
        self.fear = np.zeros(n + 1)                    # per-area "scar": 1 = just got hurt here
        self.q = np.zeros(n + 1); self.var = np.ones(n + 1); self.counts = np.zeros(n + 1)
        self.state = InternalState(dynamic, use_boredom=boredom, personality=personality, use_mood=mood)
        self.long_err = 1.0
        self.rng = np.random.default_rng(seed)
        self.memory = []                           # episodic log: (t, action, r, state snapshot)

    def act(self, energy):
        s = self.state
        novelty = 1.0 / np.sqrt(self.counts + 1)
        safe = 1.0 - self.fear                            # 1 = feels safe, 0 = just got hurt there
        base = self.q
        if self.smart:                                    # expected value = usual pay - chance of fine x fine size
            fine_rate = (self.fines + 0.02) / (self.visits + 1)
            base = self.pay - fine_rate * self.fine_size
            base[self.n] = self.q[self.n]                 # resting is never fined
        value = (base
                 - 1.5 * s.stress * np.sqrt(self.var)     # stress -> risk aversion (fades as calm returns)
                 + 0.8 * s.curiosity * novelty * safe      # curiosity -> explore, but not where it hurt
                 - 2.0 * self.fear)                        # fear memory -> avoid places that hurt
        value[self.n] += s.p.rest_drive * (1 - s.energy) ** 2   # low energy -> drive to rest (conscientiousness)
        value[self.n] -= 1.5 * s.boredom * s.energy       # bored + rested -> get back to work
        tau = 0.3 * (1 + 0.5 * s.curiosity) / (1 + s.stress)
        p = np.exp((value - value.max()) / tau); p /= p.sum()
        return self.rng.choice(len(p), p=p)

    def learn(self, a, r, energy):
        delta = r - self.q[a]
        s = self.state
        if self.smart and a != self.n:
            self.visits *= 0.998; self.fines *= 0.998
            self.visits[a] += 1
            if r < self.pay[a] - 5:                        # that was a fine, not normal pay
                self.fines[a] += 1
                self.fine_size += 0.2 * ((self.pay[a] - r) - self.fine_size)
            else:
                self.pay[a] += 0.2 * (r - self.pay[a])     # normal pay learned quickly
        if self.fear_memory:
            self.fear *= 0.998                              # scars fade slowly (half-life ~350 steps)
            if delta < -5: self.fear[a] = 1.0               # big nasty surprise -> remember this place
        alpha = 0.1 * (1 + s.curiosity + s.stress)        # surprise/stress -> faster learning
        alpha = min(alpha, 0.6)
        self.q[a] += alpha * delta
        self.var[a] += 0.1 * (delta ** 2 - self.var[a])
        self.counts *= 0.995; self.counts[a] += 1         # old knowledge fades -> novelty regrows
        self.long_err = 0.995 * self.long_err + 0.005 * abs(delta)
        s.update(delta, r, energy, self.long_err)
        self.memory.append((len(self.memory), a, r, s.stress, s.curiosity))

# ---------------------------------------------------------------- experiment
FINE = 12.0   # size of a big fine; try 25 for a more dangerous city

def run(make_agent, seed, steps=2000, n=6):
    env, ag = Env(n, seed=seed, fine=FINE), make_agent(n, seed)
    total, crashes, post_shift, since_shift, idle_fresh = 0.0, 0, [], 10**9, 0
    for _ in range(steps):
        a = ag.act(env.energy)
        idle_fresh += (a == n and env.energy > 0.9)
        r, crash, shifted = env.step(a)
        ag.learn(a, r, env.energy)
        total += r; crashes += crash
        if since_shift < 50: post_shift.append(r)
        since_shift = 0 if shifted else since_shift + 1
    return total / steps, crashes, np.mean(post_shift), 100 * idle_fresh / steps

def evaluate(name, make_agent, seeds):
    res = np.array([run(make_agent, s) for s in seeds])
    m, se = res.mean(0), res.std(0) / np.sqrt(len(seeds))
    print(f"{name:<22} reward/step {m[0]:+.3f}±{se[0]:.3f} | crashes {m[1]:5.1f}±{se[1]:.1f} "
          f"| post-shift reward {m[2]:+.3f}±{se[2]:.3f}")

def tune(cls, seeds):
    grid = [(al, t) for al in [0.05, 0.1, 0.2, 0.3] for t in [0.1, 0.3, 0.6, 1.0]]
    return max(grid, key=lambda p: np.mean([run(lambda n, s: cls(n, *p, seed=s), sd)[0] for sd in seeds]))

def trace(seed=7, steps=1000, n=6, boredom=True, fear_memory=False, smart=False, personality=None, mood=False):
    """Run agent B once and record its feelings over time."""
    env = Env(n, seed=seed, fine=FINE)
    ag = AgentB(n, seed=seed, boredom=boredom, fear_memory=fear_memory, smart=smart, personality=personality, mood=mood)
    log = {k: [] for k in ["mood", "stress", "curiosity", "energy", "boredom", "reward", "crash", "shift", "rest"]}
    for _ in range(steps):
        a = ag.act(env.energy)
        r, crash, shifted = env.step(a)
        ag.learn(a, r, env.energy)
        s = ag.state
        for k, v in [("mood", s.mood), ("stress", s.stress), ("curiosity", s.curiosity), ("energy", s.energy), ("boredom", s.boredom),
                     ("reward", r), ("crash", crash), ("shift", shifted), ("rest", a == n)]:
            log[k].append(v)
    return {k: np.array(v) for k, v in log.items()}

def plot_feelings(path="feelings_over_time.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [(trace(smart=True, mood=True, personality=Personality(**PERSONALITIES["anxious"])), "ANXIOUS rider"),
            (trace(smart=True, mood=True, personality=Personality(**PERSONALITIES["adventurous"])), "ADVENTUROUS rider")]
    rows = [("mood", "tab:blue", "Mood"), ("stress", "tab:red", "Stress"), ("curiosity", "tab:purple", "Curiosity"),
            ("boredom", "tab:brown", "Boredom"), ("energy", "tab:green", "Energy"),
            ("reward", "tab:gray", "Reward")]
    fig, ax = plt.subplots(len(rows), 2, figsize=(16, 13), sharex=True, sharey="row")
    for col, (L, title) in enumerate(runs):
        t = np.arange(len(L["stress"]))
        for i, (k, c, lab) in enumerate(rows):
            a = ax[i, col]; a.plot(t, L[k], color=c, lw=1)
            if col == 0: a.set_ylabel(lab)
            for x in t[L["shift"]]: a.axvline(x, color="k", ls="--", lw=1)
            for x in t[L["crash"]]: a.axvline(x, color="red", alpha=0.25, lw=1)
        e = ax[4, col]; e.scatter(t[L["rest"]], L["energy"][L["rest"]], s=5, color="tab:blue", label="on break", zorder=3)
        e.legend(loc="lower right")
        ax[0, col].set_title(f"{title}  |  earned {L['reward'].sum():.0f}, breaks {L['rest'].sum()}")
        ax[-1, col].set_xlabel("time step")
    fig.suptitle("Same world, same dice: black dashed = areas changed, red = big fine")
    fig.tight_layout(); fig.savefig(path, dpi=110); print(f"saved {path}")

def plot_results(results, path="results.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results); M = np.array([results[k][0] for k in names]); S = np.array([results[k][1] for k in names])
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = ["Reward per step (higher = better)", "Big losses (lower = better)",
              "Reward after world changes (higher = better)", "% time on break while fresh (lower = better)"]
    colors = ["tab:gray", "tab:olive", "tab:blue", "tab:orange", "tab:brown", "tab:pink", "tab:cyan", "tab:green", "tab:blue"]
    for i in range(4):
        ax[i].bar(names, M[:, i], yerr=S[:, i], color=colors[:len(names)], capsize=4)
        ax[i].set_title(titles[i]); ax[i].tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(path, dpi=120); print(f"saved {path}")

def personality_experiment(seeds):
    """Same engine, different personalities: do they behave CONSISTENTLY differently?"""
    print("\nPersonality experiment (smart + boredom + mood, 200 cities each):")
    print(f"  {'personality':<12} {'money':>7} {'fines':>6} {'lazy%':>6} {'breaks%':>8} {'avg mood':>9} {'avg stress':>11}")
    for name, traits in PERSONALITIES.items():
        rows = []
        for sd in seeds:
            env = Env(6, seed=sd, fine=FINE)
            ag = AgentB(6, seed=sd, boredom=True, smart=True, mood=True, personality=Personality(**traits))
            tot = fines = idle = rests = 0; moods = []; stresses = []
            for _ in range(2000):
                a = ag.act(env.energy)
                idle += (a == 6 and env.energy > 0.9); rests += (a == 6)
                r, crash, _ = env.step(a); ag.learn(a, r, env.energy)
                tot += r; fines += crash; moods.append(ag.state.mood); stresses.append(ag.state.stress)
            rows.append([tot / 2000, fines, 100 * idle / 2000, 100 * rests / 2000, np.mean(moods), np.mean(stresses)])
        m = np.array(rows).mean(0)
        print(f"  {name:<12} {m[0]:+7.3f} {m[1]:6.1f} {m[2]:6.1f} {m[3]:8.1f} {m[4]:+9.2f} {m[5]:11.2f}")

if __name__ == "__main__":
    tune_seeds, test_seeds = range(100, 130), range(200)
    bestA, bestAE = tune(AgentA, tune_seeds), tune(AgentA_E, tune_seeds)
    print(f"Tuned A: alpha={bestA[0]}, tau={bestA[1]} | Tuned A+energy: alpha={bestAE[0]}, tau={bestAE[1]}\n")
    agents = {
        "A (normal)":        lambda n, s: AgentA(n, *bestA, seed=s),
        "A+energy (fair)":   lambda n, s: AgentA_E(n, *bestAE, seed=s),
        "B_static":          lambda n, s: AgentB(n, dynamic=False, seed=s),
        "B (live feelings)": lambda n, s: AgentB(n, dynamic=True, seed=s),
        "B+boredom":         lambda n, s: AgentB(n, dynamic=True, seed=s, boredom=True),
        "B+fear":            lambda n, s: AgentB(n, dynamic=True, seed=s, fear_memory=True),
        "B+both":            lambda n, s: AgentB(n, dynamic=True, seed=s, boredom=True, fear_memory=True),
        "B+smart":           lambda n, s: AgentB(n, dynamic=True, seed=s, boredom=True, smart=True),
        "B+smart+mood":      lambda n, s: AgentB(n, dynamic=True, seed=s, boredom=True, smart=True, mood=True),
    }
    results = {}
    for name, mk in agents.items():
        res = np.array([run(mk, s) for s in test_seeds])
        m, se = res.mean(0), res.std(0) / np.sqrt(len(test_seeds))
        results[name] = (m, se)
        print(f"{name:<20} reward/step {m[0]:+.3f}±{se[0]:.3f} | crashes {m[1]:5.1f}±{se[1]:.1f} "
              f"| post-shift {m[2]:+.3f}±{se[2]:.3f} | idle-while-fresh {m[3]:4.1f}%")
    print("\nPaired check vs B+boredom (same 200 cities):")
    base = np.array([run(agents["B+boredom"], sd) for sd in test_seeds])
    for nm in ["B+fear", "B+both", "B+smart", "B+smart+mood"]:
        d = np.array([run(agents[nm], sd) for sd in test_seeds]) - base
        m, se = d.mean(0), d.std(0) / np.sqrt(len(test_seeds))
        print(f"  {nm:<8} money {m[0]:+.3f}±{se[0]:.3f} | fines {m[1]:+.2f}±{se[1]:.2f} "
              f"| post-shift {m[2]:+.3f}±{se[2]:.3f} | idle {m[3]:+.1f}±{se[3]:.1f}%")
    plot_results(results)
    personality_experiment(test_seeds)
    plot_feelings()
