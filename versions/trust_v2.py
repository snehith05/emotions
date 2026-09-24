"""
Artificial Emotion AI - stage 2b: trust that handles INCONSISTENT people.

Problem with v1 trust: it was a single slider moved by the last outcome.
A liar who got lucky once gained trust; a friend who got unlucky lost it.
And once trust fell low the rider stopped testing that person forever (belief trap).

v2 trust, closer to how people actually do it:
  1. JUDGE THE ADVICE, NOT THE LUCK
     "Was following this tip better than what I would have done anyway?"
  2. KEEP A RECORD, NOT A MOOD
     right/wrong counts -> trust = right/(right+wrong)  (Bayesian: Beta distribution)
     one lucky hit barely moves a long record.
  3. KNOW HOW SURE YOU ARE
     confidence = how much evidence you have. Low confidence -> act cautiously
     on the tip, but stay curious about the person.
  4. TIME HEALS / PEOPLE CHANGE
     old evidence fades, so opinions can be revised and nobody is written off forever.

Tipsters now include "mixed": right half the time - the inconsistent person.
"""
import numpy as np
from affect_agent_v5 import Personality, PERSONALITIES, FINE
from social_agent import SocialEnv, SocialAgent

TIPSTERS2 = ["honest", "liar", "mixed", "turncoat"]
TIP_STRENGTH = 3.0


class SocialEnv2(SocialEnv):
    """Honest advice now accounts for fines too: the best area ALL THINGS CONSIDERED."""
    def tip(self, who, t, total_steps):
        if who == "mixed":                       # tells the truth about half the time
            who = "honest" if self.rng.random() < 0.5 else "liar"
        net = self.means - self.crash_p * self.fine      # what an area is really worth
        best, worst = int(np.argmax(net)), int(np.argmin(net))
        if who == "honest":   return best
        if who == "liar":     return worst
        if who == "random":   return int(self.rng.integers(self.n))
        if who == "turncoat": return best if t < total_steps / 2 else worst
        raise ValueError(who)


class TrustV2Agent(SocialAgent):
    """Trust built from evidence, with a confidence level attached."""

    def __init__(self, n, **kw):
        super().__init__(n, **kw)
        k = len(TIPSTERS2)
        self.right = np.ones(k)        # Beta prior: 1 right, 1 wrong = "no idea yet"
        self.wrong = np.ones(k)
        self.best_alt = 0.0
        self.rbar = 0.0                # how a normal trip usually goes (my yardstick)

    # ---- what the rider believes about each person -------------------------
    def trust_of(self, k):
        return self.right[k] / (self.right[k] + self.wrong[k])          # 0..1

    def confidence_of(self, k):
        evidence = self.right[k] + self.wrong[k] - 2
        return evidence / (evidence + 8.0)                               # 0 = no idea, ->1 = sure

    def act_social(self, energy, k, tipped_area):
        self.pending = (k, tipped_area)
        t, c = self.trust_of(k), self.confidence_of(k)
        # weak opinions push the choice only weakly; a little extra pull to TEST unknown people
        belief = 2 * (t - 0.5) * c + 0.25 * (1 - c)
        bonus = np.zeros(self.n + 1)
        bonus[tipped_area] = TIP_STRENGTH * belief
        # remember what I would have done without the tip, so I can judge the ADVICE later
        self.best_alt = float(np.max(self._value_no_tip(energy)))
        return self._act_with_bonus(energy, bonus)

    def _value_no_tip(self, energy):
        s = self.state
        base = self.q
        if self.smart:
            fine_rate = (self.fines + 0.02) / (self.visits + 1)
            base = self.pay - fine_rate * self.fine_size
            base[self.n] = self.q[self.n]
        v = base.copy()
        v[self.n] += s.p.rest_drive * (1 - s.energy) ** 2
        return v

    def learn_social(self, a, r, energy):
        k, j = self.pending
        self.learn(a, r, energy)                     # update my knowledge FIRST...
        if a == j:                                   # only judge tips I actually followed
            # ...then judge the tip against how a normal trip goes for me.
            # One lucky or unlucky trip barely moves the record below, so luck averages out.
            good = r > self.rbar
            self.right *= 0.999; self.wrong *= 0.999  # old evidence fades: people can change
            if good: self.right[k] += 1
            else:    self.wrong[k] += 1
            if self.use_attachment:
                self.attach *= 0.999
                self.attach[k] = float(np.clip(self.attach[k] + (0.01 if good else -0.005), 0, 1))
                if not good and self.attach[k] > 0.3:     # betrayal by someone close
                    self.state.stress = min(1.0, self.state.stress + 0.05 * self.attach[k])
        self.rbar += 0.01 * (r - self.rbar)
        self.trust = np.array([self.trust_of(i) for i in range(len(TIPSTERS2))])
        self.conf = np.array([self.confidence_of(i) for i in range(len(TIPSTERS2))])
        self.tip_log.append((k, j, a == j, r, self.trust.copy(), self.conf.copy()))


def run2(kind, seed, steps=2000, n=20):
    """kind: 'v1' (old slider trust) or 'v2' (evidence + confidence)"""
    env = SocialEnv2(n, seed=seed, fine=FINE)
    kw = dict(seed=seed, boredom=True, smart=True, mood=True,
              personality=Personality(**PERSONALITIES["average"]), use_attachment=True)
    ag = (TrustV2Agent if kind == "v2" else SocialAgent)(n, use_trust=True, **kw)
    rng = np.random.default_rng(seed + 999)
    total, fines = 0.0, 0
    for t in range(steps):
        k = int(rng.integers(len(TIPSTERS2)))
        a = ag.act_social(env.energy, k, env.tip(TIPSTERS2[k], t, steps))
        r, crash, _ = env.step(a)
        ag.learn_social(a, r, env.energy)
        total += r; fines += crash
    return total / steps, fines, ag


def experiment(seeds=range(40)):
    print(f"Inconsistent-people world ({len(seeds)} cities, 20 areas)\n")
    print(f"  {'rider':<6} {'money':>7} {'fines':>6} | trust:  "
          + "  ".join(f"{t:>8}" for t in TIPSTERS2))
    for kind in ["v1", "v2"]:
        rows, trusts, confs = [], [], []
        for sd in seeds:
            m, f, ag = run2(kind, sd)
            rows.append([m, f]); trusts.append(ag.trust)
            confs.append(getattr(ag, "conf", np.full(len(TIPSTERS2), np.nan)))
        m, tr, cf = np.array(rows).mean(0), np.array(trusts).mean(0), np.array(confs).mean(0)
        print(f"  {kind:<6} {m[0]:+7.3f} {m[1]:6.1f} |         "
              + "  ".join(f"{v:8.2f}" for v in tr))
        if kind == "v2":
            print(f"  {'':<6} {'':>7} {'':>6} | how sure:"
                  + "  ".join(f"{v:8.2f}" for v in cf))


def plot_trust2(seed=3, steps=3000, path="trust_v2.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    _, _, a1 = run2("v1", seed, steps=steps)
    _, _, a2 = run2("v2", seed, steps=steps)
    for i, name in enumerate(TIPSTERS2):
        ax[0].plot([row[4][i] for row in a1.tip_log], lw=1.2, label=name)
        ax[1].plot([row[4][i] for row in a2.tip_log], lw=1.2, label=name)
        ax[2].plot([row[5][i] for row in a2.tip_log], lw=1.2, label=name)
    for axi, title in zip(ax, ["v1 trust (last outcome moves the slider)",
                               "v2 trust (record of good/bad ADVICE)",
                               "v2 confidence (how sure am I about this person?)"]):
        axi.axvline(len(a2.tip_log) / 2, color="k", ls="--", lw=1)
        axi.set_title(title); axi.legend(loc="lower left", ncol=4, fontsize=8)
    ax[-1].set_xlabel("tips received (dashed = turncoat starts lying)")
    fig.tight_layout(); fig.savefig(path, dpi=120); print(f"saved {path}")


if __name__ == "__main__":
    experiment()
    plot_trust2()
