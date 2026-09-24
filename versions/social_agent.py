"""
Artificial Emotion AI - stage 2: SOCIAL feelings (trust, attachment, betrayal)

The rider now has 4 "tipsters" who tell it which area is good today:
    honest    - always tells the truth
    liar      - always names a bad/risky area
    random    - means well, but is useless
    turncoat  - honest for the first half, then starts lying (tests re-learning + betrayal)

Riders compared:
    no_tips    - ignores everyone (our v5 best rider)
    gullible   - believes every tip equally, never learns who lies
    trust      - learns a trust level for each tipster
    trust+att  - trust PLUS attachment: long friendships are forgiven more,
                 but their betrayal hurts more (bigger stress hit)
"""
import numpy as np
from affect_agent_v5 import Env, AgentB, Personality, PERSONALITIES, FINE

TIPSTERS = ["honest", "liar", "random", "turncoat"]
TIP_STRENGTH = 3.0   # how loudly a tip shouts in the decision


class SocialEnv(Env):
    def tip(self, who, t, total_steps):
        """Which area does this tipster point to?"""
        best = int(np.argmax(self.means))
        worst = int(np.argmin(self.means - 12 * self.crash_p))
        if who == "honest":   return best
        if who == "liar":     return worst
        if who == "random":   return int(self.rng.integers(self.n))
        if who == "turncoat": return best if t < total_steps / 2 else worst
        raise ValueError(who)


class SocialAgent(AgentB):
    """AgentB + feelings about PEOPLE."""

    def __init__(self, n, use_trust=True, use_attachment=False, **kw):
        super().__init__(n, **kw)
        self.use_trust, self.use_attachment = use_trust, use_attachment
        k = len(TIPSTERS)
        self.trust = np.full(k, 0.5)        # "do I believe this person?"  (fast to change)
        self.attach = np.zeros(k)           # "how close are we?"          (slow to build/fade)
        self.tip_log = []                   # history for charts

    def act_social(self, energy, tipster_idx, tipped_area):
        """Same decision as before, but a tip nudges the choice - weighted by trust."""
        self.pending = (tipster_idx, tipped_area)
        self.tip_bonus = np.zeros(self.n + 1)
        if self.use_trust:
            # trust 1.0 -> "go there"; trust 0.5 -> ignore; trust 0.0 -> "actively avoid it"
            belief = 2 * (self.trust[tipster_idx] - 0.5)
        else:
            belief = 1.0                     # gullible: believes everyone fully
        self.tip_bonus[tipped_area] = TIP_STRENGTH * belief
        return self._act_with_bonus(energy, self.tip_bonus)

    def _act_with_bonus(self, energy, bonus):
        s = self.state
        novelty = 1.0 / np.sqrt(self.counts + 1)
        safe = 1.0 - self.fear
        base = self.q
        if self.smart:
            fine_rate = (self.fines + 0.02) / (self.visits + 1)
            base = self.pay - fine_rate * self.fine_size
            base[self.n] = self.q[self.n]
        value = (base - 1.5 * s.stress * np.sqrt(self.var)
                 + 0.8 * s.curiosity * novelty * safe - 2.0 * self.fear + bonus)
        value[self.n] += s.p.rest_drive * (1 - s.energy) ** 2
        value[self.n] -= 1.5 * s.boredom * s.energy
        tau = 0.3 * (1 + 0.5 * s.curiosity) / (1 + s.stress)
        p = np.exp((value - value.max()) / tau); p /= p.sum()
        return int(self.rng.choice(len(p), p=p))

    def learn_social(self, a, r, energy):
        k, j = self.pending
        followed = (a == j)
        if followed:
            good = r > 0                                    # did the tip pay off?
            # TRUST: fast update, but an old friend gets the benefit of the doubt (forgiveness)
            step = 0.25
            if self.use_attachment and not good:
                step *= (1 - 0.6 * self.attach[k])
            self.trust[k] += step * ((1.0 if good else 0.0) - self.trust[k])
            # ATTACHMENT: grows slowly from good shared experiences, fades slowly
            if self.use_attachment:
                self.attach *= 0.999
                self.attach[k] = float(np.clip(self.attach[k] + (0.01 if good else -0.005), 0, 1))
                # BETRAYAL: being let down by someone close hurts more than by a stranger
                if not good:
                    self.state.stress = min(1.0, self.state.stress
                                            + 0.05 * self.attach[k] * self.trust[k])
        self.tip_log.append((k, j, followed, r, self.trust.copy(), self.attach.copy()))
        self.learn(a, r, energy)


def run_social(kind, seed, steps=2000, n=20, personality="average"):
    env = SocialEnv(n, seed=seed, fine=FINE)
    kw = dict(seed=seed, boredom=True, smart=True, mood=True,
              personality=Personality(**PERSONALITIES[personality]))
    ag = SocialAgent(n, use_trust=(kind in ("trust", "trust+att")),
                     use_attachment=(kind == "trust+att"), **kw)
    rng = np.random.default_rng(seed + 999)
    total, fines = 0.0, 0
    for t in range(steps):
        if kind == "no_tips":
            a = ag.act(env.energy)
            r, crash, _ = env.step(a); ag.learn(a, r, env.energy)
        else:
            k = int(rng.integers(len(TIPSTERS)))
            a = ag.act_social(env.energy, k, env.tip(TIPSTERS[k], t, steps))
            r, crash, _ = env.step(a); ag.learn_social(a, r, env.energy)
        total += r; fines += crash
    return total / steps, fines, ag


def experiment(seeds=range(60)):
    print(f"Social world: 4 tipsters (honest, liar, random, turncoat), {len(seeds)} cities each\n")
    print(f"  {'rider':<11} {'money':>7} {'fines':>6} |   trust at the end: "
          + " ".join(f"{t:>9}" for t in TIPSTERS))
    out = {}
    for kind in ["no_tips", "gullible", "trust", "trust+att"]:
        rows, trusts, attachs = [], [], []
        for sd in seeds:
            m, f, ag = run_social(kind, sd)
            rows.append([m, f]); trusts.append(ag.trust); attachs.append(ag.attach)
        m = np.array(rows).mean(0); tr = np.array(trusts).mean(0)
        out[kind] = (m, tr, np.array(attachs).mean(0))
        print(f"  {kind:<11} {m[0]:+7.3f} {m[1]:6.1f} |                      "
              + " ".join(f"{v:9.2f}" for v in tr))
    print("\n  attachment at the end (trust+att rider): "
          + " ".join(f"{t}={v:.2f}" for t, v in zip(TIPSTERS, out['trust+att'][2])))
    return out


def plot_trust(seed=3, steps=3000, path="trust_over_time.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axi, kind in zip(ax, ["trust", "trust+att"]):
        _, _, ag = run_social(kind, seed, steps=steps)
        T = np.array([row[4] for row in ag.tip_log])
        for i, name in enumerate(TIPSTERS):
            axi.plot(T[:, i], label=name, lw=1.2)
        axi.axvline(len(T) / 2, color="k", ls="--", lw=1)
        axi.set_ylabel("trust"); axi.set_title(f"rider: {kind}   (dashed line = turncoat starts lying)")
        axi.legend(loc="lower left", ncol=4, fontsize=8)
    ax[-1].set_xlabel("tips received")
    fig.tight_layout(); fig.savefig(path, dpi=120); print(f"saved {path}")


if __name__ == "__main__":
    experiment()
    plot_trust()
