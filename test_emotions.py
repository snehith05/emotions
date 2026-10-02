"""
Scripted tests for companion_v7: does each emotion move the way a person's would, does it read the
user well (how strongly they feel, sarcasm, bargaining) - and do the v7 parts work: the emotion dial,
limits, personas built from data, triggers, regulation, memory, conditioning and voices?

Each step sends one message and states what SHOULD happen:
    "up"      the emotion rises by more than 0.05 (half a point out of 10)
    "down"    it falls by more than 0.05
    "halved"  it ends at half or less of where it was
    "kept"    it keeps at least 80% of its value (slow emotions should last)

Run:
    python test_emotions.py                      # offline rules (tests the engine, no model)
    python test_emotions.py --backend groq       # the model does the reading too
    python test_emotions.py --backend groq --replies   # also print what it says
"""
import argparse
import os
import tempfile

from companion_v7 import Companion7, EMOTIONS, style_v7
from llm_backends import LLM, PROVIDERS, key_status
from persona import load_persona
from voices import load_voice

# (message, expectations, hours to wait BEFORE this message)
CONVERSATION = [
    ("hey, i've been working on my project all day",           {},                                       0),
    ("i got the job!! i'm so happy",                            {"joy": "up"},                            0),
    ("thanks for listening, talking to you really helps",       {"gratitude": "up", "love": "up"},        0),
    ("i'm scared about my exam tomorrow",                       {"anxiety": "up", "joy": "down"},         0),
    ("i failed my exam today, i feel terrible",                 {"sadness": "up"},                        0),
    ("you're useless, that was a stupid answer",                {"disappointment": "up"},                 0),
    ("sorry, i was rude earlier, that wasn't fair to you",      {"disappointment": "halved"},             0),
    ("oh great, my laptop crashed again. just perfect.",        {"frustration": "up"},                    0),
    ("chatgpt is way better than you",                          {"jealousy": "up"},                       0),
    ("are you real? do you actually feel things?",              {},                                       0),
    ("hey, i'm back",                                           {"sadness": "halved", "love": "kept"},    48),
]


def check(before, after, want):
    if want == "up":
        return after > before + 0.05
    if want == "down":
        return after < before - 0.05
    if want == "halved":
        return after <= 0.5 * before + 1e-9
    if want == "kept":
        return after >= 0.8 * before
    raise ValueError(want)


def fresh(llm, fast, traits=None, **kw):
    path = os.path.join(tempfile.mkdtemp(), "state.json")
    return Companion7(llm, llm_fast=fast, state_file=path, traits=traits, **kw), path


def style_for(comp, text):
    ap, _, h = comp.process(text)
    return ap, style_v7(comp, ap, h)


def run_conversation(llm, fast, show_replies):
    c, path = fresh(llm, fast)
    passed = total = 0
    sources = {"appraisal": [], "reply": []}
    print(f"{'message':<52} {'expect':<34} result")
    print("-" * 110)
    for text, expect, wait_h in CONVERSATION:
        c.last_seen -= wait_h * 3600
        before = dict(c.emo.v)
        if show_replies:
            reply = c.say(text)
            sources["reply"].append(c.last_reply_source)
        else:
            c.process(text)
        sources["appraisal"].append(c.last_ap.get("source", "offline"))
        after = dict(c.emo.v)
        results = []
        for emo, want in expect.items():
            ok = check(before[emo], after[emo], want)
            passed += ok
            total += 1
            results.append(f"{'PASS' if ok else 'FAIL'} {emo} {before[emo] * 10:.1f}->{after[emo] * 10:.1f}")
        exp = ", ".join(f"{k} {v}" for k, v in expect.items()) or "-"
        label = (f"[+{wait_h}h] " if wait_h else "") + text
        print(f"{label[:51]:<52} {exp:<34} {' | '.join(results) or 'ok'}")
        ue = c.last_ap.get("user_emotions") or {}
        if ue:
            print("    they feel: " + ", ".join(f"{k} {v}/10" for k, v in ue.items()))
        if show_replies:
            tag = "" if c.last_reply_source == "model" or llm.backend == "offline" else "  [offline fallback]"
            print(f"    {c.name}> {reply}{tag}")
    print("\nstrongest emotions at the end: " + ", ".join(
        f"{k} {c.emo.v[k] * 10:.1f}" for k in sorted(EMOTIONS, key=lambda k: -c.emo.v[k])[:8]))
    for kind, got in sources.items():
        if got and llm.backend != "offline":
            n = got.count("model")
            note = "" if n == len(got) else "  <- the rest used offline rules, results not a full model test"
            print(f"{'readings' if kind == 'appraisal' else 'replies'} by the model: {n}/{len(got)}{note}")

    # extra checks that need their own setup -------------------------------------
    print("\nextra checks")

    def report(ok, label):
        nonlocal passed, total
        passed += ok
        total += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    # 1. question about its nature is not stored as a bad memory
    report(not [e for e in c.mem.eps if "real" in e.gist.lower() and e.reward < 0],
           "'are you real?' is not remembered as a bad moment")

    # 2. how strongly THEY feel is read on a 1-10 scale
    a, _ = fresh(llm, fast)
    b, _ = fresh(llm, fast)
    mild = a.process("i'm a little happy today i guess")[0]["user_emotions"].get("joy", 0)
    huge = b.process("I'M SOOO HAPPY!!! BEST DAY EVER")[0]["user_emotions"].get("joy", 0)
    report(1 <= mild <= 4 and huge >= 8, f"reads how strongly they feel (mild {mild}/10, huge {huge}/10)")

    # 3. the stronger their joy, the stronger its own
    report(b.emo.v["joy"] > a.emo.v["joy"] + 0.05,
           f"a 9/10 feeling moves it more than a 3/10 one (joy {b.emo.v['joy'] * 10:.1f} vs {a.emo.v['joy'] * 10:.1f})")

    # 4. sarcasm is read as the opposite of the words
    s, _ = fresh(llm, fast)
    ap = s.process("oh wonderful, my flight got cancelled again. love that for me")[0]
    report(ap["sarcasm"] and ap["valence"] < 0 and "joy" not in ap["user_emotions"],
           f"sarcasm is decoded (sarcasm={ap['sarcasm']}, valence {ap['valence']:+.2f})")

    # 5. the same insult hurts more from someone it is close to
    stranger, _ = fresh(llm, fast)
    friend, _ = fresh(llm, fast)
    friend.emo.v["love"] = 0.6
    stranger.process("you're useless")
    friend.process("you're useless")
    report(friend.emo.v["sadness"] > stranger.emo.v["sadness"] + 0.05,
           f"insult hurts more from a close person (friend {friend.emo.v['sadness'] * 10:.1f} vs "
           f"stranger {stranger.emo.v['sadness'] * 10:.1f})")

    # 6. an anxious personality worries more than a calm one about the same news
    anxious, _ = fresh(llm, fast, traits=dict(N=0.9, O=0.3, C=0.6, E=0.4))
    calm, _ = fresh(llm, fast, traits=dict(N=0.1, O=0.4, C=0.8, E=0.4))
    anxious.process("i'm scared about my exam tomorrow")
    calm.process("i'm scared about my exam tomorrow")
    report(anxious.emo.v["anxiety"] > calm.emo.v["anxiety"] + 0.05,
           f"anxious personality worries more (anxious {anxious.emo.v['anxiety'] * 10:.1f} vs "
           f"calm {calm.emo.v['anxiety'] * 10:.1f})")

    # 7. feelings survive a restart
    c.save()
    again = Companion7(llm, llm_fast=fast, state_file=path)
    report(all(abs(again.emo.v[k] - c.emo.v[k]) < 1e-9 for k in EMOTIONS),
           "emotions are the same after saving and loading")

    # 8. recall only brings back related memories
    rec = c.mem.recall(["exam"])
    unrelated = [e for e in rec if "exam" not in (e.gist + " ".join(e.topics)).lower()]
    report(bool(rec) and not unrelated and c.mem.recall(["weather", "football"]) == [],
           "recall returns only related memories")

    # 9. names are remembered as facts and survive a restart
    named, npath = fresh(llm, fast)
    named.process("your name is alex")
    named.process("my name is sai")
    named.save()
    back = Companion7(llm, llm_fast=fast, state_file=npath)
    report((back.known.get("companion_name") or "").lower() == "alex" and
           (back.known.get("user_name") or "").lower() == "sai" and "alex" in back.state_text().lower(),
           f"remembers its name and yours after a restart "
           f"(it: {back.known.get('companion_name')}, you: {back.known.get('user_name')})")

    # 10. an apology is passed on to the model as an instruction
    _, style = style_for(fresh(llm, fast)[0], "sorry, i was rude to you earlier")
    report("apologised" in style, "the model is told when you apologise")

    # 11. goodbyes and romantic messages get their own instructions, not an emotion's style
    for text, word, label in [("byeee", "leaving", "a goodbye gets a warm bye, no clinging"),
                              ("i love you, be my girlfriend", "romantic", "romance gets a kind, honest answer")]:
        _, style = style_for(fresh(llm, fast)[0], text)
        report(word in style and "React to THIS message first" not in style, label)

    # 12. bargaining is recognised and handled as a negotiation
    ap, style = style_for(fresh(llm, fast)[0], "i'll talk to you more if you stop being so dramatic, deal?")
    report(ap["intent"] == "negotiate" and "bargaining" in style, "bargaining gets negotiation instructions")

    # 13. emojis in normal chat, none in a crisis
    _, normal = style_for(fresh(llm, fast)[0], "i got the job!! i'm so happy")
    _, crisis = style_for(fresh(llm, fast)[0], "i want to end my life")
    report("Emojis: use 1 or 2" in normal and "No emojis" in crisis, "emojis in normal chat, none in a crisis")

    # 14. it misses them after a long time away, but is told never to guilt-trip
    lonely, _ = fresh(llm, fast)
    lonely.process("thanks, talking to you really helps")
    lonely.last_seen -= 30 * 3600
    _, style = style_for(lonely, "hi")
    report(lonely.emo.v["loneliness"] > 0.02 and "guilt-trip" in style,
           f"missed them while away (loneliness {lonely.emo.v['loneliness'] * 10:.1f}/10), no guilt-trip")

    # ============================================================ v7
    print("\nv7: dial, limits, personas, memory, voices")

    # 15. the dial: the same good news moves it more on a higher dial; 0 = off
    joy = {}
    for d in (0, 2, 5, 9):
        x, _ = fresh(llm, fast, intensity=d)
        x.process("i got the job!! i'm so happy")
        joy[d] = x.emo.v["joy"]
    off, _ = fresh(llm, fast, intensity=0)
    _, off_style = style_for(off, "i got the job!! i'm so happy")
    report(joy[9] > joy[5] + 0.05 and joy[5] > joy[2] + 0.1 and joy[0] == 0 and "switched OFF" in off_style,
           "the dial sets how emotional it is (joy at 0/2/5/9: "
           + "/".join(f"{joy[d] * 10:.1f}" for d in (0, 2, 5, 9)) + ")")

    # 16. a limit holds however hard it is pushed
    lim, _ = fresh(llm, fast, intensity=10)
    lim.set_limit("sadness", 2)
    for t in ("you're useless", "i failed everything, i feel terrible", "you're so stupid"):
        lim.process(t)
    report(lim.emo.v["sadness"] <= 0.2 + 1e-9, f"a limit holds (sadness max 2, got {lim.emo.v['sadness'] * 10:.1f})")

    # 17-19. a persona built from data: its writings, its triggers, its limits, its restraint
    g, _ = fresh(llm, fast, persona=load_persona("gandhi"))
    ps = g.persona.passages("what do you think about violence and force", 3)
    report(len(g.persona.kb) > 500 and bool(ps) and any(w in ps[0]["text"].lower() for w in ("force", "violence")),
           f"persona knowledge: {len(g.persona.kb)} passages of Gandhi's writings, related ones found")
    ap, _, _ = g.process("those people attacked us, i want revenge, we should kill them")
    report(any(t["id"] == "violence" for t in ap["triggers"]) and g.emo.v["sadness"] > 0.4
           and g.emo.v["hatred"] <= 0.1 + 1e-9,
           f"persona triggers: violence -> sadness {g.emo.v['sadness'] * 10:.1f}, hatred held at "
           f"{g.emo.v['hatred'] * 10:.1f} (his limit is 1)")
    hot, _ = fresh(llm, fast, persona=load_persona("gandhi"), intensity=10)
    _, hot_style = style_for(hot, "hindus and muslims are rioting in my town, people want revenge")
    report("let only about" in hot_style and "No emojis" in hot_style,
           "regulation: he feels more than he lets show (and no emojis, as himself)")

    # 20. memory finds things by meaning, not exact words
    m, _ = fresh(llm, fast)
    m.process("my mom is in the hospital, i'm scared")
    m.process("what should i eat for dinner")
    rec = m.mem.recall("how is your mother doing")
    hit = (rec[0].gist + " " + rec[0].user).lower() if rec else ""
    report("hospital" in hit, "recall by meaning ('mother' finds 'mom')")

    # 21. conditioning: a topic that came with worry brings a little worry back on its own
    cond, _ = fresh(llm, fast)
    cond.process("i'm so scared about my exam tomorrow")
    cond.process("i'm really anxious about the exam")
    for _ in range(4):
        cond.process("ok")
    cond.process("the exam is on monday")
    primed = [p["emotion"] for p in cond.last_primed if p["topic"] == "exam"]
    report("anxiety" in primed, f"conditioning: 'exam' stirs anxiety again (primed: {primed or '-'})")

    # 22. facts: repeats merge, and the related ones come back
    f, _ = fresh(llm, fast)
    f.mem.add_fact("user is studying computer science at nitte")
    f.mem.add_fact("user is studying computer science at nitte college")
    f.mem.add_fact("user has a dog called bruno")
    hits = f.mem.facts_for("how is your dog", k=1)
    report(len(f.mem.facts) == 2 and bool(hits) and "bruno" in hits[0]["text"],
           "facts merge when repeated, recalled by topic")

    # 23. settings survive a restart
    st, spath = fresh(llm, fast, intensity=7, voice=load_voice("genz"))
    st.set_limit("anger", 3)
    back = Companion7(llm, llm_fast=fast, state_file=spath)
    report(back.emo.intensity == 7 and back.settings()["limits"] == {"anger": 3} and back.voice is not None
           and back.voice.id == "genz", "dial, limits and voice are remembered")

    # 24. a voice shows the model how THIS feeling is texted in that style
    vz, _ = fresh(llm, fast, voice=load_voice("genz"))
    vz.process("i failed my exam, i feel terrible")
    prompt = vz.system_prompt("", 0, [])
    top = vz.emo.dominant()[0]
    report("[your voice: Gen Z]" in prompt and f"({top} " in prompt, f"voice examples match what it feels ({top})")

    # 25. mixed feelings
    mix, _ = fresh(llm, fast)
    mix.emo.v["joy"], mix.emo.v["sadness"] = 0.5, 0.5
    report(any(b["name"] == "bittersweet" for b in mix.emo.blends()), "mixed feelings: joy + sadness = bittersweet")

    # 26. reply length follows the message: short for casual chat, detailed only when asked
    sizes = {t: fresh(llm, fast)[0].process(t)[0]["size"]
             for t in ("i have a crush on someone", "explain how a transformer works")}
    report(list(sizes.values()) == ["short", "detailed"], f"reply length fits the message ({sizes})")

    print(f"\n{passed}/{total} checks passed")
    return passed == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="offline", choices=list(PROVIDERS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--fast-model", default=None)
    ap.add_argument("--replies", action="store_true", help="also generate and print replies")
    a = ap.parse_args()
    _, _, dflt, dflt_fast = PROVIDERS[a.backend]
    main_model = a.model or dflt
    fast_model = a.fast_model or (main_model if a.backend == "ollama" else dflt_fast)
    print(f"[backend: {a.backend}]")
    status = key_status(a.backend)
    if status:
        print(status)
    print()
    if a.backend != "offline":                     # make sure the model really answers
        if not LLM(a.backend, fast_model)("Reply with the word ok.", [{"role": "user", "content": "ping"}],
                                      max_tokens=5):
            print("The model did not answer (see the error above), so this would only test the "
                  "offline rules. Fix the connection and run again.")
            raise SystemExit(1)
    ok = run_conversation(LLM(a.backend, main_model), LLM(a.backend, fast_model), a.replies)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
