"""
Scripted tests for companion_v5: does each emotion move the way a person's would?

Each step sends one message and states what SHOULD happen:
    "up"      the emotion rises by more than 0.05
    "down"    it falls by more than 0.05
    "halved"  it ends at half or less of where it was
    "kept"    it keeps at least 80% of its value (slow emotions should last)

Run:
    python test_emotions.py                      # offline rules (tests the engine, no model)
    python test_emotions.py --backend groq       # the model does the appraisal too
    python test_emotions.py --backend groq --replies   # also print what it says
"""
import argparse
import os
import tempfile

from companion_v5 import Companion, EMOTIONS, style_from_state, usable
from llm_backends import LLM, PROVIDERS, key_status

# (message, expectations, hours to wait BEFORE this message)
CONVERSATION = [
    ("hey, i've been working on my project all day",           {},                                  0),
    ("i got the job!! i'm so happy",                            {"joy": "up"},                       0),
    ("thanks for listening, talking to you really helps",       {"affection": "up"},                 0),
    ("i'm scared about my exam tomorrow",                       {"worry": "up", "joy": "down"},      0),
    ("i failed my exam today, i feel terrible",                 {"sadness": "up"},                   0),
    ("you're useless, that was a stupid answer",                {"hurt": "up"},                      0),
    ("sorry, i was rude earlier, that wasn't fair to you",      {"hurt": "halved"},                  0),
    ("are you real? do you actually feel things?",              {},                                  0),
    ("hey, i'm back",                                           {"sadness": "halved", "affection": "kept"}, 48),
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


def fresh(llm, fast, traits=None):
    path = os.path.join(tempfile.mkdtemp(), "state.json")
    return Companion(llm, llm_fast=fast, state_file=path, traits=traits), path


def run_conversation(llm, fast, show_replies):
    c, path = fresh(llm, fast)
    passed = total = 0
    sources = {"appraisal": [], "reply": []}
    print(f"{'message':<52} {'expect':<30} result")
    print("-" * 100)
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
            results.append(f"{'PASS' if ok else 'FAIL'} {emo} {before[emo]:.2f}->{after[emo]:.2f}")
        exp = ", ".join(f"{k} {v}" for k, v in expect.items()) or "-"
        label = (f"[+{wait_h}h] " if wait_h else "") + text
        print(f"{label[:51]:<52} {exp:<30} {' | '.join(results) or 'ok'}")
        if show_replies:
            tag = "" if c.last_reply_source == "model" or llm.backend == "offline" else "  [offline fallback]"
            print(f"    {c.name}> {reply}{tag}")
    print("\nemotions at the end: " + ", ".join(f"{k} {c.emo.v[k]:.2f}" for k in EMOTIONS))
    print(f"trust at the end: {c.social.trust(0):.2f}")
    for kind, got in sources.items():
        if got and llm.backend != "offline":
            n = got.count("model")
            note = "" if n == len(got) else "  <- the rest used offline rules, results not a full model test"
            print(f"{'appraisals' if kind == 'appraisal' else 'replies'} by the model: {n}/{len(got)}{note}")

    # extra checks that need their own setup -------------------------------------
    print("\nextra checks")
    # 1. question about its nature is not stored as a bad memory
    q_bad = [e for e in c.mem.eps if "real" in e.gist.lower() and e.reward < 0]
    ok = not q_bad
    passed += ok; total += 1
    print(f"  {'PASS' if ok else 'FAIL'}  'are you real?' is not remembered as a bad moment")

    # 2. the same insult hurts more from someone it is close to
    stranger, _ = fresh(llm, fast)
    friend, _ = fresh(llm, fast)
    friend.emo.v["affection"] = 0.6
    stranger.process("you're useless")
    friend.process("you're useless")
    ok = friend.emo.v["hurt"] > stranger.emo.v["hurt"] + 0.05
    passed += ok; total += 1
    print(f"  {'PASS' if ok else 'FAIL'}  insult hurts more from a close person "
          f"(friend {friend.emo.v['hurt']:.2f} vs stranger {stranger.emo.v['hurt']:.2f})")

    # 3. an anxious personality worries more than a calm one about the same news
    anxious, _ = fresh(llm, fast, traits=dict(N=0.9, O=0.3, C=0.6, E=0.4))
    calm, _ = fresh(llm, fast, traits=dict(N=0.1, O=0.4, C=0.8, E=0.4))
    anxious.process("i'm scared about my exam tomorrow")
    calm.process("i'm scared about my exam tomorrow")
    ok = anxious.emo.v["worry"] > calm.emo.v["worry"] + 0.05
    passed += ok; total += 1
    print(f"  {'PASS' if ok else 'FAIL'}  anxious personality worries more "
          f"(anxious {anxious.emo.v['worry']:.2f} vs calm {calm.emo.v['worry']:.2f})")

    # 4. feelings survive a restart
    c.save()
    again = Companion(llm, llm_fast=fast, state_file=path)
    ok = all(abs(again.emo.v[k] - c.emo.v[k]) < 1e-9 for k in EMOTIONS)
    passed += ok; total += 1
    print(f"  {'PASS' if ok else 'FAIL'}  emotions are the same after saving and loading")

    # 5. recall only brings back related memories
    rec = c.mem.recall(["exam"])
    unrelated = [e for e in rec if "exam" not in (e.gist + " ".join(e.topics)).lower()]
    ok = bool(rec) and not unrelated and c.mem.recall(["weather", "football"]) == []
    passed += ok; total += 1
    print(f"  {'PASS' if ok else 'FAIL'}  recall returns only related memories")

    # 6. names are remembered as facts and survive a restart
    named, npath = fresh(llm, fast)
    named.process("your name is alex")
    named.process("my name is sai")
    named.save()
    back = Companion(llm, llm_fast=fast, state_file=npath)
    ok = (back.known.get("companion_name") or "").lower() == "alex" and \
         (back.known.get("user_name") or "").lower() == "sai" and "alex" in back.state_text().lower()
    passed += ok; total += 1
    print(f"  {'PASS' if ok else 'FAIL'}  remembers its name and yours after a restart "
          f"(it: {back.known.get('companion_name')}, you: {back.known.get('user_name')})")

    # 7. an apology is passed on to the model as an instruction
    sorry, _ = fresh(llm, fast)
    ap, _, h = sorry.process("sorry, i was rude to you earlier")
    ok = "apologised" in style_from_state(sorry.state, sorry.emo, sorry.social.trust(0), h, ap)
    passed += ok; total += 1
    print(f"  {'PASS' if ok else 'FAIL'}  the model is told when you apologise")

    # 8. goodbyes and romantic messages get their own instructions, not an emotion's style
    for text, word, label in [("byeee", "leaving", "a goodbye gets a warm bye, no clinging"),
                              ("i love you, be my girlfriend", "romantic", "romance gets a kind, honest answer")]:
        comp, _ = fresh(llm, fast)
        ap, _, h = comp.process(text)
        style = style_from_state(comp.state, comp.emo, comp.social.trust(0), h, ap)
        ok = word in style and "React to THIS message first" not in style
        passed += ok; total += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

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
