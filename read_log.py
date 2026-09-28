"""
Read the companion's chat log and show WHY each reply happened.

    python read_log.py                  # last 10 messages
    python read_log.py -n 30            # last 30
    python read_log.py --full           # also show the appraisal and the instructions given to the model
    python read_log.py --file logs/other.jsonl

Every line of the log is one message: what you said, how it was read (appraisal),
which emotions moved, the full state, what was remembered, and the reply.
"""
import argparse
import json
import os

from companion_v5 import LOG_FILE


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"No log at {path} yet - chat with the companion first.")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                              # skip a half-written line
                if rec.get("event") == "reset":
                    rows.append({"reset": True})
                elif "turn" in rec:
                    rows.append(rec)
    return rows


def show(t, full):
    ap = t["appraisal"]
    away = f"  (+{t['hours_away']:.0f}h away)" if t.get("hours_away", 0) >= 1 else ""
    print(f"#{t['turn']}  {t['time']}{away}")
    print(f"  you> {t['user']}")
    tag = "" if t["reply_source"] == "model" else f"   [{t['reply_source']}]"
    print(f"  it>  {t['reply']}{tag}")
    moved = [f"{k} {v:+.2f}" for k, v in t["change"].items() if abs(v) > 0.01]
    print(f"  felt: {t['reacting']:<9} moved: {', '.join(moved) or '-'}")
    if full:
        print(f"  read by {ap.get('source', '?')}: valence {ap['valence']:+.2f}, warmth {ap['warmth']:+.2f}, "
              f"cause {ap['cause']}, about {ap['about']}, uncertainty {ap['uncertainty']:.2f}, "
              f"apology {ap['apology']}")
        st = t["state"]
        print(f"  state: mood {st['mood']:+.2f}, stress {st['stress']:.2f}, energy {st['energy']:.2f}, "
              f"trust {t['trust']:.2f}")
        if t["recalled"]:
            print("  remembered: " + " | ".join(r["gist"] for r in t["recalled"]))
        for line in t["instructions"]:
            print("    - " + line)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=LOG_FILE)
    ap.add_argument("-n", type=int, default=10, help="how many recent messages")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    rows = load(a.file)
    for t in rows[-a.n:]:
        if t.get("reset"):
            print("----- reset: companion forgot everything here -----\n")
        else:
            show(t, a.full)
    print(f"({sum(1 for t in rows if not t.get('reset'))} messages in {a.file})")


if __name__ == "__main__":
    main()
