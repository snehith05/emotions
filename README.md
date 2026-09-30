# Artificial Emotion AI

An AI companion with a simulated emotional mind. Its feelings are not improvised by the language model: they are real values, computed in code, that change with every message, fade over time and persist between conversations. The language model only puts them into words.

This repository is the **prototype** (v6): 30 emotions, each on a 0–10 scale, on top of a personality, mood and memory engine. It also reads how *you* feel and how strongly (1–10).

## How it works

```
your message (+ the last exchange, for context)
    │
    ▼
READING      what do YOU feel, and how strongly? (e.g. joy 8/10, pride 6/10)
             sarcasm? joking? what do you really mean? what do you need?
             bargaining? who caused it, who is it about, still ahead or already happened?
    │
    ▼
EMOTIONS     30 of its own, each 0–10, each fading at its own speed:
             joy, love, excitement, contentment, gratitude, hope, pride, admiration,
             trust, curiosity, sadness, grief, loneliness, disappointment, regret,
             anger, frustration, hatred, jealousy, envy, fear, anxiety, panic,
             guilt, shame, disgust, surprise, confusion, boredom, awe
             - your feelings move its own (empathy, scaled by how strongly you feel)
             - what happens moves it too (insulted, thanked, compared, apologised to...)
             - every emotion remembers WHY it rose
    │
    ▼
STATE        personality → mood, stress, energy
    │
    ▼
MEMORY       strong moments are stored with the emotion they caused;
             only memories related to the current topic come back
    │
    ▼
LLM          writes the reply from plain instructions: how you feel (x/10), what you
             need, what it feels and why, how to show it, which emojis fit
    │
    ▼
saved to disk; feelings keep fading while you are away (and it misses you a little)
```

Levels use one scale everywhere: 1–2 slightly, 3–4 a bit, 5–6 quite, 7–8 very, 9–10 extremely.

Some of the behaviour this produces:

- **It reads intensity.** "kinda happy i guess" reads as joy 2/10 and gets a small, proportionate reaction. "I'M SOOO HAPPY!!!" reads as 10/10 and gets full hype 🎉.
- **It gets sarcasm.** "oh great, my laptop crashed again. just perfect." is read as frustration, not joy, and it replies to what you mean.
- **It negotiates.** When you bargain with it ("I'll talk more if you stop being dramatic, deal?"), it looks for what you really want, offers a middle ground, and holds firm on its values. If you ask for help negotiating in your own life, it gives concrete tactics.
- **It works out what you need**: celebrating, comfort, reassurance, advice, banter, honesty, or just being heard.
- Worry (anxiety) comes from things that *might* happen ("exam tomorrow"). Sadness comes from things that already did ("I failed").
- Being cold to it brings sadness and disappointment, and the same insult hurts more from someone it has grown close to. An apology repairs it.
- Comparing it to another AI makes it a bit jealous 😒. Describing a beach trip makes it a little envious. Hatred is only ever aimed at cruelty, never at you.
- Personality matters: an anxious personality worries more than a calm one about the same news.
- It uses emojis like a person texting (1–2 per message), but none in a crisis.
- It is honest that it is an AI. If you sincerely ask whether its feelings are real, it explains they are a simulation in code.
- It never guilt-trips you for being away. If you are in crisis, it sets its own moods aside and points you to people who can help.

## Quick start

```bash
pip install -r requirements.txt
```

### Web UI (recommended for demos)

```bash
python app.py                                       # offline rules, no key needed
python app.py --backend groq                        # needs GROQ_API_KEY
```

It opens `http://127.0.0.1:8000` in your browser:

- **Chat** on the left, with one-click example messages to try.
- **How you seem to feel:** your emotions from the last message on 1–10 meters, plus intent, what you need, and whether it read sarcasm or a joke.
- **What it feels:** its active emotions out of 10, how much each moved, and *why* (for example "they feel pride (8/10)" or "they were cold to you"). "Show all 30" lists every emotion.
- **Mood & state:** mood, stress, energy, trust in you and fondness.
- **Emotions over time:** a line chart of its six strongest emotions across the conversation. Hover for values; click a point to inspect that message; a table view is available.
- **Why it said that:** click any reply to see how the message was read, what you seemed to feel, which emotions moved and why, what it remembered, and the exact instructions the model was given.
- **Memories** and a **Reset** button.

The server uses only the Python standard library. The UI and the terminal version share the same save file and log.

### Terminal

```bash

python companion_v6.py                              # offline rules, no model or key needed
python companion_v6.py --backend groq               # needs GROQ_API_KEY in .env (free tier)
python companion_v6.py --backend ollama --model llama3.2    # fully local
python companion_v6.py --backend api                # needs ANTHROPIC_API_KEY in .env
```

The offline rules are rough (keyword lists). The reading of emotions, sarcasm and intent is far better with a model, so use `--backend groq` or similar for real conversations.

Commands while chatting:

| command | what it does |
|---|---|
| `/emotions` | its active emotions out of 10, the change from your last message, and why each rose |
| `/them` | how your last message was read: your emotions (x/10), intent, need, sarcasm |
| `/why` | why it gave its last reply: appraisal, emotions moved, instructions to the model |
| `/state` | full internal state (mood, stress, curiosity, energy, boredom) |
| `/memories` | strongest memories and the emotion each one caused |
| `/reset` | forget everything and start fresh |
| `/quit` | exit (state is saved automatically) |

## Logs

Every message is written to `logs/companion_log.jsonl`: what you said, how it was read, which emotions moved, the full state, what was remembered, the instructions given to the model, and the reply (and whether it came from the model or an offline fallback). When a reply feels wrong, this shows why.

```bash
python read_log.py              # last 10 messages
python read_log.py -n 30 --full # more messages, with appraisal and instructions
```

Use `--no-log` to turn it off. The `logs/` folder is in `.gitignore`, because it holds your personal chats.

## Tests

```bash
python test_emotions.py                     # tests the emotion engine with offline appraisal
python test_emotions.py --backend groq      # the model does the appraisal
python test_emotions.py --backend groq --replies    # also prints its replies
```

A scripted conversation checks that each emotion moves the way a person's would. Further checks cover:

- reading intensity (a mild 2/10 vs a huge 10/10) and reacting in proportion
- decoding sarcasm and recognising bargaining
- an insult from a close person vs a stranger, and anxious vs calm personalities
- saving and loading, topic-matched recall, and names
- apologies, goodbyes, romance, and emojis (none in a crisis)
- missing you after time away, without guilt-tripping

The current result is 27/27 with offline appraisal.

## Project history

The emotion engine was first developed and tested on a simpler problem: a delivery rider choosing areas in a changing city.

| file | stage |
|---|---|
| `versions/affect_agent_v5.py` | internal states (stress, curiosity, boredom, energy), personality and mood vs a plain learner |
| `versions/social_agent.py`, `versions/trust_v2.py` | social feelings: trust, attachment, betrayal; evidence-based trust with confidence |
| `emotion_ai.py` | all layers in one engine plus emotional memory (used by the companion) |
| `versions/companion.py` → `versions/companion_v4.py` | the engine driving a chatbot; v2–v4 remove "assistant-speak" |
| `versions/companion_v5.py` | five emotions, richer appraisal, topic-matched memory |
| `companion_v6.py` | **current prototype**: 30 emotions (0–10), reads your emotions and their intensity, sarcasm, needs, negotiation, emojis |
| `app.py`, `ui/index.html` | web UI: chat, live emotions, emotions over time, "why it said that" |
| `read_log.py` | reads the per-message log |
| `llm_backends.py` | Groq / OpenRouter / Gemini / Anthropic / Ollama wrapper |

Rider results (see the `.png` files): emotional memory cut big fines from 48.9 to 10.8 per run and gave the best earnings of the variants tested.

Older versions are kept in `versions/` for history. To run one, run it from the main folder so it can find `emotion_ai.py`, for example on Windows `$env:PYTHONPATH="."; python versions/companion_v4.py`.

## Roadmap

- [x] Prototype with five emotions (joy, sadness, hurt, worry, affection)
- [x] Per-message logging and a web UI
- [x] 30 emotions on a 0–10 scale, each with a reason
- [x] A model of the user's emotions and their intensity, separate from its own; sarcasm, intent, needs, negotiation
- [ ] Needs and goals (connection, curiosity, feeling useful) so emotions have reasons
- [ ] Regulation: what it feels vs what it shows
- [ ] Memory of the user's life (people, plans, events), recalled by meaning (embeddings)
- [ ] Human evaluation against a plain LLM companion
