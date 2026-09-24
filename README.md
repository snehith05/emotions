# Artificial Emotion AI

An AI companion with a simulated emotional mind. Its feelings are not improvised by the language model: they are real values, computed in code, that change with every message, fade over time and persist between conversations. The language model only puts them into words.

This repository is the **prototype**: five emotions on top of a personality, mood and memory engine. More complex emotions come later.

## How it works

```
your message
    │
    ▼
APPRAISAL    good or bad? kind or cold to me? who caused it? who is it about?
             still ahead (worry) or already happened (sadness)? an apology?
    │
    ▼
EMOTIONS     joy · sadness · hurt · worry        fast: rise on triggers, fade every turn
             affection                            slow: builds over many good moments
    │
    ▼
STATE        personality → mood → stress, curiosity, energy, boredom, trust
    │
    ▼
MEMORY       strong moments are stored with the emotion they caused;
             only memories related to the current topic come back
    │
    ▼
LLM          writes the reply, guided by the strongest emotions
    │
    ▼
saved to disk; feelings keep fading while you are away
```

Some of the behaviour this produces:

- Good news brings joy, and hard news pushes earlier happiness aside.
- Worry comes from things that *might* happen ("exam tomorrow"). Sadness comes from things that already did ("I failed").
- Being cold to it causes hurt, and the same insult hurts more from someone it has grown close to.
- An apology repairs hurt, and it forgives faster when affection is high.
- Personality matters: an anxious personality worries more than a calm one about the same news.
- It is honest that it is an AI. If you sincerely ask whether its feelings are real, it explains they are a simulation in code.
- It never guilt-trips you for being away. If you are in crisis, it sets its own moods aside and points you to people who can help.

## Quick start

```bash
pip install -r requirements.txt

python companion_v5.py                              # offline rules, no model or key needed
python companion_v5.py --backend groq               # export GROQ_API_KEY=...   (free tier)
python companion_v5.py --backend ollama --model llama3.2    # fully local
python companion_v5.py --backend api                # export ANTHROPIC_API_KEY=...
```

Commands while chatting:

| command | what it does |
|---|---|
| `/emotions` | the five emotions and trust as bars, with the change from your last message |
| `/state` | full internal state (mood, stress, curiosity, energy, boredom) |
| `/memories` | strongest memories and the emotion each one caused |
| `/reset` | forget everything and start fresh |
| `/quit` | exit (state is saved automatically) |

## Tests

```bash
python test_emotions.py                     # tests the emotion engine with offline appraisal
python test_emotions.py --backend groq      # the model does the appraisal
python test_emotions.py --backend groq --replies    # also prints its replies
```

A scripted conversation checks that each emotion moves the way a person's would. Further checks cover hurt from a close person vs a stranger, anxious vs calm personalities, saving and loading, and topic-matched recall. The current result is 14/14 with offline appraisal.

## Project history

The emotion engine was first developed and tested on a simpler problem: a delivery rider choosing areas in a changing city.

| file | stage |
|---|---|
| `versions/affect_agent_v5.py` | internal states (stress, curiosity, boredom, energy), personality and mood vs a plain learner |
| `versions/social_agent.py`, `versions/trust_v2.py` | social feelings: trust, attachment, betrayal; evidence-based trust with confidence |
| `emotion_ai.py` | all layers in one engine plus emotional memory (used by the companion) |
| `versions/companion.py` → `versions/companion_v4.py` | the engine driving a chatbot; v2–v4 remove "assistant-speak" |
| `companion_v5.py` | **current prototype**: five emotions, richer appraisal, topic-matched memory |
| `llm_backends.py` | Groq / OpenRouter / Gemini / Anthropic / Ollama wrapper |

Rider results (see the `.png` files): emotional memory cut big fines from 48.9 to 10.8 per run and gave the best earnings of the variants tested.

Older versions are kept in `versions/` for history. To run one, run it from the main folder so it can find `emotion_ai.py`, for example on Windows `$env:PYTHONPATH="."; python versions/companion_v4.py`.

## Roadmap

- [x] Prototype with five emotions (joy, sadness, hurt, worry, affection)
- [ ] Needs and goals (connection, curiosity, feeling useful) so emotions have reasons
- [ ] A model of the user's emotions, separate from its own (empathy, not just contagion)
- [ ] Regulation: what it feels vs what it shows
- [ ] Memory of the user's life (people, plans, events), recalled by meaning (embeddings)
- [ ] Human evaluation against a plain LLM companion
