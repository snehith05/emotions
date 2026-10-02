# Artificial Emotion AI

An AI with a simulated emotional mind. Its feelings are not improvised by the language model: they are real values, computed in code, that change with every message, fade over time and persist between conversations. The language model only puts them into words.

This repository is the **prototype** (v7). It has 30 emotions on a 0–10 scale, on top of a personality, mood and memory engine. It reads how *you* feel and how strongly (1–10). Version 7 adds three things:

- **Personas built from data.** Give it a folder of everything a person wrote or said, and it speaks *as them*. Each reply is grounded in the passages most related to your message (RAG), and the person's emotional profile is worked out from the same data. Gandhi is included, built from ~141,000 words of his own public-domain writings.
- **An emotion dial.** Choose how emotional the persona is, from **off** to **10**, and set a ceiling for any single emotion ("anger never above 3").
- **Voices.** Choose how the feelings are put into words: Gen Z, millennial, desi Hinglish or Bollywood-filmy. This is independent of who it is and what it feels.

## How it works

```
your message (+ the last exchange, for context)
    │
    ▼
READING      what do YOU feel, and how strongly? (joy 8/10, pride 6/10)
             sarcasm? joking? what do you need? bargaining? who caused it?
             + which of THIS persona's triggers does it touch? what to look up in their writings?
    │
    ▼
EMOTIONS     30 of its own, each 0–10, each fading at its own speed, each with a reason:
               dial          0 = off, 1–10 = how emotional (5 = a typical person)
               limits        a ceiling per emotion, set by you or by the persona
               baseline      what the persona feels on an ordinary day; emotions settle back to it
               sensitivity   what stirs this person easily and what hardly at all
               triggers      "violence or revenge -> sadness 6, frustration 4", from their data
               mood          a low mood makes bad news hit harder, a good mood the reverse
               opposites     rising joy eases sadness, rising hope eases fear...
               mixed         bittersweet, protective worry, nervous hope, sorrow with faith...
               regulation    what they FEEL vs what they let SHOW (a restrained person shows less)
               conditioning  topics that came with a feeling stir it again ("exam" -> anxiety)
    │
    ▼
STATE        personality → mood, stress, energy
    │
    ▼
MEMORY       every meaningful moment, facts about you, and reflections it writes about you;
             recalled by relevance + importance + recency + current mood; forgetting curve
    │
    ▼
KNOWLEDGE    passages of the persona's own writings that relate to your message (RAG)
    │
    ▼
VOICE        how this voice texts the feeling it has right now (style examples by emotion and level)
    │
    ▼
LLM          speaks as the persona: their voice, their era, these feelings, these passages
```

Levels use one scale everywhere: 1–2 slightly, 3–4 a bit, 5–6 quite, 7–8 very, 9–10 extremely.

## Quick start

```bash
pip install -r requirements.txt
```

Put your free Groq key in a file named `.env` in the project folder (see `.env.example`):

```
GROQ_API_KEY=gsk_...
```

### Web UI (recommended)

```bash
python app.py --backend groq                                     # the original companion
python app.py --backend groq --persona gandhi                    # talk to Gandhi
python app.py --backend groq --persona gandhi --intensity 8 --voice genz
```

It opens in your browser. Everything can also be changed live in the **Who and how** card:

- **Talking to:** the persona. Each persona keeps its own feelings, memories and settings.
- **Emotions:** the dial, from Off to 10. The persona's natural level is shown next to it (Gandhi's is 4).
- **Texts like:** the voice, or "their own way".
- **Emotion limits:** a ceiling for any of the 30 emotions. The persona's own limits are shown locked (Gandhi never goes above 1 in hatred).
- **From their writings:** the passages it was given for the last reply.
- **What it feels:** each emotion, why it rose, mixed feelings, and how much it lets show.
- **Why it said that:** click any reply to see the reading, the emotions that moved, the triggers it touched, the memories and passages it used, and the exact instructions the model got.
- **Memories:** facts about you, its reflections on you, the topics that stir feelings, and the moments it keeps (with when, and how often recalled).

### Terminal

```bash
python companion_v7.py --backend groq --persona gandhi --intensity 6
```

| command | what it does |
|---|---|
| `/dial 7`, `/dial off` | how emotional it is |
| `/limit anger 3`, `/limit anger off` | a ceiling for one emotion |
| `/limits` | the dial, the limits and the voice |
| `/voice genz`, `/voice none` | how it texts |
| `/sources` | the passages of their writings used for the last reply |
| `/emotions` | its emotions out of 10, the change, why each rose, mixed feelings |
| `/them` | how your last message was read |
| `/why` | why it gave its last reply |
| `/memories`, `/facts` | what it remembers, and the topic → feeling links |
| `/persona` | who it is, and which personas exist |
| `/reset`, `/quit` | forget everything / exit |

Without `--backend` it runs on the offline rules. These are rough keyword rules that are good for testing the engine; a persona then answers by quoting its most relevant passage.

## Personas: a person from their data

```
personas/gandhi/
    data/                      the person's own words: .txt / .md, or .jsonl with {"text": ...}
    persona.json               who they are and how they feel (hand-written, optional)
    persona.generated.json     the same, worked out from data/ by a model
```

Make your own:

```bash
python persona.py new marie --name "Marie Curie"          # makes personas/marie/
# put their letters, speeches, interviews, books... into personas/marie/data/
python persona.py build marie --backend groq              # works out the emotional profile from the data
python persona.py search marie "what drives you"          # see what the knowledge search finds
python app.py --backend groq --persona marie
```

`build` reads samples spread across the data. It notes the person's values, beliefs, how they react and how they speak, then turns the notes into a profile:

| key | what it does |
|---|---|
| `summary`, `era`, `values`, `voice`, `examples` | who they are and how they talk; `era` stops them "knowing" later events |
| `traits` | Big Five (N, O, C, E): how hard good and bad things hit, mood, memory |
| `default_intensity` | their natural place on the dial |
| `baseline` | emotions they carry on an ordinary day |
| `sensitivity` | 1 = typical, 0.2 = hardly stirred, 2 = very easily |
| `caps` | emotions they never let past a level |
| `display` | how much of a feeling they let show (Gandhi shows 40% of his anger) |
| `expression` | how *they* show each emotion, in words |
| `triggers` | situations that move them, with keywords and the feelings they cause |

At the persona's natural intensity, a trigger produces exactly the levels in the profile. Turning the dial up or down scales them smoothly. Anything in `persona.json` overrides the generated file, so you can correct the model's guesses. For Gandhi the generated profile was reviewed, and the triggers, limits and expressions were set by hand in `persona.json`.

More data gives a more faithful persona: the knowledge search covers every passage, however much you add. Gandhi's data is 940 passages (~141k words), and searching it takes about a millisecond.

**Use data you have the right to use.** Public-domain writings, your own material, or data about someone who has agreed to it. The persona is honest that it is an AI simulation whenever someone sincerely asks.

## Voices: how it texts

```bash
python voices.py list
python voices.py show genz sadness 7                      # the examples it would use for sadness 7/10
python voices.py generate genz --n 1000 --backend groq    # write 1000 more original examples
```

A voice is `voices/<id>/voice.json` (rules, emoji habits, the voice's own words for low, mid and high feelings) plus example messages in `data/*.jsonl`:

```json
{"emotion": "sadness", "level": 7, "text": "ngl i'm not okay rn. everything just feels heavy"}
```

For each reply it picks the examples closest to what it feels right now (same emotion, nearest level) and shows them to the model as style, never to copy. Included voices: `genz`, `millennial`, `desi` (Hinglish with Kannada and Tamil slang), and `filmy` (Bollywood-style, with original lines only). Each ships with a hand-written seed set. `generate` grows a voice to thousands of original examples across all 30 emotions × 3 levels. On Groq's free tier, 1000 examples take about half an hour.

To ground a voice in real conversations, add any chat or dialogue data you are allowed to use as more `.jsonl` files. Examples are your own chats (with the other person's consent), or research datasets such as EmpatheticDialogues, which is labelled with 32 emotions (check each dataset's licence; many are non-commercial). Film scripts and private chats scraped from the web are not suitable: the first are copyrighted and the second were never consented to.

## Memory

- **Moments:** every meaningful message is kept with the emotions it caused and how important it was. Strong, emotional, surprising moments matter more.
- **Recall:** BM25 search with stemming and synonyms finds memories by meaning ("mother" finds "mom"). Results are ranked by relevance, importance, recency, and whether they match its current mood (when sad, sad memories surface more easily). Only related memories come back.
- **Forgetting:** a dull moment fades in about a week and an important one in weeks. Every recall makes a memory last longer (the spacing effect).
- **Facts:** repeated facts merge instead of piling up, and the ones related to your message are recalled.
- **Reflections:** every 12 messages the model looks back and writes 1–3 insights about you.
- **Conditioning:** the words of each message are linked to the feelings they came with. A topic that hurt before stirs a little of that feeling again, and the link fades if the topic stops hurting.

A v6 save is brought over automatically the first time v7 starts.

## Logs

Every message is written to `logs/companion_log.jsonl`. Each record has what you said, how it was read, which emotions moved and why, the triggers, the mixed feelings, the memories and passages used, the dial and voice, the instructions given to the model, and the reply.

```bash
python read_log.py              # last 10 messages
python read_log.py -n 30 --full
```

Use `--no-log` to turn it off. `logs/` is in `.gitignore`.

## Tests

```bash
python test_emotions.py                     # the engine, with offline reading
python test_emotions.py --backend groq      # the model does the reading too
```

There are 38 checks. A scripted conversation checks that each emotion moves the way a person's would. Other checks cover intensity, sarcasm, closeness, personality, saving and recall, names, apologies, goodbyes, romance, bargaining, emojis and missing you. The v7 checks cover the dial, limits, persona knowledge, triggers and limits, regulation, recall by meaning, conditioning, facts, saved settings, voices and mixed feelings. The current result is 38/38 offline.

## Project history

| file | stage |
|---|---|
| `versions/affect_agent_v5.py` | internal states, personality and mood vs a plain learner (delivery-rider simulation) |
| `versions/social_agent.py`, `versions/trust_v2.py` | trust, attachment, betrayal |
| `emotion_ai.py` | all layers in one engine, plus emotional memory |
| `versions/companion.py` → `versions/companion_v5.py` | the engine driving a chatbot |
| `companion_v6.py` | 30 emotions (0–10), reading your emotions, sarcasm, needs, negotiation, emojis |
| `companion_v7.py` | **current**: personas, the dial, limits, regulation, mixed feelings, conditioning, voices |
| `persona.py`, `retrieval.py` | personas from data, BM25 search (no extra packages) |
| `memory.py` | moments, facts, reflections, forgetting, mood-congruent recall, conditioning |
| `voices.py`, `voices/` | texting voices |
| `app.py`, `ui/index.html` | web UI |
| `llm_backends.py` | Groq / OpenRouter / Gemini / Anthropic / Ollama |
| `mood_research/` | separate study: a reproduction of *StudentLife* (Wang et al., UbiComp 2014) on sleep, activity, light and mood, plus an ML extension - see its own README |

## Roadmap

- [x] Prototype with five emotions
- [x] Per-message logging and a web UI
- [x] 30 emotions on a 0–10 scale, each with a reason
- [x] A model of the user's emotions and their intensity; sarcasm, intent, needs, negotiation
- [x] Regulation: what it feels vs what it shows
- [x] Memory of the user's life, recalled by meaning
- [x] Personas built from data (RAG), an emotion dial and limits, texting voices
- [ ] Embedding search alongside BM25, for recall by meaning beyond shared words
- [ ] Needs and goals (connection, curiosity, feeling useful) so emotions have deeper reasons
- [ ] Human evaluation: does a persona with the emotion engine feel more like the real person than a plain LLM role-play?
