"""
LLM backends shared by the companion versions.
One small wrapper that talks to Groq, OpenRouter, Gemini, Anthropic or a local Ollama model.
If the model is unreachable it returns None and the caller falls back to offline rules.
"""
import json
import re
import time
import os
import urllib.error
import urllib.request

# ============================================================ LLM BACKENDS
# provider -> (base url, env var holding the key, default model, default cheap model)
PROVIDERS = {
    "groq":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY",
                   "qwen/qwen3.8-27b", "qwen/qwen3.8-27b"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                   "meta-llama/llama-3.3-70b-instruct:free", "meta-llama/llama-3.3-70b-instruct:free"),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
                   "gemini-2.0-flash", "gemini-2.0-flash-lite"),
    "api":        ("", "ANTHROPIC_API_KEY", "claude-sonnet-5", "claude-haiku-4-5-20251001"),
    "ollama":     ("", "", "llama3.2", "llama3.2"),
    "offline":    ("", "", "-", "-"),
}


def strip_thinking(text):
    """Some models write their reasoning inside <think>...</think>; keep only the answer."""
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()


class LLM:
    def __init__(self, backend="offline", model="llama3.2"):
        self.backend, self.model = backend, model

    def __call__(self, system, messages, max_tokens=300, retries=3, json_mode=False):
        self.json_mode = json_mode
        for attempt in range(retries + 1):
            try:
                if self.backend == "ollama":
                    return self._ollama(system, messages, max_tokens)
                if self.backend == "api":
                    return self._anthropic(system, messages, max_tokens)
                if self.backend in PROVIDERS and self.backend != "offline":
                    return self._openai_compatible(system, messages, max_tokens)
                return None
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < retries:  # busy or rate limited: wait, retry
                    wait = e.headers.get("retry-after") if e.headers else None
                    try:
                        wait = min(max(float(wait), 1.0), 60.0)
                    except (TypeError, ValueError):
                        wait = 5.0 * 2 ** attempt                  # 5s, 10s, 20s
                    why = "rate limit" if e.code == 429 else "server busy"
                    print(f"  ({why} - waiting {wait:.0f}s)")
                    time.sleep(wait)
                    continue
                body = e.read().decode()[:200] if hasattr(e, "read") else ""
                print(f"  (model error {e.code}: {body})")
            except Exception as e:                       # network/key problem -> offline fallback
                print(f"  (model unavailable: {e})")
            return None                                  # caller falls back to offline rules
        return None

    def _openai_compatible(self, system, messages, max_tokens):
        """Works with Groq, OpenRouter, Gemini, and anything else speaking the OpenAI format."""
        base, env, _, _ = PROVIDERS[self.backend]
        key = os.environ.get(env, "")
        if not key:
            raise RuntimeError(f"set {env} first")
        payload = {"model": self.model, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}] + messages}
        if getattr(self, "json_mode", False) and self.backend == "groq":
            payload["response_format"] = {"type": "json_object"}   # model must return valid JSON
        m = self.model.lower()
        if self.backend == "groq" and "qwen3" in m:
            payload["reasoning_effort"] = "none"          # chat, not puzzle-solving: skip thinking
        elif "gpt-oss" in m:
            payload["reasoning_effort"] = "low"           # these always think; keep it short and
            payload["max_tokens"] = max_tokens + 600      # leave room so the answer isn't cut off
        out = self._post(base + "/chat/completions", payload, {"Authorization": f"Bearer {key}"})
        return strip_thinking(out["choices"][0]["message"].get("content") or "")

    def _post(self, url, payload, headers):
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "emotion-ai/0.1", **headers})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())

    def _ollama(self, system, messages, max_tokens):
        msgs = [{"role": "system", "content": system}] + messages
        try:                                    # native Ollama endpoint
            out = self._post("http://localhost:11434/api/chat",
                             {"model": self.model, "stream": False, "messages": msgs,
                              "options": {"num_predict": max_tokens}}, {})
            return strip_thinking(out["message"]["content"])
        except Exception:                       # newer Ollama also speaks the OpenAI format
            out = self._post("http://localhost:11434/v1/chat/completions",
                             {"model": self.model, "messages": msgs, "max_tokens": max_tokens},
                             {"Authorization": "Bearer ollama"})
            return strip_thinking(out["choices"][0]["message"].get("content") or "")

    def _anthropic(self, system, messages, max_tokens):
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        out = self._post("https://api.anthropic.com/v1/messages",
                         {"model": self.model, "max_tokens": max_tokens,
                          "system": system, "messages": messages},
                         {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in out["content"]).strip()

