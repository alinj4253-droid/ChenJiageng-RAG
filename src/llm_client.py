import json
import time
import urllib.request
import urllib.error
from src.config import OPENAI_API_BASE, OPENAI_API_KEY, ALL_ONLINE_MODELS


class LLMClient:
    def __init__(self, preferred_models=None):
        self.models = preferred_models if preferred_models else ALL_ONLINE_MODELS
        self._model_idx = 0

    def _next_model(self):
        model = self.models[self._model_idx % len(self.models)]
        self._model_idx += 1
        return model

    def chat(self, messages, temperature=0.2, max_tokens=1024):
        for attempt in range(len(self.models) * 2):
            use_model = self._next_model()
            url = f"{OPENAI_API_BASE}/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": use_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[LLM] {use_model} 错误: {e}")
                time.sleep(1)
                continue
        return ""

    def chat_stream(self, messages, temperature=0.2, max_tokens=1024):
        use_model = self.models[0]
        url = f"{OPENAI_API_BASE}/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": use_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=120)
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                yield delta