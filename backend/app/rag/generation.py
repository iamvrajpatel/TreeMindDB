"""Generation adapters deliberately isolate provider-specific HTTP/API behavior."""
from __future__ import annotations
import os
import httpx

class UnavailableGenerationProvider:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("No generation provider is configured")

class OllamaGenerationProvider:
    def __init__(self, model: str, base_url: str | None = None): self.model=model; self.base_url=(base_url or os.getenv("OLLAMA_BASE_URL","http://localhost:11434")).rstrip("/")
    def generate(self, prompt: str) -> str:
        response=httpx.post(f"{self.base_url}/api/generate",json={"model":self.model,"prompt":prompt,"stream":False},timeout=60); response.raise_for_status(); return response.json()["response"]

class OpenAICompatibleGenerationProvider:
    def __init__(self, model: str, base_url: str, api_key: str): self.model,self.base_url,self.api_key=model,base_url.rstrip("/"),api_key
    def generate(self, prompt: str) -> str:
        response=httpx.post(f"{self.base_url}/chat/completions",headers={"Authorization":f"Bearer {self.api_key}"},json={"model":self.model,"messages":[{"role":"user","content":prompt}]},timeout=60); response.raise_for_status(); return response.json()["choices"][0]["message"]["content"]
