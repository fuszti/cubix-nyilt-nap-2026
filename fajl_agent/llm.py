"""Claude-hívások egy helyen: kulcs, modell, költség, és felvétel/visszajátszás a demóhoz.

Kulcs (az első, ami megvan), környezeti változóból vagy a repó gyökerében lévő .env fájlból:
  OPENROUTER_API_KEY  a Claude-ot az OpenRouter Anthropic-kompatibilis végpontján hívjuk
  ANTHROPIC_API_KEY   közvetlenül a Claude API-t hívjuk

DEMO_MODE=live    (alap) élő hívás
DEMO_MODE=record  élő hívás + a válaszok mentése a recordings/ mappába
DEMO_MODE=replay  nincs hálózat: a korábban felvett válaszokat játssza vissza
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
from anthropic.types.beta import BetaMessage

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def _load_env_file() -> None:
    """Egyszerű .env olvasó (soronként KULCS=érték). A már beállított változót nem írja felül."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            v = v.strip().strip("\"'")
            if sep and v and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v)


_load_env_file()
OPENROUTER = bool(os.getenv("OPENROUTER_API_KEY"))
MODEL = os.getenv("DEMO_MODEL", "anthropic/claude-opus-5.5" if OPENROUTER else "claude-opus-5-5")
EFFORT = os.getenv("DEMO_EFFORT", "medium")
MODE = os.getenv("DEMO_MODE", "live")
REC_DIR = ROOT / "recordings"

# USD / 1M token (input, output), a Claude API listaárai 2026 novemberében (az OpenRouteren ugyanennyi)
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if OPENROUTER:
            # Ugyanaz az SDK és ugyanaz a kérésformátum, csak az OpenRouter végpontján.
            # Az explicit auth_token miatt az SDK nem olvas ANTHROPIC_* kulcsot a környezetből.
            _client = anthropic.Anthropic(base_url="https://openrouter.ai/api",
                                          auth_token=os.environ["OPENROUTER_API_KEY"])
        else:
            _client = anthropic.Anthropic()
    return _client


def _backend_args() -> dict:
    """Az OpenRouteren az Anthropic szolgáltatót kérjük elsőnek: a PDF-et és a search_result
    hivatkozásokat ott mértük ki. A Claude API-n szerveroldali fallback: ha a modell elutasítja
    a kérést, az API ugyanabban a hívásban újrafuttatja egy fallback modellel."""
    if OPENROUTER:
        return {"extra_body": {"provider": {"order": ["anthropic"]}}}
    return {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}


class Recorder:
    """Egy demó API-válaszait sorban menti / visszajátssza (JSONL)."""

    def __init__(self, name: str):
        self.path = REC_DIR / f"{name}.jsonl"
        self.replay = []
        if MODE == "replay":
            if not self.path.exists():
                raise SystemExit(f"Nincs felvétel: {self.path}. Futtasd előbb DEMO_MODE=record módban.")
            self.replay = [json.loads(l) for l in self.path.read_text().splitlines() if l]
        elif MODE == "record":
            REC_DIR.mkdir(exist_ok=True)
            self.path.write_text("")

    def _save(self, kind: str, data: dict) -> None:
        if MODE == "record":
            with self.path.open("a") as f:
                f.write(json.dumps({"kind": kind, "data": data}, ensure_ascii=False) + "\n")

    def _next(self, kind: str) -> dict:
        item = self.replay.pop(0)
        assert item["kind"] == kind, f"A felvétel sorrendje eltér: {item['kind']} != {kind}"
        return item["data"]

    def create(self, **kwargs) -> BetaMessage:
        if MODE == "replay":
            return BetaMessage.model_validate(self._next("message"))
        resp = client().beta.messages.create(
            model=MODEL, max_tokens=16000, output_config={"effort": EFFORT}, **_backend_args(), **kwargs,
        )
        self._save("message", resp.to_dict())
        return resp

    def count_tokens(self, **kwargs) -> int:
        """Tokenszám. A Claude API-n generálás nélkül, ingyen (count_tokens). Az OpenRouteren nincs
        ilyen végpont, ott egy rövid, fizetős kérés usage-éből olvassuk ki. Az elutasítást is felvesszük."""
        if MODE == "replay":
            data = self._next("count")
            if "error" in data:
                raise RuntimeError(data["error"])
            return data["input_tokens"]
        try:
            if OPENROUTER:
                n = client().beta.messages.create(model=MODEL, max_tokens=16, **_backend_args(),
                                                  **kwargs).usage.input_tokens
            else:
                n = client().messages.count_tokens(model=MODEL, **kwargs).input_tokens
        except anthropic.APIStatusError as e:
            self._save("count", {"error": e.message})
            raise RuntimeError(e.message) from e
        self._save("count", {"input_tokens": n})
        return n


def cost_usd(input_tokens: int, output_tokens: int = 0) -> float:
    key = MODEL.split("/")[-1].replace(".", "-")  # anthropic/claude-opus-5.5 -> claude-opus-5-5
    p_in, p_out = PRICES.get(key, PRICES["claude-opus-5-5"])
    return input_tokens / 1e6 * p_in + output_tokens / 1e6 * p_out


def run_demo(main) -> None:
    """Érthető hibaüzenet, ha nincs API-kulcs vagy hálózat."""
    try:
        main()
    except (anthropic.AuthenticationError, anthropic.APIConnectionError, TypeError) as e:
        if isinstance(e, TypeError) and "api_key" not in str(e).lower() and "auth" not in str(e).lower():
            raise
        raise SystemExit(f"Nem érem el a Claude-ot ({type(e).__name__}).\n"
                         f"Írd be a kulcsot ide: {ENV_FILE} (OPENROUTER_API_KEY=... vagy ANTHROPIC_API_KEY=...),\n"
                         "vagy futtasd felvételből: DEMO_MODE=replay")
