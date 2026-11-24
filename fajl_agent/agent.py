"""Kézzel írt agent loop. Szándékosan nem a SDK tool runnerét használjuk:
a demó lényege, hogy látszódjon, mi megy oda-vissza a modell és a toolok között."""

from __future__ import annotations

import json
from typing import Callable

from .llm import Recorder, cost_usd
from .sandbox import ToolResult

CYAN, YELLOW, GREY, BOLD, RESET = "\033[36m", "\033[33m", "\033[90m", "\033[1m", "\033[0m"


def short(s: str, n: int = 600) -> str:
    return s if len(s) <= n else s[:n] + f" ...(+{len(s) - n} karakter)"


def format_answer(content) -> str:
    """Szöveg + a hivatkozások (citations) a válasz végén, forrással."""
    out, refs = [], []
    for b in content:
        if b.type != "text":
            continue
        out.append(b.text)
        for c in b.citations or []:
            src = getattr(c, "source", None) or getattr(c, "document_title", None) or "?"
            page = getattr(c, "start_page_number", None)
            refs.append(f"  [{len(refs) + 1}] {src}{f', {page}. oldal' if page else ''}: "
                        f"„{c.cited_text.strip()[:120]}”")
            out.append(f"[{len(refs)}]")
    return "".join(out) + ("\n\nForrások:\n" + "\n".join(refs) if refs else "")


def run_agent(rec: Recorder, *, system: str, tools: list[dict],
              handlers: dict[str, Callable[..., ToolResult]], user_content,
              max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": user_content}]
    total_in = total_out = 0
    for turn in range(max_turns):
        resp = rec.create(system=system, tools=tools, messages=messages)
        total_in += resp.usage.input_tokens
        total_out += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            return "[A modell elutasította a kérést.]"
        messages.append({"role": "assistant", "content": resp.content})
        calls = [b for b in resp.content if b.type == "tool_use"]
        for b in resp.content:
            if b.type == "text" and b.text.strip() and calls:
                print(f"{GREY}{b.text.strip()}{RESET}")
        if not calls:  # end_turn: kész a válasz
            print(f"{GREY}[{turn + 1} kör · {total_in} input + {total_out} output token · "
                  f"~${cost_usd(total_in, total_out):.3f}]{RESET}")
            return format_answer(resp.content)
        results = []
        for call in calls:  # párhuzamos tool-hívások: minden eredmény EGY user üzenetbe
            args = call.input
            shown = args.get("code") or json.dumps(args, ensure_ascii=False)
            print(f"{CYAN}▶ {call.name}{RESET}\n{short(shown, 1200)}")
            try:
                r = handlers[call.name](**args)
            except Exception as e:  # a tool hibája is megfigyelés a modellnek
                r = ToolResult(f"Hiba: {e}", True)
            shown_r = r.content if isinstance(r.content, str) else "\n".join(
                f"[{b['source']}] {b['content'][0]['text'][:160]}" for b in r.content)
            print(f"{YELLOW}{'✗' if r.is_error else '◀'} {short(shown_r)}{RESET}\n")
            results.append({"type": "tool_result", "tool_use_id": call.id,
                            "content": r.content, "is_error": r.is_error})
        messages.append({"role": "user", "content": results})
    return "[Elfogyott a körök száma.]"
