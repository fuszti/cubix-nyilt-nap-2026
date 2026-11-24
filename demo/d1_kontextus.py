"""1. út: a fájl közvetlenül a kontextusba kerül (document block).

  uv run python demo/d1_kontextus.py
"""

import base64
import json
import re

from fajl_agent.agent import format_answer
from fajl_agent.corpus import DATA
from fajl_agent.data_gen import new_pdf, write_markdown_pdf
from fajl_agent.llm import MODEL, Recorder, cost_usd, run_demo

# A nagy fájlokból csak egy mintát mérünk meg, és felszorozzuk. A Claude API-n ez ingyenes
# (count_tokens), az OpenRouteren viszont valódi kérés, ezért nem küldjük el az egészet.
SAMPLE_PAGES = 10
SAMPLE_ROWS = 1000


def pdf_block(pdf: bytes, title: str) -> dict:
    data = base64.standard_b64encode(pdf).decode()
    return {
        "type": "document",
        "title": title,
        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
    }


def show_request(content) -> None:
    """A nyers API-kérés, a base64 rövidítve: ennyi a 'fájlfeltöltés'."""
    shown = json.loads(json.dumps(content))
    for b in shown:
        if b["type"] == "document":
            d = b["source"]["data"]
            b["source"]["data"] = f"{d[:24]}... ({len(d):,} karakter base64)"
    req = {"model": MODEL, "messages": [{"role": "user", "content": shown}]}
    print(json.dumps(req, ensure_ascii=False, indent=2))


def sample_pdf(pages: int) -> bytes:
    """Az első tájékoztatókból ugyanúgy épített PDF, mint a nagy (minden tájékoztató kétszer)."""
    pdf = new_pdf()
    for md in sorted((DATA / "tajekoztatok").glob("*.md"))[: pages // 2]:
        for _ in range(2):
            write_markdown_pdf(pdf, md.read_text(encoding="utf-8"))
    return bytes(pdf.output())


def main() -> None:
    rec = Recorder("d1_kontextus")
    question = "Mennyi a tandíj, és meddig kell befizetni?"

    print("\n=== 1) Kis PDF (1 oldal) a messages tömbben ===")
    small = DATA / "docs" / "tanfolyami-tajekoztato.pdf"
    content = [pdf_block(small.read_bytes(), small.name), {"type": "text", "text": question}]
    show_request(content)
    resp = rec.create(messages=[{"role": "user", "content": content}])
    print("\n" + format_answer(resp.content))
    u = resp.usage
    print(f"\n[{u.input_tokens:,} input + {u.output_tokens:,} output token · "
          f"~${cost_usd(u.input_tokens, u.output_tokens):.3f}]")

    print("\n=== 2) Ugyanez a 240 oldalas PDF-fel (csak tokenszám, válasz nélkül) ===")
    big = DATA / "docs" / "tajekoztatok-osszes.pdf"
    pages = len(re.findall(rb"/Type\s*/Page\b(?!s)", big.read_bytes()))
    pdf, factor = sample_pdf(SAMPLE_PAGES), pages / SAMPLE_PAGES
    note = f" (becslés: {SAMPLE_PAGES} oldalt mértünk meg, ×{factor:.0f})"
    try:
        n = round(factor * rec.count_tokens(messages=[{"role": "user", "content": [
            pdf_block(pdf, big.name), {"type": "text", "text": question}]}]))
        print(f"{n:,} input token{note} · ~${cost_usd(n):.2f} MINDEN egyes kérdésnél, "
              f"és a beszélgetés minden további körében újra (prompt caching nélkül)")
    except RuntimeError as e:
        print(f"Az API elutasította: {e}")

    print("\n=== 3) A 30 ezer soros vizsgaeredmény-CSV szövegként ===")
    csv_text = (DATA / "tables" / "vizsgaeredmenyek.csv").read_text(encoding="utf-8")
    lines = csv_text.splitlines(keepends=True)
    csv_text, factor = "".join(lines[: SAMPLE_ROWS + 1]), (len(lines) - 1) / SAMPLE_ROWS
    note = f" (becslés: {SAMPLE_ROWS:,} sort mértünk meg, ×{factor:.0f})"
    try:
        n = round(factor * rec.count_tokens(messages=[{"role": "user", "content": [
            {"type": "document", "title": "vizsgaeredmenyek.csv",
             "source": {"type": "text", "media_type": "text/plain", "data": csv_text}},
            {"type": "text", "text": "Hány százalék a sikeres forgalmi vizsgák aránya?"}]}]))
        print(f"{n:,} input token{note} · ~${cost_usd(n):.2f} kérdésenként")
    except RuntimeError as e:
        print(f"Az API elutasította: {e}")


if __name__ == "__main__":
    run_demo(main)
