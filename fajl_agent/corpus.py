"""A demók közös adatai: a tájékoztató-korpusz indexe, a search tool és a "valódi" válaszok."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .chunking import chunk_markdown_sections
from .index import HybridIndex
from .sandbox import ToolResult

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@lru_cache
def doc_index() -> HybridIndex:
    chunks = []
    for p in sorted((DATA / "tajekoztatok").glob("*.md")):
        chunks += chunk_markdown_sections(p.stem, p.read_text(encoding="utf-8"))
    return HybridIndex(chunks)


SEARCH_TOOL = {
    "name": "search_documents",
    "description": (
        "Szemantikus + kulcsszavas keresés az autósiskolák tanfolyami tájékoztatóinak SZÖVEGÉBEN. "
        "A legrelevánsabb néhány részletet adja vissza forrással. Jó: mit mond egy tájékoztató, melyik pontban. "
        "Nem jó: számolás, összesítés, 'hány darab', 'összesen mennyi' (csak top-k találatot lát)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keresőkifejezés magyarul"},
            "k": {"type": "integer", "description": "Találatok száma (1-10)", "default": 5},
        },
        "required": ["query"],
    },
}


def search_documents(query: str, k: int = 5) -> ToolResult:
    hits = doc_index().search(query, k=min(max(k, 1), 10))
    if not hits:
        return ToolResult("Nincs találat erre a kifejezésre a tájékoztatókban.")
    # Natív search_result blokkok: a modell ezekre tud hivatkozni (citations).
    blocks = [{
        "type": "search_result",
        "source": f"{h.chunk.meta['doc_id']}.md",
        "title": f"{h.chunk.meta['title']} – {h.chunk.meta['section']}",
        "content": [{"type": "text", "text": h.chunk.text}],
        "citations": {"enabled": True},
    } for h in hits]
    return ToolResult(blocks)


def registry() -> pd.DataFrame:
    return pd.read_excel(DATA / "tables" / "iskola-nyilvantartas.xlsx")
