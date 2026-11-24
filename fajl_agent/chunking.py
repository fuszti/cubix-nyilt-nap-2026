"""Chunkolási stratégiák: szöveghez és táblázathoz.

A tanulság: a chunk a keresés egysége. Ami egy chunkból kimarad (a dokumentum
címe, a táblázat fejléce), azt a keresés nem fogja megtalálni.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Chunk:
    id: str
    text: str
    meta: dict = field(default_factory=dict)


# ---------- Szöveg ----------

def chunk_fixed(doc_id: str, text: str, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Naiv: fix karakterszámonként vág, a struktúrát nem nézi."""
    chunks, i, n = [], 0, 0
    while i < len(text):
        chunks.append(Chunk(f"{doc_id}#{n}", text[i:i + size], {"doc_id": doc_id}))
        i += size - overlap
        n += 1
    return chunks


def chunk_markdown_sections(doc_id: str, text: str, max_chars: int = 1200) -> list[Chunk]:
    """Struktúra szerint vág (## fejezetek), és minden chunk elejére odaírja,
    melyik dokumentumból és fejezetből jön (contextual header)."""
    title = text.splitlines()[0].lstrip("# ").strip()
    intro = next((l for l in text.splitlines() if l.startswith("**")), "")  # pl. **Iskola:** ...
    parts = re.split(r"\n(?=## )", text)
    chunks = []
    for n, part in enumerate(parts[1:]):  # parts[0] a cím + a bevezető sor
        section = part.splitlines()[0].lstrip("# ").strip()
        body = "\n".join(part.splitlines()[1:]).strip()
        header = f"[{title} | {intro.replace('**', '')} | {section}]"
        chunks.append(Chunk(
            f"{doc_id}#{n}", f"{header}\n{body}"[:max_chars],
            {"doc_id": doc_id, "title": title, "section": section},
        ))
    return chunks


# ---------- Táblázat ----------

def chunk_table_naive(table_name: str, csv_text: str, size: int = 500) -> list[Chunk]:
    """A CSV-t sima szövegként vágja: sorok kettévágódnak, a fejléc csak az elsőben van."""
    return [
        Chunk(f"{table_name}#{n}", csv_text[i:i + size], {"table": table_name})
        for n, i in enumerate(range(0, len(csv_text), size))
    ]


def chunk_table_rows(table_name: str, df: pd.DataFrame, rows_per_chunk: int = 1,
                     prefix: bool = False) -> list[Chunk]:
    """Soronként (vagy pár soronként) vág, és minden sorba beírja az oszlopneveket.
    Így a 'vezetett_orak: 42' a chunkban marad, nem egy gazdátlan '42'."""
    chunks = []
    cols = list(df.columns)
    for start in range(0, len(df), rows_per_chunk):
        rows = df.iloc[start:start + rows_per_chunk]
        lines = ["; ".join(f"{c}: {r[c]}" for c in cols) for _, r in rows.iterrows()]
        chunks.append(Chunk(
            f"{table_name}#r{start}", (f"[{table_name}]\n" if prefix else "") + "\n".join(lines),
            {"table": table_name, "row": start},
        ))
    return chunks


def table_card(table_name: str, df: pd.DataFrame) -> Chunk:
    """Egyetlen 'adatlap' chunk a táblázatról: séma, sorszám, tartományok.
    Ezt a kereső megtalálja, és az agent tudja: ehhez a kérdéshez a sandbox kell."""
    lines = [f"[TÁBLÁZAT: {table_name}] {len(df)} sor, {len(df.columns)} oszlop."]
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            lines.append(f"- {c} (szám): {s.min()} – {s.max()}")
        else:
            sample = ", ".join(map(str, s.dropna().unique()[:3]))
            lines.append(f"- {c} (szöveg, {s.nunique()} különböző): pl. {sample}")
    lines.append("Összesítéshez, szűréshez, számoláshoz töltsd be a sandboxba.")
    return Chunk(f"{table_name}#card", "\n".join(lines), {"table": table_name, "card": True})
