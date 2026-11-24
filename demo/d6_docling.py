"""Feldolgozás (parse) Doclinggal: PDF -> szerkezet -> chunk, és egy nehéz, valódi táblázat.

  uv sync --group docling
  uv run --group docling python demo/d6_docling.py                                  # csak a tájékoztató-PDF
  uv run --group docling python demo/d6_docling.py utvonalak.pdf [--oldalak 1-3]    # + a táblázat mérése

Lokálisan fut, API-kulcs nélkül. Az első futás letölti a Docling modelljeit (pár száz MB).

1. rész: a tájékoztató-PDF-ből a Docling szerkezetet épít, a HybridChunker pedig a felismert
   címsorokat teszi a chunk elé, mint a saját chunk_markdown_sections-ünk a fejezetútvonalat.
   Amit a Docling nem ismer fel címsornak, az a chunk elől is kimarad.
2. rész: az „Országos vizsgaútvonalak” PDF (Excel-export, 191 oldal, nyilvános adat). Egy sor:
   azonosító, kiindulási hely, többsoros útvonal, és 17 kategóriaoszlop, ahol egy „X” jelzi,
   melyik vizsgára jó az útvonal. A helyes választ a PDF szavainak koordinátáiból számoljuk ki,
   és ehhez mérjük a Docling táblázatát, valamint egy sima szövegkinyerést.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

from fajl_agent.chunking import chunk_markdown_sections
from fajl_agent.corpus import DATA
from fajl_agent.index import EMBED_MODEL

try:
    import pdfplumber
    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
except ImportError:
    raise SystemExit("Ehhez a demóhoz kell a docling csoport: uv sync --group docling")

logging.getLogger("docling").setLevel(logging.ERROR)

CATS = ["AM", "A1", "A1 B-vel", "A2", "A", "B", "BE", "C1", "C", "D1", "D", "C1E", "CE", "D1E", "DE", "T", "Tr"]
ROW_ID = re.compile(r"^[A-Z]+/[^/\s]+/\d+$")


def converter(mode: TableFormerMode) -> DocumentConverter:
    po = PdfPipelineOptions(do_ocr=False)  # a PDF-ben van szövegréteg, OCR nem kell
    po.table_structure_options.mode = mode
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)})


def norm(s: str) -> str:
    """Összehasonlításhoz: gondolatjel = kötőjel, szóközök nélkül."""
    return re.sub(r"\s+", "", s.replace("–", "-").replace("—", "-"))


# ---------- 1. rész: tájékoztató-PDF -> Markdown -> chunk ----------

def brochure_part() -> None:
    pdf = DATA / "docs" / "tajekoztatok-osszes.pdf"
    print("\n=== 1) Tájékoztató-PDF Doclinggal (1. oldal) ===")
    t0 = time.time()
    doc = converter(TableFormerMode.FAST).convert(pdf, page_range=(1, 1)).document
    md = doc.export_to_markdown()
    print(f"[{time.time() - t0:.0f} s]\n" + "\n".join(md.splitlines()[:14]) + "\n...")

    # A chunk méretét az embedding modell tokenizere szabja meg (ez a modell 128 tokent lát).
    tok = HuggingFaceTokenizer.from_pretrained(EMBED_MODEL, max_tokens=128)
    chunker = HybridChunker(tokenizer=tok)
    chunks = list(chunker.chunk(dl_doc=doc))
    lemondas = next(c for c in chunks if "lemondani" in c.text)
    print(f"\nDocling HybridChunker ({len(chunks)} chunk), a lemondásról szóló chunk a contextualize() után:")
    print("  " + chunker.contextualize(lemondas).replace("\n", "\n  "))
    heads = lemondas.meta.headings or []
    print(f"  (a chunk elé tett címsorok: {heads})")

    doc_id = re.search(r"ISK-\d{4}", md).group(0)
    ours = chunk_markdown_sections(doc_id, (DATA / "tajekoztatok" / f"{doc_id}.md").read_text())
    mine = next(c for c in ours if "lemondás" in c.text.split("\n")[0])
    print("\nA saját chunk_markdown_sections ugyanerre (a Markdown-forrásból):")
    print("  " + mine.text.replace("\n", "\n  "))
    if not any("lemondás" in h for h in heads):
        print("\nA Docling a dokumentum címét címsornak ismerte fel, a fejezetcímeket („3. Pótóra és lemondás”) nem:")
        print("azok itt csak kicsit nagyobb betűs sorok, ezért sima szövegként maradtak. Így a chunk elé csak")
        print("a cím kerül. Amit a feldolgozó nem ismer fel szerkezetnek, azt a chunkoló sem tudja felhasználni.")


# ---------- 2. rész: vizsgaútvonal-táblázat ----------

def ground_truth(pdf_path: Path, pages: range) -> list[dict]:
    """A helyes tábla a szavak koordinátáiból: melyik X melyik kategória oszlopa alá esik."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno in pages:
            pg = pdf.pages[pno - 1]
            words = pg.extract_words(x_tolerance=1.5)
            footer = min((w["top"] for w in words if w["text"] == "Méret:"), default=pg.height)
            head = [w for w in words if 105 < w["top"] < 130]
            bvel = [w for w in head if w["text"] == "B-vel"]
            centers = {}
            for w in head:
                m = re.fullmatch(r'"([A-Z0-9]+)"', w["text"])
                if not m:
                    continue
                name = m.group(1)
                if name == "A1" and any(b["x0"] < w["x1"] and w["x0"] < b["x1"] for b in bvel):
                    name = "A1 B-vel"
                centers[name] = (w["x0"] + w["x1"]) / 2
            body = [w for w in words if 130 < w["top"] < footer - 2]
            ids = sorted((w for w in body if w["x0"] < 95 and ROW_ID.match(w["text"])), key=lambda w: w["top"])
            tops = [w["top"] for w in ids]
            bounds = [130] + [(a + b) / 2 for a, b in zip(tops, tops[1:])] + [footer]
            page_rows = [{"id": w["text"], "page": pno, "cats": set(), "route": []} for w in ids]
            for w in body:
                if not page_rows:
                    break
                i = next(k for k in range(len(page_rows)) if bounds[k] <= w["top"] < bounds[k + 1]) \
                    if bounds[0] <= w["top"] < bounds[-1] else None
                if i is None:
                    continue
                if w["text"] == "X":
                    cat = min(centers, key=lambda c: abs(centers[c] - (w["x0"] + w["x1"]) / 2))
                    page_rows[i]["cats"].add(cat)
                elif 425 < w["x0"] and w["x1"] < 668:  # az Útvonal oszlop sávja
                    page_rows[i]["route"].append(w)
            for r in page_rows:
                r["route"] = " ".join(w["text"] for w in sorted(r["route"], key=lambda w: (round(w["top"]), w["x0"])))
            rows += page_rows
    return rows


def header_labels(text: str) -> set[str]:
    """Egy fejléccella szövegéből a benne szereplő kategóriák ('"A1" B-vel' = A1 B-vel)."""
    return {name + (" B-vel" if bvel else "") for name, bvel in re.findall(r'"([A-Z0-9]+)"(\s*B-vel)?', text)}


def docling_rows(doc) -> tuple[dict[str, dict], int]:
    """Azonosító -> az X-ek oszlopának kategóriái és a sor cellái; plusz hány cellában olvadt össze két sor.

    Egy oszlop kategóriáit kétféleképpen olvassuk ki:
      szigorú: az összes fölötte lévő fejléccella együtt (így néz ki a DataFrame oszlopneve),
      alsó:    csak a legalsó fejléccella, a legjobb esetet feltételezve."""
    rows, merged = {}, 0
    for t in doc.tables:
        strict: dict[int, set[str]] = {}
        lowest: dict[int, tuple[int, set[str]]] = {}
        for c in t.data.table_cells:
            if c.column_header:
                for col in range(c.start_col_offset_idx, c.end_col_offset_idx):
                    strict.setdefault(col, set()).update(header_labels(c.text))
                    if col not in lowest or c.start_row_offset_idx > lowest[col][0]:
                        lowest[col] = (c.start_row_offset_idx, header_labels(c.text))
        merged += sum(1 for c in t.data.table_cells if len(re.findall(r"[A-Z]+/[^/\s]+/\d+", c.text)) > 1)
        by_row: dict[int, list] = {}
        for c in t.data.table_cells:
            if not c.column_header:
                by_row.setdefault(c.start_row_offset_idx, []).append(c)
        for cells in by_row.values():
            rid = next((c.text.strip() for c in cells if ROW_ID.match(c.text.strip())), None)
            if rid:
                xs = [c.start_col_offset_idx for c in cells if c.text.strip() == "X"]
                rows[rid] = {"strict": [strict.get(x, set()) for x in xs],
                             "lowest": [lowest.get(x, (0, set()))[1] for x in xs],
                             "cells": [c.text for c in cells]}
    return rows, merged


def score(gt: list[dict], rows: dict[str, dict], merged: int) -> dict:
    marks = sum(len(r["cats"]) for r in gt)
    found = strict_ok = lowest_ok = full_rows = route_ok = 0
    for r in gt:
        d = rows.get(r["id"])
        if not d:
            continue
        found += 1
        strict_ok += sum(1 for s in d["strict"] if len(s) == 1 and s <= r["cats"])
        lowest = [next(iter(s)) for s in d["lowest"] if len(s) == 1]
        lowest_ok += len(set(lowest) & r["cats"])
        full_rows += sorted(lowest) == sorted(r["cats"]) and len(d["lowest"]) == len(r["cats"])
        route_ok += any(norm(r["route"]) in norm(c) for c in d["cells"])
    return {"megtalált sor": f"{found}/{len(gt)} (összeolvadt sorpár: {merged})",
            "X egyértelmű oszlopnév alatt": f"{strict_ok}/{marks}",
            "X jó oszlopban (csak alsó fejléc)": f"{lowest_ok}/{marks}",
            "hibátlan sor (csak alsó fejléc)": f"{full_rows}/{len(gt)}",
            "teljes útvonal egy cellában": f"{route_ok}/{len(gt)}"}


def table_part(pdf_path: Path, pages: range) -> None:
    print(f"\n=== 2) Vizsgaútvonalak: {pdf_path.name}, {pages.start}–{pages.stop - 1}. oldal ===")
    gt = ground_truth(pdf_path, pages)
    print(f"A helyes tábla (szókoordinátákból): {len(gt)} sor, {sum(len(r['cats']) for r in gt)} X.")
    r0 = gt[0]
    print(f"  pl. {r0['id']}: kategória {sorted(r0['cats'])}, útvonal: {r0['route'][:70]}…")

    with pdfplumber.open(pdf_path) as pdf:
        line = next(l for l in pdf.pages[pages.start - 1].extract_text().splitlines() if r0["id"] in l)
    print("\nSima szövegkinyerés, ugyanez a sor:")
    print(f"  {line[:150]}")
    print("  Az X ott van, de hogy melyik kategória oszlopában, az a szövegből nem derül ki.")

    results = {}
    for label, mode in [("Docling, alapbeállítás (accurate)", TableFormerMode.ACCURATE),
                        ("Docling, fast mód", TableFormerMode.FAST)]:
        t0 = time.time()
        res = converter(mode).convert(pdf_path, page_range=(pages.start, pages.stop - 1))
        dt = time.time() - t0
        shapes = [f"{t.data.num_rows}×{t.data.num_cols}" for t in res.document.tables]
        rows, merged = docling_rows(res.document)
        results[label] = score(gt, rows, merged)
        print(f"\n{label}: státusz={res.status.value}, {dt:.0f} s, táblák: {', '.join(shapes) or 'nincs'}")
        for k, v in results[label].items():
            print(f"  {k:32s} {v}")

    print("\nA konverzió státusza mindkét esetben „success”, kivétel nincs, a rossz táblára legfeljebb egy")
    print("naplósor utal. Ezt csak a kimenet ellenőrzése fogja meg: sorszám, oszlopszám, és néhány sor")
    print("összevetése az eredetivel.")


def main() -> None:
    args = sys.argv[1:]
    pages = range(1, 4)
    if "--oldalak" in args:
        i = args.index("--oldalak")
        a, b = (int(x) for x in args[i + 1].split("-"))
        pages = range(a, b + 1)
        del args[i:i + 2]
    brochure_part()
    pdf_path = Path(args[0]) if args else next(iter(sorted((DATA / "docs").glob("*vizsga*tvonal*.pdf"))), None)
    if not pdf_path or not pdf_path.exists():
        print("\n(A vizsgaútvonal-PDF nincs meg. Add meg az útvonalát argumentumként, vagy tedd ide: "
              f"{DATA / 'docs'}/)")
        return
    table_part(pdf_path, pages)


if __name__ == "__main__":
    main()
