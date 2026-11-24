"""Bónusz (Q&A-hoz): a három út egy agentben.

1) Feltöltéskor mechanikus útválasztás: méret ÉS becsült tartalom szerint (a bájtméret hazudik).
2) Kérdéskor a modell választ toolt; a szabályok a tool-leírásokban és a system promptban vannak.

  uv run python demo/bonus_router.py            # csak a feltöltési útválasztás, offline
  uv run python demo/bonus_router.py 1|2|3      # + az agent az adott kérdéssel (API-kulcs kell)
"""

import json
import sys
from pathlib import Path

from fajl_agent.agent import run_agent
from fajl_agent.catalog import Catalog
from fajl_agent.corpus import DATA, SEARCH_TOOL, search_documents
from fajl_agent.llm import Recorder, run_demo
from fajl_agent.sandbox import DockerSandbox, ToolResult

CONTEXT_BUDGET = 50_000  # token / kör, demó-érték
PDF_TOKENS_PER_PAGE = 3_000 + 1_568  # szöveg (felső becslés) + oldalkép: inkább túlbecsüljük


def estimate_tokens(p: Path) -> int:
    if p.suffix == ".pdf":
        raw = p.read_bytes()
        pages = raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")
        return pages * PDF_TOKENS_PER_PAGE
    return int(len(p.read_text(encoding="utf-8", errors="ignore")) / 3)  # magyar szöveg: pesszimista


def route_upload(p: Path) -> tuple[str, str]:
    if p.suffix in {".csv", ".xlsx", ".json"}:
        return "sandbox", "strukturált: számolni kóddal kell, nem fejben"
    est = estimate_tokens(p)
    if est <= CONTEXT_BUDGET:
        return "kontextus", f"~{est:_} token, belefér".replace("_", " ")
    return "RAG + sandbox", f"~{est:_} token, túl nagy: indexelés, és kódból is elérhető".replace("_", " ")


QUESTIONS = {
    "1": "Megvan a vizsgaeredmények fájl? Ha igen, hány sikertelen forgalmi vizsga volt 2026-ban?",
    "2": "Mit mond a Zöldhullám Autósiskola tájékoztatója a lemondásról?",
    "3": "Hány iskola tart motoros képzést?",
}
SYSTEM = """Autósiskolák dokumentumairól és adatairól válaszolsz. Válaszd ki a kérdéshez illő utat:
- Megvan-e egy fájl, milyen fájlok vannak: find_files (adatbázis-lekérdezés fájlnévre). Ne keress tartalomban.
- Mit mond egy dokumentum, melyik pontban: search_documents, és hivatkozz a találatokra.
- Számolás, összesítés, szűrés, rangsor ("hány", "összesen", "legnagyobb"): keresd meg a táblázatot
  a find_files-szal, töltsd be (load_file), és számolj kóddal (run_python, pandas, /data mappa).
  Keresési találatokból SOHA ne számolj: azok csak a legjobb néhány részlet, nem a teljes adat."""


def main() -> None:
    print("Feltöltéskori útválasztás:")
    for p in [*sorted((DATA / "docs").glob("*")), *sorted((DATA / "tables").glob("*")),
              DATA / "tajekoztatok" / "ISK-0085.md"]:
        route, why = route_upload(p)
        print(f"  {p.name:<34} {p.stat().st_size / 1e6:>6.2f} MB → {route:<14} ({why})")

    q = QUESTIONS.get(sys.argv[1] if len(sys.argv) > 1 else "")
    if not q:
        return
    catalog = Catalog()
    catalog.add_dir("pelda-tenant", DATA)  # élesben a tenant a bejelentkezésből jön, NEM a modelltől

    def find_files(pattern: str) -> ToolResult:
        rows = catalog.find("pelda-tenant", pattern)
        return ToolResult(json.dumps(rows, ensure_ascii=False, indent=1) if rows
                          else f"Nincs '{pattern}' nevű fájl ebben a projektben.")

    tools = [
        {"name": "find_files", "description": "Fájlok keresése NÉV szerint a projektben (SQL LIKE). "
         "Táblázatoknál az adatlapot (oszlopok, sorszám) is visszaadja.",
         "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}},
                          "required": ["pattern"]}},
        SEARCH_TOOL,
        {"name": "load_file", "description": "Fájl bemásolása a sandboxba (/data/<név>). Keretes.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "run_python", "description": "Python futtatása a sandboxban. Csak az eredményt printeld.",
         "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    ]
    with DockerSandbox(DATA, show_numbers=False) as sbx:
        answer = run_agent(Recorder(f"bonus_router_{sys.argv[1]}"), system=SYSTEM, tools=tools,
                           handlers={"find_files": find_files, "search_documents": search_documents,
                                     "load_file": lambda path: sbx.load_file(path),
                                     "run_python": lambda code: sbx.run_python(code)},
                           user_content=q)
    print(f"\nKÉRDÉS: {q}\nVÁLASZ:\n{answer}")


if __name__ == "__main__":
    run_demo(main)
