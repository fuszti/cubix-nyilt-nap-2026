"""A fordulat: a RAG nem tud aggregálni. "Hány iskola tart motoros képzést?"

A retrieval csak a legjobb k találatot adja vissza, és nem jelzi, hogy a lista részleges.
A modell ezt csak a tool leírásából tudhatja. Ezt modell nélkül, determinisztikusan megmérjük.

  uv run python demo/d3_rag_hatar.py          # mérés, offline
  uv run python demo/d3_rag_hatar.py --llm    # + a RAG-agent válasza (API-kulcs kell)
"""

import sys

from fajl_agent.agent import run_agent
from fajl_agent.corpus import SEARCH_TOOL, doc_index, registry, search_documents
from fajl_agent.data_gen import ft
from fajl_agent.llm import Recorder, run_demo

QUESTION = "Hány iskola tart motoros (A) képzést, és mennyi náluk átlagosan a B kategóriás tandíj?"


def main() -> None:
    reg = registry()
    motoros = reg[reg.motoros == "igen"]
    truth = set(motoros.azonosito)
    avg = ft(round(motoros.tandij_b_ft.mean()))
    print(f"A valóság (a nyilvántartásból): {len(truth)} iskola, átlagos B tandíj {avg} Ft.\n")

    index = doc_index()
    print(f"Mit lát a RAG ugyanerre a kérdésre? ({QUESTION})")
    for k in [5, 10, 20, 50]:
        hits = index.search(QUESTION, k=k)
        found = {h.chunk.meta["doc_id"] for h in hits}
        print(f"  top-{k:>2}: {len(found & truth):>2} / {len(truth)} motoros iskola van a találatok között"
              f"  ({len(found - truth)} másik iskola is bejött)")
    print("\nA top-k ablak kicsi, a „tartunk” és a „nem tartunk” mondat szinte ugyanaz a keresőnek,"
          "\nés a keresés nem jelzi, hogy amit visszaad, az nem a teljes lista. A modell ezt csak a tool"
          "\nleírásából tudhatja. Ez a felépítés korlátja, nem a modell hibája.")

    if "--llm" in sys.argv:
        answer = run_agent(Recorder("d3_rag_hatar"),
                           system="Autósiskolák tájékoztatóiról válaszolsz a search_documents toollal.",
                           tools=[SEARCH_TOOL], handlers={"search_documents": search_documents},
                           user_content=QUESTION)
        print(f"\nA RAG-AGENT VÁLASZA:\n{answer}\n\n(A helyes: {len(truth)} iskola, átlagosan {avg} Ft.)")


if __name__ == "__main__":
    run_demo(main)
