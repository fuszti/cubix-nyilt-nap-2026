"""2. út: RAG. Chunkolás, embedding, hibrid keresés, és egy search tool, amit a MODELL hív meg.

  uv run python demo/d2_rag.py               # agent + hivatkozások (API-kulcs kell)
  uv run python demo/d2_rag.py --csak-kereses # csak a retrieval, offline
"""

import sys

from fajl_agent.agent import run_agent
from fajl_agent.corpus import SEARCH_TOOL, doc_index, search_documents
from fajl_agent.llm import Recorder, run_demo

QUESTION = "Mennyi a pótóra díja a Zöldhullám Autósiskolánál, és meddig érvényes ez az ár?"
SYSTEM = ("Autósiskolák tanfolyami tájékoztatóiról válaszolsz. A tájékoztatók szövegében a "
          "search_documents toollal keresel. Csak a találatokból válaszolj, és hivatkozz rájuk.")


def main() -> None:
    index = doc_index()
    print(f"Index: {len(index.chunks)} chunk {len({c.meta['doc_id'] for c in index.chunks})} tájékoztatóból.")
    print(f"Egy chunk így néz ki:\n---\n{index.chunks[1].text}\n---")

    if "--csak-kereses" in sys.argv:
        for mode in ["dense", "bm25", "hybrid"]:
            print(f"\n[{mode}] {QUESTION}")
            for h in index.search(QUESTION, k=3, mode=mode):
                print(f"  {h.chunk.meta['doc_id']} | {h.chunk.meta['section']}")
        return

    answer = run_agent(Recorder("d2_rag"), system=SYSTEM, tools=[SEARCH_TOOL],
                       handlers={"search_documents": search_documents}, user_content=QUESTION)
    print(f"\nVÁLASZ:\n{answer}")


if __name__ == "__main__":
    run_demo(main)
