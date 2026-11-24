"""Miért nehéz táblázatban keresni RAG-gal? Mérés modell nélkül, offline.

Három chunkolás ugyanarra a vizsgaeredmény-táblára, három keresési mód, és háromféle kérdés:
  1. rekord-keresés azonosítóval      ("VIZS-2025-000123")
  2. rekord-keresés leírással          (iskola + kategória + vizsgatípus + dátum)
  3. számszerű feltétel                 ("50 óránál többet vezetett, mégis sikertelen")
És egy negyedik, amire egyik sem jó: az összesítés.

  uv run python demo/d4_tablazat.py
"""

import random

import pandas as pd

from fajl_agent.chunking import chunk_table_naive, chunk_table_rows, table_card
from fajl_agent.corpus import DATA
from fajl_agent.index import HybridIndex

N_ROWS = 1500  # a teljes 30 ezer sor embeddingje percekig tartana; a jelenség így is látszik
K = 5
TABLE = "vizsgaeredmenyek.csv"


def main() -> None:
    df = pd.read_csv(DATA / "tables" / TABLE).head(N_ROWS)
    csv_text = df.to_csv(index=False)
    strategies = {
        "naiv (500 karakter)": chunk_table_naive(TABLE, csv_text),
        "soronként, fejléccel": chunk_table_rows(TABLE, df),
        "soronként + tábla-előtag": chunk_table_rows(TABLE, df, prefix=True),
    }
    print("Ugyanaz a sor háromféleképpen:")
    for name, chunks in strategies.items():
        print(f"  {name:<26} {chunks[3].text[:110]!r}")

    rng = random.Random(7)
    targets = df.sample(20, random_state=7)
    queries = []
    for _, r in targets.iterrows():
        queries.append(("azonosító", f"Mi lett a {r.vizsgaazonosito} vizsga eredménye?", r.vizsgaazonosito))
        queries.append(("leírás", f"{r.iskola} {r.kategoria} kategóriás {r.vizsgatipus} vizsgája, {r.datum}",
                        r.vizsgaazonosito))
    rng.shuffle(queries)

    print(f"\nTalálati arány (hit@{K}): benne van-e a keresett vizsga az első {K} chunkban? "
          f"({len(queries)} kérdés, {N_ROWS} sor)")
    print(f"  {'chunkolás':<26} {'mód':<7} {'azonosító':>10} {'leírás':>8}")
    for name, chunks in strategies.items():
        index = HybridIndex(chunks)
        for mode in ["dense", "bm25", "hybrid"]:
            score = {"azonosító": 0, "leírás": 0}
            for kind, q, target in queries:
                if any(target in h.chunk.text for h in index.search(q, k=K, mode=mode)):
                    score[kind] += 1
            print(f"  {name:<26} {mode:<7} {score['azonosító']:>7}/20 {score['leírás']:>6}/20")

    print("  → Azonosítóra, névre a BM25 a király; az embedding a táblasorokra gyenge, és a"
          "\n    hibrid fúzió egy gyenge vektor-ággal a BM25 alá is húzhat. Mérd meg a saját adatodon!")

    print("\nSzámszerű feltétel: 'sikertelen forgalmi vizsgák, ahol a vizsgázó 50 óránál többet vezetett'")
    rows = strategies["soronként, fejléccel"]
    index = HybridIndex(rows)
    hits = index.search("sikertelen forgalmi vizsga, 50 óránál több vezetett óra", k=10)
    ok = sum(1 for h in hits
             if df.iloc[h.chunk.meta["row"]].vezetett_orak > 50
             and df.iloc[h.chunk.meta["row"]].eredmeny == "sikertelen")
    real = ((df.vezetett_orak > 50) & (df.eredmeny == "sikertelen")).sum()
    print(f"  A top-10 találatból {ok} teljesíti a feltételt. A táblában {real} ilyen sor van.")

    print("\nÖsszesítés: 'Hány százalék a sikeres forgalmi vizsgák aránya?'")
    forg = df[df.vizsgatipus == "forgalmi"]
    rate = (forg.eredmeny == "sikeres").mean() * 100
    print(f"  Helyes válasz: {rate:.1f}% ".replace(".", ",")
          + f"({len(forg)} forgalmi vizsgából). A RAG ebből legfeljebb top-k sort lát.")

    card = table_card(TABLE, df)
    index = HybridIndex(rows + [card])
    rank = next((i for i, h in enumerate(index.search("vizsgaeredmények összesítése vizsgahely szerint", k=50))
                 if h.chunk.meta.get("card")), None)
    print(f"\nA tábla adatlapja (séma, tartományok, pár példaérték):\n{card.text}")
    print(f"  Ha az adatlapot is beindexeljük, a kereső a(z) "
          f"{rank + 1 if rank is not None else '50. helyen sem'} adja vissza: 1500 sor-chunk elnyomja.")
    print("  → Az adatlapot a fájl-lekérdezés (katalógus) adja vissza, ne a kereső. Lásd bonus_router.py.")


if __name__ == "__main__":
    main()
