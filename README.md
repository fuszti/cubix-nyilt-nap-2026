# Fájlfeltöltés egy AI agentnek: kontextus, RAG és sandbox

Példakód a Cubix nyílt napi előadáshoz. Ugyanazokat a kitalált autósiskolás adatokat (tanfolyami tájékoztatók,
iskolanyilvántartás, vizsgaeredmények) dolgozzuk fel háromféleképpen, és megmérjük, melyik út mikor jó és mikor
bukik el. Minden iskola, vizsgahely, ár és eredmény kitalált.

| Út | Mikor jó | Hol bukik el | Demó |
|---|---|---|---|
| **1. Közvetlenül a kontextusba** (document block) | kicsi fájl, a modell egyben lássa, PDF-nél az oldalképeket is | nagy fájl: tokenszám és költség minden körben; a bájtméret nem tokenszám | `d1_kontextus.py` |
| **2. RAG** (chunk → embedding + BM25 → search tool) | „mit mond a tájékoztató X-ről?”, sok dokumentum, hivatkozással | összesítés, számolás, dátum- és számfeltétel, „megvan-e a fájl?” | `d2_rag.py`, `d3_rag_hatar.py`, `d4_tablazat.py` |
| **3. Sandbox + kódíró agent** | táblázat, CSV, Excel: szűrés, aggregálás, pontos szám | lassabb (több kör), nincs mire hivatkozni, nem „lát” (szkennelt PDF) | `d5_sandbox.py` |

A `bonus_router.py` a hármat egy agentben köti össze (Q&A-hoz).

## Előkészítés (egyszer, az előadás előtt)

```bash
uv sync
docker build -t fajl-agent-sandbox sandbox/       # a sandbox image
uv run python -m fajl_agent.data_gen              # kitalált adatok a data/ mappába
uv run python demo/d2_rag.py --csak-kereses       # letölti az embedding modellt, cache-el
uv run python demo/d4_tablazat.py                 # első futás ~5 perc (embedding), utána ~3 s
uv run pytest -q                                  # a mechanika tesztjei (API-kulcs nélkül)
```

Élő hívásokhoz írd be a kulcsot a repó gyökerében lévő `.env` fájlba (a git nem követi):

```
OPENROUTER_API_KEY=sk-or-...      # vagy közvetlenül a Claude API-hoz: ANTHROPIC_API_KEY=sk-ant-...
```

Modell: `DEMO_MODEL`, alapból `anthropic/claude-opus-5.5` (OpenRouter) vagy `claude-opus-5-5` (Claude API).
Gondolkodási mélység: `DEMO_EFFORT`, alapból `medium`.

**Tartalék az előadásra:** vedd fel minden élő demó válaszait, és ha nincs net, játszd vissza:

```bash
DEMO_MODE=record uv run python demo/d1_kontextus.py      # ugyanígy: d2_rag, d3_rag_hatar --llm, d5_sandbox
DEMO_MODE=replay uv run python demo/d1_kontextus.py      # hálózat nélkül, a felvételből
```

A `recordings/` mappában ott vannak egy OpenRouteres futás felvételei, így `DEMO_MODE=replay` módban
kulcs nélkül is végignézheted a demókat.

## OpenRouterrel: mi más, mint a Claude API-n

Ugyanaz az `anthropic` SDK és ugyanaz a kérésformátum, csak az OpenRouter Anthropic-kompatibilis végpontján
(`base_url="https://openrouter.ai/api"`, `fajl_agent/llm.py`). Az Anthropic szolgáltatót kérjük elsőnek.
Amit 2026 novemberében mértünk:

- **Nincs `count_tokens` végpont** (404). A tokenszámot egy rövid, fizetős kérés `usage` mezőjéből olvassuk ki,
  ezért a `d1_kontextus.py` a nagy fájlokból csak mintát küld (10 oldal, 1000 sor). Egy futás kb. 0,4 $.
- **A PDF oldalszámos hivatkozásai nem jönnek vissza.** A PDF eljut a modellhez (oldalképpel együtt),
  de a válaszban üres a `citations`, ezért a `d1` nem is kér hivatkozást. A RAG `search_result`
  hivatkozásai viszont megjönnek, ezeket a `d2` használja.
- **A szerveroldali fallback beta helyett** az OpenRouter saját útválasztása dönt.

## A demók

| Script | API-kulcs? | Mit mutat |
|---|---|---|
| `demo/d1_kontextus.py` | igen | a nyers API-kérés: a PDF base64-ként a `messages` tömbben; a válasz; a 240 oldalas PDF és a 30 ezer soros CSV tokenszáma, mintából becsülve |
| `demo/d2_rag.py` | igen (`--csak-kereses`: nem) | chunk contextual headerrel; dense vs BM25 vs hibrid; a modell maga hívja a search toolt, `search_result` blokkokra hivatkozik |
| `demo/d3_rag_hatar.py` | nem (`--llm`: igen) | „Hány iskola tart motoros képzést?” — a top-k találatban a valódi 59-ből csak néhány van |
| `demo/d4_tablazat.py` | nem | táblázatkeresés: háromféle chunkolás × háromféle keresés, számfeltétel, összesítés, tábla-adatlap |
| `demo/d5_sandbox.py` | igen (`--offline`: nem) | izolált Docker sandbox: betöltés kerettel, hibából javítás, túl hosszú kimenet hibaként, nincs hálózat |
| `demo/bonus_router.py` | nem (`1`/`2`/`3`: igen) | feltöltéskori útválasztás becsült token alapján; agent négy toollal |
| `demo/d6_docling.py` | nem (`--group docling`) | PDF feldolgozása Doclinggal: szerkezet, chunk, és egy valódi, nehéz táblázat mérése |

## Feldolgozás Doclinggal (`d6_docling.py`, opcionális)

```bash
uv sync --group docling     # Docling + pdfplumber; nagy letöltés (PyTorch és a Docling modelljei)
uv run --group docling python demo/d6_docling.py
uv run --group docling python demo/d6_docling.py utvonalak.pdf --oldalak 1-5
```

Lokálisan fut, API-kulcs nélkül. Két dolgot mutat:

- **Tájékoztató-PDF → chunk.** A Docling a PDF-ből szerkezetet épít, a `HybridChunker` pedig a felismert
  címsorokat teszi a chunk elé (`contextualize()`), ahogy a mi contextual headerünk is. A tokenizálót az
  embedding modelltől kapja, így a chunk belefér a modell 128 tokenjébe. A generált PDF-ben a
  fejezetcímeket nem ismeri fel címsornak (csak kicsit nagyobb betűsek), ezért ott a chunk elé csak a
  dokumentum címe kerül.
- **Egy valódi, nehéz táblázat:** az „Országos vizsgaútvonalak” PDF (Excel-export, 191 oldal, nyilvános
  adat). A repóban nincs benne, az útvonalát argumentumként add meg, vagy tedd a `data/docs/` mappába.
  Egy sor: azonosító, kiindulási hely, többsoros útvonal és 17 kategóriaoszlop, ahol egy „X” jelzi a
  vizsgakategóriát. A helyes táblát a PDF szavainak koordinátáiból számoljuk ki, és ehhez mérjük a
  Doclingot és a sima szövegkinyerést.

Mért eredmény az 1–5. oldalon (102 sor, 506 X, Docling 2.152, CPU):

| | sima szöveg | Docling, alap (accurate) | Docling, fast |
|---|---|---|---|
| futásidő | – | 100 s | 23 s |
| tábla | – | oldalanként 1×1: minden szöveg egy cellában | oldalanként egy rács, 15–21 oszloppal (a valódi 24) |
| megtalált sor | – | 0/102 | 85/102, 8 cellában két sor összeolvadt |
| X a jó kategóriában | 0: a szövegben az X ott van, az oszlopa nem | 0/506 | 0/506 az oszlopnév szerint (a kategóriafejlécek összeolvadtak); 42/506, ha csak a legalsó fejlécsort nézzük |
| teljes útvonal egy cellában | – | 0/102 | 48/102 |

A 100–102. oldalon ugyanez jött ki: accurate módban 1×1-es táblák, fast módban 39/47 sor, 19/163 X a jó
oszlopban a legalsó fejléc szerint. A konverzió státusza mindegyik futásban `success`. A rossz táblára
legfeljebb egy naplósor utal, ezért a kinyert táblát ellenőrizni kell: sorszám, oszlopszám, és néhány sor
összevetése az eredetivel.

## Ami élesben is kell (a kódban megoldva, a kommentekben indokolva)

- **A bájtméret nem tokenszám.** A PDF oldalanként szöveg + oldalkép (oldalképenként legfeljebb 1 568 token), a tömörített DOCX/XLSX sokkal több szöveget rejt, mint amekkora a fájl. A `d1_kontextus.py` egy mintán méri meg, és felszorozza: a 240 oldalas, 0,46 MB-os PDF kb. 560 ezer token, a 3,3 MB-os CSV kb. 2 millió. Becsülj tartalomra, inkább felfelé.
- **A chunk a keresés egysége.** Ami kimarad belőle (dokumentumcím, oszlopnév), azt nem találja meg a kereső → contextual header, táblázatnál soronként `oszlop: érték`.
- **Azonosítóra, névre BM25.** Az embedding a vizsgaazonosítót és a táblasorokat rosszul találja; a hibrid fúzió egy gyenge vektor-ággal le is ronthatja a BM25-öt. Mérd meg a saját adatodon (`d4_tablazat.py`).
- **„Megvan-e a fájl?” lekérdezés, nem keresés.** A tartalomkeresés nem tudja bizonyítani, hogy valami nincs (`catalog.py`).
- **A RAG nem tud aggregálni, és a keresés nem jelzi, hogy a találati lista részleges.** A modell csak a tool leírásából tudhatja. Számolás → sandbox, kóddal a teljes fájlon.
- **Sandbox:** egy beszélgetés = egy konténer; nincs hálózat; read-only; nem root; CPU/memória/idő limit; csak a tenant mappájából tölthet be; fájl-, méret- és futtatási keret, amit minden válasz jelez; a túl hosszú kimenet hiba, nem csendes vágás; hibánál a traceback megy előre; a sandbox saját hibája nem „kódhiba”.
- **A tenant szűrő sosem a modelltől jön**, hanem a bejelentkezésből (`index.search(where=...)`, `catalog.find(tenant, ...)`).

## Szerkezet

```
fajl_agent/   chunking, index (hibrid), catalog (SQL), sandbox (Docker), llm (record/replay), agent (kézi loop)
demo/         a demók, az előadás sorrendjében
sandbox/      a sandbox image Dockerfile-ja
tests/        a mechanika tesztjei
```
