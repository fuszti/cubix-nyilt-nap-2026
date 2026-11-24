"""Kitalált demóadatok generálása (determinisztikus, seed=42): autósiskolák és vizsgaeredmények.

Minden iskola, vizsgahely, név, ár és eredmény kitalált. Futtatás:  uv run python -m fajl_agent.data_gen
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
N_EXAMS = 30_000
SEED = 42

HU_MONTHS = [
    "január", "február", "március", "április", "május", "június",
    "július", "augusztus", "szeptember", "október", "november", "december",
]
# 20 előtag × 6 típus = 120 iskola. Egy előtag több iskolában is szerepel, mint a valóságban.
PREFIXES = [
    "Zöldhullám", "Sebváltó", "Kuplung", "Indexfény", "Sávváltó", "Körforgalom", "Fényszóró",
    "Tempomat", "Kézifék", "Útjelző", "Rajtvonal", "Szélvédő", "Napfény", "Duna", "Tisza",
    "Bakony", "Mátra", "Balaton", "Pannon", "Csillag",
]
KINDS = ["Autósiskola", "Jogsi Stúdió", "Vezetőképző", "Oktatóközpont", "KRESZ-Iskola", "Autósuli"]
CITIES = ["Budapest", "Debrecen", "Szeged", "Pécs", "Győr", "Miskolc", "Kecskemét", "Székesfehérvár",
          "Nyíregyháza", "Szombathely", "Eger", "Veszprém"]
# Kitalált vizsgahelyek: a sikerarányok generáltak, ne kössük valódi helyhez.
SITES = ["Északi", "Déli", "Keleti", "Nyugati", "Belvárosi", "Ipari parki", "Repülőtéri", "Tóparti"]
SITE_PASS = {"Északi": 0.64, "Déli": 0.61, "Keleti": 0.66, "Nyugati": 0.59, "Belvárosi": 0.57,
             "Ipari parki": 0.45, "Repülőtéri": 0.62, "Tóparti": 0.68}  # B forgalmi, kb.
HEADS = ["Kiss Anna", "Nagy Péter", "Szabó Eszter", "Tóth Gábor", "Varga Júlia", "Horváth Bence"]


@dataclass
class School:
    azon: str
    nev: str
    varos: str
    vizsgahely: str
    motoros: bool
    kategoriak: list[str]
    tandij: int
    elso_reszlet: int
    potora: int
    lemondas_ora: int
    ervenyes: date
    vezeto: str


def ft(n: int) -> str:
    """1234567 -> '1 234 567'"""
    return f"{n:,}".replace(",", " ")


def hu_date(d: date) -> str:
    return f"{d.year}. {HU_MONTHS[d.month - 1]} {d.day}."


def site_name(site: str) -> str:
    return f"{site} vizsgaközpont"


def article(word: str) -> str:
    """Határozott névelő: az Északi, a Déli."""
    return "az" if word[0].lower() in "aáeéiíoóöőuúüű" else "a"


def make_schools(rng: random.Random) -> list[School]:
    names = [f"{p} {k}" for p in PREFIXES for k in KINDS]
    rng.shuffle(names)
    schools = []
    for i, nev in enumerate(names):
        motoros = rng.random() < 0.4
        kat = ["B"] + (["A2", "A"] if motoros else []) + [k for k in ["BE", "C"] if rng.random() < 0.3]
        tandij = rng.randrange(329, 489, 10) * 1000
        schools.append(School(
            azon=f"ISK-{i + 1:04d}", nev=nev, varos=rng.choice(CITIES),
            vizsgahely=site_name(rng.choice(SITES)), motoros=motoros, kategoriak=kat,
            tandij=tandij, elso_reszlet=rng.choice([100, 120, 150]) * 1000,
            potora=rng.randrange(90, 145, 5) * 100, lemondas_ora=rng.choice([24, 48, 72]),
            ervenyes=date(2026, 12, 31) + timedelta(days=rng.choice([0, 90, 181, 365])),
            vezeto=rng.choice(HEADS),
        ))
    return schools


def school_markdown(s: School) -> str:
    motoros = (
        "Motoros képzést (A2 és A kategória) is tartunk, saját gyakorlópályán, iskolai motorokkal."
        if s.motoros
        else "Motoros képzést (A2 és A kategória) nem tartunk, ezekre partneriskolához irányítjuk a jelentkezőket."
    )
    return f"""# Tanfolyami tájékoztató – {s.nev} ({s.azon})

**Iskola:** {s.nev}, {s.varos} · oktatásvezető: {s.vezeto}

## 1. Képzéseink
Kategóriák: {", ".join(s.kategoriak)}. Az elméleti oktatás e-learningben folyik, a gyakorlati órákat egyeztetett időpontban, a tanuló lakóhelyéről indulva tartjuk. {motoros}

## 2. Tandíj és befizetés
A B kategóriás tanfolyam díja {ft(s.tandij)} Ft, amely az elméleti oktatást és az alap óraszámú vezetést tartalmazza. Két részletben is fizethető: {ft(s.elso_reszlet)} Ft a beiratkozáskor, a fennmaradó {ft(s.tandij - s.elso_reszlet)} Ft legkésőbb az első vezetési óra előtt. A hatósági vizsgadíjakat a tandíj nem tartalmazza.

## 3. Pótóra és lemondás
Az alap óraszámon felüli vezetési óra (pótóra) díja {ft(s.potora)} Ft / 50 perc. A vezetési órát legkésőbb {s.lemondas_ora} órával előtte lehet díjmentesen lemondani; késői lemondásnál a teljes óradíjat felszámítjuk.

## 4. Vizsgák
Az elméleti és a forgalmi vizsgát {article(s.vizsgahely)} {s.vizsgahely}ban tesszük le. A vizsgára az iskola jelentkezteti a tanulót, ha az elméleti tananyagot elvégezte, illetve ha az oktató szerint a forgalmi vizsgára felkészült.

## 5. Az árlista érvényessége
Az árak {hu_date(s.ervenyes)} napjáig érvényesek. A változásról a tanulókat legalább 30 nappal előre értesítjük; a már befizetett tandíjat nem emeljük.

## 6. Panaszkezelés
Panaszt írásban, az oktatásvezetőnél ({s.vezeto}) lehet tenni, amelyre 15 napon belül válaszolunk.
"""


def find_font() -> str:
    for p in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            return p
    raise SystemExit("Nem találok Unicode TTF fontot (Arial Unicode vagy DejaVuSans).")


def new_pdf() -> FPDF:
    pdf = FPDF(format="A4")
    pdf.add_font("U", "", find_font())
    pdf.set_font("U", size=11)
    pdf.set_margins(20, 20, 20)
    return pdf


def write_markdown_pdf(pdf: FPDF, md: str) -> None:
    pdf.add_page()
    for line in md.splitlines():
        if line.startswith("# "):
            pdf.set_font_size(16)
            pdf.multi_cell(0, 9, line[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font_size(11)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font_size(13)
            pdf.multi_cell(0, 7, line[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font_size(11)
        elif line.strip():
            pdf.multi_cell(0, 6, line.replace("**", ""), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.ln(2)


COURSE = """# Tanfolyami tájékoztató – Kilométerkő Autósiskola, B kategória

**Iskola:** Kilométerkő Autósiskola, Budapest XIII. kerület, Példa utca 12. · 2026. tavaszi tanfolyam

## 1. A tanfolyam
Az elméleti oktatás e-learningben folyik, 90 napos hozzáféréssel. A gyakorlati oktatás az alap óraszámú vezetésből áll; a vezetési órák 50 percesek, és a tanuló lakóhelyéről vagy munkahelyéről indulnak.

## 2. Tandíj
A tanfolyam teljes díja 412 000 Ft. Ez tartalmazza az e-learning hozzáférést, az alap óraszámú vezetést és az első forgalmi vizsgára kísérést. A pótóra díja 11 500 Ft / 50 perc.

## 3. Befizetés
A tandíj első részletét, 150 000 Ft-ot a beiratkozáskor kell befizetni. A fennmaradó 262 000 Ft-ot legkésőbb az első vezetési óra előtt 8 munkanappal kell átutalni az iskola számlájára. Amíg a teljes tandíj nem érkezett meg, vezetési órát nem tudunk tartani.

## 4. Lemondás
A vezetési órát legkésőbb 48 órával előtte lehet díjmentesen lemondani. Késői lemondásnál vagy meg nem jelenésnél a pótóra díját felszámítjuk.

## 5. Vizsgadíjak
A hatósági vizsgadíjakat a tanuló közvetlenül fizeti, ezeket a tandíj nem tartalmazza. Sikertelen vizsga után a pótórák és az újabb vizsga díja is a tanulót terheli.

## 6. Visszalépés
Ha a tanuló az első vezetési óra előtt visszalép, a befizetett összeget 10 000 Ft kezelési költséggel csökkentve, 30 napon belül visszafizetjük. Az első vezetési óra után a tandíjat nem térítjük vissza.
"""


def pass_prob(site: str, tipus: str, kategoria: str, orak: int, year: int) -> float:
    if tipus == "elmélet":
        return 0.58
    p = SITE_PASS[site] + (orak - 40) * 0.006
    if kategoria != "B":
        p += 0.05
    if year == 2026 and site == "Ipari parki":
        p -= 0.03
    return min(max(p, 0.05), 0.95)


def main() -> None:
    rng = random.Random(SEED)
    for sub in ["tajekoztatok", "tables", "docs"]:
        (DATA / sub).mkdir(parents=True, exist_ok=True)

    schools = make_schools(rng)

    # 1) Tájékoztatók szövegként: ez lesz a RAG-korpusz
    for s in schools:
        (DATA / "tajekoztatok" / f"{s.azon}.md").write_text(school_markdown(s), encoding="utf-8")

    # 2) Ugyanez táblázatként (a "valódi" nyilvántartás): Excel
    reg = pd.DataFrame([{
        "azonosito": s.azon, "iskola": s.nev, "varos": s.varos, "vizsgahely": s.vizsgahely,
        "kategoriak": ", ".join(s.kategoriak), "motoros": "igen" if s.motoros else "nem",
        "tandij_b_ft": s.tandij, "potora_ft": s.potora, "lemondas_ora": s.lemondas_ora,
        "arlista_ervenyes": s.ervenyes, "oktatasvezeto": s.vezeto,
    } for s in schools])
    reg.to_excel(DATA / "tables" / "iskola-nyilvantartas.xlsx", index=False)

    # 3) Vizsgaeredmények: nagy CSV (30 ezer sor)
    start = date(2024, 1, 1)
    with open(DATA / "tables" / "vizsgaeredmenyek.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vizsgaazonosito", "datum", "vizsgahely", "iskola", "iskola_azonosito",
                    "kategoria", "vizsgatipus", "vezetett_orak", "eredmeny"])
        for i in range(N_EXAMS):
            s = rng.choice(schools)
            datum = start + timedelta(days=rng.randint(0, 1033))  # 2024-01-01 .. 2026-10-29
            kategoria = "B" if rng.random() < 0.75 else rng.choice(s.kategoriak)
            tipus = "elmélet" if rng.random() < 0.4 else "forgalmi"
            orak = 0 if tipus == "elmélet" else int(rng.triangular(30, 64, 38))
            site = s.vizsgahely.removesuffix(" vizsgaközpont")
            ok = rng.random() < pass_prob(site, tipus, kategoria, orak, datum.year)
            w.writerow([f"VIZS-{datum.year}-{i + 1:06d}", datum.isoformat(), s.vizsgahely, s.nev, s.azon,
                        kategoria, tipus, orak, "sikeres" if ok else "sikertelen"])

    # 4) Kis PDF (közvetlenül a kontextusba) és nagy PDF (az összes tájékoztató)
    small = new_pdf()
    write_markdown_pdf(small, COURSE)
    small.output(str(DATA / "docs" / "tanfolyami-tajekoztato.pdf"))

    big = new_pdf()
    for s in schools:
        for _ in range(2):  # minden tájékoztató + egy "melléklet" oldal, hogy vastag legyen
            write_markdown_pdf(big, school_markdown(s))
    big.output(str(DATA / "docs" / "tajekoztatok-osszes.pdf"))

    print(f"Kész: {len(schools)} iskola, {N_EXAMS} vizsgaeredmény, 2 PDF "
          f"({small.page} oldal a kicsi, {big.page} oldal a nagy).")
    print(f"Ellenőrző érték: {sum(s.motoros for s in schools)} iskola tart motoros képzést.")


if __name__ == "__main__":
    main()
