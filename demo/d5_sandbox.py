"""3. út: kódíró agent sandboxban. A fájl NEM megy a kontextusba: az agent kódot ír,
lefuttatja egy izolált konténerben, a hibából javít, és csak az eredmény jön vissza.

  docker build -t fajl-agent-sandbox sandbox/       # egyszer
  uv run python demo/d5_sandbox.py                  # agent (API-kulcs kell)
  uv run python demo/d5_sandbox.py --offline        # a mechanika modell nélkül
"""

import sys

from fajl_agent.agent import run_agent
from fajl_agent.corpus import DATA
from fajl_agent.llm import Recorder, run_demo
from fajl_agent.sandbox import DockerSandbox

QUESTION = ("Hány iskola tart motoros (A) képzést, és mennyi náluk átlagosan a B kategóriás tandíj? "
            "És melyik vizsgahelyen volt 2026-ban a legalacsonyabb a B kategóriás forgalmi vizsgák sikeraránya?")
SYSTEM = """Adatelemző agent vagy. A projekt fájljai:
- tables/iskola-nyilvantartas.xlsx (az autósiskolák nyilvántartása)
- tables/vizsgaeredmenyek.csv (kb. 30 ezer vizsgaeredmény)
Számoláshoz töltsd be a fájlt a load_file toollal, majd futtass Python kódot (pandas) a run_python toollal.
A fájlok a /data mappában lesznek. Minden futtatás új folyamat: egy futtatásban legyen a teljes script.
Ne írd ki a nyers adatot, csak az eredményt. Nincs internet, nincs pip install."""

TOOLS = [
    {"name": "load_file", "description": "Egy projektfájl bemásolása a sandboxba (/data/<név>). "
     "Keretes: fájlszám- és méretlimit van, a válasz jelzi a maradékot.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "run_python", "description": "Python script futtatása az izolált sandboxban. "
     "A stdout-ot kapod vissza; ha túl hosszú, hibát kapsz: akkor aggregálj.",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
]


def offline(sbx: DockerSandbox) -> None:
    """Ugyanazok a lépések, amiket egy agent tenne, kézzel: így látszik a mechanika."""
    steps = [
        ("load_file", "tables/vizsgaeredmenyek.csv"),
        ("load_file", "../../../etc/passwd"),                       # jogosultság: nem engedi
        ("run_python", "import pandas as pd\ndf = pd.read_csv('/data/vizsgaeredmenyek.csv')\n"
                       "print(df[df.vizsga_tipus == 'forgalmi'].eredmeny.value_counts())"),  # rossz oszlopnév
        ("run_python", "import pandas as pd\nprint(pd.read_csv('/data/vizsgaeredmenyek.csv').to_string())"),  # dump
        ("run_python", "import pandas as pd\ndf = pd.read_csv('/data/vizsgaeredmenyek.csv')\n"
                       "f = df[(df.vizsgatipus == 'forgalmi') & (df.kategoria == 'B') & df.datum.str.startswith('2026')]\n"
                       "print((f.eredmeny == 'sikeres').groupby(f.vizsgahely).mean().nsmallest(3).round(3))"),
        ("run_python", "import urllib.request\nurllib.request.urlopen('https://example.com', timeout=3)"),
    ]
    for tool, arg in steps:
        print(f"\033[36m▶ {tool}\033[0m\n{arg}")
        r = sbx.load_file(arg) if tool == "load_file" else sbx.run_python(arg)
        print(f"\033[33m{'✗' if r.is_error else '◀'} {r.content[-700:]}\033[0m\n")


def main() -> None:
    with DockerSandbox(DATA) as sbx:
        if "--offline" in sys.argv:
            offline(sbx)
            return
        answer = run_agent(Recorder("d5_sandbox"), system=SYSTEM, tools=TOOLS,
                           handlers={"load_file": lambda path: sbx.load_file(path),
                                     "run_python": lambda code: sbx.run_python(code)},
                           user_content=QUESTION)
        print(f"\nVÁLASZ:\n{answer}")


if __name__ == "__main__":
    run_demo(main)
