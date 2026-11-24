"""Docker sandbox egy beszélgetéshez: az agent Python kódot futtat benne a betöltött fájlokon.

Amit élesben is meg kell oldani, és itt meg van oldva:
- izoláció: nincs hálózat, read-only fájlrendszer, nem root user, CPU/memória/processz-limit, timeout;
- jogosultság: csak a tenant saját mappájából lehet fájlt betölteni (path traversal ellen is);
- keret (budget): max fájlszám, max összméret, max futtatás, és MINDEN válasz jelzi a maradékot;
  élesben inkább minőségileg ("van hely" / "megtelt"), mert a modell a számot idézi a usernek;
- túl hosszú kimenet: nem vágjuk le csendben, hanem hibaként visszaadjuk a modellnek,
  hogy írjon jobb kódot (aggregáljon, ne dumpoljon); hibánál a traceback megy előre;
- ha a sandbox maga hibázik, azt nem kódhibaként adjuk vissza (különben a modell "javítja").
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE = "fajl-agent-sandbox"


@dataclass
class ToolResult:
    content: str | list  # szöveg, vagy content blokkok listája (pl. search_result)
    is_error: bool = False


class SandboxError(Exception):
    pass


class DockerSandbox:
    def __init__(self, tenant_root: Path, *, max_files: int = 5, max_bytes: int = 50_000_000,
                 max_runs: int = 10, max_output_chars: int = 4_000, timeout_s: int = 20,
                 image: str = IMAGE, show_numbers: bool = True):
        self.tenant_root = tenant_root.resolve()
        self.max_files, self.max_bytes, self.max_runs = max_files, max_bytes, max_runs
        self.max_output_chars, self.timeout_s, self.image = max_output_chars, timeout_s, image
        self.show_numbers = show_numbers  # demóban látványos; élesben False
        self.name = f"sbx-{uuid.uuid4().hex[:8]}"  # egy beszélgetés = egy konténer
        self.workdir = Path(tempfile.mkdtemp(prefix=f"{self.name}-"))
        self.loaded: dict[str, int] = {}
        self.runs = 0

    # ---- életciklus ----
    def __enter__(self) -> "DockerSandbox":
        self.workdir.chmod(0o755)
        cmd = [
            "docker", "run", "-d", "--rm", "--name", self.name,
            "--network", "none",                 # nincs internet: nem tud adatot kivinni
            "--read-only", "--tmpfs", "/tmp:size=64m",
            "--memory", "512m", "--cpus", "1", "--pids-limit", "128",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534",             # nobody
            "-v", f"{self.workdir}:/data:ro",    # a betöltött fájlok, csak olvasásra
            "-w", "/tmp", self.image, "sleep", "infinity",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SandboxError(f"Nem indult a sandbox: {r.stderr.strip()}\n"
                               f"Építsd meg: docker build -t {self.image} sandbox/")
        return self

    def __exit__(self, *exc) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        shutil.rmtree(self.workdir, ignore_errors=True)

    # ---- keret ----
    def budget_line(self) -> str:
        used = sum(self.loaded.values())
        if not self.show_numbers:
            full = len(self.loaded) >= self.max_files or used >= self.max_bytes
            return "[keret] A sandbox megtelt." if full else "[keret] Van még hely fájloknak."
        return (f"[keret] fájlok: {len(self.loaded)}/{self.max_files} · "
                f"méret: {used / 1e6:.1f}/{self.max_bytes / 1e6:.0f} MB · "
                f"futtatások: {self.runs}/{self.max_runs}")

    # ---- toolok ----
    def load_file(self, rel_path: str) -> ToolResult:
        """Átmásol egy fájlt a tenant tárhelyéről a sandboxba (/data/<név>)."""
        src = (self.tenant_root / rel_path).resolve()
        if not src.is_relative_to(self.tenant_root) or not src.is_file():
            return ToolResult(f"Nincs ilyen fájl ebben a projektben: {rel_path}\n{self.budget_line()}", True)
        if src.name in self.loaded:
            return ToolResult(f"Már betöltve: /data/{src.name}\n{self.budget_line()}")
        size = src.stat().st_size
        if len(self.loaded) >= self.max_files or sum(self.loaded.values()) + size > self.max_bytes:
            return ToolResult(f"A keret nem engedi ({size / 1e6:.1f} MB). Dolgozz a már betöltött "
                              f"fájlokkal.\n{self.budget_line()}", True)
        shutil.copy(src, self.workdir / src.name)
        self.loaded[src.name] = size
        return ToolResult(f"Betöltve: /data/{src.name} ({size / 1e6:.2f} MB)\n{self.budget_line()}")

    def run_python(self, code: str) -> ToolResult:
        if self.runs >= self.max_runs:
            return ToolResult(f"Elfogyott a futtatási keret. Válaszolj a meglévő eredményekből.\n"
                              f"{self.budget_line()}", True)
        self.runs += 1
        try:
            r = subprocess.run(
                ["docker", "exec", "-i", self.name, "timeout", str(self.timeout_s), "python", "-"],
                input=code, capture_output=True, text=True, timeout=self.timeout_s + 5,
            )
        except FileNotFoundError:
            return ToolResult("A sandbox nem elérhető. Ez NEM a kódod hibája, ne javítsd; "
                              "jelezd a felhasználónak.", True)
        except subprocess.TimeoutExpired:
            return ToolResult(f"Időtúllépés ({self.timeout_s} s).\n{self.budget_line()}", True)
        if r.returncode == 124:
            return ToolResult(f"Időtúllépés ({self.timeout_s} s). Írj gyorsabb kódot.\n"
                              f"{self.budget_line()}", True)
        if r.returncode == 125 or "No such container" in r.stderr:
            return ToolResult("A sandbox nem fut. Ez NEM a kódod hibája, ne javítsd.", True)
        # Hiba esetén a traceback megy ELŐRE: ha vágni kell, ne a hiba oka vesszen el.
        out = (("[stderr]\n" + r.stderr[-2000:] + "\n[stdout]\n") if r.returncode else "") + r.stdout
        out = out.strip()
        if r.returncode == 0 and len(out) > self.max_output_chars:
            # NEM vágjuk le csendben: a modellnek tudnia kell, hogy a kódja rossz választ adott.
            return ToolResult(
                f"A kimenet túl hosszú ({len(out)} karakter), nem adom vissza. "
                f"Ne írd ki a nyers adatot: szűrj, aggregálj, és csak az eredményt printeld.\n"
                f"Eleje: {out[:300]}...\n{self.budget_line()}", True)
        if len(out) > self.max_output_chars:  # hibás futás: traceback marad, a stdout vége megy
            out = out[:self.max_output_chars] + "\n[...a kimenet többi része levágva: printelj kevesebbet]"
        return ToolResult(f"{out or '(nincs kimenet: használj print-et)'}\n{self.budget_line()}",
                          r.returncode != 0)
