"""Fájlkatalógus: a "megvan-e az X fájl?" kérdés nem keresés, hanem lekérdezés.

Az embedding a fájlnévre rossz: a "vizsgaeredmenyek.csv" és a "vizsgaeredmények 2026" jelentésben
közel vannak, de a felhasználó egy konkrét fájlt kérdez. Erre egy SQL LIKE pontos és olcsó.
"""

from __future__ import annotations

import mimetypes
import sqlite3
from pathlib import Path

import pandas as pd

from .chunking import table_card

TABLE_EXT = {".csv", ".xlsx", ".xls", ".json"}


class Catalog:
    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.execute(
            "CREATE TABLE files (tenant TEXT, name TEXT, path TEXT, size_bytes INT, mime TEXT, "
            "profile TEXT)"
        )

    def add_dir(self, tenant: str, root: Path) -> None:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                self.db.execute(
                    "INSERT INTO files VALUES (?,?,?,?,?,?)",
                    (tenant, p.name, str(p.relative_to(root)), p.stat().st_size,
                     mimetypes.guess_type(p.name)[0] or "application/octet-stream", profile(p)),
                )

    def find(self, tenant: str, pattern: str, limit: int = 20) -> list[dict]:
        """Fájlnév-keresés egy tenanton belül. A tenant szűrés nem opcionális paraméter."""
        rows = self.db.execute(
            "SELECT name, path, size_bytes, mime, profile FROM files "
            "WHERE tenant = ? AND lower(name) LIKE ? ORDER BY name LIMIT ?",
            (tenant, f"%{pattern.lower()}%", limit),
        ).fetchall()
        return [dict(zip(["name", "path", "size_bytes", "mime", "profile"], r)) for r in rows]


def profile(p: Path) -> str | None:
    """Táblázatos fájlnál egy rövid adatlap (séma, sorszám), amit a fájl-lekérdezés visszaad:
    az agent ebből tudja, hogy számoláshoz a sandbox kell, és melyik oszlop mit jelent."""
    if p.suffix.lower() not in TABLE_EXT:
        return None
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)
    return table_card(p.name, df).text
