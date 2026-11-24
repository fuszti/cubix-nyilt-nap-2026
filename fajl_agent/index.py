"""Hibrid keresés: dense (embedding) + BM25 (kulcsszó), Reciprocal Rank Fusion-nel összefésülve.

- Az embedding a jelentést találja meg ("lemondás" ~ "visszamondás").
- A BM25 a pontos tokeneket: azonosító, iskolanév, vizsgaszám.
Egyik sem tud számolni, szűrni dátumra, vagy összeadni.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE = Path(__file__).resolve().parent.parent / ".cache"

_model = None


def embed(texts: list[str]) -> np.ndarray:
    """Lokális, ingyenes, offline embedding (fastembed, ONNX). Lemezre cache-el."""
    global _model
    CACHE.mkdir(exist_ok=True)
    key = hashlib.sha256(("\x00".join(texts) + EMBED_MODEL).encode()).hexdigest()[:16]
    path = CACHE / f"emb-{key}.npy"
    if path.exists():
        return np.load(path)
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(EMBED_MODEL)
    vecs = np.array(list(_model.embed(texts)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(path, vecs)
    return vecs


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wáéíóöőúüű-]+", text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None
    bm25_rank: int | None


class HybridIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vecs = embed([c.text for c in chunks])
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               where: dict | None = None) -> list[Hit]:
        """mode: 'dense' | 'bm25' | 'hybrid'. A `where` metaadat-szűrő MINDIG a rangsorolás
        előtt fut (pl. tenant, projekt): ami nem a felhasználóé, be sem kerül a jelöltek közé."""
        allowed = np.array([
            all(c.meta.get(k_) == v for k_, v in (where or {}).items()) for c in self.chunks
        ])
        q = embed([query])[0]
        dense = np.where(allowed, self.vecs @ q, -np.inf)
        sparse = np.where(allowed, self.bm25.get_scores(tokenize(query)), -np.inf)
        d_rank = {int(i): r for r, i in enumerate(np.argsort(-dense))}
        b_rank = {int(i): r for r, i in enumerate(np.argsort(-sparse))}

        if mode == "dense":
            order = np.argsort(-dense)
            scores = dense
        elif mode == "bm25":
            order = np.argsort(-sparse)
            scores = sparse
        else:  # Reciprocal Rank Fusion: 1/(60+rang) összege
            rrf = np.array([1 / (60 + d_rank[i]) + 1 / (60 + b_rank[i]) for i in range(len(self.chunks))])
            rrf = np.where(allowed, rrf, -np.inf)
            order = np.argsort(-rrf)
            scores = rrf
        return [
            Hit(self.chunks[i], float(scores[i]), d_rank[int(i)], b_rank[int(i)])
            for i in order[:k] if np.isfinite(scores[i])
        ]
