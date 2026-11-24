"""A mechanika tesztjei. API-kulcs nem kell; a sandbox-tesztekhez Docker és a megépített image."""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from anthropic.types.beta import BetaMessage

from fajl_agent import agent as agent_mod
from fajl_agent.catalog import Catalog
from fajl_agent.chunking import chunk_markdown_sections, chunk_table_naive, chunk_table_rows
from fajl_agent.sandbox import IMAGE, DockerSandbox, ToolResult

DOC = """# Tanfolyami tájékoztató – Próba Autósiskola (ISK-9999)

**Iskola:** Próba Autósiskola, Eger · oktatásvezető: Kiss Anna

## 1. Képzéseink
Kategóriák: B.

## 3. Pótóra és lemondás
A vezetési órát legkésőbb 48 órával előtte lehet díjmentesen lemondani.
"""


def test_section_chunk_carries_document_context():
    chunks = chunk_markdown_sections("ISK-9999", DOC)
    assert len(chunks) == 2
    lemondas = chunks[1]
    assert "ISK-9999" in lemondas.text and "Eger" in lemondas.text
    assert lemondas.meta["section"] == "3. Pótóra és lemondás"


def test_row_chunks_keep_column_names_naive_does_not():
    df = pd.DataFrame({"vizsgaazonosito": ["V-1", "V-2"], "vezetett_orak": [31, 42]})
    rows = chunk_table_rows("t.csv", df)
    assert rows[1].text == "vizsgaazonosito: V-2; vezetett_orak: 42"
    naive = chunk_table_naive("t.csv", df.to_csv(index=False), size=20)
    assert "vezetett_orak" not in naive[-1].text


def test_catalog_is_tenant_scoped_and_profiles_tables(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "vizsgaeredmenyek.csv").write_text("id,orak\n1,30\n2,40\n")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "vizsgaeredmenyek-titkos.csv").write_text("id\n1\n")
    cat = Catalog()
    cat.add_dir("a", tmp_path / "a")
    cat.add_dir("b", tmp_path / "b")
    found = cat.find("a", "VIZSGA")
    assert [f["name"] for f in found] == ["vizsgaeredmenyek.csv"]
    assert "2 sor" in found[0]["profile"]


def _msg(content, stop):
    return BetaMessage.model_validate({
        "id": "msg_x", "type": "message", "role": "assistant", "model": "teszt",
        "content": content, "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })


def test_agent_loop_runs_tool_and_returns_error_as_observation():
    scripted = [
        _msg([{"type": "tool_use", "id": "tu_1", "name": "boom", "input": {}}], "tool_use"),
        _msg([{"type": "text", "text": "Kész."}], "end_turn"),
    ]
    sent = []

    class FakeRecorder:
        def create(self, **kw):
            sent.append([dict(m) if isinstance(m, dict) else m for m in kw["messages"]])
            return scripted.pop(0)

    def boom():
        raise ValueError("rossz oszlop")

    answer = agent_mod.run_agent(FakeRecorder(), system="", tools=[], handlers={"boom": boom},
                                 user_content="?")
    assert answer == "Kész."
    tool_result = sent[1][-1]["content"][0]
    assert tool_result["is_error"] is True and "rossz oszlop" in tool_result["content"]


docker_ok = shutil.which("docker") and subprocess.run(
    ["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
needs_docker = pytest.mark.skipif(not docker_ok, reason=f"Docker vagy a {IMAGE} image hiányzik")


@pytest.fixture
def sbx(tmp_path):
    (tmp_path / "adat.csv").write_text("a,b\n1,2\n3,4\n")
    (tmp_path / "masik.csv").write_text("x\n1\n")
    with DockerSandbox(tmp_path, max_files=1, max_output_chars=200, timeout_s=8) as s:  # a pandas import ~2,5 s
        yield s


@needs_docker
def test_sandbox_computes_on_loaded_file(sbx):
    assert not sbx.load_file("adat.csv").is_error
    r = sbx.run_python("import pandas as pd\nprint(pd.read_csv('/data/adat.csv').b.sum())")
    assert not r.is_error and r.content.startswith("6")
    assert "[keret]" in r.content


@needs_docker
def test_sandbox_rejects_path_traversal_and_enforces_file_budget(sbx):
    assert sbx.load_file("../../etc/passwd").is_error
    assert not sbx.load_file("adat.csv").is_error
    r = sbx.load_file("masik.csv")
    assert r.is_error and "keret" in r.content


@needs_docker
def test_sandbox_too_long_output_is_an_error_not_silent_truncation(sbx):
    r = sbx.run_python("print('x' * 5000)")
    assert r.is_error and "túl hosszú" in r.content


@needs_docker
def test_sandbox_has_no_network_and_times_out(sbx):
    r = sbx.run_python("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)")
    assert r.is_error
    r = sbx.run_python("while True: pass")
    assert r.is_error and "Időtúllépés" in r.content


@needs_docker
def test_sandbox_failure_puts_traceback_first(sbx):
    r = sbx.run_python("print('y' * 500)\nraise KeyError('eredmeny')")
    assert r.is_error and r.content.startswith("[stderr]") and "KeyError" in r.content


def test_docling_demo_reads_category_headers():
    """A d6 demó fejléc-értelmezése (csak ha a docling csoport telepítve van)."""
    pytest.importorskip("docling")
    import importlib.util
    spec = importlib.util.spec_from_file_location("d6", Path(__file__).parent.parent / "demo" / "d6_docling.py")
    d6 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d6)
    assert d6.header_labels('"AM" "A1" "A1"  B-vel "A2"') == {"AM", "A1", "A1 B-vel", "A2"}
    assert d6.norm("Bács–Kiskun  6500") == "Bács-Kiskun6500"
