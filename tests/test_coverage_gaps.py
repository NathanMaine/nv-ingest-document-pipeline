# SPDX-FileCopyrightText: Copyright (c) 2026 Nathan Maine
# SPDX-License-Identifier: Apache-2.0

"""Targeted tests to close the gap from 85% → 100% statement coverage.

These tests cover paths not exercised by the main test files:
  - extractor._extract_with_nv_ingest (real method, mocked nv_ingest_client)
  - converter._chunk_text edge cases (empty paragraphs, exact-boundary chunks)
  - converter._split_long_paragraph fallback (single sentence > max_length)
  - benchmark error paths (PdfReader exceptions, nv-ingest exceptions)
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.converter import DocumentConverter
from src.extractor import (
    DocumentExtractor,
    ExtractedContent,
    ExtractionResult,
)
from src.benchmark import (
    _compute_metrics,
    _write_results,
    BenchmarkComparison,
    BenchmarkMetrics,
    extract_baseline,
    format_results_table,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# extractor._extract_with_nv_ingest — exercise the real method via mocks
# ---------------------------------------------------------------------------
def _install_fake_nv_ingest_module(raw_results):
    """Install a fake nv_ingest_client module so the import inside
    _extract_with_nv_ingest resolves to a controllable stub.
    """
    fake_module = types.ModuleType("nv_ingest_client.client.interface")

    class FakeIngestor:
        def __init__(self):
            self._files = []
            self._extract_kwargs = {}

        def files(self, paths):
            self._files = paths
            return self

        def extract(self, **kwargs):
            self._extract_kwargs = kwargs
            return self

        def ingest(self):
            return raw_results

    fake_module.Ingestor = FakeIngestor

    parent_module = types.ModuleType("nv_ingest_client.client")
    parent_module.interface = fake_module
    grandparent_module = types.ModuleType("nv_ingest_client")
    grandparent_module.client = parent_module

    return {
        "nv_ingest_client": grandparent_module,
        "nv_ingest_client.client": parent_module,
        "nv_ingest_client.client.interface": fake_module,
    }


def test_extract_with_nv_ingest_single_pdf(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    raw = [
        {
            "content": [
                {"type": "text", "content": "page 1 text", "page_number": 1},
                {"type": "table", "content": "| a | b |", "page_number": 1},
            ],
            "processing_metadata": {"total_pages": 1},
        }
    ]

    fake_modules = _install_fake_nv_ingest_module(raw)
    with patch.dict(sys.modules, fake_modules):
        e = DocumentExtractor()
        results = e._extract_with_nv_ingest([pdf])

    assert len(results) == 1
    assert results[0].text_count == 1
    assert results[0].table_count == 1
    assert results[0].source_file == str(pdf)
    assert results[0].processing_time_ms >= 0


def test_extract_with_nv_ingest_multiple_pdfs(tmp_path):
    pdf1 = tmp_path / "a.pdf"
    pdf2 = tmp_path / "b.pdf"
    pdf1.write_bytes(b"%PDF-1.4")
    pdf2.write_bytes(b"%PDF-1.4")

    raw = [
        {
            "content": [{"type": "text", "content": "doc a", "page_number": 1}],
            "processing_metadata": {"total_pages": 1},
        },
        {
            "content": [{"type": "text", "content": "doc b", "page_number": 1}],
            "processing_metadata": {"total_pages": 1},
        },
    ]

    fake_modules = _install_fake_nv_ingest_module(raw)
    with patch.dict(sys.modules, fake_modules):
        e = DocumentExtractor()
        results = e._extract_with_nv_ingest([pdf1, pdf2])

    assert len(results) == 2
    # Each PDF gets the elapsed_ms split across input count
    assert results[0].processing_time_ms == results[1].processing_time_ms


def test_extract_with_nv_ingest_handles_short_raw_results(tmp_path):
    """If nv-ingest returns fewer results than inputs, missing slots default to {}."""
    pdf1 = tmp_path / "a.pdf"
    pdf2 = tmp_path / "b.pdf"
    pdf1.write_bytes(b"%PDF-1.4")
    pdf2.write_bytes(b"%PDF-1.4")

    raw = [
        {
            "content": [{"type": "text", "content": "only a", "page_number": 1}],
            "processing_metadata": {"total_pages": 1},
        }
    ]

    fake_modules = _install_fake_nv_ingest_module(raw)
    with patch.dict(sys.modules, fake_modules):
        e = DocumentExtractor()
        results = e._extract_with_nv_ingest([pdf1, pdf2])

    assert len(results) == 2
    assert results[0].text_count == 1
    assert results[1].text_count == 0  # missing raw -> empty content


def test_extract_passes_extractor_flags_to_ingestor(tmp_path):
    """Verify extract_charts and extract_images flags propagate."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    captured = {}

    fake_module = types.ModuleType("nv_ingest_client.client.interface")

    class FakeIngestor:
        def files(self, paths):
            return self

        def extract(self, **kwargs):
            captured.update(kwargs)
            return self

        def ingest(self):
            return [{"content": [], "processing_metadata": {"total_pages": 0}}]

    fake_module.Ingestor = FakeIngestor

    fakes = {
        "nv_ingest_client": types.ModuleType("nv_ingest_client"),
        "nv_ingest_client.client": types.ModuleType("nv_ingest_client.client"),
        "nv_ingest_client.client.interface": fake_module,
    }
    fakes["nv_ingest_client"].client = fakes["nv_ingest_client.client"]
    fakes["nv_ingest_client.client"].interface = fake_module

    with patch.dict(sys.modules, fakes):
        e = DocumentExtractor(extract_charts=False, extract_images=True)
        e._extract_with_nv_ingest([pdf])

    assert captured["extract_text"] is True
    assert captured["extract_tables"] is True
    assert captured["extract_charts"] is False
    assert captured["extract_images"] is True


# ---------------------------------------------------------------------------
# converter._chunk_text — edge cases
# ---------------------------------------------------------------------------
def test_chunk_text_at_exact_max_length():
    """A paragraph exactly at max_length should not split."""
    c = DocumentConverter(min_length=10, max_length=100)
    text = "a" * 100
    chunks = c._chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_two_paragraphs_combined_under_limit():
    """Two paragraphs that together fit under max_length combine into one chunk."""
    c = DocumentConverter(min_length=10, max_length=200)
    text = ("a" * 50) + "\n\n" + ("b" * 50)
    chunks = c._chunk_text(text)
    assert len(chunks) == 1
    assert "a" * 50 in chunks[0]
    assert "b" * 50 in chunks[0]


def test_chunk_text_two_paragraphs_combined_over_limit():
    """Two paragraphs that together exceed max_length split into two chunks."""
    c = DocumentConverter(min_length=10, max_length=80)
    text = ("a" * 50) + "\n\n" + ("b" * 50)
    chunks = c._chunk_text(text)
    assert len(chunks) == 2


def test_chunk_text_drops_chunks_below_min_length():
    """A chunk that ends up shorter than min_length is dropped."""
    c = DocumentConverter(min_length=200, max_length=400)
    text = ("a" * 100) + "\n\n" + ("b" * 300)
    chunks = c._chunk_text(text)
    # The first 100-char chunk should be dropped (below min_length=200)
    assert all(len(ch) >= 200 for ch in chunks)


# ---------------------------------------------------------------------------
# converter._split_long_paragraph — sentence-boundary fallback
# ---------------------------------------------------------------------------
def test_split_long_paragraph_on_sentences():
    """A paragraph too long for max_length splits on sentence boundaries."""
    c = DocumentConverter(min_length=10, max_length=50)
    text = "First sentence. Second sentence. Third sentence."
    chunks = c._split_long_paragraph(text)
    assert len(chunks) >= 1
    # No chunk should grossly exceed max_length
    for ch in chunks:
        assert len(ch) <= 100  # generous allowance for sentence boundaries


def test_split_long_paragraph_hard_split_on_giant_sentence():
    """A single sentence longer than max_length gets hard-split into chunks."""
    c = DocumentConverter(min_length=10, max_length=20)
    text = "a" * 100  # one giant "sentence" (no terminator)
    chunks = c._split_long_paragraph(text)
    assert len(chunks) >= 5  # 100 / 20 = 5 chunks at minimum
    for ch in chunks:
        assert len(ch) <= 20


def test_split_long_paragraph_combines_short_sentences():
    """Multiple short sentences combine into a chunk under max_length."""
    c = DocumentConverter(min_length=5, max_length=50)
    text = "Short. Also short. Tiny."
    chunks = c._split_long_paragraph(text)
    # Total content fits in a single chunk
    assert len(chunks) == 1


def test_chunk_text_routes_giant_paragraph_to_split_long():
    """Verify the integration: a giant paragraph triggers _split_long_paragraph."""
    c = DocumentConverter(min_length=10, max_length=30)
    text = "Normal paragraph here.\n\n" + ("x" * 200)
    chunks = c._chunk_text(text)
    # Should produce multiple chunks, with the giant paragraph hard-split
    assert len(chunks) >= 5


# ---------------------------------------------------------------------------
# benchmark — error paths
# ---------------------------------------------------------------------------
def test_run_benchmark_handles_baseline_oserror(tmp_path):
    """If baseline raises OSError, benchmark continues with nv_ingest only."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with patch("src.benchmark.extract_baseline", side_effect=OSError("disk")):
        with patch("src.benchmark.DocumentExtractor") as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.extract.return_value = []
            mock_extractor.return_value = mock_instance
            comparisons = run_benchmark([pdf])

    assert comparisons[0].baseline is None


def test_run_benchmark_handles_nv_ingest_oserror(tmp_path):
    """If nv-ingest raises OSError, benchmark continues with baseline only."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    fake_baseline = ExtractionResult("x.pdf", 1, "pypdf2", [], 50.0)
    with patch("src.benchmark.extract_baseline", return_value=fake_baseline):
        with patch("src.benchmark.DocumentExtractor") as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.extract.side_effect = OSError("nv-ingest disk error")
            mock_extractor.return_value = mock_instance
            comparisons = run_benchmark([pdf])

    assert comparisons[0].nv_ingest is None
    assert comparisons[0].baseline is not None


def test_run_benchmark_handles_nv_ingest_returning_empty_list(tmp_path):
    """If nv-ingest succeeds but returns [], nv_ingest metrics stay None."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    fake_baseline = ExtractionResult("x.pdf", 1, "pypdf2", [], 50.0)
    with patch("src.benchmark.extract_baseline", return_value=fake_baseline):
        with patch("src.benchmark.DocumentExtractor") as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.extract.return_value = []
            mock_extractor.return_value = mock_instance
            comparisons = run_benchmark([pdf])

    assert comparisons[0].nv_ingest is None


def test_write_results_includes_both_methods(tmp_path):
    """_write_results renders both baseline and nv_ingest entries with speedup."""
    out = tmp_path / "results.json"

    baseline = BenchmarkMetrics(
        method="pypdf2", source_file="x.pdf", extraction_time_ms=1000.0,
        text_blocks=10, table_blocks=0, total_chars=5000, total_pages=10,
        whitespace_ratio=0.1,
    )
    nv = BenchmarkMetrics(
        method="nv-ingest", source_file="x.pdf", extraction_time_ms=200.0,
        text_blocks=10, table_blocks=2, total_chars=6000, total_pages=10,
        whitespace_ratio=0.1,
    )
    comp = BenchmarkComparison(source_file="x.pdf", nv_ingest=nv, baseline=baseline)
    _write_results([comp], out)

    import json
    data = json.loads(out.read_text())
    assert data[0]["baseline"]["method"] == "pypdf2"
    assert data[0]["nv_ingest"]["method"] == "nv-ingest"
    assert data[0]["speedup_x"] == 5.0
    assert data[0]["text_yield_ratio"] == 1.2


def test_extract_baseline_real_invocation_with_mocked_pypdf2(tmp_path):
    """Cover the extract_baseline body — including the time accounting."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    fake_page1 = MagicMock()
    fake_page1.extract_text.return_value = "page one text"
    fake_page2 = MagicMock()
    fake_page2.extract_text.return_value = "  "  # whitespace-only, should be skipped
    fake_page3 = MagicMock()
    fake_page3.extract_text.return_value = ""  # empty, should be skipped

    fake_reader = MagicMock()
    fake_reader.pages = [fake_page1, fake_page2, fake_page3]

    fake_pypdf2 = types.ModuleType("PyPDF2")
    fake_pypdf2.PdfReader = MagicMock(return_value=fake_reader)

    with patch.dict(sys.modules, {"PyPDF2": fake_pypdf2}):
        result = extract_baseline(pdf)

    assert result.total_pages == 3
    assert result.text_count == 1  # Only page 1 had non-empty content
    assert result.contents[0].text == "page one text"
    assert result.processing_time_ms >= 0
