# SPDX-FileCopyrightText: Copyright (c) 2026 Nathan Maine
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for src.benchmark — all run on CPU, no GPU or nv-ingest needed."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.benchmark import (
    BenchmarkComparison,
    BenchmarkMetrics,
    _compute_metrics,
    extract_baseline,
    format_results_table,
    run_benchmark,
)
from src.extractor import ExtractedContent, ExtractionResult


# ---------------------------------------------------------------------------
# BenchmarkComparison properties
# ---------------------------------------------------------------------------
def _metrics(method: str, time_ms: float, chars: int = 1000) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        method=method,
        source_file="x.pdf",
        extraction_time_ms=time_ms,
        text_blocks=10,
        table_blocks=2,
        total_chars=chars,
        total_pages=5,
        whitespace_ratio=0.15,
    )


def test_speedup_ratio():
    comp = BenchmarkComparison(
        source_file="x.pdf",
        nv_ingest=_metrics("nv-ingest", 200.0),
        baseline=_metrics("pypdf2", 1000.0),
    )
    assert comp.speedup == 5.0


def test_speedup_when_nv_ingest_is_zero():
    comp = BenchmarkComparison(
        source_file="x.pdf",
        nv_ingest=_metrics("nv-ingest", 0.0),
        baseline=_metrics("pypdf2", 1000.0),
    )
    assert comp.speedup is None


def test_speedup_when_only_one_present():
    comp = BenchmarkComparison(
        source_file="x.pdf", nv_ingest=_metrics("nv-ingest", 100.0), baseline=None
    )
    assert comp.speedup is None


def test_text_yield_ratio():
    comp = BenchmarkComparison(
        source_file="x.pdf",
        nv_ingest=_metrics("nv-ingest", 100.0, chars=2000),
        baseline=_metrics("pypdf2", 500.0, chars=1000),
    )
    assert comp.text_yield_ratio == 2.0


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------
def test_compute_metrics_empty():
    result = ExtractionResult("x.pdf", 0, "test", [], 50.0)
    m = _compute_metrics(result, "test")
    assert m.total_chars == 0
    assert m.whitespace_ratio == 0.0


def test_compute_metrics_basic():
    contents = [
        ExtractedContent("text", "Hello world", 1, "x.pdf"),
        ExtractedContent("table", "row 1", 2, "x.pdf"),
    ]
    result = ExtractionResult("x.pdf", 5, "pdfium_hybrid", contents, 100.0)
    m = _compute_metrics(result, "nv-ingest")
    assert m.text_blocks == 1
    assert m.table_blocks == 1
    assert m.total_chars == len("Hello world row 1")
    assert m.total_pages == 5


# ---------------------------------------------------------------------------
# extract_baseline (mocked PyPDF2)
# ---------------------------------------------------------------------------
def test_extract_baseline_raises_without_pypdf2(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with patch.dict("sys.modules", {"PyPDF2": None}):
        with pytest.raises(ImportError):
            extract_baseline(pdf)


def test_extract_baseline_with_mock_pypdf2(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "page text"
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page, fake_page]
    with patch("src.benchmark.PdfReader", create=True, return_value=fake_reader):
        # Simulate the import inside extract_baseline
        with patch.dict("sys.modules", {"PyPDF2": MagicMock(PdfReader=lambda _: fake_reader)}):
            result = extract_baseline(pdf)
    assert result.total_pages == 2
    assert result.text_count == 2
    assert result.contents[0].text == "page text"


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------
def test_run_benchmark_handles_missing_pypdf2(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with patch("src.benchmark.extract_baseline", side_effect=ImportError):
        with patch("src.benchmark.DocumentExtractor") as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.extract.return_value = []
            mock_extractor.return_value = mock_instance
            comparisons = run_benchmark([pdf])
    assert len(comparisons) == 1
    assert comparisons[0].baseline is None


def test_run_benchmark_handles_missing_nv_ingest(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    fake_baseline = ExtractionResult("x.pdf", 1, "pypdf2", [], 50.0)
    with patch("src.benchmark.extract_baseline", return_value=fake_baseline):
        with patch("src.benchmark.DocumentExtractor") as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.extract.side_effect = ImportError
            mock_extractor.return_value = mock_instance
            comparisons = run_benchmark([pdf])
    assert comparisons[0].nv_ingest is None
    assert comparisons[0].baseline is not None


def test_run_benchmark_writes_json_when_path_given(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    out = tmp_path / "results.json"
    fake_baseline = ExtractionResult("x.pdf", 1, "pypdf2", [], 50.0)
    with patch("src.benchmark.extract_baseline", return_value=fake_baseline):
        with patch("src.benchmark.DocumentExtractor") as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.extract.return_value = []
            mock_extractor.return_value = mock_instance
            run_benchmark([pdf], output_path=out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert data[0]["source_file"].endswith("x.pdf")


# ---------------------------------------------------------------------------
# format_results_table
# ---------------------------------------------------------------------------
def test_format_results_table_with_both_methods():
    comp = BenchmarkComparison(
        source_file="example.pdf",
        nv_ingest=_metrics("nv-ingest", 200.0),
        baseline=_metrics("pypdf2", 1000.0),
    )
    out = format_results_table([comp])
    assert "PyPDF2" in out
    assert "nv-ingest" in out
    assert "5.0x" in out  # speedup column


def test_format_results_table_empty():
    out = format_results_table([])
    assert out.startswith("| Document |")
