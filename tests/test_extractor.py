# SPDX-FileCopyrightText: Copyright (c) 2026 Nathan Maine
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for src.extractor — all run on CPU, no GPU or nv-ingest needed."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.extractor import (
    DocumentExtractor,
    ExtractedContent,
    ExtractionResult,
)


# ---------------------------------------------------------------------------
# ExtractedContent
# ---------------------------------------------------------------------------
def test_extracted_content_defaults():
    c = ExtractedContent(
        content_type="text", text="hello", page_number=1, source_file="x.pdf"
    )
    assert c.content_type == "text"
    assert c.metadata == {}


def test_extracted_content_with_metadata():
    c = ExtractedContent(
        content_type="table",
        text="| a | b |",
        page_number=2,
        source_file="x.pdf",
        metadata={"rows": 2, "cols": 2},
    )
    assert c.metadata["rows"] == 2


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------
def test_extraction_result_counts_empty():
    r = ExtractionResult(
        source_file="x.pdf", total_pages=10, extraction_method="pdfium_hybrid"
    )
    assert r.text_count == 0
    assert r.table_count == 0
    assert r.chart_count == 0


def test_extraction_result_counts_mixed():
    contents = [
        ExtractedContent("text", "a", 1, "x.pdf"),
        ExtractedContent("text", "b", 2, "x.pdf"),
        ExtractedContent("table", "| t |", 3, "x.pdf"),
        ExtractedContent("chart", "fig", 4, "x.pdf"),
    ]
    r = ExtractionResult(
        source_file="x.pdf",
        total_pages=4,
        extraction_method="pdfium_hybrid",
        contents=contents,
    )
    assert r.text_count == 2
    assert r.table_count == 1
    assert r.chart_count == 1


# ---------------------------------------------------------------------------
# DocumentExtractor — input validation
# ---------------------------------------------------------------------------
def test_extractor_default_init():
    e = DocumentExtractor()
    assert e.extraction_method == "pdfium_hybrid"
    assert e.extract_charts is True
    assert e.extract_images is False


def test_extractor_custom_init():
    e = DocumentExtractor(
        extraction_method="pdfium",
        extract_charts=False,
        extract_images=True,
    )
    assert e.extraction_method == "pdfium"
    assert e.extract_charts is False
    assert e.extract_images is True


def test_extract_raises_on_missing_file(tmp_path):
    e = DocumentExtractor()
    missing = tmp_path / "nope.pdf"
    with pytest.raises(FileNotFoundError):
        e.extract([missing])


def test_extract_raises_on_non_pdf(tmp_path):
    e = DocumentExtractor()
    txt = tmp_path / "doc.txt"
    txt.write_text("not a pdf")
    with pytest.raises(ValueError):
        e.extract([txt])


def test_extract_raises_when_nv_ingest_missing(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    e = DocumentExtractor()
    with patch.object(e, "_extract_with_nv_ingest", side_effect=ImportError("no")):
        with pytest.raises(ImportError):
            e.extract([pdf])


# ---------------------------------------------------------------------------
# _parse_nv_ingest_output — exercises the JSON shape parser
# ---------------------------------------------------------------------------
def test_parse_empty_output():
    e = DocumentExtractor()
    result = e._parse_nv_ingest_output({}, "x.pdf", 100.0)
    assert result.source_file == "x.pdf"
    assert result.total_pages == 0
    assert result.contents == []
    assert result.processing_time_ms == 100.0


def test_parse_with_text_content():
    raw = {
        "content": [
            {"type": "text", "content": "Hello world", "page_number": 1},
        ],
        "processing_metadata": {"total_pages": 5},
    }
    e = DocumentExtractor()
    result = e._parse_nv_ingest_output(raw, "x.pdf", 200.0)
    assert result.total_pages == 5
    assert len(result.contents) == 1
    assert result.contents[0].text == "Hello world"


def test_parse_with_mixed_content():
    raw = {
        "content": [
            {"type": "text", "content": "para 1", "page_number": 1},
            {"type": "table", "content": "| a | b |", "page_number": 2},
            {"type": "chart", "content": "figure 1", "page_number": 3},
        ],
        "processing_metadata": {"total_pages": 3},
    }
    e = DocumentExtractor()
    result = e._parse_nv_ingest_output(raw, "x.pdf", 300.0)
    assert result.text_count == 1
    assert result.table_count == 1
    assert result.chart_count == 1


def test_parse_skips_empty_content():
    raw = {
        "content": [
            {"type": "text", "content": "", "page_number": 1},
            {"type": "text", "content": "   ", "page_number": 2},
            {"type": "text", "content": "real content", "page_number": 3},
        ],
        "processing_metadata": {"total_pages": 3},
    }
    e = DocumentExtractor()
    result = e._parse_nv_ingest_output(raw, "x.pdf", 100.0)
    assert len(result.contents) == 1
    assert result.contents[0].text == "real content"


def test_parse_strips_whitespace():
    raw = {
        "content": [
            {"type": "text", "content": "  hello  \n", "page_number": 1},
        ],
        "processing_metadata": {"total_pages": 1},
    }
    e = DocumentExtractor()
    result = e._parse_nv_ingest_output(raw, "x.pdf", 100.0)
    assert result.contents[0].text == "hello"


def test_parse_preserves_metadata():
    raw = {
        "content": [
            {
                "type": "table",
                "content": "| a | b |",
                "page_number": 1,
                "metadata": {"rows": 2, "cols": 2},
            },
        ],
        "processing_metadata": {"total_pages": 1},
    }
    e = DocumentExtractor()
    result = e._parse_nv_ingest_output(raw, "x.pdf", 100.0)
    assert result.contents[0].metadata == {"rows": 2, "cols": 2}


def test_parse_default_type_is_text():
    raw = {
        "content": [{"content": "no type", "page_number": 1}],
        "processing_metadata": {"total_pages": 1},
    }
    e = DocumentExtractor()
    result = e._parse_nv_ingest_output(raw, "x.pdf", 100.0)
    assert result.contents[0].content_type == "text"


# ---------------------------------------------------------------------------
# extract_to_json
# ---------------------------------------------------------------------------
def test_extract_to_json_writes_file(tmp_path):
    out = tmp_path / "subdir" / "out.json"
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    fake_results = [
        ExtractionResult(
            source_file=str(pdf),
            total_pages=2,
            extraction_method="pdfium_hybrid",
            contents=[ExtractedContent("text", "hi", 1, str(pdf))],
            processing_time_ms=50.0,
        )
    ]
    e = DocumentExtractor()
    with patch.object(e, "extract", return_value=fake_results):
        result_path = e.extract_to_json([pdf], out)
    assert result_path == out
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["text_count"] == 1
    assert data[0]["contents"][0]["text"] == "hi"


def test_extract_to_json_creates_parent_dir(tmp_path):
    out = tmp_path / "deep" / "nested" / "out.json"
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    e = DocumentExtractor()
    with patch.object(e, "extract", return_value=[]):
        e.extract_to_json([pdf], out)
    assert out.parent.exists()


# ---------------------------------------------------------------------------
# Nice-to-have integration smoke
# ---------------------------------------------------------------------------
def test_extract_passes_paths_as_strings(tmp_path):
    """nv-ingest's Ingestor.files() expects strings, not Path objects."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    e = DocumentExtractor()
    captured = {}

    def fake_extract(paths):
        captured["paths"] = paths
        return [ExtractionResult(str(paths[0]), 1, "pdfium_hybrid")]

    with patch.object(e, "_extract_with_nv_ingest", side_effect=fake_extract):
        e.extract([pdf])
    assert captured["paths"][0] == pdf
    assert isinstance(captured["paths"][0], Path)
