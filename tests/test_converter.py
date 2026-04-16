# SPDX-FileCopyrightText: Copyright (c) 2026 Nathan Maine
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for src.converter — all run on CPU, no GPU or nv-ingest needed."""
from __future__ import annotations

import json

from src.converter import (
    DEFAULT_SYSTEM_PROMPT,
    DocumentConverter,
    TrainingExample,
)
from src.extractor import ExtractedContent, ExtractionResult


# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------
def test_training_example_to_dict():
    ex = TrainingExample(
        system="sys", user="q", assistant="a", source="src_x"
    )
    d = ex.to_dict()
    assert d["messages"][0] == {"role": "system", "content": "sys"}
    assert d["messages"][1] == {"role": "user", "content": "q"}
    assert d["messages"][2] == {"role": "assistant", "content": "a"}
    assert d["source"] == "src_x"


# ---------------------------------------------------------------------------
# DocumentConverter — defaults
# ---------------------------------------------------------------------------
def test_converter_defaults():
    c = DocumentConverter()
    assert c.source_prefix == "nv_ingest"
    assert c.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert c.min_length == 100
    assert c.max_length == 4000


def test_converter_custom_init():
    c = DocumentConverter(
        source_prefix="my_corpus",
        system_prompt="custom prompt",
        min_length=50,
        max_length=2000,
    )
    assert c.source_prefix == "my_corpus"
    assert c.system_prompt == "custom prompt"
    assert c.min_length == 50
    assert c.max_length == 2000


# ---------------------------------------------------------------------------
# _source_tag
# ---------------------------------------------------------------------------
def test_source_tag_basic():
    c = DocumentConverter()
    assert c._source_tag("foo.pdf") == "nv_ingest_foo"


def test_source_tag_with_path():
    c = DocumentConverter()
    assert c._source_tag("/path/to/My Document.pdf") == "nv_ingest_my_document"


def test_source_tag_normalization():
    c = DocumentConverter()
    assert c._source_tag("doc-v1.2.pdf") == "nv_ingest_doc_v1_2"


def test_source_tag_custom_prefix():
    c = DocumentConverter(source_prefix="legal")
    assert c._source_tag("contract.pdf") == "legal_contract"


# ---------------------------------------------------------------------------
# _extract_topic — generic patterns
# ---------------------------------------------------------------------------
def test_extract_topic_numbered_heading():
    c = DocumentConverter()
    text = "1.2.3 Network Architecture\n\nDetails follow here."
    topic = c._extract_topic(text)
    assert topic == "Network Architecture"


def test_extract_topic_first_sentence_fallback():
    c = DocumentConverter()
    text = "This document describes the deployment process. It includes config samples."
    topic = c._extract_topic(text)
    assert topic and "deployment process" in topic


def test_extract_topic_returns_none_for_empty():
    c = DocumentConverter()
    assert c._extract_topic("") is None


def test_extract_topic_returns_none_for_short_text():
    c = DocumentConverter()
    assert c._extract_topic("hi.") is None


# ---------------------------------------------------------------------------
# _generate_question — keyword routing
# ---------------------------------------------------------------------------
def test_generate_question_requirements():
    c = DocumentConverter()
    q = c._generate_question("topic", "The system shall provide encryption.")
    assert q.startswith("What are the requirements for")


def test_generate_question_implementation():
    c = DocumentConverter()
    q = c._generate_question("topic", "Implement the procedure as follows.")
    assert q.startswith("How should one implement")


def test_generate_question_assessment():
    c = DocumentConverter()
    q = c._generate_question("topic", "Auditors will assess the controls.")
    assert q.startswith("How is")
    assert "evaluated" in q


def test_generate_question_risk():
    c = DocumentConverter()
    q = c._generate_question("topic", "Mitigates risk of data breach.")
    assert q.startswith("What are the risks")


def test_generate_question_default():
    c = DocumentConverter()
    q = c._generate_question("topic", "Plain factual content.")
    assert q == "Explain topic."


# ---------------------------------------------------------------------------
# _find_identifiers — base class returns empty
# ---------------------------------------------------------------------------
def test_find_identifiers_default_empty():
    c = DocumentConverter()
    assert c._find_identifiers("any text 123 ABC") == []


def test_find_identifiers_extension_subclass():
    """Demonstrates the subclassing extension point."""

    class CustomConverter(DocumentConverter):
        def _find_identifiers(self, text):
            import re
            return re.findall(r"REF-\d{4}", text)

    c = CustomConverter()
    ids = c._find_identifiers("see REF-1234 and REF-5678")
    assert ids == ["REF-1234", "REF-5678"]


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------
def test_chunk_text_short_returns_one():
    c = DocumentConverter(max_length=100)
    chunks = c._chunk_text("short text under limit")
    assert chunks == ["short text under limit"]


def test_chunk_text_paragraph_splits():
    c = DocumentConverter(min_length=20, max_length=80)
    text = "First paragraph here.\n\n" + "Second paragraph here, longer than limit, with more content."
    chunks = c._chunk_text(text)
    assert len(chunks) >= 1


def test_chunk_text_drops_below_min_length():
    c = DocumentConverter(min_length=50, max_length=200)
    text = "tiny\n\n" + "This is a normal-sized paragraph that meets the minimum length threshold."
    chunks = c._chunk_text(text)
    assert all(len(ch) >= 50 for ch in chunks)


# ---------------------------------------------------------------------------
# convert / convert_to_jsonl integration
# ---------------------------------------------------------------------------
def _fake_result(text: str, ctype: str = "text") -> ExtractionResult:
    return ExtractionResult(
        source_file="test.pdf",
        total_pages=1,
        extraction_method="pdfium_hybrid",
        contents=[ExtractedContent(ctype, text, 1, "test.pdf")],
    )


def test_convert_skips_short_content():
    short_result = _fake_result("too short")  # <100 chars
    c = DocumentConverter()
    examples = c.convert([short_result])
    assert examples == []


def test_convert_emits_text_examples():
    long_text = (
        "1.0 Introduction\n\n"
        + "The system must provide encryption at rest. " * 10
    )
    result = _fake_result(long_text)
    c = DocumentConverter()
    examples = c.convert([result])
    assert len(examples) >= 1
    assert all(ex.system == DEFAULT_SYSTEM_PROMPT for ex in examples)


def test_convert_emits_table_example():
    table_text = (
        "Header A | Header B | Header C\n"
        + "Row 1 col 1 | Row 1 col 2 | Row 1 col 3\n"
        + "Row 2 col 1 | Row 2 col 2 | Row 2 col 3\n"
        * 5
    )
    result = _fake_result(table_text, ctype="table")
    c = DocumentConverter()
    examples = c.convert([result])
    assert len(examples) == 1
    assert "details of" in examples[0].user.lower()


def test_convert_to_jsonl_writes_file(tmp_path):
    long_text = "1.0 Introduction Section\n\n" + "Required content. " * 20
    result = _fake_result(long_text)
    c = DocumentConverter()
    out = tmp_path / "subdir" / "out.jsonl"
    written = c.convert_to_jsonl([result], out)
    assert written == out
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 1
    parsed = json.loads(lines[0])
    assert "messages" in parsed
    assert parsed["source"].startswith("nv_ingest_")


def test_convert_to_jsonl_empty(tmp_path):
    c = DocumentConverter()
    out = tmp_path / "empty.jsonl"
    c.convert_to_jsonl([], out)
    assert out.exists()
    assert out.read_text() == ""


def test_subclass_topic_extraction_used():
    """Verifies the override hook actually runs."""
    class CustomConverter(DocumentConverter):
        def _extract_topic(self, text):
            return "CUSTOM_TOPIC"

    text = "1.0 Introduction Section\n\n" + "Some content. " * 30
    result = _fake_result(text)
    c = CustomConverter()
    examples = c.convert([result])
    assert any("CUSTOM_TOPIC" in ex.user for ex in examples)


def test_subclass_question_generation_used():
    class CustomConverter(DocumentConverter):
        def _generate_question(self, topic, content):
            return f"CUSTOM Q: {topic}"

    text = "1.0 Introduction Section\n\n" + "Some content. " * 30
    result = _fake_result(text)
    c = CustomConverter()
    examples = c.convert([result])
    assert any(ex.user.startswith("CUSTOM Q:") for ex in examples)


def test_subclass_identifiers_emit_extra_table_example():
    class CustomConverter(DocumentConverter):
        def _find_identifiers(self, text):
            return ["ID-001"] if "ID-001" in text else []

    table_text = "Topic Header\nID-001: This row " + "x " * 100
    result = _fake_result(table_text, ctype="table")
    c = CustomConverter()
    examples = c.convert([result])
    assert len(examples) == 2
    assert any("ID-001" in ex.user for ex in examples)
