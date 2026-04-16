# SPDX-FileCopyrightText: Copyright (c) 2026 Nathan Maine
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: nv-ingest vs PyPDF2 for PDF extraction.

Compares GPU-accelerated extraction (NVIDIA nv-ingest) against a CPU-only
baseline (PyPDF2 text extraction) on any PDF corpus.

Metrics collected:
- Extraction time (wall clock)
- Content yield (text blocks, tables extracted)
- Text quality (character count, whitespace ratio)

Usage:
    >>> from src.benchmark import run_benchmark, format_results_table
    >>> comparisons = run_benchmark(["data/sample_pdfs/example.pdf"])
    >>> print(format_results_table(comparisons))
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.extractor import DocumentExtractor, ExtractedContent, ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Metrics for a single extraction run."""

    method: str
    source_file: str
    extraction_time_ms: float
    text_blocks: int
    table_blocks: int
    total_chars: int
    total_pages: int
    whitespace_ratio: float
    errors: list[str] = field(default_factory=list)


@dataclass
class BenchmarkComparison:
    """Side-by-side comparison of two extraction methods."""

    source_file: str
    nv_ingest: BenchmarkMetrics | None
    baseline: BenchmarkMetrics | None

    @property
    def speedup(self) -> float | None:
        if self.nv_ingest and self.baseline and self.nv_ingest.extraction_time_ms > 0:
            return self.baseline.extraction_time_ms / self.nv_ingest.extraction_time_ms
        return None

    @property
    def text_yield_ratio(self) -> float | None:
        if self.nv_ingest and self.baseline and self.baseline.total_chars > 0:
            return self.nv_ingest.total_chars / self.baseline.total_chars
        return None


def extract_baseline(pdf_path: str | Path) -> ExtractionResult:
    """Extract PDF content using PyPDF2 (CPU baseline).

    Traditional approach: PyPDF2 for text, no table detection, no GPU
    acceleration. Represents the lowest-bar comparison for any nv-ingest
    speedup claim.
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PyPDF2 required for baseline. Install: pip install PyPDF2"
        ) from exc

    pdf_path = Path(pdf_path)
    start = time.perf_counter()

    reader = PdfReader(str(pdf_path))
    contents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            contents.append(
                ExtractedContent(
                    content_type="text",
                    text=text.strip(),
                    page_number=i + 1,
                    source_file=str(pdf_path),
                )
            )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return ExtractionResult(
        source_file=str(pdf_path),
        total_pages=len(reader.pages),
        extraction_method="pypdf2",
        contents=contents,
        processing_time_ms=elapsed_ms,
    )


def _compute_metrics(
    result: ExtractionResult,
    method: str,
) -> BenchmarkMetrics:
    """Compute benchmark metrics from an extraction result."""
    all_text = " ".join(c.text for c in result.contents)
    total_chars = len(all_text)
    whitespace = sum(1 for c in all_text if c.isspace())

    return BenchmarkMetrics(
        method=method,
        source_file=result.source_file,
        extraction_time_ms=result.processing_time_ms,
        text_blocks=result.text_count,
        table_blocks=result.table_count,
        total_chars=total_chars,
        total_pages=result.total_pages,
        whitespace_ratio=whitespace / total_chars if total_chars > 0 else 0.0,
        errors=result.errors,
    )


def run_benchmark(
    pdf_paths: list[str | Path],
    output_path: str | Path | None = None,
) -> list[BenchmarkComparison]:
    """Run side-by-side benchmark on PDFs.

    Args:
        pdf_paths: PDFs to benchmark.
        output_path: Optional JSON file to write results.

    Returns:
        List of BenchmarkComparison, one per PDF.
    """
    comparisons = []

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        logger.info("Benchmarking: %s", pdf_path.name)

        # Baseline (PyPDF2)
        baseline_metrics = None
        try:
            baseline_result = extract_baseline(pdf_path)
            baseline_metrics = _compute_metrics(baseline_result, "pypdf2")
            logger.info(
                "  Baseline: %.0fms, %d blocks, %d chars",
                baseline_metrics.extraction_time_ms,
                baseline_metrics.text_blocks,
                baseline_metrics.total_chars,
            )
        except ImportError:
            logger.warning("  PyPDF2 not available, skipping baseline benchmark")
        except (OSError, ValueError) as e:
            logger.error("  Baseline failed: %s", e)

        # nv-ingest (GPU)
        nv_metrics = None
        try:
            extractor = DocumentExtractor()
            nv_results = extractor.extract([pdf_path])
            if nv_results:
                nv_metrics = _compute_metrics(nv_results[0], "nv-ingest")
                logger.info(
                    "  nv-ingest: %.0fms, %d text + %d tables, %d chars",
                    nv_metrics.extraction_time_ms,
                    nv_metrics.text_blocks,
                    nv_metrics.table_blocks,
                    nv_metrics.total_chars,
                )
        except ImportError:
            logger.warning("  nv-ingest not available, skipping GPU benchmark")
        except (OSError, ValueError, ConnectionError) as e:
            logger.error("  nv-ingest failed: %s", e)

        comparisons.append(
            BenchmarkComparison(
                source_file=str(pdf_path),
                nv_ingest=nv_metrics,
                baseline=baseline_metrics,
            )
        )

    if output_path:
        _write_results(comparisons, Path(output_path))

    return comparisons


def _write_results(
    comparisons: list[BenchmarkComparison],
    output_path: Path,
) -> None:
    """Write benchmark results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for comp in comparisons:
        entry = {"source_file": comp.source_file}
        if comp.baseline:
            entry["baseline"] = {
                "method": comp.baseline.method,
                "time_ms": comp.baseline.extraction_time_ms,
                "text_blocks": comp.baseline.text_blocks,
                "table_blocks": comp.baseline.table_blocks,
                "total_chars": comp.baseline.total_chars,
                "total_pages": comp.baseline.total_pages,
            }
        if comp.nv_ingest:
            entry["nv_ingest"] = {
                "method": comp.nv_ingest.method,
                "time_ms": comp.nv_ingest.extraction_time_ms,
                "text_blocks": comp.nv_ingest.text_blocks,
                "table_blocks": comp.nv_ingest.table_blocks,
                "total_chars": comp.nv_ingest.total_chars,
                "total_pages": comp.nv_ingest.total_pages,
            }
        if comp.speedup:
            entry["speedup_x"] = round(comp.speedup, 2)
        if comp.text_yield_ratio:
            entry["text_yield_ratio"] = round(comp.text_yield_ratio, 2)
        data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Wrote benchmark results to %s", output_path)


def format_results_table(comparisons: list[BenchmarkComparison]) -> str:
    """Format benchmark results as a markdown table."""
    lines = [
        "| Document | Method | Time (ms) | Text Blocks | Tables | Chars | Speedup |",
        "|----------|--------|-----------|-------------|--------|-------|---------|",
    ]

    for comp in comparisons:
        name = Path(comp.source_file).stem[:30]
        speedup_str = f"{comp.speedup:.1f}x" if comp.speedup else "N/A"

        if comp.baseline:
            b = comp.baseline
            lines.append(
                f"| {name} | PyPDF2 | {b.extraction_time_ms:.0f} | "
                f"{b.text_blocks} | {b.table_blocks} | {b.total_chars:,} | - |"
            )
        if comp.nv_ingest:
            n = comp.nv_ingest
            lines.append(
                f"| {name} | nv-ingest | {n.extraction_time_ms:.0f} | "
                f"{n.text_blocks} | {n.table_blocks} | {n.total_chars:,} | {speedup_str} |"
            )

    return "\n".join(lines)
