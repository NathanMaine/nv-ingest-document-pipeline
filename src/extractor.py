# SPDX-FileCopyrightText: Copyright (c) 2026 Nathan Maine
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated document extraction using NVIDIA nv-ingest.

Wraps the nv-ingest Ingestor API to extract text, tables, charts, and metadata
from any PDF corpus. Domain-agnostic by design — bring your own documents and
let downstream code decide what to do with the extracted content.

Example:
    >>> from src.extractor import DocumentExtractor
    >>> ex = DocumentExtractor()
    >>> results = ex.extract(["data/sample_pdfs/example.pdf"])
    >>> for r in results:
    ...     print(f"{r.source_file}: {r.text_count} text + {r.table_count} tables")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Single piece of extracted content from a document."""

    content_type: str  # "text", "table", "chart"
    text: str
    page_number: int
    source_file: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Complete extraction result for a single document."""

    source_file: str
    total_pages: int
    extraction_method: str
    contents: list[ExtractedContent] = field(default_factory=list)
    processing_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def text_count(self) -> int:
        return sum(1 for c in self.contents if c.content_type == "text")

    @property
    def table_count(self) -> int:
        return sum(1 for c in self.contents if c.content_type == "table")

    @property
    def chart_count(self) -> int:
        return sum(1 for c in self.contents if c.content_type == "chart")


class DocumentExtractor:
    """Extracts document content using NVIDIA nv-ingest.

    Wraps the nv-ingest Ingestor API with sensible defaults:
    - Text and table extraction enabled (primary content for most use cases)
    - Chart extraction enabled (useful for documents with diagrams)
    - Image extraction disabled by default (set extract_images=True if you need it)
    - pdfium_hybrid extraction method (handles both native and scanned PDFs)

    Args:
        extraction_method: Which nv-ingest extraction backend to use. Default
            "pdfium_hybrid" works well for mixed corpora.
        extract_charts: Extract charts and diagrams as text. Default True.
        extract_images: Extract embedded images. Default False (rarely needed
            for text-based downstream pipelines).
    """

    def __init__(
        self,
        extraction_method: str = "pdfium_hybrid",
        extract_charts: bool = True,
        extract_images: bool = False,
    ):
        self.extraction_method = extraction_method
        self.extract_charts = extract_charts
        self.extract_images = extract_images

    def extract(self, pdf_paths: list[str | Path]) -> list[ExtractionResult]:
        """Extract content from one or more PDFs.

        Args:
            pdf_paths: Paths to PDF files to process.

        Returns:
            List of ExtractionResult, one per input file.

        Raises:
            FileNotFoundError: If any input PDF does not exist.
            ValueError: If any input file is not a PDF.
            ImportError: If nv-ingest is not installed.
        """
        resolved_paths: list[Path] = [Path(p) for p in pdf_paths]
        for p in resolved_paths:
            if not p.exists():
                raise FileNotFoundError(f"PDF not found: {p}")
            if p.suffix.lower() != ".pdf":
                raise ValueError(f"Expected PDF file, got: {p.suffix}")

        try:
            return self._extract_with_nv_ingest(resolved_paths)
        except ImportError as exc:
            logger.warning(
                "nv-ingest not available. Install with: pip install nv-ingest nv-ingest-client"
            )
            raise ImportError(
                "nv-ingest not installed. Install with: pip install nv-ingest nv-ingest-client"
            ) from exc

    def _extract_with_nv_ingest(
        self, pdf_paths: list[Path]
    ) -> list[ExtractionResult]:
        """Run extraction via nv-ingest Ingestor API."""
        import time

        from nv_ingest_client.client.interface import Ingestor

        str_paths = [str(p) for p in pdf_paths]

        ingestor = Ingestor().files(str_paths).extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=self.extract_charts,
            extract_images=self.extract_images,
        )

        start = time.perf_counter()
        raw_results = ingestor.ingest()
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        for i, pdf_path in enumerate(pdf_paths):
            result = self._parse_nv_ingest_output(
                raw_results[i] if i < len(raw_results) else {},
                str(pdf_path),
                elapsed_ms / len(pdf_paths),
            )
            results.append(result)

        return results

    def _parse_nv_ingest_output(
        self,
        raw: dict[str, Any],
        source_file: str,
        processing_time_ms: float,
    ) -> ExtractionResult:
        """Convert raw nv-ingest JSON output to ExtractionResult."""
        contents = []

        for item in raw.get("content", []):
            item_type = item.get("type", "text")
            text = item.get("content", "")
            page = item.get("page_number", 0)
            metadata = item.get("metadata", {})

            if not text or not text.strip():
                continue

            contents.append(
                ExtractedContent(
                    content_type=item_type,
                    text=text.strip(),
                    page_number=page,
                    source_file=source_file,
                    metadata=metadata,
                )
            )

        return ExtractionResult(
            source_file=source_file,
            total_pages=raw.get("processing_metadata", {}).get("total_pages", 0),
            extraction_method=self.extraction_method,
            contents=contents,
            processing_time_ms=processing_time_ms,
        )

    def extract_to_json(
        self,
        pdf_paths: list[str | Path],
        output_path: str | Path,
    ) -> Path:
        """Extract and write results to a JSON file.

        Args:
            pdf_paths: Paths to PDF files to process.
            output_path: Where to write the JSON output.

        Returns:
            Path to the written JSON file.
        """
        results = self.extract(pdf_paths)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serialized = []
        for result in results:
            serialized.append(
                {
                    "source_file": result.source_file,
                    "total_pages": result.total_pages,
                    "extraction_method": result.extraction_method,
                    "processing_time_ms": result.processing_time_ms,
                    "text_count": result.text_count,
                    "table_count": result.table_count,
                    "chart_count": result.chart_count,
                    "contents": [
                        {
                            "content_type": c.content_type,
                            "text": c.text,
                            "page_number": c.page_number,
                            "metadata": c.metadata,
                        }
                        for c in result.contents
                    ],
                    "errors": result.errors,
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

        logger.info("Wrote extraction results to %s", output_path)
        return output_path
