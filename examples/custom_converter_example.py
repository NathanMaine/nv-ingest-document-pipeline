# SPDX-FileCopyrightText: Copyright (c) 2026 Nathan Maine
# SPDX-License-Identifier: Apache-2.0

"""Example: subclass DocumentConverter for an academic-paper corpus.

This is a complete worked example showing how to override the three extension
hooks in DocumentConverter for a specific domain. Run this file as-is to see
output structure (no PDF needed — it uses a fake ExtractionResult).

To use with real PDFs:
    1. Install nv-ingest (see docs/SETUP.md)
    2. Start the docker-compose service
    3. Replace the fake_results below with:
           from src.extractor import DocumentExtractor
           ex = DocumentExtractor()
           results = ex.extract(["data/sample_pdfs/your_paper.pdf"])
"""
from __future__ import annotations

import json
import re

from src.converter import DocumentConverter
from src.extractor import ExtractedContent, ExtractionResult


class AcademicPaperConverter(DocumentConverter):
    """A converter specialized for academic papers (arXiv, conference proceedings).

    Three overrides:
      _extract_topic       — detect arXiv IDs, equation labels, section headings
      _generate_question   — paper-specific question framing (contributions, experiments, limitations)
      _find_identifiers    — extract DOIs and arXiv IDs for cross-reference questions
    """

    def _extract_topic(self, text: str) -> str | None:
        # arXiv ID
        arxiv = re.search(r"arXiv:(\d{4}\.\d{4,5})", text)
        if arxiv:
            return f"the paper arXiv:{arxiv.group(1)}"

        # Equation label: "(eq. 3)" or "Equation 3.4"
        eq = re.search(r"(?:\(eq\.?\s*([\d.]+)\)|Equation\s+([\d.]+))", text, re.IGNORECASE)
        if eq:
            number = eq.group(1) or eq.group(2)
            return f"equation ({number})"

        # Section heading: "3.2 Methods"
        section = re.match(r"^[\d.]+\s+(.{5,80}?)(?:\n|$)", text)
        if section:
            return section.group(1).strip()

        # Fall back to base class (numbered headings + first sentence)
        return super()._extract_topic(text)

    def _generate_question(self, topic: str, content: str) -> str:
        cl = content.lower()

        if any(p in cl for p in ["we propose", "we introduce", "our method", "our approach"]):
            return f"What is the main contribution of {topic}?"
        if any(p in cl for p in ["experiment", "evaluat", "benchmark"]):
            return f"What experiments were run for {topic}?"
        if any(p in cl for p in ["limitation", "future work"]):
            return f"What are the limitations of {topic}?"
        if any(p in cl for p in ["related work", "prior art", "previous"]):
            return f"How does {topic} relate to prior work?"
        if any(p in cl for p in ["conclude", "conclusion", "summary"]):
            return f"What is the main conclusion of {topic}?"

        # Defer to base class for generic templates
        return super()._generate_question(topic, content)

    def _find_identifiers(self, text: str) -> list[str]:
        ids = []

        # DOIs
        ids.extend(re.findall(r"10\.\d{4,9}/[-._;()/:a-z0-9A-Z]+", text))

        # arXiv IDs
        ids.extend(re.findall(r"arXiv:\d{4}\.\d{4,5}", text))

        # Dedupe, preserve order
        return list(dict.fromkeys(ids))


def main():
    """Demo: run the custom converter against a fake ExtractionResult."""

    # Fake extraction result simulating output from DocumentExtractor.extract()
    fake_results = [
        ExtractionResult(
            source_file="attention_is_all_you_need.pdf",
            total_pages=15,
            extraction_method="pdfium_hybrid",
            contents=[
                ExtractedContent(
                    content_type="text",
                    text=(
                        "We propose the Transformer, a new model architecture based "
                        "solely on attention mechanisms, dispensing with recurrence "
                        "and convolutions entirely. arXiv:1706.03762\n\n"
                        "The Transformer relies on scaled dot-product attention as "
                        "defined in equation (3). Our method achieves state of the "
                        "art results on machine translation tasks while being more "
                        "parallelizable than prior recurrent architectures."
                    ),
                    page_number=1,
                    source_file="attention_is_all_you_need.pdf",
                ),
                ExtractedContent(
                    content_type="text",
                    text=(
                        "We evaluate the Transformer on the WMT 2014 English-to-German "
                        "translation task and the WMT 2014 English-to-French translation "
                        "task. Our experiments show that the Transformer outperforms "
                        "previous state-of-the-art models by 2.0 BLEU on English-to-German "
                        "and matches the prior best on English-to-French while training "
                        "in a fraction of the time."
                    ),
                    page_number=8,
                    source_file="attention_is_all_you_need.pdf",
                ),
                ExtractedContent(
                    content_type="table",
                    text=(
                        "Table 2: Variations on the Transformer architecture\n\n"
                        "Model       | Layers | d_model | Heads | BLEU\n"
                        "Base        | 6      | 512     | 8     | 27.3\n"
                        "Big         | 6      | 1024    | 16    | 28.4\n"
                        "Reference: 10.5555/3295222.3295349"
                    ),
                    page_number=10,
                    source_file="attention_is_all_you_need.pdf",
                ),
            ],
        )
    ]

    # Run the custom converter
    conv = AcademicPaperConverter(
        source_prefix="arxiv_demo",
        system_prompt=(
            "You are a research assistant trained on academic papers. Cite "
            "equation numbers, sections, and author claims precisely. "
            "Distinguish between what the paper proves and what it conjectures."
        ),
    )

    examples = conv.convert(fake_results)

    print(f"Generated {len(examples)} training examples\n")
    print("=" * 70)
    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Source: {ex.source}")
        print(f"User:   {ex.user}")
        print(f"Asst:   {ex.assistant[:200]}{'...' if len(ex.assistant) > 200 else ''}")

    # Also show the JSONL output format
    print("\n" + "=" * 70)
    print("Sample JSONL output:")
    print(json.dumps(examples[0].to_dict(), indent=2))


if __name__ == "__main__":
    main()
