# SPDX-FileCopyrightText: Copyright (c) 2026 Nathan Maine
# SPDX-License-Identifier: Apache-2.0

"""Convert nv-ingest extraction output to chat-format JSONL training data.

Transforms extracted text and tables from PDFs into the chat-format JSONL
used by most modern fine-tuning toolchains (axolotl, unsloth, transformers TRL):

    {"messages": [system, user, assistant], "source": "nv_ingest_<doc>"}

This converter is **domain-agnostic by default**. The base class produces
generic Q&A pairs from extracted content. To specialize for your domain
(legal, medical, technical, scientific, etc.), subclass `DocumentConverter`
and override the extension hooks listed in `docs/CUSTOMIZATION.md`:

    - `_extract_topic(text)` — pull the main topic / identifier from a chunk
    - `_generate_question(topic, content)` — frame the question for your domain
    - `_find_identifiers(text)` — return canonical IDs for cross-reference Qs

See `examples/custom_converter_example.py` for a worked subclass example.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.extractor import ExtractedContent, ExtractionResult

logger = logging.getLogger(__name__)

# Generic system prompt. Override via the constructor for your domain.
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert assistant trained to provide accurate, well-cited answers "
    "based on technical documentation. Cite specific sections, identifiers, and "
    "references where relevant. Prefer precise, structured responses over vague ones."
)

# Minimum content length for a training example (chars).
MIN_CONTENT_LENGTH = 100

# Maximum content length before splitting into chunks.
MAX_CONTENT_LENGTH = 4000


@dataclass
class TrainingExample:
    """Single chat-format training example."""

    system: str
    user: str
    assistant: str
    source: str

    def to_dict(self) -> dict:
        return {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ],
            "source": self.source,
        }


class DocumentConverter:
    """Converts nv-ingest extraction results to chat-format training data.

    Generates question-answer pairs from extracted content:
    - Text blocks become explanations grounded in the source content
    - Tables become structured descriptions or lookups
    - Each piece of content generates 1-3 training examples with varied
      question framing (what / how / why)

    Args:
        source_prefix: Prefix used in the `source` field of each training example.
            Defaults to "nv_ingest". Useful when you want to tag the origin of
            data for downstream filtering (e.g., "legal_corpus_v1").
        system_prompt: System message included in every training example. The
            default is generic; override with your domain's role description.
        min_length: Minimum chars for a chunk to be emitted as a training
            example. Default 100.
        max_length: Maximum chars per chunk. Long content is split on
            paragraph then sentence boundaries. Default 4000.

    Subclassing for your domain:
        Override `_extract_topic`, `_generate_question`, and `_find_identifiers`
        to add domain-specific topic detection and question framing. See
        `examples/custom_converter_example.py` and `docs/CUSTOMIZATION.md`.
    """

    def __init__(
        self,
        source_prefix: str = "nv_ingest",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        min_length: int = MIN_CONTENT_LENGTH,
        max_length: int = MAX_CONTENT_LENGTH,
    ):
        self.source_prefix = source_prefix
        self.system_prompt = system_prompt
        self.min_length = min_length
        self.max_length = max_length

    def convert(self, results: list[ExtractionResult]) -> list[TrainingExample]:
        """Convert extraction results to training examples.

        Args:
            results: Output from DocumentExtractor.extract().

        Returns:
            List of TrainingExample in chat format.
        """
        examples = []
        for result in results:
            source_tag = self._source_tag(result.source_file)
            for content in result.contents:
                if len(content.text) < self.min_length:
                    continue
                new_examples = self._content_to_examples(content, source_tag)
                examples.extend(new_examples)

        logger.info(
            "Generated %d training examples from %d documents",
            len(examples),
            len(results),
        )
        return examples

    def convert_to_jsonl(
        self,
        results: list[ExtractionResult],
        output_path: str | Path,
    ) -> Path:
        """Convert and write to a JSONL file.

        Args:
            results: Output from DocumentExtractor.extract().
            output_path: Where to write the JSONL file.

        Returns:
            Path to the written file.
        """
        examples = self.convert(results)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

        logger.info("Wrote %d examples to %s", len(examples), output_path)
        return output_path

    # ------------------------------------------------------------------
    # Extension points — override these in a subclass for your domain
    # ------------------------------------------------------------------

    def _extract_topic(self, text: str) -> str | None:
        """Pull the main topic from a text chunk.

        Default behavior:
        1. Look for a numbered section heading at the start (e.g., "1.2.3 Title")
        2. Fall back to the first complete sentence (20-200 chars)

        Override this for your domain to detect things like:
            - Legal: case citations, statute references
            - Medical: ICD codes, drug names
            - Technical: API names, RFC numbers
            - Scientific: equation labels, figure captions

        See examples/custom_converter_example.py for a worked example.
        """
        # Numbered section heading at start
        heading_match = re.match(r"^[\d.]+\s+(.{10,60}?)(?:\n|$)", text)
        if heading_match:
            return heading_match.group(1).strip().rstrip(".")

        # First complete sentence
        sentence_match = re.match(r"(.{20,200}?[.!?])(?:\s|$)", text)
        if sentence_match:
            return sentence_match.group(1).strip().rstrip(".")

        return None

    def _generate_question(self, topic: str, content: str) -> str:
        """Generate a natural question for the topic.

        Default behavior is generic and works for most domains. The function
        looks for keywords in the content and picks an appropriate framing:

            - "requirement", "shall", "must" → "What are the requirements for X?"
            - "implement", "procedure", "step" → "How does one implement X?"
            - "evaluate", "assess", "audit" → "How is X evaluated?"
            - "risk", "threat" → "What are the risks related to X?"
            - default → "Explain X."

        Override this to use domain-specific question framing.
        """
        content_lower = content.lower()

        if any(w in content_lower for w in ["requirement", "shall", "must"]):
            return f"What are the requirements for {topic}?"
        if any(w in content_lower for w in ["implement", "procedure", "step"]):
            return f"How should one implement {topic}?"
        if any(w in content_lower for w in ["assess", "evaluat", "audit"]):
            return f"How is {topic} evaluated?"
        if any(w in content_lower for w in ["risk", "threat", "vulnerab"]):
            return f"What are the risks related to {topic}?"

        return f"Explain {topic}."

    def _find_identifiers(self, text: str) -> list[str]:
        """Find canonical identifiers in the text for cross-reference questions.

        Default returns empty — most generic corpora don't have a stable
        identifier scheme. Override for your domain:

            - Legal: case citations like "410 U.S. 113"
            - Medical: ICD codes like "I50.42"
            - Technical: RFC numbers like "RFC 9110"
            - Scientific: DOIs like "10.1038/nature12373"

        Returned identifiers are used to generate additional lookup-style
        training examples for tables.
        """
        return []

    # ------------------------------------------------------------------
    # Internal — usually no need to override
    # ------------------------------------------------------------------

    def _source_tag(self, source_file: str) -> str:
        """Generate a source tag from the filename."""
        name = Path(source_file).stem
        name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return f"{self.source_prefix}_{name}"

    def _content_to_examples(
        self,
        content: ExtractedContent,
        source_tag: str,
    ) -> list[TrainingExample]:
        """Generate training examples from a single extracted content block."""
        if content.content_type == "table":
            return self._table_to_examples(content, source_tag)
        return self._text_to_examples(content, source_tag)

    def _text_to_examples(
        self,
        content: ExtractedContent,
        source_tag: str,
    ) -> list[TrainingExample]:
        """Convert a text block to training examples."""
        text = content.text
        chunks = self._chunk_text(text)
        examples = []

        for chunk in chunks:
            topic = self._extract_topic(chunk)
            if not topic:
                continue

            question = self._generate_question(topic, chunk)
            examples.append(
                TrainingExample(
                    system=self.system_prompt,
                    user=question,
                    assistant=chunk,
                    source=source_tag,
                )
            )

        return examples

    def _table_to_examples(
        self,
        content: ExtractedContent,
        source_tag: str,
    ) -> list[TrainingExample]:
        """Convert a table to training examples."""
        text = content.text
        if len(text) < self.min_length:
            return []

        topic = self._extract_topic(text) or "this topic"

        question = f"What are the details of {topic}?"
        examples = [
            TrainingExample(
                system=self.system_prompt,
                user=question,
                assistant=text,
                source=source_tag,
            )
        ]

        # If the table mentions specific identifiers, generate a lookup question
        identifiers = self._find_identifiers(text)
        if identifiers:
            id_str = ", ".join(identifiers[:3])
            examples.append(
                TrainingExample(
                    system=self.system_prompt,
                    user=f"Explain {id_str}.",
                    assistant=text,
                    source=source_tag,
                )
            )

        return examples

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks that fit within max_length.

        Splits on paragraph boundaries first. If a single paragraph exceeds
        max_length, falls back to sentence-boundary splitting.
        """
        if len(text) <= self.max_length:
            return [text]

        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(para) > self.max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.extend(self._split_long_paragraph(para))
                continue

            if len(current_chunk) + len(para) + 2 > self.max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [c for c in chunks if len(c) >= self.min_length]

    def _split_long_paragraph(self, text: str) -> list[str]:
        """Split a long paragraph on sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(sentence) > self.max_length:
                if current:
                    chunks.append(current.strip())
                    current = ""
                for i in range(0, len(sentence), self.max_length):
                    chunk = sentence[i:i + self.max_length].strip()
                    if chunk:
                        chunks.append(chunk)
                continue

            if len(current) + len(sentence) + 1 > self.max_length:
                if current:
                    chunks.append(current.strip())
                current = sentence
            else:
                current = current + " " + sentence if current else sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks
