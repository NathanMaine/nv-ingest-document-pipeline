# Customization Guide

How to adapt the pipeline for your specific document domain. The base `DocumentConverter` is intentionally generic — it produces reasonable output for any corpus, but you'll get much better training data by overriding the three extension hooks below.

See `examples/custom_converter_example.py` for a complete working subclass.

## Overview of extension hooks

`DocumentConverter` exposes three methods specifically designed for subclassing:

| Method | Default behavior | Override to... |
|---|---|---|
| `_extract_topic(text)` | Look for numbered headings or first sentence | Detect your domain's identifiers (case citations, RFC numbers, drug codes, etc.) |
| `_generate_question(topic, content)` | Generic "what / how / why" framing | Use domain-specific question phrasing |
| `_find_identifiers(text)` | Returns empty list | Return canonical IDs for cross-reference questions |

You can override one, two, or all three. They're independent.

## Override #1 — `_extract_topic(text)`

The base class looks for things like:

```
1.2.3 Network Architecture       ← matches numbered heading regex
This document describes...        ← falls back to first sentence
```

For a domain with stable identifiers, override to detect them first. Examples:

### Legal documents

```python
import re
from src.converter import DocumentConverter

class LegalConverter(DocumentConverter):
    def _extract_topic(self, text):
        # US case citation: "Brown v. Board of Education, 347 U.S. 483 (1954)"
        case_match = re.search(r"([A-Z][a-z]+\s+v\.\s+[A-Z][^,]+,\s+\d+\s+U\.S\.\s+\d+)", text)
        if case_match:
            return case_match.group(1)

        # USC citation: "42 U.S.C. § 1983"
        usc_match = re.search(r"\b\d+\s+U\.S\.C\.\s+§?\s*\d+", text)
        if usc_match:
            return usc_match.group(0)

        # Fall back to base class behavior
        return super()._extract_topic(text)
```

### Technical documentation (RFCs)

```python
class RFCConverter(DocumentConverter):
    def _extract_topic(self, text):
        # RFC reference: "RFC 9110" or "RFC9110"
        rfc_match = re.search(r"RFC\s*(\d{3,5})", text)
        if rfc_match:
            return f"RFC {rfc_match.group(1)}"

        # Section heading inside an RFC: "Section 3.4.2."
        section_match = re.match(r"Section\s+([\d.]+)", text)
        if section_match:
            return f"Section {section_match.group(1)}"

        return super()._extract_topic(text)
```

### Medical / pharmaceutical

```python
class MedicalConverter(DocumentConverter):
    def _extract_topic(self, text):
        # ICD-10 code: "I50.42" or "J45.901"
        icd_match = re.search(r"\b[A-Z]\d{2}\.\d{1,3}\b", text)
        if icd_match:
            return f"ICD code {icd_match.group(0)}"

        # Drug brand name pattern (rough): capitalized + dose
        drug_match = re.search(r"\b([A-Z][a-z]+(?:in|ol|am|ide|ate))\s+\d+\s*mg", text)
        if drug_match:
            return f"medication {drug_match.group(1)}"

        return super()._extract_topic(text)
```

### Scientific publications

```python
class ScientificConverter(DocumentConverter):
    def _extract_topic(self, text):
        # DOI: "10.1038/nature12373"
        doi_match = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9A-Z]+", text)
        if doi_match:
            return f"DOI {doi_match.group(0)}"

        # arXiv ID
        arxiv_match = re.search(r"arXiv:\d{4}\.\d{4,5}", text)
        if arxiv_match:
            return arxiv_match.group(0)

        # Equation reference: "(eq. 3.4)"
        eq_match = re.search(r"\(eq\.\s+([\d.]+)\)", text, re.IGNORECASE)
        if eq_match:
            return f"equation {eq_match.group(1)}"

        return super()._extract_topic(text)
```

## Override #2 — `_generate_question(topic, content)`

The base class picks a question template based on keywords in the content:

| Content contains... | Question template |
|---|---|
| "requirement", "shall", "must" | "What are the requirements for X?" |
| "implement", "procedure", "step" | "How should one implement X?" |
| "assess", "evaluate", "audit" | "How is X evaluated?" |
| "risk", "threat", "vulnerab" | "What are the risks related to X?" |
| (default) | "Explain X." |

This works fine for technical / regulatory documents. For other domains you'll want different framing:

### Legal documents

```python
class LegalConverter(DocumentConverter):
    def _generate_question(self, topic, content):
        content_lower = content.lower()

        if "holding" in content_lower or "ruled" in content_lower:
            return f"What was the court's holding in {topic}?"
        if "dissent" in content_lower:
            return f"What did the dissent argue in {topic}?"
        if "precedent" in content_lower or "stare decisis" in content_lower:
            return f"What precedent does {topic} establish?"
        if "facts" in content_lower or "background" in content_lower:
            return f"What are the facts of {topic}?"

        return f"Summarize {topic}."
```

### Customer support / FAQ

```python
class FAQConverter(DocumentConverter):
    def _generate_question(self, topic, content):
        content_lower = content.lower()

        if "error" in content_lower or "fix" in content_lower or "resolve" in content_lower:
            return f"How do I fix the {topic} issue?"
        if "configure" in content_lower or "setting" in content_lower:
            return f"How do I configure {topic}?"
        if "install" in content_lower or "setup" in content_lower:
            return f"How do I install or set up {topic}?"
        if "limitation" in content_lower or "known issue" in content_lower:
            return f"What are the limitations of {topic}?"

        return f"How does {topic} work?"
```

### Cooking / recipes

```python
class RecipeConverter(DocumentConverter):
    def _generate_question(self, topic, content):
        content_lower = content.lower()

        if "ingredient" in content_lower:
            return f"What ingredients does {topic} need?"
        if "step" in content_lower or "instruction" in content_lower:
            return f"How do you make {topic}?"
        if "calorie" in content_lower or "nutrition" in content_lower:
            return f"What are the nutrition facts for {topic}?"

        return f"Tell me about {topic}."
```

## Override #3 — `_find_identifiers(text)`

The base class returns `[]`. Override to return canonical identifiers from your text. These are used to generate **additional** lookup-style questions for tables.

### Legal

```python
class LegalConverter(DocumentConverter):
    def _find_identifiers(self, text):
        ids = []
        # USC citations
        ids.extend(re.findall(r"\b\d+\s+U\.S\.C\.\s+§?\s*\d+\b", text))
        # CFR citations
        ids.extend(re.findall(r"\b\d+\s+C\.F\.R\.\s+§?\s*[\d.]+\b", text))
        # Case citations (US Reports)
        ids.extend(re.findall(r"\b\d+\s+U\.S\.\s+\d+\b", text))
        # Dedupe, preserve order
        return list(dict.fromkeys(ids))
```

### Technical

```python
class APIConverter(DocumentConverter):
    def _find_identifiers(self, text):
        ids = []
        # API endpoints: GET /api/v1/users
        ids.extend(re.findall(r"\b(?:GET|POST|PUT|DELETE|PATCH)\s+/[a-zA-Z0-9/_{}-]+", text))
        # HTTP status codes
        ids.extend(re.findall(r"\b[1-5]\d{2}\s+(?:OK|Created|Bad Request|Unauthorized)", text))
        return list(dict.fromkeys(ids))
```

## Override #4 (advanced) — Custom system prompt

Don't subclass for this — just pass `system_prompt=` to the constructor:

```python
conv = DocumentConverter(
    system_prompt=(
        "You are a senior software architect with deep knowledge of distributed "
        "systems, networking protocols, and reliability engineering. Cite specific "
        "RFCs, sections, and design rationale where relevant."
    )
)
```

The prompt is included in every training example's `messages[0]`. Pick something that matches the role you want the fine-tuned model to play.

## Putting it all together

```python
from src.converter import DocumentConverter
from src.extractor import DocumentExtractor
import re


class LegalConverter(DocumentConverter):
    def _extract_topic(self, text):
        case_match = re.search(r"([A-Z][a-z]+\s+v\.\s+[A-Z][^,]+,\s+\d+\s+U\.S\.\s+\d+)", text)
        if case_match:
            return case_match.group(1)
        usc_match = re.search(r"\b\d+\s+U\.S\.C\.\s+§?\s*\d+\b", text)
        if usc_match:
            return usc_match.group(0)
        return super()._extract_topic(text)

    def _generate_question(self, topic, content):
        content_lower = content.lower()
        if "holding" in content_lower or "ruled" in content_lower:
            return f"What was the court's holding in {topic}?"
        if "dissent" in content_lower:
            return f"What did the dissent argue in {topic}?"
        return f"Summarize {topic}."

    def _find_identifiers(self, text):
        return list(dict.fromkeys(
            re.findall(r"\b\d+\s+U\.S\.C\.\s+§?\s*\d+\b", text)
        ))


# Use it
ex = DocumentExtractor()
results = ex.extract(["data/sample_pdfs/case_law.pdf"])

conv = LegalConverter(
    source_prefix="legal_corpus",
    system_prompt="You are a legal research assistant with expertise in US federal law. Cite specific cases and statutes.",
)
conv.convert_to_jsonl(results, "output/legal_training.jsonl")
```

## Testing your subclass

The unit test pattern in `tests/test_converter.py` shows how to verify your overrides actually run. Add tests like:

```python
def test_legal_converter_detects_case_citation():
    conv = LegalConverter()
    text = "In Brown v. Board of Education, 347 U.S. 483 (1954), the Court held..."
    topic = conv._extract_topic(text)
    assert "Brown v. Board of Education" in topic
```

Run with `python -m pytest tests/test_my_legal_converter.py -v`.

## When NOT to subclass

Don't override these methods if:

- You're prototyping and the base behavior is "good enough"
- Your corpus has no stable identifier scheme
- You'll filter the output downstream anyway and don't care about the question framing

The chunker (`_chunk_text`, `_split_long_paragraph`) is generally not worth overriding unless you have unusual document structure (e.g., poetry, code listings, dense math).
