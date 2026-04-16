# Worked Examples

Concrete end-to-end examples for different document types. Each shows what to override and what defaults to keep.

## Example 1 — Academic papers (arXiv-style PDFs)

**Goal:** fine-tune an LLM that can answer questions about a corpus of research papers, citing equation labels and author names.

```python
from src.extractor import DocumentExtractor
from src.converter import DocumentConverter
import re


class AcademicConverter(DocumentConverter):
    def _extract_topic(self, text):
        # arXiv ID
        arxiv = re.search(r"arXiv:(\d{4}\.\d{4,5})", text)
        if arxiv:
            return f"the paper arXiv:{arxiv.group(1)}"

        # Equation label: "(eq. 3)" or "(3.4)"
        eq = re.search(r"\(eq\.?\s*([\d.]+)\)", text, re.IGNORECASE)
        if eq:
            return f"equation ({eq.group(1)})"

        # Section heading: "3.2 Methods"
        section = re.match(r"^[\d.]+\s+(.{5,80}?)(?:\n|$)", text)
        if section:
            return section.group(1).strip()

        return super()._extract_topic(text)

    def _generate_question(self, topic, content):
        cl = content.lower()
        if "we propose" in cl or "we introduce" in cl or "our method" in cl:
            return f"What is the main contribution of {topic}?"
        if "experiment" in cl or "evaluat" in cl or "benchmark" in cl:
            return f"What experiments were run for {topic}?"
        if "limitation" in cl or "future work" in cl:
            return f"What are the limitations of {topic}?"
        return super()._generate_question(topic, content)


ex = DocumentExtractor()
results = ex.extract(["data/sample_pdfs/attention_is_all_you_need.pdf"])

conv = AcademicConverter(
    source_prefix="arxiv",
    system_prompt=(
        "You are a research assistant trained on academic papers. Cite equation "
        "numbers, sections, and author claims precisely. Distinguish between "
        "what the paper proves and what it conjectures."
    ),
)
conv.convert_to_jsonl(results, "output/arxiv_training.jsonl")
```

**What you'll get:** Training examples like:
- `What is the main contribution of arXiv:1706.03762?` → "The paper proposes the Transformer architecture..."
- `What experiments were run for equation (3)?` → "The authors evaluate scaled dot-product attention against..."

## Example 2 — Legal contracts

**Goal:** answer questions about contract clauses, citing section numbers.

```python
class ContractConverter(DocumentConverter):
    def _extract_topic(self, text):
        # Section reference: "Section 4.2(a)" or "§ 4.2(a)"
        section = re.search(r"(?:Section|§)\s+([\d.]+(?:\([a-z]\))?)", text)
        if section:
            return f"Section {section.group(1)}"

        # Defined term in caps: "TERMINATION" or "INDEMNIFICATION"
        defined = re.match(r"^([A-Z][A-Z\s]{3,30})\.\s", text)
        if defined:
            return f"the {defined.group(1).strip().title()} clause"

        return super()._extract_topic(text)

    def _generate_question(self, topic, content):
        cl = content.lower()
        if "shall not" in cl or "prohibited" in cl or "must not" in cl:
            return f"What is prohibited by {topic}?"
        if "shall" in cl or "must" in cl:
            return f"What does {topic} require?"
        if "terminate" in cl:
            return f"What are the termination provisions in {topic}?"
        if "warrant" in cl or "represent" in cl:
            return f"What does {topic} warrant?"
        return f"Summarize {topic}."


conv = ContractConverter(
    source_prefix="contracts_v1",
    system_prompt=(
        "You are a contract analyst. When asked about a section, cite the section "
        "number and quote the operative language. Distinguish 'shall' (mandatory) "
        "from 'may' (permissive)."
    ),
)
```

## Example 3 — Technical specifications (RFCs, IEEE standards)

**Goal:** fine-tune a model to answer protocol-design questions, citing RFC numbers and section references.

```python
class RFCConverter(DocumentConverter):
    def _extract_topic(self, text):
        rfc = re.search(r"\bRFC\s*(\d{3,5})\b", text)
        if rfc:
            return f"RFC {rfc.group(1)}"

        # Inside-RFC section reference
        section = re.match(r"^([\d.]+)\s+(.{5,80}?)(?:\n|$)", text)
        if section:
            return f"Section {section.group(1)} ({section.group(2).strip()})"

        return super()._extract_topic(text)

    def _generate_question(self, topic, content):
        cl = content.lower()
        if "must" in cl or "shall" in cl or "required" in cl:
            return f"What does {topic} require implementations to do?"
        if "should" in cl or "recommended" in cl:
            return f"What does {topic} recommend?"
        if "may" in cl or "optional" in cl:
            return f"What does {topic} permit as optional?"
        if "deprecated" in cl or "obsolete" in cl:
            return f"What is deprecated in {topic}?"
        return f"Explain {topic}."

    def _find_identifiers(self, text):
        return list(dict.fromkeys(re.findall(r"\bRFC\s*\d{3,5}\b", text)))


conv = RFCConverter(
    source_prefix="rfcs",
    system_prompt=(
        "You are a network protocol expert. Cite RFC numbers and section references. "
        "Distinguish MUST/SHALL/REQUIRED from SHOULD/RECOMMENDED from MAY/OPTIONAL "
        "per RFC 2119."
    ),
)
```

## Example 4 — Medical guidelines

**Goal:** answer clinical questions citing ICD codes and drug names.

```python
class MedicalConverter(DocumentConverter):
    def _extract_topic(self, text):
        # ICD-10
        icd = re.search(r"\b([A-Z]\d{2}(?:\.\d{1,3})?)\b", text)
        if icd:
            return f"diagnosis {icd.group(1)}"

        # Drug + dose pattern
        drug = re.search(r"\b([A-Z][a-z]+(?:in|ol|am|ide|ate|pin|cin))\s+(\d+)\s*(mg|mcg|ml)\b", text)
        if drug:
            return f"{drug.group(1)} {drug.group(2)}{drug.group(3)}"

        return super()._extract_topic(text)

    def _generate_question(self, topic, content):
        cl = content.lower()
        if "contraindicated" in cl or "do not" in cl:
            return f"What are the contraindications for {topic}?"
        if "side effect" in cl or "adverse" in cl:
            return f"What are the side effects of {topic}?"
        if "dose" in cl or "dosage" in cl:
            return f"What is the dosing for {topic}?"
        if "diagnos" in cl or "criteria" in cl:
            return f"What are the diagnostic criteria for {topic}?"
        return f"Describe {topic}."


conv = MedicalConverter(
    source_prefix="clinical_guidelines",
    system_prompt=(
        "You are a clinical decision support assistant. Cite ICD codes, drug names "
        "with doses, and source guidelines. Distinguish absolute contraindications "
        "from relative ones. Always recommend consulting a licensed clinician for "
        "patient-specific advice."
    ),
)
```

## Example 5 — Internal company documentation

**Goal:** fine-tune on your company's wiki, runbooks, and architecture docs.

```python
class CompanyDocsConverter(DocumentConverter):
    def _extract_topic(self, text):
        # Common heading patterns in internal docs
        heading = re.match(r"^#+\s+(.{5,80}?)$", text, re.MULTILINE)
        if heading:
            return heading.group(1).strip()

        return super()._extract_topic(text)

    def _generate_question(self, topic, content):
        cl = content.lower()
        if "runbook" in cl or "incident" in cl or "alert" in cl:
            return f"What's the runbook for {topic}?"
        if "architecture" in cl or "design" in cl:
            return f"How is {topic} architected?"
        if "deploy" in cl or "release" in cl:
            return f"How do we deploy {topic}?"
        if "owner" in cl or "team" in cl:
            return f"Who owns {topic}?"
        return f"Tell me about {topic}."


conv = CompanyDocsConverter(
    source_prefix="internal_docs",
    system_prompt=(
        "You are an internal company assistant. Answer based on internal "
        "documentation. If a question requires information you don't have, "
        "say so explicitly rather than guessing."
    ),
)
```

## Pattern — When to use the base class as-is

If your corpus is a mix of document types and you don't have stable identifiers, the base `DocumentConverter` is fine. The generic question framing covers most "what / how / why" patterns and the JSONL output is still useful training data. You can always specialize later.

```python
from src.extractor import DocumentExtractor
from src.converter import DocumentConverter

ex = DocumentExtractor()
results = ex.extract(["data/sample_pdfs/mixed_corpus.pdf"])

conv = DocumentConverter(
    source_prefix="mixed_v1",
    system_prompt="You are a helpful assistant trained on technical documentation.",
)
conv.convert_to_jsonl(results, "output/mixed_training.jsonl")
```

## Pattern — Generate multiple training pipelines from one corpus

Useful if you want to fine-tune different models for different roles using the same source documents.

```python
ex = DocumentExtractor()
results = ex.extract(["data/sample_pdfs/big_corpus.pdf"])

# Pipeline 1 — assistant-style
DocumentConverter(
    source_prefix="assistant",
    system_prompt="You are a helpful assistant.",
).convert_to_jsonl(results, "output/assistant.jsonl")

# Pipeline 2 — expert-style with stricter framing
DocumentConverter(
    source_prefix="expert",
    system_prompt="You are a domain expert. Refuse to answer questions outside your training scope.",
).convert_to_jsonl(results, "output/expert.jsonl")

# Pipeline 3 — Socratic teaching style
class SocraticConverter(DocumentConverter):
    def _generate_question(self, topic, content):
        return f"What questions should a student ask to understand {topic}?"

SocraticConverter(
    source_prefix="socratic",
    system_prompt="You are a teacher. Help students think through problems by asking clarifying questions.",
).convert_to_jsonl(results, "output/socratic.jsonl")
```

Same extraction, three different training corpora, 1x the GPU cost.

## Pattern — Validate output before training

Before spending GPU hours on fine-tuning, sanity-check the JSONL:

```python
import json
from collections import Counter

with open("output/training.jsonl") as f:
    examples = [json.loads(line) for line in f]

print(f"Total examples: {len(examples)}")
print(f"Sources: {Counter(ex['source'] for ex in examples)}")
print(f"Avg user prompt length: {sum(len(ex['messages'][1]['content']) for ex in examples) // len(examples)}")
print(f"Avg assistant length: {sum(len(ex['messages'][2]['content']) for ex in examples) // len(examples)}")

# Spot-check first 3
for ex in examples[:3]:
    print("---")
    print("Q:", ex["messages"][1]["content"])
    print("A:", ex["messages"][2]["content"][:300])
```

If the avg assistant length is way too short (< 100 chars) or way too long (> 3000 chars), tune `min_length` and `max_length` in the converter.
