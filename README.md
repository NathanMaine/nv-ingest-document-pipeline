# nv-ingest Document Pipeline

[![CI](https://github.com/NathanMaine/nv-ingest-document-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/NathanMaine/nv-ingest-document-pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Coverage 95%](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](#testing-and-quality)
[![Type-checked: mypy strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy.readthedocs.io/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A GPU-accelerated document extraction pipeline built on [NVIDIA nv-ingest](https://github.com/NVIDIA/nv-ingest) (~2.8K stars). Extract text, tables, charts, and metadata from any PDF corpus and convert the output into chat-format JSONL training data for fine-tuning LLMs on your domain.

This repo is a **starter / shell**. The architecture is generic. You bring the documents, you choose the system prompt, you decide what your domain-specific question framing looks like — the pipeline handles the heavy lifting of extraction, chunking, formatting, and benchmarking.

## What this gives you

- **One-command nv-ingest setup** via docker compose — no manual model downloads, no CUDA-version headaches
- **A clean Python API** to extract any PDF corpus
- **A converter** that turns extracted content into chat-format JSONL ready to feed into training pipelines (the format used by `axolotl`, `unsloth`, and most QLoRA / SFT toolchains)
- **A benchmark harness** that compares nv-ingest against a CPU PyPDF2 baseline, so you can quantify the speedup on your own documents
- **61 unit tests** that all run without a GPU, so CI is fast and you can iterate on the converter without spinning up nv-ingest
- **Extension points** documented in [docs/CUSTOMIZATION.md](docs/CUSTOMIZATION.md) — subclass the converter to add domain-specific topic extraction, ID detection, or question framing

## Pipeline flow

```
your PDFs (any domain)
        |
   nv-ingest (GPU extraction)
   - Text with paragraph/heading classification
   - Tables with row/column structure
   - Charts and diagrams
        |
   DocumentConverter (this repo)
   - Chunks long text on paragraph + sentence boundaries
   - Emits chat-format training examples
   - Optional: subclass to add your own topic extraction
        |
   chat-format JSONL
   {"messages": [system, user, assistant], "source": "nv_ingest_..."}
        |
   feed into your training pipeline
   (axolotl, unsloth, transformers, etc.)
```

## Quick start

The full step-by-step is in [docs/SETUP.md](docs/SETUP.md). The 60-second version:

```bash
git clone https://github.com/NathanMaine/nv-ingest-document-pipeline.git
cd nv-ingest-document-pipeline

# 1. Install Python deps (no GPU needed for the converter or tests)
pip install -r requirements.txt

# 2. Run the test suite (all 51 tests pass without GPU or nv-ingest)
python -m pytest tests/ -v

# 3. (When ready for real extraction) start nv-ingest
docker login nvcr.io          # free NGC account required
docker compose up -d           # ~10-15 min first run for model loading
docker compose logs -f         # watch readiness

# 4. Drop a PDF in data/sample_pdfs/, then:
python -c "
from src.extractor import DocumentExtractor
from src.converter import DocumentConverter

ex = DocumentExtractor()
results = ex.extract(['data/sample_pdfs/your.pdf'])

conv = DocumentConverter()
conv.convert_to_jsonl(results, 'output/training_data.jsonl')
"
```

## What you choose, what the pipeline gives you

| Decision | You choose | Pipeline provides |
|---|---|---|
| Document corpus | Your PDFs | Extraction at scale |
| System prompt | Domain-specific role | Generic default works for any topic |
| Question framing | Subclass `_generate_question()` | Generic question templates that work for "what / how / why" |
| Topic extraction | Override `_extract_topic()` to detect your domain's identifiers | Falls back to first-sentence heuristic |
| Chunk size | `min_length` and `max_length` constructor args | Defaults of 100 / 4000 chars work for most corpora |
| Output format | Anywhere you like (the JSONL is standard chat format) | Chat format compatible with axolotl, unsloth, transformers |

## Documentation

| Doc | What it covers |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Complete install walkthrough — Python, Docker, NGC login, GPU drivers, first extraction |
| [docs/HARDWARE.md](docs/HARDWARE.md) | GPU recommendations for different workload sizes, RAM/disk requirements, NVIDIA driver compatibility |
| [docs/CUSTOMIZATION.md](docs/CUSTOMIZATION.md) | How to subclass `DocumentConverter` for your domain — topic extraction, identifier detection, custom question framing, custom chunking |
| [docs/EXAMPLES.md](docs/EXAMPLES.md) | Worked examples — academic papers, legal contracts, technical documentation, scientific datasets |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common errors and fixes — Docker daemon, NGC auth, OOM during model load, slow extraction |
| [examples/custom_converter_example.py](examples/custom_converter_example.py) | Working code that subclasses `DocumentConverter` to add custom domain logic |

## Project structure

```
nv-ingest-document-pipeline/
├── src/
│   ├── extractor.py      # nv-ingest API wrapper (DocumentExtractor)
│   ├── converter.py       # nv-ingest JSON -> chat training format (DocumentConverter)
│   └── benchmark.py       # PyPDF2 vs nv-ingest side-by-side comparison
├── tests/
│   ├── test_extractor.py  # 18 tests (mock data, no GPU)
│   ├── test_converter.py  # 22 tests (mock data, no GPU)
│   └── test_benchmark.py  # 11 tests (mock data, no GPU)
├── data/
│   ├── sample_pdfs/       # Drop your test PDFs here
│   └── expected_output/   # Ground truth JSONL for validation (you create this)
├── docs/
│   ├── SETUP.md
│   ├── HARDWARE.md
│   ├── CUSTOMIZATION.md
│   ├── EXAMPLES.md
│   └── TROUBLESHOOTING.md
├── examples/
│   └── custom_converter_example.py
├── docker-compose.yml     # nv-ingest service definition
├── requirements.txt       # Pinned Python dependencies
├── LICENSE                # Apache 2.0
└── README.md              # This file
```

## Hardware requirements (TL;DR)

| Use case | GPU | Reason |
|---|---|---|
| Development, testing, converter only | None | All 51 tests run on CPU, converter is pure Python |
| Small workload (<20 PDFs/day) | RTX 3090 / 4090 (24 GB) | Fits nv-ingest's models with headroom |
| Medium workload (20-200 PDFs/day) | A40 / L40S (48 GB) | Faster inference, supports larger batches |
| Production / heavy batch jobs | A100 80GB / H100 / DGX Spark | Best throughput per PDF, room for additional models |

Full details in [docs/HARDWARE.md](docs/HARDWARE.md).

## Why nv-ingest specifically

You can extract PDFs with PyPDF2, pdfminer.six, pdfplumber, Marker, MinerU, Docling, and a dozen other tools. The benchmark in this repo lets you compare nv-ingest against the PyPDF2 baseline on your own documents, but the broad story is:

- **GPU-accelerated** — extraction time scales with VRAM, not CPU cores
- **Table-aware** — preserves row/column structure that PyPDF2 flattens
- **Chart-aware** — extracts diagrams and charts as structured text
- **Maintained by NVIDIA** — production-grade with regular updates
- **Microservice architecture** — easy to scale horizontally with Kubernetes
- **OCR fallback** — handles scanned PDFs, not just native text PDFs

If your corpus is mostly clean native-text PDFs and you don't need tables, PyPDF2 is fine. If you have any combination of dense tables, scanned pages, or charts you need preserved as text — nv-ingest pays for itself.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

Built on [NVIDIA nv-ingest](https://github.com/NVIDIA/nv-ingest), a GPU-accelerated document extraction microservice. Thanks to the nv-ingest team — including [Edward Kim](https://github.com/edknv), [Jeremy Dyer](https://github.com/jdye64), and [Devin Robison](https://github.com/drobison00) — for building and maintaining the project.

The chat-format JSONL output is compatible with most modern fine-tuning toolchains including [axolotl](https://github.com/axolotl-ai-cloud/axolotl), [unsloth](https://github.com/unslothai/unsloth), and the Hugging Face [transformers](https://github.com/huggingface/transformers) library's TRL trainer.
