# Setup Guide

Complete walkthrough from "I just cloned the repo" to "I have JSONL training data extracted from my PDFs." Should take **30-60 minutes** end to end depending on your hardware and download speed.

## Prerequisites checklist

Before you start, confirm you have:

- [ ] **Linux, macOS (Apple Silicon), or Windows + WSL2**
- [ ] **Docker Engine 24.0+** ([install instructions](https://docs.docker.com/engine/install/))
- [ ] **Python 3.10+** (`python3 --version`)
- [ ] **NVIDIA GPU with at least 24 GB VRAM** for production extraction (RTX 3090, RTX 4090, A40, A100, etc.) — *not required for tests or the Python converter, only for actually running nv-ingest*
- [ ] **NVIDIA Container Toolkit** ([install instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) — *only needed if you have a GPU*
- [ ] **NGC account** (free at https://ngc.nvidia.com) — *only needed to pull the nv-ingest container*
- [ ] **20 GB free disk space** for the nv-ingest container images and model cache

## Step 1 — Clone and install Python deps

```bash
git clone https://github.com/NathanMaine/nv-ingest-document-pipeline.git
cd nv-ingest-document-pipeline

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

Expected install time: 1-3 minutes.

## Step 2 — Run the test suite

```bash
python -m pytest tests/ -v
```

Expected: **51 tests, all pass**, in under 10 seconds. **No GPU or Docker required for this step** — all tests use mocks. If any tests fail at this stage, the install is broken; do not proceed to nv-ingest setup.

If `nv_ingest_client` is missing and pytest collection fails, that's fine — those tests use mocks and don't actually call the real client. Look for `ImportError: nv_ingest_client` only in tests that explicitly test the import-error path; that's expected.

## Step 3 — NGC login (one-time)

NVIDIA's container registry requires authentication, even for free tier downloads.

1. Create a free NGC account at https://ngc.nvidia.com
2. Generate an API key: profile menu → "Setup" → "Generate API Key"
3. Login from your shell:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <paste your API key>
```

If you skip this step, `docker compose up` will fail with `pull access denied`.

## Step 4 — Verify GPU and Docker

```bash
# Check NVIDIA driver
nvidia-smi
# Should show your GPU and driver version

# Check NVIDIA Container Toolkit is configured for Docker
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# Should print the same nvidia-smi output from inside a container
```

If the container `nvidia-smi` fails, the NVIDIA Container Toolkit isn't configured. See the [official troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html).

## Step 5 — Start nv-ingest

```bash
docker compose up -d
docker compose logs -f
# Keep this open in a second terminal — first run takes 10-15 min for model loading
```

You'll see output like:
```
nv-ingest-1  | INFO:nv_ingest:loading models...
nv-ingest-1  | INFO:nv_ingest:cached_paddle_ocr loaded
nv-ingest-1  | INFO:nv_ingest:nim_yolox loaded
nv-ingest-1  | INFO:nv_ingest:ready to serve
```

When you see `ready to serve`, you can stop watching logs (Ctrl+C — this only stops the log follow, not the container).

Verify the service is up:
```bash
curl http://localhost:7670/health
# Should return: {"status":"ready"} or similar
```

## Step 6 — First extraction

Drop a PDF in `data/sample_pdfs/`:

```bash
cp /path/to/your/test.pdf data/sample_pdfs/
```

Run a smoke test:

```bash
python3 -c "
from src.extractor import DocumentExtractor
ex = DocumentExtractor()
results = ex.extract(['data/sample_pdfs/test.pdf'])
for r in results:
    print(f'{r.source_file}: {r.text_count} text + {r.table_count} tables + {r.chart_count} charts')
    print(f'  processed in {r.processing_time_ms:.0f}ms')
"
```

Expected output:
```
data/sample_pdfs/test.pdf: 47 text + 12 tables + 0 charts
  processed in 8420ms
```

If you see `nv-ingest not installed` even though you ran `pip install -r requirements.txt`, you're probably running outside your virtualenv. Re-activate it.

## Step 7 — Generate training JSONL

```bash
python3 -c "
from src.extractor import DocumentExtractor
from src.converter import DocumentConverter

ex = DocumentExtractor()
results = ex.extract(['data/sample_pdfs/test.pdf'])

conv = DocumentConverter()
conv.convert_to_jsonl(results, 'output/test_training.jsonl')

# Quick stats
import json
with open('output/test_training.jsonl') as f:
    examples = [json.loads(line) for line in f]
print(f'Generated {len(examples)} training examples')
print(f'First example user prompt: {examples[0][\"messages\"][1][\"content\"][:200]}')
"
```

Expected output:
```
Generated 89 training examples
First example user prompt: What are the requirements for...
```

You now have chat-format JSONL ready for fine-tuning.

## Step 8 — Run the benchmark (optional)

Compare nv-ingest against PyPDF2 baseline on your PDF:

```bash
python3 -c "
from src.benchmark import run_benchmark, format_results_table
comparisons = run_benchmark(['data/sample_pdfs/test.pdf'], output_path='benchmarks/results.json')
print(format_results_table(comparisons))
"
```

Expected output:
```
| Document | Method    | Time (ms) | Text Blocks | Tables | Chars  | Speedup |
|----------|-----------|-----------|-------------|--------|--------|---------|
| test     | PyPDF2    | 1234      | 47          | 0      | 124,567 | -      |
| test     | nv-ingest | 8420      | 47          | 12     | 156,890 | 0.1x   |
```

Wait — *PyPDF2 is faster?* Yes, on small text-only PDFs PyPDF2 wins on raw wall-clock because it doesn't pay the GPU model-loading overhead. The nv-ingest advantage shows on:
- Table extraction (PyPDF2 returns 0 tables; nv-ingest returns structured tables)
- Total characters extracted (nv-ingest catches content PyPDF2 misses)
- Large PDFs and batches (model load is amortized)
- OCR'd / scanned PDFs (PyPDF2 returns nothing; nv-ingest does OCR)

Run the benchmark on a 100-page table-heavy PDF and the speedup ratio reverses dramatically.

## Step 9 — Stop the service when done

```bash
docker compose down              # stop containers, keep model cache
docker compose down --volumes    # also delete model cache (re-downloads next start)
```

## What's next

- **Customize for your domain:** [docs/CUSTOMIZATION.md](CUSTOMIZATION.md)
- **Pick the right GPU:** [docs/HARDWARE.md](HARDWARE.md)
- **Worked examples for different document types:** [docs/EXAMPLES.md](EXAMPLES.md)
- **If something goes wrong:** [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)
