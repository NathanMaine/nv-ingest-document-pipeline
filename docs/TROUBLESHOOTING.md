# Troubleshooting

Common errors and their fixes, in roughly the order you'll hit them during setup.

## Install / dependency errors

### `pip install -r requirements.txt` fails on `nv-ingest`

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement nv-ingest>=26.1.2
```

**Cause:** `nv-ingest` is not on PyPI by default. NVIDIA publishes it via NGC's pip index.

**Fix:**
```bash
pip install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-nemo-microservices-pypi/simple/ nv-ingest nv-ingest-api nv-ingest-client
```

If you only need the converter and tests (not actual extraction), skip the install entirely — the tests use mocks.

### `import nv_ingest_client` fails inside the project

**Symptom:** Tests fail collection with `ModuleNotFoundError: nv_ingest_client`.

**Cause:** You're not in the virtualenv where you installed dependencies.

**Fix:**
```bash
source .venv/bin/activate    # Re-enter virtualenv
which python                 # Should show .venv path
python -m pytest tests/ -v   # Should now pass
```

## Docker / NGC errors

### `docker compose up` fails with `pull access denied for nvcr.io/...`

**Symptom:**
```
Error response from daemon: pull access denied for nvcr.io/nvidia/nemo-microservices/nv-ingest
```

**Cause:** You haven't logged into NGC.

**Fix:**
1. Create free NGC account at https://ngc.nvidia.com
2. Generate API key (profile → "Setup" → "Generate API Key")
3. Login:
   ```bash
   docker login nvcr.io
   # Username: $oauthtoken      (literally $oauthtoken, not your username)
   # Password: <paste API key>
   ```
4. Retry `docker compose up -d`

### `docker compose up` fails with `could not select device driver "nvidia"`

**Symptom:**
```
ERROR: could not select device driver "" with capabilities: [[gpu]]
```

**Cause:** NVIDIA Container Toolkit is not installed or not configured.

**Fix:**
```bash
# Install (Ubuntu / Debian)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

For other Linux distros, see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

### nv-ingest container starts but never reaches "ready to serve"

**Symptom:** `docker compose logs -f nv-ingest` shows model-loading messages but eventually OOM-kills or hangs.

**Possible causes:**

| Cause | Symptom in logs | Fix |
|---|---|---|
| Insufficient VRAM | `CUDA out of memory` | Use a 24+ GB GPU or wait for a smaller model variant |
| Insufficient system RAM | `OOMKilled` in `docker inspect` | Increase RAM or reduce `shm_size` in docker-compose.yml |
| Slow disk | Logs show 5+ min between download messages | First-time pull is slow on slow disks. Wait it out (10-20 min normal) |
| NGC auth expired | `401 Unauthorized` | Re-run `docker login nvcr.io` |
| Driver too old | `CUDA error: driver version is insufficient` | Update NVIDIA driver to 535+ |

### `curl http://localhost:7670/health` returns connection refused

**Symptom:** Container is `Up` per `docker ps` but health check fails.

**Cause:** Models are still loading. The `start_period: 600s` in docker-compose.yml gives a 10-minute grace.

**Fix:** Wait. Check progress with `docker compose logs -f nv-ingest`. When you see `ready to serve`, retry the curl.

## Extraction errors

### `extract()` raises `ImportError: nv-ingest not installed`

**Symptom:**
```python
>>> ex.extract(["x.pdf"])
ImportError: nv-ingest not installed. Install with: pip install nv-ingest nv-ingest-client
```

**Cause:** `nv-ingest` isn't in your active Python environment.

**Fix:** See "Install / dependency errors" above. Make sure the virtualenv is active.

### `extract()` hangs with no output

**Symptom:** Python call to `extractor.extract([...])` runs for 10+ minutes with no output.

**Possible causes:**

1. **First call after container start.** Initial PDF triggers model warm-up. Subsequent calls are much faster.
2. **PDF is huge or scanned.** OCR'd 500-page PDFs can take 5-10 minutes legitimately.
3. **nv-ingest container died.** Check `docker compose ps` — if status isn't "Up", restart it.

**Fix:** Add timeout to your call, or check container health:
```python
import requests
print(requests.get("http://localhost:7670/health").json())
```

### `extract()` returns empty `contents`

**Symptom:** Result has `total_pages > 0` but `contents == []`.

**Possible causes:**

1. **PDF is image-only and OCR is disabled.** Default extraction includes OCR (`pdfium_hybrid`), but if the OCR model failed to load, you'll get no text.
2. **PDF is encrypted.** PyPDF2 and nv-ingest both fail silently on password-protected PDFs.
3. **PDF text is in an unsupported language.** OCR models default to English; non-Latin scripts may need a different model variant.

**Fix:**
```python
# Check the raw output
import json
with open("debug.json", "w") as f:
    f.write(ex.extract_to_json(["x.pdf"], "debug.json"))
# Inspect debug.json for clues
```

## Converter errors

### Converter produces 0 training examples from a non-empty PDF

**Symptom:** `len(examples) == 0` even though `extractor.extract()` returned content.

**Possible causes:**

1. **All chunks below `min_length`.** Default is 100 chars. If your PDF has very short paragraphs, lower it.
2. **No topics detected.** If `_extract_topic` returns None for every chunk, no examples are emitted.

**Fix:**
```python
# Lower the minimum length
conv = DocumentConverter(min_length=30)

# Or override topic extraction to always return something
class PermissiveConverter(DocumentConverter):
    def _extract_topic(self, text):
        # Always extract first 50 chars as topic
        first_line = text.split("\n")[0][:50]
        return first_line if first_line.strip() else "this content"
```

### Training examples are too short / too long

**Symptom:** Average assistant length is way off (< 100 chars or > 3000 chars).

**Fix:** Tune `max_length`:
```python
conv = DocumentConverter(
    min_length=200,    # Drop tiny chunks
    max_length=2000,   # Tighter chunks for shorter assistant responses
)
```

### Examples have repetitive question framing

**Symptom:** All training examples start with "What are the requirements for..."

**Cause:** Your content has lots of "shall" / "must" / "requirement" keywords.

**Fix:** Override `_generate_question` for more variety. See [docs/CUSTOMIZATION.md](CUSTOMIZATION.md).

## Test errors

### Tests fail with `nv_ingest_client` import error

**Symptom:**
```
ERROR collecting tests/test_extractor.py
ModuleNotFoundError: No module named 'nv_ingest_client'
```

**Cause:** Tests use mocks but a top-level import in `extractor.py` is failing.

**Fix:** This shouldn't happen because the import is inside a function, but if it does:
```bash
pip install nv-ingest nv-ingest-client --no-deps
```

### Tests pass locally but fail in CI

**Symptom:** Local `pytest` is green, GitHub Actions / GitLab CI shows failures.

**Common causes:**
- CI runner doesn't have Python 3.10+ (`actions/setup-python@v5` with `python-version: '3.10'` or higher)
- CI runner doesn't have the `src` package on the path. Add to `tests/__init__.py` or use `pyproject.toml` with `packages = ["src"]`

## Performance / cost issues

### Extraction is slower than PyPDF2

**Symptom:** Benchmark shows `nv-ingest` takes longer than `pypdf2`.

**This is expected** for small native-text PDFs. nv-ingest pays a constant model-load cost regardless of PDF size; PyPDF2 is essentially free for text extraction. The tradeoff is:
- nv-ingest extracts tables (PyPDF2 doesn't)
- nv-ingest does OCR for scanned PDFs (PyPDF2 doesn't)
- nv-ingest scales much better for batch jobs (10 PDFs is barely slower than 1)

Run the benchmark on a 100-page table-heavy PDF; the speedup ratio reverses.

### Spending too much on cloud GPUs

**Symptom:** Bills are high relative to extraction volume.

**Fixes:**
1. **Batch your extractions.** Don't call `extract([one_pdf])` 100 times — call `extract([pdf1, pdf2, ..., pdf100])` once.
2. **Use spot instances.** RunPod / Vast.ai spot pricing is 2-3x cheaper than on-demand.
3. **Scale down between batches.** `docker compose down` between work sessions, then `up -d` when needed.
4. **Cache the model layer.** The named volume `nv-ingest-cache` persists models across container restarts. Don't delete it accidentally.

## Still stuck?

1. Check `docker compose logs -f nv-ingest` for the actual nv-ingest error
2. Search the [nv-ingest GitHub issues](https://github.com/NVIDIA/nv-ingest/issues)
3. Open an issue on this repo with: nv-ingest version, GPU model, driver version, command that failed, full error output
