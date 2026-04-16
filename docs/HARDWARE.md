# Hardware Guide

What GPU, RAM, and disk you need for different workload sizes.

## Quick decision matrix

| Workload | Recommended | Why | Approx cost |
|---|---|---|---|
| Development, tests, converter only | **No GPU needed** | All 51 unit tests run on CPU. Pure Python converter doesn't need GPU. | $0 |
| Small workload (<20 PDFs/day) | **RTX 3090 (24 GB)** or **RTX 4090 (24 GB)** | Fits nv-ingest's vision models with headroom for one concurrent extraction. | ~$0.30/hr cloud |
| Medium workload (20-200 PDFs/day) | **A40 (48 GB)** or **L40S (48 GB)** | More VRAM allows larger batches and faster startup. | ~$0.40/hr cloud |
| Heavy workload (200-1000 PDFs/day) | **A100 80 GB** or **DGX Spark** | Best throughput per PDF. Room for additional models alongside (e.g., a re-ranker). | ~$1-2/hr cloud |
| Enterprise / continuous | **DGX Station** or **H100 cluster** | 24/7 production deployments at scale. | varies |

## VRAM math

nv-ingest loads several vision models at startup:

| Model | Approx VRAM |
|---|---|
| YOLOX (page layout detection) | ~2 GB |
| PaddleOCR (OCR for scanned text) | ~1 GB |
| DePlot (chart understanding) | ~3 GB |
| Cached embeddings + buffers | ~2-4 GB |

**Steady-state baseline: ~10-12 GB VRAM.** Concurrent extractions add another ~1-3 GB each. So:

- 16 GB GPU: works for single extractions only, no concurrency, no margin
- 24 GB GPU: 2-3 concurrent extractions comfortable
- 48 GB GPU: ~6-8 concurrent extractions
- 80 GB GPU: ~12-16 concurrent extractions

## System RAM

Less critical than VRAM since most heavy compute happens on the GPU.

- **Minimum:** 16 GB system RAM
- **Recommended:** 32 GB system RAM (allows OS + Docker overhead + larger PDFs in flight)
- **Production:** 64 GB system RAM

The Docker compose file sets `shm_size: "16g"` which is shared memory for inter-process communication. If you run on a system with less than 16 GB system RAM, lower this in `docker-compose.yml` or expect OOMs.

## Disk space

| Item | Approx size |
|---|---|
| nv-ingest container images (pulled by docker compose) | ~20 GB |
| Model cache (`nv-ingest-cache` volume) | ~10 GB |
| Your PDFs | varies — plan for 5-10x the raw PDF size for extracted JSONL |

**Recommended free space: 60 GB minimum** for a clean install with room to grow.

## NVIDIA driver compatibility

nv-ingest 26.1.x requires:

- **NVIDIA driver:** 535+ (CUDA 12.1+ compatible)
- **NVIDIA Container Toolkit:** 1.14+
- **CUDA:** 12.1+ on the host (only matters if you run things outside Docker)

Check:
```bash
nvidia-smi | grep "Driver Version"
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If your driver is older than 535, upgrade before pulling the container or you'll get cryptic errors during model load.

### Apple Silicon caveat

You **cannot** run nv-ingest on Apple Silicon. NVIDIA containers require NVIDIA GPUs with CUDA. Apple M1/M2/M3/M4 chips have no path to running nv-ingest natively.

What you can do on Apple Silicon:

1. **Develop the converter** — all unit tests run on Apple Silicon
2. **SSH into a remote GPU machine** and run nv-ingest there
3. **Use a cloud GPU** (RunPod, Lambda Labs, AWS, GCP) for the actual extraction step
4. **Use a Linux box on your network** with a GPU

The converter, benchmark formatter, and tests all work on Apple Silicon. Only the actual `extractor.extract()` call needs CUDA.

## Cloud rental quick reference

If you don't own a GPU and want to do a one-off extraction job:

| Provider | Best deal for nv-ingest | Per-hour |
|---|---|---|
| **RunPod (spot)** | A40 48 GB or RTX 4090 24 GB | $0.20 - $0.35 |
| **Lambda Labs** | A100 40 GB or H100 80 GB | $1.10 - $2.50 |
| **TensorDock** | Various consumer + datacenter | $0.20 - $1.50 |
| **AWS EC2** | g5.2xlarge (A10G) | ~$1.20 |
| **Vast.ai** | Marketplace, varies wildly | $0.15 - $1.00 |

Spot/interruptible instances are fine for batch extraction work. For interactive use, spot can be frustrating because they can be reclaimed.

## DGX Spark / DGX Station

NVIDIA's "developer desktop" workstations:

- **DGX Spark (GB10):** 128 GB unified memory, ARM64. Single-purpose AI dev box, ~$3000-4000. **NemoClaw and nv-ingest both list DGX Spark as a tested platform.**
- **DGX Station (Blackwell Ultra, 2026):** 252 GB HBM3e, x86. ~$50K+ tier, true workstation power.

Both are excellent for running nv-ingest 24/7 alongside other AI services (Ollama, Qdrant, MCP servers, etc.).

## Recommendation by use case

**"I just want to try it once"** → RunPod RTX 4090 spot, ~$0.30/hr, 1-2 hour rental

**"I'm developing a custom converter"** → No GPU yet. Develop and test locally; rent a GPU when you need to run a real extraction

**"I'm building a pipeline for my company's regular documents"** → A40 48 GB on RunPod or Lambda. ~$300/month if you run it 8 hours/day

**"This is going into production"** → A100 80 GB or H100 dedicated, possibly multi-GPU. Or an on-prem DGX Spark / DGX Station if you can't use cloud

**"I have an old GPU lying around"** → If it's at least 16 GB VRAM and supports CUDA 12.1, give it a try. Below 16 GB you'll hit OOM during model load.
