# GPU Infrastructure

GPU worker images and provisioning land in Phase 2 (ACE-Step integration)
and Phase 7 (GPU job system).

Planned contents:

- `gpu-worker.Dockerfile` — CUDA base image for the generation worker
  running the ACE-Step model server (separate from the CPU worker image).
- Model server process supervision (crash isolation from web/API).
- Target hardware: RTX 4090 24GB / RTX 5090 32GB / L40S / A100 / H100.
- Local development on Apple Silicon uses MPS; production targets NVIDIA CUDA.
