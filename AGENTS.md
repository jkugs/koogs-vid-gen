# Project Guidance

## Project Context

This repository is an experimental AI video-generation project. Read
`dev-docs/001.md` before making substantial changes.

The short-term goal is to prove a minimal image-to-video workflow on a RunPod
GPU: take a real photograph and motion prompt, run Wan 2.2, and save an MP4.
Iteration 1 should remain a small, explicit Python script rather than becoming
an application or general model framework.

The long-term goal is a personal AI filmmaking system with consistent people
and characters. Possible later capabilities include identity-preserving image
generation, LoRA training, face refinement, lip sync, voice, scenes, shots,
sequencing, job orchestration, and a web UI. Do not build these until an
iteration specifically calls for them.

## Development Documentation

Use `dev-docs/` as durable working context shared across development sessions.
At the beginning of a session, list the directory and read `dev-docs/001.md`,
the latest applicable handoff, and any documents relevant to the current topic
before proposing substantial work.

Documents may follow a numbered sequence for chronological sessions or
iterations, such as `002.md` and `003.md`, or use descriptive one-off names for
specific investigations and decisions. A strict numbering scheme is not
required; prefer filenames that make the intended context easy to find.

When updating or creating a handoff, record confirmed decisions, completed
work, important paths and commands, observed results, unresolved issues, and
the exact next checkpoint. Clearly distinguish completed work from planned
work. Do not include credentials, private images, generated private media, or
other secrets. These documents supplement the code and Git history rather than
replacing them.

## Current Technical Direction

- Use Python 3.11 and `uv` for dependencies and commands.
- Use `Wan-AI/Wan2.2-TI2V-5B-Diffusers` through Diffusers.
- Target an NVIDIA RunPod GPU with CUDA 12.8. Torch and Torchvision are pinned
  to the CUDA 12.8 builds in `pyproject.toml`.
- Keep model weights, caches, personal photos, and generated videos out of Git.
- On RunPod, keep persistent data under `/workspace` and use:
  - `HF_HOME=/workspace/.cache/huggingface`
  - `UV_CACHE_DIR=/workspace/.cache/uv`
- Prefer direct, understandable code and minimal dependencies. Learn from real
  runs before introducing abstractions.

## Working Style

- Inspect relevant files and explain the recommended next step first.
- Suggest exact commands and changes, including likely costs or risks.
- Get the user's confirmation before material code changes, dependency changes,
  commits, destructive actions, model downloads, or paid GPU inference runs.
- Read-only inspection and routine verification do not require confirmation.
- The user runs commands in remote RunPod SSH sessions unless direct remote
  access has explicitly been provided. Give commands that are safe to paste and
  state whether they run locally or on the pod.
- Work in small, testable increments. After each change, run proportionate
  checks and report what passed, what remains untested, and whether files are
  committed.
- Do not commit or push unless the user asks.

## Iteration 1 Boundaries

Focus on `generate_video.py`, input validation, Wan inference, MP4 export, and
basic timing/VRAM observability. Avoid a web UI, API, database, authentication,
serverless orchestration, queues, plugins, model abstraction layers, or other
future-facing architecture.
