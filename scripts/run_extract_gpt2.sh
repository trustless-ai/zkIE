#!/usr/bin/env bash
# Extract GPT-2's fixed-point weights + lookup tables into models/gpt2/weights/.
#
# The previous version hard-coded a checkout at /data/jimmyshi/ie and shelled out
# to `docker run`; neither exists on a normal dev box, so the script could not
# run anywhere. It now locates the repo root itself and uses a local venv:
#
#   python3 -m venv .venv-gpt2
#   .venv-gpt2/bin/python -m pip install torch transformers onnx onnxscript onnxruntime
#
# Set PYTHON=/path/to/python to use a different interpreter (any Python >= 3.10
# with those packages works).
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv-gpt2/bin/python}
if [ ! -x "$PYTHON" ]; then
    echo "error: $PYTHON not found; create it with:" >&2
    echo "  python3 -m venv .venv-gpt2" >&2
    echo "  .venv-gpt2/bin/pip install torch transformers onnx onnxscript onnxruntime" >&2
    exit 1
fi

# huggingface.co / cdn-lfs.huggingface.co are unreachable from some networks
# (plain connection timeouts, not 4xx). hf-mirror.com serves the same files, so
# default to it; export HF_ENDPOINT=... to point somewhere else.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

"$PYTHON" -m pip install -q onnx onnxruntime numpy

# extract_gpt2.py reads models/gpt2/gpt2.onnx, which export_gpt2.py produces.
if [ ! -f models/gpt2/gpt2.onnx ]; then
    echo "models/gpt2/gpt2.onnx missing; exporting it first (GPT2_SEQ=${GPT2_SEQ:-16})"
    "$PYTHON" scripts/export_gpt2.py
fi

"$PYTHON" scripts/extract_gpt2.py
