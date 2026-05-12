#!/bin/bash

set -euo pipefail

SLIME_DIR="/root/slime"
MEGATRON_PATH="/root/Megatron-LM"

cd "${SLIME_DIR}"

PYTHONPATH="${MEGATRON_PATH}" python tools/convert_torch_dist_to_hf.py \
  --input-dir $1 \
  --output-dir $2 \
  --origin-hf-dir $3
