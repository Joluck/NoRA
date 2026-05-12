#!/bin/bash
# Batch convert all iter_* checkpoints under a directory to HF format.
#
# Usage:
#   bash batch_convert_to_hf.sh <ckpt-dir> <origin-hf-dir>
#
# Example:
#   bash batch_convert_to_hf.sh \
#     /jpfs-5p/chenyanxu.9/model/Qwen3-8B-Base-sft-dolci-think \
#     /jpfs/chenyanxu.9/model/zhennanshen/Qwen3-8B-Base

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <ckpt-dir> <origin-hf-dir>"
    exit 1
fi

CKPT_DIR="$1"
ORIGIN_HF_DIR="$2"

SLIME_DIR="/root/slime"
MEGATRON_PATH="/root/Megatron-LM"

cd "${SLIME_DIR}"

for input_dir in "${CKPT_DIR}"/iter_*; do
    # skip non-directories
    [ -d "${input_dir}" ] || continue
    # skip already-converted -hf directories and _torch_dist directories
    case "${input_dir}" in
        *-hf|*_torch_dist) continue ;;
    esac

    output_dir="${input_dir}-hf"

    if [ -d "${output_dir}" ]; then
        echo "[SKIP] ${output_dir} already exists"
        continue
    fi

    echo "[CONVERT] ${input_dir} -> ${output_dir}"
    PYTHONPATH="${MEGATRON_PATH}" python tools/convert_torch_dist_to_hf.py \
        --input-dir "${input_dir}" \
        --output-dir "${output_dir}" \
        --origin-hf-dir "${ORIGIN_HF_DIR}"
done

echo "Done."
