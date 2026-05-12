# SFT Recipes (SLIME)

SFT training scripts for Moonlight-16B-A3B and Qwen3-8B-Base via the SLIME backend, submitted as Ray jobs.

## Scripts

| Script | Model | Optimizer | Context | Nodes | Hardware |
|--------|-------|-----------|---------|-------|----------|
| `run-moonlight-*-sft-muon.sh` | Moonlight 16B-A3B | Muon | 8k | 8 | A800 |
| `run-moonlight-*-hard-32k.sh` | Moonlight 16B-A3B | Muon | 32k (no YaRN) | 4 | A800 |
| `run-moonlight-*-yarn-32k.sh` | Moonlight 16B-A3B | Muon | 32k (YaRN) | 16 | A800 |
| `run-moonlight-*-yarn-32k_4991.sh` | Moonlight 16B-A3B | Muon | 32k (YaRN cont.) | 8 | A800 |
| `run-moonlight-*-1node-B300.sh` | Moonlight 16B-A3B | Muon | 32k (YaRN) | 1 | B300 |
| `run-qwen3-8B-base-sft.sh` | Qwen3 8B Base | Adam | 32k | 8 | A800 |

## B300 / Blackwell (sm_103a) Pitfalls

- **TRITON_PTXAS_PATH** must point to system ptxas (`/usr/local/cuda/bin/ptxas`, CUDA 12.9+). Triton's bundled ptxas does not support sm_103a and will silently produce invalid kernels.
- **`--attention-backend auto`** is required instead of `flash`. Transformer Engine auto-selects the right backend; flash-attn may not yet support sm_103a.
- **TORCHDYNAMO_DISABLE=1** is needed to avoid torch.compile failures on the new architecture.
- **Muon's `use_syrk` Triton kernel** does not list sm_103a in its allowed architectures; it silently falls back to a slower path.
- **DeepEP** (`--moe-enable-deepep`) and **`--moe-permute-fusion`** depend on CUDA/TE kernels that may not yet support sm_103a.

## B300 vs Standard (yarn-32k) Differences

The B300 script uses TP=1 (no sequence-parallel) instead of TP=2, raises `max-tokens-per-gpu` from 64k to 128k, and parameterizes all paths via environment variables rather than hardcoding them.

## Required Environment Variables (B300)

`PROMPT_DATA`, `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_BASE_URL`, `WANDB_PROJECT`, `WANDB_GROUP`, `PROJECT_DIR`, `SCRIPT_DIR`, `PYTHONPATH`, `HF_CHECKPOINT`, `MCORE_CHECKPOINT`, `SAVE_DIR`

## Resume Training with Changed Parallelism (OOM Recovery)

When training OOMs, you often want to resume from the last checkpoint with larger TP/PP. Megatron checkpoints are sharded by the original parallelism and **cannot be loaded directly with different TP/PP**. The workflow is:

### 1. Find the last checkpoint

```bash
cat /your/save_dir/latest_checkpointed_iteration.txt
# 64  ← this is the rollout_id

ls /your/save_dir/
# iter_0000064_torch_dist/  ← directory name = rollout_id
```

### 2. Convert to HF format (parallelism-agnostic)

```bash
python modules/slime/tools/convert_torch_dist_to_hf.py \
  --input-dir /your/save_dir/iter_0000064_torch_dist \
  --output-dir /your/save_dir/iter_0000064-hf \
  --origin-hf-dir /path/to/original/hf/model
```

Or batch-convert all checkpoints:

```bash
bash recipes/slime/utils/batch_convert_to_hf.sh /your/save_dir /path/to/original/hf/model
```

### 3. Determine `--start-rollout-id`

The number in the checkpoint directory name (`iter_XXXXXXX`) **is** the rollout_id. Set `--start-rollout-id` to that number **+ 1**:

```
Checkpoint: iter_0000064  →  --start-rollout-id 65
```

This must be set manually because loading an HF checkpoint forces `start_rollout_id = 0` by default.

### 4. Launch training with new parallelism

```bash
python3 train.py \
  --hf-checkpoint /path/to/original/hf/model \
  --load /your/save_dir/iter_0000064-hf \
  --save /new/save_dir \
  --start-rollout-id 65 \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 2 \
  ...
```

### Caveats

- **Optimizer state is lost.** Converting to HF drops momentum/variance; the optimizer restarts fresh. Expect a brief convergence dip.
- **Learning rate schedule** continues from `start-rollout-id`, so the LR should be roughly correct if you set the rollout ID properly.

### OOM Mitigation Parameters

If increasing parallelism alone isn't enough, tune these knobs:

| Parameter | Effect |
|-----------|--------|
| `--max-tokens-per-gpu <N>` | Reduce per-GPU batch size (e.g. 64000 → 32000) |
| `--recompute-num-layers <N>` | Recompute N layers instead of caching activations (trade compute for memory) |
| `--recompute-granularity full` | Enable full activation recomputation |
| `--offload-train` | Offload actor weights to CPU during training |
| `--offload-rollout` | Offload rollout generator weights to CPU |
| `--train-memory-margin-bytes <N>` | Increase reserved memory margin (default 1 GB) |
