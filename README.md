# NoRA: Rank-wise Normalized LoRA

NoRA is a LoRA variant that L2-normalizes the columns of `lora_A.weight` along the rank
(r) dimension, implemented as a fork of [PEFT](https://github.com/huggingface/peft) together
with an SFT training pipeline.

Constrained to the unit sphere, each column of `A` acts as a direction rather than an
arbitrary vector: magnitude is decoupled from `A` and carried entirely by `lora_B` (and the
scaling factor). This keeps every rank direction balanced throughout training and makes the
effective update scale easier to reason about.

Two modes are supported via `use_nora`:

- `use_nora=True` — normalize **once**, right after adapter initialization. Afterwards `A`
  is an ordinary parameter; merging is the standard `B @ A`.
- `use_nora="alltime"` — normalize on **every forward pass** (reparameterization-style;
  gradients flow through the normalization). `merge_and_unload()` applies the same
  normalization, so the merged model matches the adapter model exactly.

**BIMI** (Block Identity Matrix Initialization, `init_lora_weights="bimi"`) is a special
case of this family: `lora_A` is initialized as `r × r` identity blocks tiled along the
diagonal (`A[:, kr:(k+1)r] = I`), whose columns are already unit-norm and mutually
orthogonal — i.e. a maximally balanced point on the NoRA constraint manifold — while
`lora_B = 0` keeps `ΔW = 0` at step 0. No SVD over the base weights is needed (unlike
PiSSA/MiLoRA).

## Installation

```bash
git clone <this-repo>
cd BIMI/peft
uv pip install .   # or: pip install .
```

This installs the modified `peft` package with `use_nora` and `init_lora_weights="bimi"`
support. **Note:** it replaces any existing `peft` installation in the environment.

## Usage

### With PEFT directly

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_nora="alltime",         # per-forward normalization; or True for init-only
    init_lora_weights="bimi",   # optional: block identity init (already unit-norm)
)
model = get_peft_model(base_model, config)
```

`init_lora_weights` also accepts the standard PEFT options
(`True`/`"gaussian"`, `"pissa"`, `"olora"`, `"corda"`, `"loftq"`, ...), all of which compose
with `use_nora`.

### SFT training

The `sft/` directory contains a full supervised fine-tuning pipeline
(HuggingFace Trainer + DeepSpeed, adapted from the PiSSA codebase):

```bash
cd sft
deepspeed --include=localhost:0,1,2,3 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --full_finetune False --bf16 \
    --nora alltime \
    --lora_rank 32 --lora_alpha 32 \
    --data_path fxmeng/pissa-dataset --sub_task metamath:100000 \
    --dataset_field instruction output \
    --output_dir <out>
```

See `sft/scripts/` for complete runnable examples.

Key arguments in `sft/train.py`:

| Argument | Description |
|---|---|
| `--nora` | `True` = normalize `lora_A` once after init; `alltime` = normalize every forward |
| `--init_weights` | `True` = vanilla LoRA; `bimi` = BIMI; also `pissa`, `pissa_niter_N`, `olora`, `gaussian` |
| `--lora_rank` / `--lora_alpha` | LoRA rank / alpha |
| `--target_modules` | Comma-separated modules to adapt |
| `--full_finetune` | Disable adapters and train the full model |
| `--merge` | Merge adapter into base weights when saving |
| `--bits` | `16` for full precision, `4`/`8` for quantized training |

DeepSpeed configs are provided in `sft/configs/` (ZeRO-2 and ZeRO-3).

### Evaluation

Task accuracy with vLLM generation:

```bash
python utils/gen_vllm.py --model <ckpt> --data_path <dataset> --sub_task <task> \
    --output_file model_response.jsonl
python utils/test_acc.py --input_file model_response.jsonl
```

Benchmarks with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(see `sft/eval.sh`):

```bash
lm_eval --model vllm \
    --model_args pretrained=<ckpt>,tensor_parallel_size=4,dtype=bfloat16,gpu_memory_utilization=0.8 \
    --tasks mmlu,agieval,arc_challenge \
    --batch_size auto
```

## Repository structure

```
BIMI/
├── peft/     # PEFT fork: use_nora + init_lora_weights="bimi"
│             # (core logic: peft/src/peft/tuners/lora/layer.py)
└── sft/      # SFT training + evaluation pipeline
    ├── train.py
    ├── eval.sh    # lm_eval entry (mmlu / agieval / arc_challenge)
    ├── configs/   # DeepSpeed configs
    ├── scripts/   # example training scripts
    └── utils/     # vLLM generation & accuracy scripts
```

## License

The `peft/` directory retains the original PEFT license (Apache 2.0).
