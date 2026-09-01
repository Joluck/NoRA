# NoRA: Normalized Low-Rank Adaptation
<a href="https://spherelab.ai/NoRA/">Project Page</a>

NoRA is a LoRA variant that normalizes the columns of `lora_A.weight` along the rank
(r) dimension, implemented as a fork of [PEFT](https://github.com/huggingface/peft) together
with an SFT training pipeline.

Constrained to the unit sphere, each column of `A` acts as a direction rather than an
arbitrary vector: magnitude is decoupled from `A` and carried entirely by `lora_B` (and the
scaling factor). This keeps every rank direction balanced throughout training and makes the
effective update scale easier to reason about.

Two modes are supported via `use_nora`:

- `use_nora=True` — normalize on **every forward pass** (reparameterization-style;
  gradients flow through the normalization). `merge_and_unload()` applies the same
  normalization, so the merged model matches the adapter model exactly.
- `use_nora="init"` — normalize **once**, right after adapter initialization. Afterwards `A`
  is an ordinary parameter; merging is the standard `B @ A`.

`use_nora` composes with `use_dora=True`: in the default (per-forward) mode the normalization
is applied inside DoRA's forward (including its weight-norm computation), and the DoRA
merge/unmerge path uses the same normalized delta via `get_delta_weight()`.

> **Tip:** with NoRA we recommend setting `lora_alpha = r` (i.e. scaling = 1). Since every
> column of `A` is unit-norm, the update magnitude is governed by `lora_B` alone, and
> `alpha = rank` keeps the effective step size comparable across ranks.

**BIMI** (Block Identity Matrix Initialization, `init_lora_weights="bimi"`) is a special
case of this family: `lora_A` is initialized as `r × r` identity blocks tiled along the
diagonal (`A[:, kr:(k+1)r] = I`), whose columns are already unit-norm and mutually
orthogonal — i.e. a maximally balanced point on the NoRA constraint manifold — while
`lora_B = 0` keeps `ΔW = 0` at step 0. No SVD over the base weights is needed (unlike
PiSSA/MiLoRA).

## Installation

> **Note:** we are working on merging NoRA into the upstream
> [PEFT](https://github.com/huggingface/peft) library. Once the PR lands, a plain
> `pip install peft` will be enough — this fork is only needed in the meantime.

```bash
git clone <this-repo>
cd NoRA/peft
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
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_nora=True,            # per-forward normalization; or "init" for init-only
)
model = get_peft_model(base_model, config)
```

#### if you want to use BIMI
  
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    init_lora_weights="bimi",   # optional: block identity init (already unit-norm)
)
model = get_peft_model(base_model, config)
```

### SFT training

The `sft/` directory contains a full supervised fine-tuning pipeline
(HuggingFace Trainer + DeepSpeed), adapted from
[PiSSA](https://github.com/MuLabPKU/PiSSA.git) and
[PEFT-Arena](https://github.com/Sphere-AI-Lab/PEFT-Arena.git):

```bash
cd sft
sh nora.sh
```

See `sft/scripts/` for complete runnable examples.

Key arguments in `sft/train.py`:

| Argument | Description |
|---|---|
| `--use_nora` | `True` = normalize `lora_A` every forward (default mode); `init` = normalize once after init |
| `--init_weights` | `True` = vanilla LoRA; `bimi` = BIMI; also `pissa`, `pissa_niter_N`, `olora`, `gaussian` |
| `--lora_rank` / `--lora_alpha` | LoRA rank / alpha |
| `--target_modules` | Comma-separated modules to adapt |

DeepSpeed configs are provided in `sft/configs/` (ZeRO-2 and ZeRO-3).

### RL training

For RL training, we directly build on [PeRL](https://github.com/MikaStars39/PeRL.git) —
follow that repo for the RL pipeline; NoRA is enabled via the same `use_nora` flag in the
PEFT config.

## Repository structure

```
NoRA/
├── peft/     # PEFT fork: use_nora + init_lora_weights="bimi"
│             # (core logic: peft/src/peft/tuners/lora/layer.py)
└── sft/      # SFT training + evaluation pipeline
    ├── train.py
    ├── eval.sh    # lm_eval entry (mmlu / agieval / arc_challenge)
    ├── configs/   # DeepSpeed configs
    ├── scripts/   # example training scripts
    └── utils/     # vLLM generation & accuracy scripts
```

## 📚 Citation

```bibtex
  @article{kang2026nora,
      title={Normalized Low-Rank Adaptation}, 
      author={Jiale Kang and Ziyin Yue and Zheng Zhan and Yangyi Huang and Weiyang Liu},
      journal={arXiv preprint arXiv:2608.31036},
      year={2026}}
```
