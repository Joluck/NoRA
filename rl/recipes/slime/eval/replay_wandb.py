#!/usr/bin/env python3
"""
从 SLIME 的 output.log 解析训练/rollout/perf/eval 指标，replay 到 W&B。

用法:
    python replay_wandb.py \
        --log /jpfs-5p/.../output.log \
        --project slime-rl-optim \
        --name "Qwen3-8B-cispo-rl-20260408_142430"
"""

import argparse
import ast
import os
import re
from collections import defaultdict

os.environ.setdefault("WANDB_BASE_URL", "http://11.71.1.218:8082")
os.environ.setdefault("WANDB_API_KEY", "local-b0d90ad40bfaa2dd58fa4525f18c82ccb8aca2c6")

import wandb

# ── 日志行正则 ──────────────────────────────────────────────
# model.py:  "{role_tag}step {N}: {dict}"   role_tag = "" | "critic-" | ...
STEP_RE = re.compile(r"model\.py:\d+ - (?:\w+-)?step (\d+): (\{.+\})")
# rollout.py: "perf {N}: {dict}"  (rollout metrics + perf 合并在一行)
PERF_ROLLOUT_RE = re.compile(r"rollout\.py:\d+ - perf (\d+): (\{.+\})")
# train_metric_utils.py: "perf {N}: {dict}"
PERF_TRAIN_RE = re.compile(r"train_metric_utils\.py:\d+ - perf (\d+): (\{.+\})")
# rollout.py: "eval {N}: {dict}"
EVAL_RE = re.compile(r"rollout\.py:\d+ - eval (\d+): (\{.+\})")
# data.py: "rollout {N}: {dict}"  (legacy, 旧版 slime)
ROLLOUT_RE = re.compile(r"data\.py:\d+ - rollout (\d+): (\{.+\})")

ALL_PATTERNS = [
    (STEP_RE, "step"),
    (ROLLOUT_RE, "rollout"),
    (PERF_ROLLOUT_RE, "perf_rollout"),
    (PERF_TRAIN_RE, "perf_train"),
    (EVAL_RE, "eval"),
]


def parse_output_log(path: str):
    """解析 output.log，返回按全局顺序排列的 (line_no, type, idx, dict) 列表。"""
    events = []

    with open(path, "r", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            for pattern, etype in ALL_PATTERNS:
                m = pattern.search(line)
                if m:
                    idx = int(m.group(1))
                    try:
                        data = ast.literal_eval(m.group(2))
                        if isinstance(data, dict):
                            events.append((line_no, etype, idx, data))
                    except Exception:
                        pass
                    break  # 一行只匹配一种

    return events


def main():
    parser = argparse.ArgumentParser(description="Replay SLIME output.log to W&B")
    parser.add_argument("--log", required=True, help="Path to output.log")
    parser.add_argument("--project", default="slime-rl-optim")
    parser.add_argument("--entity", default="automl")
    parser.add_argument("--name", default=None)
    parser.add_argument("--host", default="http://11.71.1.218:8082")
    parser.add_argument("--api-key", default=os.environ.get("WANDB_API_KEY"))
    args = parser.parse_args()

    os.environ["WANDB_BASE_URL"] = args.host

    # 1. 解析
    print(f"Parsing {args.log} ...")
    events = parse_output_log(args.log)

    # 统计
    counts = {}
    for _, etype, _, _ in events:
        counts[etype] = counts.get(etype, 0) + 1
    print(f"Total events: {len(events)}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print()

    if not events:
        print("No events found, exiting")
        return

    # 去重：同类型同 idx 只保留第一条
    seen = set()
    deduped = []
    for e in events:
        key = (e[1], e[2])  # (type, idx)
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    events = deduped
    print(f"After dedup: {len(events)} events\n")

    # 2. 登录
    if not args.api_key:
        print("[ERROR] 请设置 --api-key 或 export WANDB_API_KEY=...")
        return
    wandb.login(key=args.api_key, host=args.host)
    run = wandb.init(project=args.project, entity=args.entity, name=args.name)
    print(f"Created new run: {run.url}\n")

    # 3. 按 rollout_id 聚合所有事件类型，一个 rollout_id 发一次 log
    #
    # step/perf_train 使用 rollout_id 作 idx (accumulated_step_id)
    # perf_rollout/rollout/eval 也使用 rollout_id 作 idx
    # 同一个 rollout_id 的所有 metric 合并到一个 dict，确保 W&B x 轴对齐。
    #
    # 注意: step 日志的 idx 是 accumulated_step_id (= rollout_id * n_inner_steps + step_id)
    #       与 rollout/eval 的 rollout_id 不同。我们分两组处理:
    #         - "rollout 轴": rollout, perf_rollout, eval  (用 rollout_id)
    #         - "step 轴": step, perf_train  (用 accumulated_step_id)

    rollout_axis = defaultdict(dict)  # rollout_id -> merged dict
    step_axis = defaultdict(dict)     # accumulated_step_id -> merged dict

    for _, etype, idx, data in events:
        if etype in ("step", "perf_train"):
            step_axis[idx].update(data)
        else:
            # rollout, perf_rollout, eval
            rollout_axis[idx].update(data)

    # 上传 rollout 轴
    uploaded = 0
    for rid in sorted(rollout_axis.keys()):
        d = rollout_axis[rid]
        run.log(d, step=rid)
        uploaded += 1
        if uploaded % 50 == 0:
            print(f"  uploaded {uploaded} rollout-axis entries (rollout_id={rid})")

    # 上传 step 轴: 用 define_metric 让 W&B 识别 train/step 为自定义 x 轴
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("perf/*", step_metric="train/step")

    for sid in sorted(step_axis.keys()):
        d = step_axis[sid]
        d["train/step"] = sid
        run.log(d)
        uploaded += 1
        if uploaded % 50 == 0:
            print(f"  uploaded {uploaded} total entries (train/step={sid})")

    run.finish()
    print(f"\nDone! {uploaded} total entries uploaded.")
    print(f"  rollout-axis: {len(rollout_axis)} entries")
    print(f"  step-axis:    {len(step_axis)} entries")
    print(f"Run URL: {run.url}")


if __name__ == "__main__":
    main()
