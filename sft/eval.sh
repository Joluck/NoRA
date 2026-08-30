# HF_ENDPOINT="https://hf-mirror.com" lm_eval --model hf \
#     --model_args /home/rwkv/jl/outmodel/peft/gaussian-k \
#     --tasks gsm8k \
#     --device cuda:2 \
#     --cache_requests true \


lm_eval --model vllm \
    --model_args pretrained="/home/rwkv/jl/outmodel/peft/math/nora",tensor_parallel_size=4,dtype=bfloat16,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks mmlu,agieval,arc_challenge \
    --batch_size auto