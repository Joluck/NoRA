
BASE_MODEL=llama-3.2-3B
OUTPUT_PATH="./outmodel/nora"
DATA_PATH=fxmeng/pissa-dataset

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --include=localhost:0,1,2,3 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --use_nora True  \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --data_path $DATA_PATH \
    --sub_task metamath:100000 \
    --dataset_split "train"\
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.3 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --merge True \
    --report_to "none" \
    --run_name "nora"


python utils/gen_vllm.py --model $OUTPUT_PATH --data_path $DATA_PATH --sub_task metamath --output_file $OUTPUT_PATH/metamath_response.jsonl
python utils/test_acc.py --input_file $OUTPUT_PATH/metamath_response.jsonl
