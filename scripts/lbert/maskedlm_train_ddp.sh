# Experimental environment: 4 * A100
# 4 * 60GB GPU memory
nproc_per_node=4
batch_size=1
total_batch_size=16

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
fics train \
    --model_type lbert \
    --train_dataset_sample -1 \
    --task_type maskedlm \
    --dtype fp16 \
    --ddp_backend nccl \
    --dataset tenk-pretrained \
    --preprocess_num_proc 4 \
    --max_length 131072 \
    --output_dir output \
    --gradient_checkpointing true \
    --num_train_epochs 16 \
    --batch_size $batch_size \
    --weight_decay 0.1 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps $(expr $total_batch_size / $nproc_per_node / $batch_size) \
    --max_grad_norm 1 \
    --warmup_ratio 0.03 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --save_only_model true \
    --lr_scheduler_type linear \
    --lbert_window_size 512 \
    --lbert_num_global_token 16 \
