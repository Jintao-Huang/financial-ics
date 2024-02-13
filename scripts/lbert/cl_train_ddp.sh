# Experimental environment: 4 * A100
# 4 *60GB GPU memory
nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
fics train \
    --model_type lbert-no-head \
    --train_dataset_sample 2000 \
    --task_type contrastive-learning \
    --model_cache_dir output/lbert/vx-xxx/checkpoint-xxx \
    --dtype fp16 \
    --ddp_backend nccl \
    --dataset tenk-pretrained \
    --preprocess_num_proc 1 \
    --dataset_test_ratio 0 \
    --max_length 131072 \
    --output_dir output \
    --gradient_checkpointing true \
    --num_train_epochs 1 \
    --batch_size 2 \
    --weight_decay 0.1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1 \
    --warmup_ratio 0. \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --save_only_model true \
    --pooling mean \
    --lr_scheduler_type constant \
    --lbert_window_size 512 \
    --lbert_num_global_token 16 \
    --num_prototype_with_grad 1 \
