# Experimental environment: A100
# 18GB GPU memory

CUDA_VISIBLE_DEVICES=0 \
fics eval \
    --model_type lbert-no-head \
    --ckpt_dir output/lbert-xxx/vx-xxx/checkpoint-xxx \
    --dtype fp16 \
    --dataset tenk-eval \
    --eval_dataset_sample -1 \
    --max_length 131072 \
    --eval_batch_size 1 \
    --lbert_window_size 512 \
    --lbert_num_global_token 16 \
    --pooling mean \
