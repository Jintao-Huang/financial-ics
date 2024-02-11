import os
from .utils import (
    TrainArguments, get_logger, long_doc_dataset_map, get_task
)
import torch
import json
from swift.llm import get_model_tokenizer, get_dataset

from swift.utils import (
    seed_everything, is_dist, is_ddp_plus_mp, get_dist_setting,
    show_layers, get_model_info, is_master, check_json_format, plot_images, 
)
from swift.llm import (
    sort_by_max_length, stat_dataset
)
from swift.trainers import TrainingArguments
from swift.utils import get_main
import numpy as np


logger = get_logger()

def train(args: TrainArguments) -> str:
    logger.info(f'args: {args}')
    print(f'device_count: {torch.cuda.device_count()}')
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    print(f'rank: {rank}, local_rank: {local_rank}, '
          f'world_size: {world_size}, local_world_size: {local_world_size}')
    seed_everything(args.seed)
    # Loading Model and Tokenizer
    kwargs = {}
    model_kwargs = {'low_cpu_mem_usage': True}
    if is_dist() and not is_ddp_plus_mp():
        model_kwargs['device_map'] = {'': local_rank}
    else:
        model_kwargs['device_map'] = 'auto'
    if args.model_cache_dir is not None:
        kwargs['model_dir'] = args.model_cache_dir
    kwargs['args'] = args
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, model_kwargs=model_kwargs, **kwargs)
    task = get_task(args.task_type, tokenizer, args.max_length)

    show_layers(model)
    model_info = get_model_info(model)
    logger.info(model_info)
    logger.info(model)

    # Loading Dataset
    random_state = np.random.RandomState(args.dataset_seed)
    train_dataset, val_dataset = get_dataset(args.dataset,
                                             args.dataset_test_ratio,
                                             random_state)
    if args.train_dataset_sample >= 0:
        args.train_dataset_sample = min(args.train_dataset_sample,
                                        len(train_dataset))
        val_dataset_sample = max(
            int(args.train_dataset_sample * args.dataset_test_ratio), 1)
        train_idxs = random_state.permutation(args.train_dataset_sample)
        train_dataset = train_dataset.select(train_idxs)
        if val_dataset.shape[0] > val_dataset_sample:
            val_idxs = random_state.permutation(val_dataset_sample)
            val_dataset = val_dataset.select(val_idxs)
    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    train_dataset = long_doc_dataset_map(train_dataset, task.preprocess, args.preprocess_num_proc)
    val_dataset = long_doc_dataset_map(val_dataset, task.preprocess, args.preprocess_num_proc)
    # Data analysis
    stat_dataset(train_dataset)
    stat_dataset(val_dataset)

    # Setting training_args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        optim=args.optim,
        adam_beta2=args.adam_beta2,
        resume_from_checkpoint=args.resume_from_checkpoint,
        ddp_backend=args.ddp_backend,
        gradient_checkpointing=args.gradient_checkpointing,
        local_rank=local_rank,
        remove_unused_columns=False,
        save_only_model=args.save_only_model,
        logging_first_step=True)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    if is_dist():
        # Compatible with https://github.com/huggingface/transformers/pull/25903
        training_args._frozen = False
        if args.gradient_checkpointing:
            training_args.ddp_find_unused_parameters = False
            training_args.ddp_broadcast_buffers = False
        else:
            training_args.ddp_find_unused_parameters = True
            training_args.ddp_broadcast_buffers = True

    logger.info(f'training_args: {training_args}')

    trainer = task.trainer_class(
        model=model,
        args=training_args,
        data_collator=task.get_collate_fn(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=task.compute_metrics,
        preprocess_logits_for_metrics=task.preprocess_logits_for_metrics
    )
    trainer.x_args = args
    if is_master():
        for args_obj, fname in zip([args, training_args],
                                   ['train_args.json', 'training_args.json']):
            fpath = os.path.join(args.output_dir, fname)
            with open(fpath, 'w') as f:
                json.dump(
                    check_json_format(args_obj.__dict__),
                    f,
                    ensure_ascii=False,
                    indent=2)
    trainer.train(training_args.resume_from_checkpoint)
    logger.info(
        f'best_model_checkpoint: {trainer.state.best_model_checkpoint}')

    # Visualization
    if is_master():
        images_dir = os.path.join(args.output_dir, 'images')
        logger.info(f'images_dir: {images_dir}')
        tb_dir = os.path.join(args.output_dir, 'runs')
        folder_name = os.listdir(tb_dir)[0]
        tb_dir = os.path.join(tb_dir, folder_name)
        plot_images(images_dir, tb_dir, ['train/loss'], 0.9)
    return trainer.state.best_model_checkpoint


train_main = get_main(TrainArguments, train)
