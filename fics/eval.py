
from .utils import (
    EvalArguments, get_logger, get_task, long_doc_dataset_map
)
from swift.llm import get_model_tokenizer, get_dataset, stat_dataset
from swift.trainers import TrainingArguments
import torch
import numpy as np
from pprint import pprint
from swift.utils import (
    is_dist, get_dist_setting, get_main, seed_everything, get_main
)


logger = get_logger()

def eval(args: EvalArguments) -> None:
    logger.info(f'args: {args}')
    print(f'device_count: {torch.cuda.device_count()}')
    seed_everything(args.seed)

    # Loading Model and Tokenizer
    kwargs = {}
    model_kwargs = {'low_cpu_mem_usage': True}
    _, local_rank, _, _ = get_dist_setting()
    if is_dist():
        model_kwargs['device_map'] = {'': local_rank}
    else:
        model_kwargs['device_map'] = 'auto'
    if args.ckpt_dir is not None:
        kwargs['model_dir'] = args.ckpt_dir
    kwargs['args'] = args
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, model_kwargs=model_kwargs, **kwargs)
    task = get_task(args.task_type, tokenizer, args.max_length)

    # Loading Dataset
    random_state = np.random.RandomState(args.dataset_seed)
    eval_dataset, _ = get_dataset(args.dataset, 0, random_state)
    if args.eval_dataset_sample >= 0:
        args.eval_dataset_sample = min(args.eval_dataset_sample,
                                        len(eval_dataset))
        eval_dataset = eval_dataset.select(range(args.eval_dataset_sample))
    logger.info(f'eval_dataset: {eval_dataset}')
    eval_dataset = long_doc_dataset_map(eval_dataset, task.preprocess, 
                                        args.preprocess_num_proc)
    # Data analysis
    stat_dataset(eval_dataset)

    # Setting eval_args
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        per_device_eval_batch_size=args.eval_batch_size)
    logger.info(f'eval_args: {eval_args}')

    trainer = task.trainer_class(
        model=model,
        args=eval_args,
        data_collator=task.get_collate_fn(tokenizer),
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=task.compute_metrics,
        preprocess_logits_for_metrics=task.preprocess_logits_for_metrics
    )
    trainer.x_args = args
    trainer.evaluate()
    ics_metrics = trainer._custom_metrics['ics_metrics']
    if args.need_compute:
        metrics = ics_metrics.compute()
        pprint(metrics)
        return metrics
    else:
        return ics_metrics

eval_main = get_main(EvalArguments, eval)
