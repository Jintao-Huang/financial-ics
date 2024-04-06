import os
from dataclasses import dataclass, field
import torch
from swift.llm.utils.argument import select_dtype
from transformers.utils.versions import require_version

from typing import Optional, List, Literal
from .utils import get_logger
from .model import CustomModelType
from .dataset import CustomDatasetName
from swift.llm import DATASET_MAPPING, MODEL_MAPPING
from .task import TASK_MAPPING, TaskType
from swift.utils import (
    is_dist, get_dist_setting, is_master, add_version_to_work_dir, broadcast_string
)
import torch.distributed as dist
import json

logger = get_logger()

@dataclass
class TrainArguments:
    model_type: str = field(
        metadata={'choices': list(MODEL_MAPPING.keys())})
    dataset: List[str] = field(
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    task_type: str = field(
        default=None,
        metadata={'choices': list(TASK_MAPPING.keys())})

    model_cache_dir: Optional[str] = None
    dtype: str = field(
        default='fp16', metadata={'choices': ['bf16', 'fp16', 'fp32']})
    ddp_backend: str = field(
        default='nccl', metadata={'choices': ['nccl', 'gloo', 'mpi', 'ccl']})
    seed: int = 42

    dataset_seed: int = 42
    dataset_test_ratio: float = 0.01
    train_dataset_sample: int = -1  # -1: all dataset
    max_length: Optional[int] = None

    output_dir: str = 'output'
    gradient_checkpointing: bool = True
    batch_size: int = 1
    eval_batch_size: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    num_train_epochs: int = 1
    # if max_steps >= 0, override num_train_epochs
    max_steps: int = -1
    optim: str = 'adamw_torch'
    adam_beta2: float = 0.999
    learning_rate: Optional[float] = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.
    lr_scheduler_type: str = 'linear'
    warmup_ratio: float = 0.05
    eval_steps: int = 50
    save_steps: Optional[int] = None
    save_only_model: Optional[bool] = True
    save_total_limit: int = 10  # save last and best. -1: all checkpoints
    logging_steps: int = 5
    dataloader_num_workers: int = 1
    preprocess_num_proc: int = 1
    deepspeed: Optional[str] = None

    # other
    ignore_args_error: bool = False  # True: notebook compatibility
    pooling: str = field(
        default='mean',
        metadata={'choices': ['cls', 'mean']})
    dropout_p: float = 0.1
    # lbert
    lbert_window_size: int = 512
    lbert_max_position_embeddings: Optional[int] = None
    lbert_num_global_token: int = 1
    # prototype loss
    temperature: float = 0.1
    num_prototype_with_grad: Optional[int] = None

    def __post_init__(self):
        self.output_dir = os.path.join(self.output_dir, self.model_type)
        if is_master():
            self.output_dir = add_version_to_work_dir(self.output_dir)
        ds_config_folder = os.path.join(__file__, '..', '..','ds_config')
        if self.deepspeed == 'default-zero2':
            self.deepspeed = os.path.abspath(
                os.path.join(ds_config_folder, 'zero2.json'))
        if self.deepspeed is not None:
            require_version('deepspeed')
            if self.deepspeed.endswith('.json') or os.path.isfile(
                    self.deepspeed):
                with open(self.deepspeed, 'r', encoding='utf-8') as f:
                    self.deepspeed = json.load(f)
            logger.info(f'Using deepspeed: {self.deepspeed}')
        self.torch_dtype, self.fp16, self.bf16 = select_dtype(self)
        if self.torch_dtype == torch.float16:
            self.torch_dtype = torch.float32
            logger.warning('Setting torch_dtype: torch.float32')
        if is_dist():
            rank, local_rank, _, _ = get_dist_setting()
            torch.cuda.set_device(local_rank)
            self.seed += rank  # Avoid the same dropout
            # Initialize in advance
            if not dist.is_initialized():
                dist.init_process_group(backend=self.ddp_backend)
            # Make sure to set the same output_dir when using DDP.
            self.output_dir = broadcast_string(self.output_dir)

        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        if self.num_prototype_with_grad is None:
            self.num_prototype_with_grad = self.batch_size
        if self.save_steps is None:
            self.save_steps = self.eval_steps
        if self.save_total_limit == -1:
            self.save_total_limit = None

        model_info = MODEL_MAPPING[self.model_type]
        if self.max_length is None:
            self.max_length = model_info['max_length']
        if self.task_type is None:
            if 'no-head' in self.model_type:
                self.task_type = 'contrastive-learning'
            else:
                self.task_type = 'maskedlm'


@dataclass
class EvalArguments:
    model_type: str = field(
        metadata={'choices': list(MODEL_MAPPING.keys())})
    dataset: List[str] = field(
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    ckpt_dir: Optional[str] = None
    task_type: str = field(
        default=TaskType.eval_tenk,
        metadata={'choices': list(TASK_MAPPING.keys())})

    dtype: str = field(
        default='fp32', metadata={'choices': ['bf16', 'fp16', 'fp32']})
    seed: int = 42
    dataset_seed: int = 42
    eval_dataset_sample: int = -1  # -1: all dataset
    max_length: Optional[int] = None
    eval_batch_size: Optional[int] = 1
    preprocess_num_proc: int = 1
    need_compute: bool = True

    # other
    ignore_args_error: bool = False  # True: notebook compatibility
    pooling: str = field(
        default='mean', metadata={'choices': ['cls', 'mean']})
    # lbert
    lbert_window_size: int = 512
    lbert_max_position_embeddings: Optional[int] = None
    lbert_num_global_token: int = 1

    def __post_init__(self):
        self.torch_dtype, _, _ = select_dtype(self)
        if isinstance(self.dataset, str):
            self.dataset = [self.dataset]
        assert 'no-head' in self.model_type and 'pretrained' not in self.dataset[0]
        if self.ckpt_dir is not None and not os.path.isdir(self.ckpt_dir):
            raise ValueError(f'Please enter a valid ckpt_dir: {self.ckpt_dir}')
        logger.info(f'ckpt_dir: {self.ckpt_dir}')
        if self.ckpt_dir is not None:
            self.output_dir = os.path.join(self.ckpt_dir, 'eval')
        else:
            self.output_dir =  'output_eval'
            self.output_dir = os.path.join(self.output_dir, self.model_type)
        logger.info(f'self.output_dir: {self.output_dir}')

        model_info = MODEL_MAPPING[self.model_type]
        if self.max_length is None:
            self.max_length = model_info['max_length']

