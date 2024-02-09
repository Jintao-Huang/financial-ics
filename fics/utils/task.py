from transformers import PreTrainedTokenizerBase
from typing import Callable, Dict, Any, List, Optional, Type
import numpy as np
from functools import partial
from .trainer import (
    MaskedLMTrainer, EvalTenkTrainer, ContrastiveLearningTrainer
)
from .metric import compute_maskedlm_metrics
from .collate_fn import (
    get_padding_collate_fn, get_long_doc_preprocess, get_maskedlm_collate_fn,
    get_contrastive_learning_collate_fn
)
from swift.utils import preprocess_logits_for_metrics
from swift.trainers import Trainer
from torch import Tensor
from transformers.trainer_utils import EvalPrediction


GetPreprocess = Callable[[PreTrainedTokenizerBase, int, int], Callable[[Dict[str, Any]], List[Dict[str, List[int]]]]]
GetCollateFn = Callable[[PreTrainedTokenizerBase], Callable[[List[Dict[str, Any]]], Dict[str, Tensor]]]
ComputeMetrics = Callable[[EvalPrediction], Dict[str, Tensor]]
PreprocessLogitsForMetrics = Callable[[Tensor, Tensor], Tensor]


class TaskType:
    maskedlm = 'maskedlm'
    contrastive_learning = 'contrastive-learning'
    eval_tenk = 'eval-tenk'


class Task:
    def __init__(self, get_preprocess: GetPreprocess,
                get_collate_fn: GetCollateFn,
                compute_metrics: ComputeMetrics,
                trainer_class: Type[Trainer] = Trainer,
                preprocess_logits_for_metrics: Optional[PreprocessLogitsForMetrics] = None) -> None:
        self.get_preprocess = get_preprocess
        self.get_collate_fn = get_collate_fn
        self.trainer_class = trainer_class

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

    def init_task(self, tokenizer: PreTrainedTokenizerBase, min_length: Optional[int]= None, 
                  max_length: Optional[int]=None) -> None:
        self.tokenizer = tokenizer
        self.min_length = min_length
        self.max_length = max_length

        self.preprocess = self.get_preprocess(tokenizer, self.min_length, self.max_length)
        self.collate_fn = self.get_collate_fn(tokenizer)

TASK_MAPPING: Dict[str, Task] = {}

def register_task(task_type: str, task: Task):
    TASK_MAPPING[task_type] = task

register_task(TaskType.maskedlm, Task(
    get_long_doc_preprocess,
    get_maskedlm_collate_fn,
    compute_maskedlm_metrics, MaskedLMTrainer,
    preprocess_logits_for_metrics
))

register_task(TaskType.contrastive_learning, Task(
    get_long_doc_preprocess,
    get_contrastive_learning_collate_fn,
    None, ContrastiveLearningTrainer,
    None
))

register_task(TaskType.eval_tenk, Task(
    partial(get_long_doc_preprocess, for_evaluate=True),
    get_padding_collate_fn,
    None, EvalTenkTrainer,
    None
))


def get_task(task_type: str, 
             tokenizer: PreTrainedTokenizerBase, 
             min_length: Optional[int]= None, 
             max_length: Optional[int]=None) -> Task:
    task = TASK_MAPPING[task_type]
    task.init_task(tokenizer, min_length, max_length)
    return task
