import os
import logging
from queue import Queue, Empty
import multiprocess
from logging import Logger, Handler
import numpy as np
from tqdm import tqdm
import torch
from swift.llm import LLMDataset
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset
from typing import Tuple, Callable, List, Dict, Optional, Any, Iterator, Union
import torch.distributed as dist
from swift.utils import is_master
from torch.autograd.function import Function, FunctionCtx
from torch import Tensor

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
_logger_mapping = {}


def get_logger(verbose_format: bool = False) -> Logger:
    if verbose_format in _logger_mapping:
        return _logger_mapping[verbose_format]

    if is_master():
        level = logging.INFO
    else:
        level = logging.ERROR
    name = 'financial-ics'

    logger: Logger = logging.getLogger(name)
    logger.setLevel(level)
    handler: Handler = logging.StreamHandler()
    if verbose_format:
        _format = f'[%(levelname)s:{logger.name}] %(message)s [%(filename)s:%(lineno)d - %(asctime)s]'
    else:
        _format = f'[%(levelname)s:{logger.name}] %(message)s'
    handler.setFormatter(logging.Formatter(_format))
    handler.setLevel(level)
    logger.addHandler(handler)
    _logger_mapping[verbose_format] = logger
    return logger

logger = get_logger()

class GatherLayer(Function):
    """ref: https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py"""

    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tuple[Tensor]:
        res = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(res, x)
        return tuple(res)

    @staticmethod
    def backward(ctx: FunctionCtx, *grads: Tensor) -> Tensor:
        res = grads[dist.get_rank()]
        res *= dist.get_world_size()  # for same grad with 2 * batch_size; mean operation in ddp across device.
        return res

def _map_mp_single(subset: HfDataset, map_func, queue: Queue,
                   start_idx: int):
    for i, d in enumerate(subset, start=start_idx):
        queue.put(map_func(d))  # idx, result


def _map_mp_i(dataset: HfDataset, map_func,
              num_proc: int) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with multiprocess.Pool(
            num_proc) as pool, multiprocess.Manager() as manager:
        queue = manager.Queue()
        async_results = []
        split_idx = np.linspace(0, len(dataset), num_proc + 1, dtype=np.int32)
        for i in range(num_proc):
            subset = dataset.select(range(split_idx[i], split_idx[i + 1]))
            async_results.append(
                pool.apply_async(
                    _map_mp_single,
                    args=(subset, map_func, queue, split_idx[i])))
        while True:
            try:
                yield queue.get(timeout=0.05)
            except Empty:
                if all(async_result.ready()
                       for async_result in async_results) and queue.empty():
                    break


def _map_mp(dataset: HfDataset, map_func,
            num_proc: int) -> List[Dict[str, Any]]:
    # Solving the unordered problem
    data = []
    num_proc = min(num_proc, len(dataset))
    for d in tqdm(_map_mp_i(dataset, map_func, num_proc), total=len(dataset)):
        data += d
    return data


class LLMDataset(Dataset):

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.data = data

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
        if isinstance(idx, int):
            data = self.data[idx]
            return data
        elif isinstance(idx, str):
            return [d[0][idx] for d in self.data]
        else:
            raise ValueError(f'idx: {idx}')

    def select(self, idx_list: List[int]) -> 'LLMDataset':
        data = [self.data[i] for i in idx_list]
        return self.__class__(data)

    def __len__(self) -> int:
        return len(self.data)

def long_doc_dataset_map(
    dataset: HfDataset, 
    preprocess_func: Callable[[Dict[str, Any]], 
                               List[Dict[str, Optional[List[int]]]]],
    num_proc: int = 1,
) -> HfDataset:
    # long_doc to short_doc
    logger.info(f'num_proc: {num_proc}')
    if num_proc == 1:
        res = []
        for d in tqdm(dataset):
            preprocessed_d = preprocess_func(d)
            res += preprocessed_d
    else:
        assert num_proc > 1
        res =  _map_mp(dataset, preprocess_func, num_proc)
    return LLMDataset(res)
