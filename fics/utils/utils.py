import os
import logging
from logging import Logger, Handler
import torch
from typing import Tuple
import torch.distributed as dist
from swift.utils import is_master
from torch.autograd.function import Function, FunctionCtx
from torch import Tensor

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def get_logger(verbose_format: bool = False) -> Logger:
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
    return logger


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
