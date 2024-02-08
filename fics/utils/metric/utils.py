from torch import Tensor
import torch
from ._rank_loop_fast import calc_rank_loop as _calc_rank_loop
from torchmetrics.functional import pearson_corrcoef

def _calc_rank_fast(x: Tensor) -> Tensor:
    """faster"""
    N = x.shape[0]
    x, idx = x.sort()
    rank = torch.empty_like(x)
    rank[idx] = torch.arange(1, N+1, dtype=x.dtype, device=x.device)
    x, rank, idx = x.cpu(), rank.cpu(), idx.cpu()
    _calc_rank_loop(x.numpy(), rank.numpy(), idx.numpy())
    return rank

def spearman_corrcoef_fast(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    y_pred: [N] or [N, F]
    y_true: [N] or [N, F]
    return: [] or [F]
    """
    if y_pred.ndim == 1:
        y_pred = _calc_rank_fast(y_pred)
        y_true = _calc_rank_fast(y_true)
    else:
        y_pred = torch.stack([_calc_rank_fast(yp) for yp in y_pred.unbind(dim=1)], dim=1)
        y_true = torch.stack([_calc_rank_fast(yt) for yt in y_true.unbind(dim=1)], dim=1)
    return pearson_corrcoef(y_pred, y_true)
