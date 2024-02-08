from fics.utils import spearman_corrcoef_fast
import torch


if __name__ == '__main__':
    y_pred = torch.randn(100, 2)
    y_true = torch.randn(100, 2)
    res = spearman_corrcoef_fast(y_pred, y_true)
    print(f'res.shape: {res.shape}')
