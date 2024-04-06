from fics.utils import spearman_corrcoef_fast, pairwise_corrcoef
import torch


if __name__ == '__main__':
    y_pred = torch.randn(100, 2)
    y_true = torch.randn(100, 2)
    res = spearman_corrcoef_fast(y_pred, y_true)
    print(f'spearman_corrcoef: {res}')

    x1 = torch.tensor([[0.1, 0.2, 0.1, 0.2], [0.2, -0.2, 0.2, -0.2], [0.1, -0.2, 0.1, -0.2]])
    x2 = torch.tensor([[0.1, -0.1, 0.1, -0.1], [-0.2, 0.2, -0.2, 0.2]])
    res = pairwise_corrcoef(x1, x2)
    print(f'pairwise_corrcoef: {res}')
