from fics import IcsMetrics, get_dataset, CustomDatasetName
import torch

if __name__ == '__main__':
    ics_metric = IcsMetrics()
    dataset = get_dataset(CustomDatasetName.tenk_eval_mini)[0]
    cik = torch.tensor(dataset['cik'])
    ebd = torch.randn(cik.shape[0], 768)
    token_length = torch.ones_like(cik)
    date = torch.tensor(dataset['date'])
    sic = torch.tensor(dataset['sic'])
    naics = torch.tensor(dataset['naics'])
    stock = torch.tensor(dataset['stock'])
    ics_metric.update(cik, ebd, token_length, date, sic, naics, stock)
    res = ics_metric.compute()
    print(f'ics_metric: {res}')
