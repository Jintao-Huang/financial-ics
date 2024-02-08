import os
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import math
import json


file_folder = 'finance_10k/files'
train_dataset_path = 'finance_10k/preprocessed/train/train.csv'
eval_dataset_path = 'finance_10k/preprocessed/eval/eval.jsonl'
demo_dataset_path = 'finance_10k/preprocessed/demo/demo.csv'

def check_dataset() -> None:
    """check file exists"""
    text_dataset_path = 'finance_10k/readable.csv'
    df = pd.read_csv(text_dataset_path, na_filter=False)
    item1_list = df['item1']
    item1a_list = df['item1a']
    for itemx_list in [item1_list, item1a_list]:
        for itemx in tqdm(itemx_list):
            if len(itemx) == 0:
                continue
            file_path = os.path.join(file_folder, itemx)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(file_path)

def get_cik2sic_mapping() -> Dict[int, int]:
    cik2sic_path = 'finance_10k/data/cik2sic.csv'
    cik2sic_df = pd.read_csv(cik2sic_path)
    cik2sic_mapping = {}
    for i in range(cik2sic_df.shape[0]):
        line = cik2sic_df.iloc[i]
        cik, sic = line['cik'], line['sic']
        cik2sic_mapping[cik] = sic
    return cik2sic_mapping

def get_cik2name_mapping() -> Dict[int, str]:
    cik2name_path = 'finance_10k/data/cik2name.csv'
    cik2name_df = pd.read_csv(cik2name_path)
    cik2name_mapping = {}
    for i in range(cik2name_df.shape[0]):
        line = cik2name_df.iloc[i]
        cik, name = line['cik'], line['name']
        cik2name_mapping[cik] = name
    return cik2name_mapping

def _read_sic2naics():
    sic_naics_mapping_path = 'finance_10k/data/sic2naics.json'
    with open(sic_naics_mapping_path, 'r') as f:
        data = json.load(f)
    return data

def _get_sic2desc_mapping() -> Dict[int, str]:
    sic2desc_path = 'finance_10k/data/sic2desc.csv'
    sic2desc_df = pd.read_csv(sic2desc_path)
    sic2desc_mapping = {}
    for i in range(sic2desc_df.shape[0]):
        line = sic2desc_df.iloc[i]
        sic, desc = line['sic'], line['desc']
        sic2desc_mapping[sic] = desc
    return sic2desc_mapping

def get_sic2desc_mapping():
    data = _read_sic2naics()
    mapping = {}
    for d in data:
        sic = d.get('sic')
        if sic is None:
            continue
        sic = int(sic)
        sic_desc = d.get('sicDescription')
        if sic_desc is None:
            continue
        mapping[sic] = sic_desc
    mapping2 = _get_sic2desc_mapping()
    mapping.update(mapping2)
    return mapping


def get_sic2naics_mapping():
    data = _read_sic2naics()
    mapping = {}
    for d in data:
        sic = d.get('sic')
        if sic is None:
            continue
        sic = int(sic)
        naics = d.get('naics')
        if naics is None:
            continue
        naics = int(naics)
        mapping[sic] = naics
    return mapping

def get_naics2desc_mapping():
    data = _read_sic2naics()
    mapping = {}
    for d in data:
        naics = d.get('naics')
        if naics is None:
            continue
        naics = int(naics)
        naics_desc = d.get('naicsDescription')
        if naics_desc is None:
            continue
        mapping[naics] = naics_desc
    return mapping


def _get_ticker2cik_mapping():
    ticker2cik_path = 'finance_10k/data/ticker2cik.csv'
    ticker2cik_df = pd.read_csv(ticker2cik_path)
    ticker2cik_mapping = {}
    for i in range(ticker2cik_df.shape[0]):
        line = ticker2cik_df.iloc[i]
        ticker, cik = line['ticker'], line['cik']
        ticker2cik_mapping[ticker] = cik
    return ticker2cik_mapping

def get_cik_stock_mapping() -> Dict[int, List[float]]:
    stock_path = 'finance_10k/data/stock.csv'
    ticker2cik_mapping = _get_ticker2cik_mapping()
    stock_df = pd.read_csv(stock_path)
    cik_stock_mapping: Dict[int, List[float]] = {}
    RET = stock_df['RET'].to_numpy()  # faster than pandas
    TICKER = stock_df['TICKER'].to_numpy()
    DATE = stock_df['date'].to_numpy()
    for i in tqdm(range(stock_df.shape[0])):
        ret = RET[i]
        try:
            ret = float(ret)
            if math.isnan(ret):
                continue
        except ValueError:
            continue
        ticker = TICKER[i]
        date = DATE[i]
        if isinstance(ticker, float):
            continue
        ticker = ticker.lower()
        if ticker not in ticker2cik_mapping:
            continue
        cik = ticker2cik_mapping[ticker]
        year, month = date // 10000, date % 10000 // 100
        if year != 2018:
            continue
        if cik not in cik_stock_mapping:
            cik_stock_mapping[cik] = [None] * 12
        stock_list: List[float] = cik_stock_mapping[cik]
        stock_list[month - 1] = ret
    new_cik_stock_mapping = {}
    for cik, stock_list in cik_stock_mapping.items():
        if None in stock_list:
            continue
        new_cik_stock_mapping[cik] = stock_list
    return new_cik_stock_mapping
