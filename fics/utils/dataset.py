from datasets import Dataset as HfDataset
from numpy.random import RandomState
import pandas as pd
from pandas import DataFrame
from swift.llm import register_dataset
from swift.utils import read_from_jsonl, transform_jsonl_to_df
from typing import List
from tqdm import tqdm
from ..dataset_utils import train_dataset_path, eval_dataset_path, demo_dataset_path, file_folder
import os


class CustomDatasetName:
    tenk_pretrained = 'tenk-pretrained'
    tenk_eval = 'tenk-eval'
    tenk_demo = 'tenk-demo'
    tenk_pretrained_mini = 'tenk-pretrained-mini'  # for test
    tenk_eval_mini = 'tenk-eval-mini'  # for test


def random_select_df(df: DataFrame, dataset_sample: int) -> DataFrame:
    if dataset_sample < 0:
        return df
    random_state = RandomState(42)
    idxs = random_state.permutation(dataset_sample)
    df = df.iloc[idxs]
    return df

def read_raw_text(df: DataFrame) -> List[str]:
    raw_text_path = file_folder
    text = []
    for i in tqdm(range(df.shape[0])):
        line = df.iloc[i]
        item1, item1a = line['item1'], line['item1a']
        assert isinstance(item1, str)
        item1 = os.path.join(raw_text_path, item1)
        with open(item1, 'r') as f:
            t = f.read()
        t = t.strip(' ')
        if not t.endswith('\n'):
            t += '\n\n'
        if isinstance(item1a, str) and item1a != '':
            item1a = os.path.join(raw_text_path, item1a)
            with open(item1a, 'r') as f:
                t += f.read()
        text.append(t)
    return text

@register_dataset(CustomDatasetName.tenk_pretrained_mini, 
                  train_dataset_path,
                  function_kwargs={'dataset_sample': 100})
@register_dataset(CustomDatasetName.tenk_pretrained, 
                  train_dataset_path)
def get_tenk_pretrained_dataset(
    dataset_id_or_path: str, 
    **kwargs
) -> HfDataset:
    dataset_sample = kwargs.get('dataset_sample', -1)
    df = pd.read_csv(dataset_id_or_path)
    df = random_select_df(df, dataset_sample)  # for mini
    text_list = read_raw_text(df)
    return HfDataset.from_dict({'text': text_list})

@register_dataset(CustomDatasetName.tenk_eval_mini, eval_dataset_path,
                  function_kwargs={'dataset_sample': 100})
@register_dataset(CustomDatasetName.tenk_eval, eval_dataset_path)
def get_tenk_eval_dataset(dataset_id_or_path: str, 
                          **kwargs) -> HfDataset:
    dataset_sample = kwargs.get('dataset_sample', -1)
    obj_list = read_from_jsonl(dataset_id_or_path)
    df = transform_jsonl_to_df(obj_list)
    df = random_select_df(df, dataset_sample)  # for mini
    text_list = read_raw_text(df)
    del df['item1'], df['item1a']
    return HfDataset.from_dict({
        'cik': df['cik'].tolist(),
        'date': df['date'].tolist(),
        'sic': df['sic'].tolist(),
        'naics': df['naics'].tolist(),
        'stock': df['stock'].tolist(),
        'text': text_list, 
    })

@register_dataset(CustomDatasetName.tenk_demo,
                  demo_dataset_path)
def get_tenk_demo_dataset(dataset_id_or_path: str, 
                          **kwargs) -> HfDataset:
    df = pd.read_csv(dataset_id_or_path, keep_default_na=False)
    text_list = read_raw_text(df)
    del df['item1'], df['item1a']
    return HfDataset.from_dict({
        'cik': df['cik'].tolist(),
        'date': df['date'].tolist(),
        'sic': df['sic'].tolist(),
        'naics': df['naics'].tolist(),
        'text': text_list, 
    })
