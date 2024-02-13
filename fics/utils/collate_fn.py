from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from typing import Optional, List, Callable, Dict, Any
import numpy as np
from functools import partial
from torch import Tensor
import torch
import torch.nn.functional as F
import math

def get_long_doc_preprocess(
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        for_evaluate: bool = False
) -> Callable[[Dict[str, Any]], List[Dict[str, List[int]]]]:
    assert max_length is not None
    def preprocess(example: Dict[str, Any]) -> List[Dict[str, List[int]]]:
        num_bos_token = 1
        if hasattr(tokenizer, 'num_global_token'):
            num_bos_token = 0
        text = example['text']
        token_list = tokenizer(text, return_attention_mask=False, 
                               add_special_tokens=False)['input_ids']
        res = []
        lo = 0
        while True:
            hi = lo + max_length - num_bos_token
            tokens = []
            tokens += [tokenizer.bos_token_id] * num_bos_token
            tokens += token_list[lo:hi]
            if len(tokens) <= num_bos_token:
                break
            r = {'input_ids': tokens}
            if for_evaluate:
                r.update(example)
                r['token_length'] = len(tokens)
                r.pop('text')
            res.append(r)
            lo = hi
        return res
    return preprocess


def get_maskedlm_collate_fn(
        tokenizer: PreTrainedTokenizerBase,
        mlm: bool = True,
)-> Callable[[List[Dict[str, Any]]], Dict[str, Tensor]]:
    data_collator_mlm = DataCollatorForLanguageModeling(tokenizer, mlm=mlm, 
                                                        mlm_probability=0.15)
    def collate_fn(example_list: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        res = data_collator_mlm(example_list)
        maybe_padding_to(tokenizer, res)
        return res

    return collate_fn

get_padding_collate_fn = partial(get_maskedlm_collate_fn, mlm=False)


def get_contrastive_learning_collate_fn(tokenizer: PreTrainedTokenizerBase) -> Callable[[List[Dict[str, Any]]], Dict[str, Tensor]]:
    padding_data_collator = get_padding_collate_fn(tokenizer)
    def collate_fn(example_list: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        res = padding_data_collator(example_list)
        for k, v in res.items():
            res[k] = v.repeat_interleave(2, dim=0)
        return res
    return collate_fn

def maybe_padding_to(tokenizer, inputs) -> None:
    if hasattr(tokenizer, 'padding_to'):
        padding_to = tokenizer.padding_to
        length = inputs['input_ids'].shape[-1]
        if hasattr(tokenizer, 'num_bos_token'):
            length = length - tokenizer.num_bos_token
        padding_to = math.ceil(length / padding_to) * padding_to
        padding_length = padding_to - length
        inputs['input_ids'] = F.pad(inputs['input_ids'], [0, padding_length],
                                    value=tokenizer.pad_token_id)
        inputs['attention_mask'] = F.pad(inputs['attention_mask'], [0, padding_length],
                                         value=0)
        inputs['labels']  = F.pad(inputs['labels'], [0, padding_length],
                                  value=-100)
    if tokenizer.__class__.__name__.lower().startswith('longformer'):
        input_ids= inputs['input_ids']
        inputs['global_attention_mask'] = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        inputs['global_attention_mask'][:, 0] = 1
