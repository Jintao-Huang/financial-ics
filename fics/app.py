
from .utils import EvalArguments, CustomDatasetName, get_dataset
from .eval import eval
from typing import Dict, List, Any
from swift.utils import get_main
from torch import Tensor
import gradio as gr
import os
import pandas as pd
import torch
import numpy as np
from pandas import DataFrame
from torchmetrics.functional import pairwise_cosine_similarity
from .dataset_utils import (
    get_sic2desc_mapping, get_cik2name_mapping
)
import pickle

def add_additional_keys(inputs: Dict[str, Tensor]) -> None:
    cik2name_mapping = get_cik2name_mapping()
    sic2desc_mapping = get_sic2desc_mapping()
    cik = inputs['cik']
    sic = inputs['sic']
    name = []
    sic_desc = []
    for i in range(len(cik)):
        name.append(cik2name_mapping[cik[i].item()])
        sic_desc.append(sic2desc_mapping[sic[i].item()])
    inputs['name'] = name
    inputs['sic_desc'] = sic_desc

def build_mapping(inputs: Dict[str, Any]) -> None:
    cik = inputs['cik']  # cik -> idx
    mapping: Dict[str, int] = {c.item(): i for i, c in enumerate(cik)}
    inputs['mapping'] = mapping

def build_df(inputs: Dict[str, Any], sorted_idx: Tensor, ebd_cos_sim: Tensor) -> DataFrame:
    return DataFrame.from_dict({
        'cik': [inputs['cik'][idx].item() for idx in sorted_idx],
        'name': [inputs['name'][idx] for idx in sorted_idx],
        'sic': [inputs['sic'][idx].item() for idx in sorted_idx],
        'sic_desc': [inputs['sic_desc'][idx] for idx in sorted_idx],
        'ebd_cos_sim': [f'{ebd_cos_sim[idx].item():.6f}' for idx in sorted_idx],
        'date': [inputs['date'][idx].item() for idx in sorted_idx]
    })


def build_search_peer_firm_demo(inputs: Dict[str, Any]) -> gr.Blocks:
    options = [f'{c} {n}' for c, n in zip(inputs['cik'], inputs['name'])]
    embedding_cos_sim = inputs['embedding_cos_sim']
    def search_peer_firm(cik_name: str, n_peer_firm: float) -> DataFrame:
        if cik_name == '':
            return None
        cik = int(cik_name.split(' ', 1)[0])
        mapping = inputs['mapping']
        idx = mapping[cik]
        ebd_cos_sim: Tensor = embedding_cos_sim[idx]
        sorted_idx = ebd_cos_sim.argsort(descending=True)[:n_peer_firm]
        return build_df(inputs, sorted_idx, ebd_cos_sim)
    
    default_n_peer_firm = min(embedding_cos_sim.shape[0], 50)
    with gr.Blocks() as demo:
        dropdown = gr.Dropdown(choices=options, label="Select a Firm")
        n_peer_firm = gr.Slider(0, embedding_cos_sim.shape[0], default_n_peer_firm, label='Number of Peer Firms')
        search = gr.Button('🚀 Search Peer Firm')
        dataframe = gr.Dataframe(headers=['cik', 'name', 'sic', 'sic_desc', 'ebd_cos_sim', 'date'],
                                 label='Peer Firm')
        search.click(search_peer_firm, inputs=[dropdown, n_peer_firm],
                     outputs=[dataframe])
    return demo


def build_view_tenk_report_demo(inputs: Dict[str, Any]) -> gr.Blocks:
    options = [f'{c} {n}' for c, n in zip(inputs['cik'], inputs['name'])]

    def view_tenk_report(cik_name: str) -> str:
        if cik_name is None or cik_name.strip() == '':
            return None
        mapping = inputs['mapping']
        cik = int(cik_name.split(' ', 1)[0])
        idx = mapping[cik]
        date = inputs['date'][idx].item()
        return date, inputs['text'][idx]

    with gr.Blocks() as demo:
        dropdown = gr.Dropdown(choices=options, label="Select a Firm")
        view = gr.Button('🚀 View 10K Report')
        date = gr.Textbox(lines=1, max_lines=1, label='Date', visible=False)
        text_box = gr.Textbox(lines=20, max_lines=50, label='10K Report', autoscroll=False)
        view.click(view_tenk_report, inputs=[dropdown], outputs=[date, text_box])
    return demo


def gradio_demo(inputs: Dict[str, Any]) -> None:
    with gr.Blocks() as demo:
        gr.Markdown(f'<center><font size=8>Financial ICS</center>')
        with gr.Tab('Search Peer Firm'):
            build_search_peer_firm_demo(inputs)
        with gr.Tab('View 10K Report'):
            build_view_tenk_report_demo(inputs)

    demo.queue().launch(height=1000)


def add_text(inputs: Dict[str, Any]) -> None:
    demo_dataset, _ = get_dataset([CustomDatasetName.tenk_demo], 0)
    mapping = inputs['mapping']
    res_text = [None] * len(mapping)
    for d in demo_dataset:
        if d['cik'] not in mapping:
            continue
        idx = mapping[d['cik']]
        res_text[idx] = d['text']
    assert None not in res_text
    inputs['text'] = res_text

def app_ui(args: EvalArguments) -> None:
    assert 'demo' in args.dataset[0]
    pkl_path = None
    if args.output_dir is not None:
        pkl_path = os.path.join(args.output_dir, 'demo', 'app.pkl')
    if pkl_path is None or not os.path.exists(pkl_path):
        args.need_compute = False
        ics_metrics = eval(args)
        if pkl_path is None:
            return
        inputs = ics_metrics._prepare_inputs()
        add_additional_keys(inputs)
        build_mapping(inputs)
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump(inputs, f)
    else:
        with open(pkl_path, 'rb') as f:
            inputs = pickle.load(f)

    embedding = inputs['embedding']
    embedding_cos_sim = pairwise_cosine_similarity(embedding, embedding)
    inputs['embedding_cos_sim'] = embedding_cos_sim
    add_text(inputs)
    gradio_demo(inputs)

app_ui_main = get_main(EvalArguments, app_ui)
