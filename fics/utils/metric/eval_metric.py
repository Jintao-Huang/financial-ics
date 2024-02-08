import numpy as np
import torch
from typing import Dict, Optional, Tuple, Literal
from torch import Tensor
from torchmetrics.functional.regression import spearman_corrcoef
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

import pickle
import os
import json
from typing import Counter
from numpy import ndarray
from torch import device as Device
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from swift.utils import stat_array
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from fics.dataset_utils import get_cik2name_mapping, get_sic2desc_mapping
from ..utils import get_logger
from .utils import pairwise_corrcoef_amend, spearman_corrcoef_fast as spearman_corrcoef

logger = get_logger()

Index = Tuple[int, int]  # cik, date
EbdInfo = Tuple[Tensor, int]

class EmbeddingPool:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._ebd_mapping_cache: Dict[Index, EbdInfo] = {}
        self.ebd_mapping: Dict[Index, Tensor]  = {}

    def update_single(self, cik: int, ebd: Tensor, 
                      token_length: int, date: int) -> None:
        ebd = ebd.cpu().to(torch.float64).mul_(token_length)
        k = (cik, date)
        if k not in self._ebd_mapping_cache:
            self._ebd_mapping_cache[k] = (ebd, token_length)
        else:
            ebd_info: EbdInfo = self._ebd_mapping_cache[k]
            ebd_info[0].add_(ebd)
            self._ebd_mapping_cache[k] = (ebd_info[0], ebd_info[1] + token_length)
    
    def _compute_doc_ebd(self) -> None:
        for k, v in self._ebd_mapping_cache.items():
            self.ebd_mapping[k] = v[0] / v[1]

    def update(self, cik_tensor: Tensor, ebd_tensor: Tensor, 
               token_length_tensor: Tensor, date_tensor: Tensor) -> None:
        for cik, ebd, token_length, date in zip(cik_tensor, ebd_tensor, token_length_tensor, 
                                                date_tensor):
            self.update_single(cik.item(), ebd, token_length.item(), date.item())

    def compute(self) -> Dict[Index, Tensor]:
        self._compute_doc_ebd()
        return self.ebd_mapping


class IcsMetrics:
    def __init__(self, output_dir: Optional[str] = None, compute_device: Device = Device('cpu')):
        self.ebd_pool = EmbeddingPool()
        self.output_dir = output_dir
        self.compute_device = compute_device
        self.reset()

        self.peer_firm_threshold = 50
        if self.output_dir is None:
            self.save_to_jsonl = False
            self.save_to_pickle = False
        else:
            self.save_to_jsonl = True
            self.save_to_pickle = True
            self.pkl_fname = 'metrics_inputs.pkl'
            self.jsonl_fname = 'peer_firm.jsonl'
            os.makedirs(self.output_dir, exist_ok=True)
            self.pkl_path = os.path.join(self.output_dir, self.pkl_fname)
            self.jsonl_path = os.path.join(self.output_dir, self.jsonl_fname)
        self.cik2name_mapping = get_cik2name_mapping()
        self.sic2desc_mapping = get_sic2desc_mapping()

        self.logistic_regression_min_labels = 5
        self.logistic_regression_k_fold = 5
        self.logistic_regression = LogisticRegression(C=1, n_jobs=min(os.cpu_count(), 128),
                                                      max_iter=100000)

    def update(self, cik_tensor: Tensor, ebd_tensor: Tensor, 
               token_length_tensor: Tensor, date_tensor: Tensor,
               sic_tensor: Tensor, naics_tensor: Tensor,
               stock_tensor: Tensor) -> None:
        self.ebd_pool.update(cik_tensor, ebd_tensor, token_length_tensor, date_tensor)
        stock_tensor = stock_tensor.cpu()
        for cik, date, sic, naics, stock in zip(
                cik_tensor, date_tensor, sic_tensor, naics_tensor, stock_tensor):
            self._info_mapping[(cik.item(), date.item())] = (sic.item(), naics.item(), stock)

    def reset(self):
        self.ebd_pool.reset()
        self._info_mapping: Dict[Index, Tuple[int, int, Tensor]] = {}

    def _prepare_inputs(self) -> Dict[str, Tensor]:
        ebd_mapping: Dict[Index, Tensor] = self.ebd_pool.compute()
        res = {'cik': [], 'date': [], 'sic': [], 'naics': [], 'embedding': [], 'stock': []}
        for k, v in ebd_mapping.items():
            res['cik'].append(k[0])
            res['date'].append(k[1])
            res['embedding'].append(v)
            info = self._info_mapping[k]
            res['sic'].append(info[0])
            res['naics'].append(info[1])
            res['stock'].append(info[2])
        for k, v in res.items():
            if isinstance(v[0], Tensor):
                res[k] = torch.stack(v)
            else:
                res[k] = torch.tensor(v)
        return res

    def _remove_diag(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        device = x.device
        return x[torch.eye(N, dtype=torch.bool, device=device).logical_not_()].reshape(N, N-1)

    def _save_peer_firm_to_jsonl(self, inputs: Dict[str, Tensor]) -> None:
        cik, date, sic = inputs['cik'], inputs['date'], inputs['sic']
        embedding_cos_sim = inputs['embedding_cos_sim']
        stock_corrcoef = inputs['stock_corrcoef']
        embedding_cos_sim_sorted: Tensor
        embedding_cos_sim_sorted, idx = embedding_cos_sim.sort(dim=1, descending=True)
        cik_sorted, date_sorted, sic_sorted = cik[idx], date[idx], sic[idx]
        N = cik.shape[0]
        stock_corrcoef_sorted = stock_corrcoef[torch.arange(N)[:, None], idx]

        # create file
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            pass
        for cik, date, sic, embedding_cos_sim, stock_corrcoef in zip(
            cik_sorted, date_sorted, sic_sorted, embedding_cos_sim_sorted, 
            stock_corrcoef_sorted
        ):
            keys = ['cik', 'name', 'date', 'sic', 'desc', 
                    'embedding_cos_sim', 'stock_corrcoef']
            res = {k: [] for k in keys}
            cik: Tensor = cik[masks]
            date: Tensor = date[masks]
            sic: Tensor = sic[masks]
            embedding_cos_sim: Tensor = embedding_cos_sim[masks]
            stock_corrcoef: Tensor = stock_corrcoef[masks]

            for i in range(cik.shape[0]):
                cik_item = cik[i].item()
                sic_item = sic[i].item()
                name = self.cik2name_mapping.get(cik_item)
                desc = self.sic2desc_mapping.get(sic_item)
                for k, v in zip(keys, [cik_item, name, date[i].item(), sic_item, desc, 
                                       round(embedding_cos_sim[i].item(), 6), 
                                       round(stock_corrcoef[i].item(), 6)]):
                    res[k].append(v)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")

    def _compute_inputs_matrix(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        embedding = inputs['embedding']
        stock = inputs['stock']
        embedding_cos_sim = pairwise_cosine_similarity(embedding, embedding)
        stock_corrcoef = pairwise_corrcoef_amend(stock, stock)
        inputs['embedding_cos_sim'] = embedding_cos_sim
        inputs['stock_corrcoef'] = stock_corrcoef


    def _compute_ebd_stock_spearman(self, inputs: Dict[str, Tensor]) -> Dict[str, float]:
        N = inputs['cik'].shape[0]
        embedding_cos_sim = self._remove_diag(inputs['embedding_cos_sim'])
        stock_corrcoef = self._remove_diag(inputs['stock_corrcoef'])
        sorted_idx = embedding_cos_sim.argsort(dim=-1, descending=True)[:, :self.peer_firm_threshold]
        embedding_cos_sim = embedding_cos_sim[torch.arange(sorted_idx.shape[0])[:, None], sorted_idx]
        stock_corrcoef = stock_corrcoef[torch.arange(sorted_idx.shape[0])[:, None], sorted_idx]
        res = {}
        res['ebd_stock_spearman'] = spearman_corrcoef(embedding_cos_sim, stock_corrcoef).mean().item()
        res['ebd_stock_mean'] = stock_corrcoef.mean().item()
        return res

    def _generate_pairwise_sicx(self, inputs: Dict[str, Tensor]) -> Tensor:
        sic = inputs['sic']
        pairwise_sic = (sic == sic[:, None]).to(torch.float32)
        for _ in range(3):
            sic = sic // 10
            pairwise_sic += (sic == sic[:, None]).to(torch.float32)
        return pairwise_sic


    def _generate_pairwise_naicsx(self, inputs: Dict[str, Tensor]) -> Tensor:
        naics = inputs['naics']
        naics = naics[naics != -1]
        pairwise_naics = (naics == naics[:, None]).to(torch.float32)
        for _ in range(5):
            naics = naics // 10
            pairwise_naics += (naics == naics[:, None]).to(torch.float32)
        return pairwise_naics

    def _compute_sicx_stock_spearman(self, inputs: Dict[str, Tensor]) -> Dict[str, float]:
        pairwise_sic = self._remove_diag(self._generate_pairwise_sicx(inputs))
        stock_corrcoef = self._remove_diag(inputs['stock_corrcoef'])
        sorted_idx = pairwise_sic.argsort(dim=-1, descending=True)[:self.peer_firm_threshold]
        pairwise_sic = pairwise_sic[torch.arange(sorted_idx.shape[0])[:, None], sorted_idx]
        stock_corrcoef = stock_corrcoef[torch.arange(sorted_idx.shape[0])[:, None], sorted_idx]
        res = {}
        res['sicx_stock_mean'] = stock_corrcoef.mean().item()
        return res

    def _compute_naicsx_stock_spearman(self, inputs: Dict[str, Tensor]) -> Dict[str, float]:
        stock_corrcoef = inputs['stock_corrcoef']
        stock_corrcoef = stock_corrcoef[inputs['naics'] != -1][:, inputs['naics'] != -1]
        stock_corrcoef = self._remove_diag(stock_corrcoef)
        pairwise_naics = self._remove_diag(self._generate_pairwise_naicsx(inputs))
        sorted_idx = pairwise_naics.argsort(dim=-1, descending=True)[:self.peer_firm_threshold]
        pairwise_naics = pairwise_naics[torch.arange(sorted_idx.shape[0])[:, None], sorted_idx]
        stock_corrcoef = stock_corrcoef[torch.arange(sorted_idx.shape[0])[:, None], sorted_idx]
        res = {}
        res['naicsx_stock_mean'] = stock_corrcoef.mean().item()
        return res

    def _compute_ebd_sicx_spearman(self, inputs: Dict[str, Tensor]) -> Dict[str, float]:
        embedding_cos_sim = self._remove_diag(inputs['embedding_cos_sim'])
        pairwise_sic = self._remove_diag(self._generate_pairwise_sicx(inputs))
        res = spearman_corrcoef(embedding_cos_sim.flatten(), pairwise_sic.flatten()).item()
        return {f'ebd_sicx_spearman': res}

    def _compute_ebd_naicsx_spearman(self, inputs: Dict[str, Tensor]) -> Dict[str, float]:
        embedding_cos_sim = inputs['embedding_cos_sim']
        embedding_cos_sim = embedding_cos_sim[inputs['naics'] != -1][:, inputs['naics'] != -1]
        embedding_cos_sim = self._remove_diag(embedding_cos_sim)
        pairwise_naics = self._remove_diag(self._generate_pairwise_naicsx(inputs))
        res = spearman_corrcoef(embedding_cos_sim.flatten(), pairwise_naics.flatten()).item()
        return {f'ebd_naicsx_spearman': res}

    def _make_sic3_cls_dataset(self, inputs: Dict[str, Tensor]) -> Tuple[ndarray, ndarray]:
        embedding = inputs['embedding'].numpy()
        sic3 = inputs['sic'].numpy() // 10
        counter = Counter[int](sic3)
        labels_set = {k for k, v in counter.items() if v >= self.logistic_regression_min_labels}
        mapping = {k: i  for i, k in enumerate(labels_set)}
        res_embedding = []
        res_sic = []
        for e, s in zip(embedding, sic3):
            if s in mapping:
                res_embedding.append(e)
                res_sic.append(mapping[s])

        return np.stack(res_embedding, axis=0), np.array(res_sic)

    def _compute_logistic_regression_sic3(self, inputs: Dict[str, Tensor]) -> Dict[str, float]:
        embedding, labels = self._make_sic3_cls_dataset(inputs)
        print(f'sic3 labels的个数: {max(labels) + 1}')
        pred = cross_val_predict(self.logistic_regression, embedding, labels, 
                                 cv=self.logistic_regression_k_fold, method="predict")
        acc = accuracy_score(labels, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='macro')
        return {'lr_ebd_sic3_acc': acc, 'lr_ebd_sic3_f1': f1}

    def _make_naics4_cls_dataset(self, inputs: Dict[str, Tensor]) -> Tuple[ndarray, ndarray]:
        embedding = inputs['embedding'].numpy()
        naics4 = inputs['naics'].numpy() // 100
        counter = Counter[int](naics4)
        counter.pop(-1)
        labels_set = {k for k, v in counter.items() if v >= self.logistic_regression_min_labels}
        mapping = {k: i  for i, k in enumerate(labels_set)}
        res_embedding = []
        res_naics = []
        for e, s in zip(embedding, naics4):
            if s in mapping:
                res_embedding.append(e)
                res_naics.append(mapping[s])

        return np.stack(res_embedding, axis=0), np.array(res_naics)

    def _compute_logistic_regression_naics4(self, inputs: Dict[str, Tensor]) -> Dict[str, float]:
        embedding, labels = self._make_naics4_cls_dataset(inputs)
        print(f'naics4 labels的个数: {max(labels) + 1}')
        pred = cross_val_predict(self.logistic_regression, embedding, labels, 
                                 cv=self.logistic_regression_k_fold, method="predict")
        acc = accuracy_score(labels, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='macro')
        return {'lr_ebd_naics4_acc': acc, 'lr_ebd_naics4_f1': f1}

    def _to_device(self, inputs: Dict[str, Tensor]) -> None:
        for k, v in inputs.items():
            inputs[k] = v.to(device=self.compute_device)

    def compute(self, from_output_dir: Optional[str] = None) -> Dict[str, Tensor]:
        logger.info('compute metrics')
        if from_output_dir is None:
            inputs = self._prepare_inputs()
        else:
            from_pkl_path = os.path.join(from_output_dir, self.pkl_fname)
            with open(from_pkl_path, 'rb') as f:
                inputs = pickle.load(f)
        if self.save_to_pickle and from_output_dir != self.output_dir:
            with open(self.pkl_path, 'wb') as f:
                pickle.dump(inputs, f)
        self._to_device(inputs)
        self._compute_inputs_matrix(inputs)
        if self.save_to_jsonl:
            self._save_peer_firm_to_jsonl(inputs)
        res = {}
        #
        res.update(self._compute_ebd_stock_spearman(inputs))
        res.update(self._compute_sicx_stock_spearman(inputs))
        res.update(self._compute_ebd_sicx_spearman(inputs))
        res.update(self._compute_naicsx_stock_spearman(inputs))
        res.update(self._compute_ebd_naicsx_spearman(inputs))

        print(res)
        res.update(self._compute_logistic_regression_sic3(inputs))
        res.update(self._compute_logistic_regression_naics4(inputs))
        res.update({'ebd_cos_sim_mean': inputs['embedding_cos_sim'].mean().item()})
        return res
