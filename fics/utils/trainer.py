from torch import Tensor
import torch
import os
from torch.nn import CrossEntropyLoss
from swift.trainers import Trainer
from torchmetrics.functional import pairwise_cosine_similarity
import torch.nn.functional as F
from typing import Tuple
from swift.utils import is_dist, is_master
import torch.distributed as dist
from .utils import GatherLayer
from .metric import IcsMetrics

def _masked_lm_forward(model, inputs) -> Tuple[Tensor, Tensor]:
    """for save memory"""
    labels = inputs.pop('labels')
    outputs = model.roberta(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'])
    sequence_output = outputs[0]
    labels = labels.to(sequence_output.device)
    masked_idx = labels != -100
    labels = labels[masked_idx]
    inputs['labels'] = labels
    sequence_output = sequence_output[masked_idx]
    logits = model.lm_head(sequence_output)
    loss_fct = CrossEntropyLoss()
    masked_lm_loss = loss_fct(logits, labels)
    return masked_lm_loss, logits

class MaskedLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=None):
        if not hasattr(self, '_custom_metrics'):
            self._custom_metrics = {}
        inputs.pop('token_length', None)
        loss, outputs = _masked_lm_forward(model, inputs)
        preds = outputs.logits.argmax(dim=1)
        labels = inputs['labels']
        masks = labels != -100
        if return_outputs:
            logits = torch.zeros((*labels.shape, outputs.logits.shape[-1]), 
                    device=outputs.logits.device, dtype=outputs.logits.dtype)
            logits[masks] = outputs.logits
            outputs.logits = logits
        acc: Tensor = (preds == labels[masks]).float().mean()
        if model.training:
            if 'acc' not in self._custom_metrics:
                self._custom_metrics['acc'] = torch.tensor(0., device=self.args.device)
            self._custom_metrics[
                'acc'] += acc / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss

class PrototypeLoss:
    def __init__(self, temperature: float) -> None:
        self.temperature = temperature

    def __call__(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, ebd_size = logits.shape
        prototype = logits.reshape((batch_size // 2, 2, ebd_size)).mean(1)
        # if not hasattr(self, 'prototype_pool'):
        #     self.prototype_pool = prototype
        # else:
        #     self.prototype_pool = torch.concat([prototype, self.prototype_pool.detach()], 0)
        #     self.prototype_pool = self.prototype_pool[:self.prototype_pool_size]
        cos_sim = pairwise_cosine_similarity(logits, prototype)
        cos_sim_mean = cos_sim.mean()
        cos_sim = cos_sim / self.temperature
        labels = torch.arange(batch_size // 2).repeat_interleave(2, dim=0).to(cos_sim.device)
        loss = F.cross_entropy(cos_sim, labels)
        # if self.prototype_pool.shape[0] < self.prototype_pool_size:
        #     loss = loss.clamp_max(0)
        #     return loss, cos_sim_mean
        return loss, cos_sim_mean
_need_eval = False
class ContrastiveLearningTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=None):
        from fics.run import eval_main
        from fics import EvalArguments
        global _need_eval
        if model.training:
            if _need_eval:
                output_dir = os.path.join(self.args.output_dir, f'checkpoint-{self.state.global_step}')
                eval_batch_size = 1
                if 'split' in self.x_args.dataset:
                    dataset = 'tenk-eval-split'
                else:
                    dataset = 'tenk-eval'
                if 'roberta' in self.x_args.model_type:
                    eval_batch_size = 16
                if is_master():
                    eval_args = EvalArguments(self.x_args.model_type,
                        task_type='eval-tenk',
                        position_embedding_type=self.x_args.position_embedding_type,
                        ckpt_dir =output_dir,
                        dataset=dataset, 
                        max_length=self.x_args.max_length,
                        min_length=self.x_args.min_length,
                        eval_dataset_sample=-1,
                        eval_batch_size=eval_batch_size,
                        normalize=self.x_args.normalize,
                        pooling=self.x_args.pooling,
                        window_size=self.x_args.window_size)
                    torch.cuda.empty_cache()
                    eval_main(eval_args)
                    torch.cuda.empty_cache()
                    if is_dist():
                        dist.barrier()
                else:
                    if is_dist():
                        dist.barrier()
                _need_eval = False
        else:
            _need_eval = True
        if not hasattr(self, '_custom_metrics'):
            self._custom_metrics = {}
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = PrototypeLoss(self.x_args.temperature)
        inputs.pop('labels')
        attention_mask = inputs['attention_mask']
        inputs.pop('token_length', None)
        batch_size = inputs['input_ids'].shape[0]
        logits_no_grad = []
        num_prototype_with_grad = self.x_args.num_prototype_with_grad * 2
        _batch_size = 2 * num_prototype_with_grad
        for i in range(num_prototype_with_grad, batch_size, _batch_size):
            inputs_no_grad = {k: v[i:i+_batch_size] for k, v in inputs.items()}
            with torch.no_grad():
                logits_no_grad.append(model(**inputs_no_grad).logits)
        inputs_with_grad = {k: v[:num_prototype_with_grad] for k, v in inputs.items()}
        logits_with_grad = model(**inputs_with_grad).logits
        if len(logits_no_grad) > 0:
            logits_no_grad = torch.concat(logits_no_grad)
            logits = torch.concat([logits_with_grad, logits_no_grad])
        else:
            logits = logits_with_grad
        pooling = self.x_args.pooling
        # if pooling == 'cls-32':
        #     logits = logits[:, 0:32].mean(dim=1)
        if pooling == 'cls':
            logits = logits[:, 0]
        elif pooling == 'mean':
            logits = torch.einsum("ijk,ij->ik", logits, attention_mask.to(logits.dtype))
            logits.div_(attention_mask.sum(dim=1, keepdim=True))
        elif pooling == 'last':
            idx = attention_mask.sum(dim=1) - 1
            logits = logits[torch.arange(idx.shape[0]), idx]
        if self.x_args.normalize:
            logits = F.normalize(logits, dim=1)
        if is_dist():
            logits = logits.contiguous()
            logits = torch.concat(GatherLayer.apply(logits))
        loss, cos_sim_mean = self.loss_fn(logits)
        if model.training:
            if 'cos_sim_mean' not in self._custom_metrics:
                self._custom_metrics['cos_sim_mean'] = torch.tensor(0., device=self.args.device)
            self._custom_metrics[
                'cos_sim_mean'] += cos_sim_mean / self.args.gradient_accumulation_steps
        return (loss, logits) if return_outputs else loss


class EvalTenkTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=None):
        if not hasattr(self, '_custom_metrics'):
            # assert len(signature(self.compute_metrics).parameters) > 1
            # self.compute_metrics = MethodType(self.compute_metrics, self)
            self._custom_metrics = {}
        assert not model.training
        assert return_outputs is True
        ignore_keys = ['cik', 'date', 'sic', 'stock', 'token_length', 'naics']
        if 'labels' in inputs:
            inputs.pop('labels')
        model_inputs = {k: v for (k, v) in inputs.items() if k not in ignore_keys}
        attention_mask = model_inputs['attention_mask']
        logits = model(**model_inputs).logits
        pooling = self.x_args.pooling
        # if pooling == 'cls-32':
        #     logits = logits[:, 0:32].mean(dim=1)
        if pooling == 'cls':
            logits = logits[:, 0]
        elif pooling == 'mean':
            logits = torch.einsum("ijk,ij->ik", logits, attention_mask.to(logits.dtype))
            logits.div_(attention_mask.sum(dim=1, keepdim=True))
        elif pooling == 'last':
            idx = attention_mask.sum(dim=1) - 1
            logits = logits[torch.arange(idx.shape[0]), idx]

        if self.x_args.normalize:
            logits = F.normalize(logits, dim=1)
        if 'ics_metrics' not in self._custom_metrics:
            self._custom_metrics['ics_metrics'] = IcsMetrics(self.args.output_dir, 
                                                          use_split=self.x_args.use_split)
        self._custom_metrics['ics_metrics'].update(
            inputs['cik'], logits, inputs['token_length'], inputs['date'],
            inputs['sic'], inputs['naics'], inputs.get('stock')
        )
        placeholder = torch.tensor([0.], device=logits.device)
        return (placeholder, placeholder) if return_outputs else placeholder

