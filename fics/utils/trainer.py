from torch import Tensor
import torch
import os
from torch.nn import CrossEntropyLoss
from swift.trainers import Trainer
from torchmetrics.functional import pairwise_cosine_similarity
import torch.nn.functional as F
from typing import Tuple, NamedTuple, Optional
from swift.utils import is_dist, is_master
from torch import Tensor
import torch.distributed as dist
from .metric import IcsMetrics
MaskedLMOutput = NamedTuple('MaskedLMOutput', loss=Optional[Tensor], logits=Optional[Tensor])


class MaskedLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=None):
        if not hasattr(self, '_custom_metrics'):
            self._custom_metrics = {}
        inputs.pop('token_length', None)
        loss, outputs = super().compute_loss(model, inputs, True)
        labels = inputs['labels']
        masks = labels != -100
        preds = outputs.logits.argmax(dim=1)
        if model.training:
            acc: Tensor = (preds == labels[masks]).float().mean()
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
        cos_sim = pairwise_cosine_similarity(logits, prototype)
        cos_sim_mean = cos_sim.mean()
        cos_sim = cos_sim / self.temperature
        labels = torch.arange(batch_size // 2).repeat_interleave(2, dim=0).to(cos_sim.device)
        loss = F.cross_entropy(cos_sim, labels)
        return loss, cos_sim_mean

class ContrastiveLearningTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=None):
        if not hasattr(self, '_custom_metrics'):
            self._custom_metrics = {}
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = PrototypeLoss(self.x_args.temperature)
        inputs.pop('labels', None)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        inputs.pop('token_length', None)
        batch_size = input_ids.shape[0]
        logits_no_grad = []
        num_prototype_with_grad = self.x_args.num_prototype_with_grad * 2
        _batch_size = 2 * num_prototype_with_grad  # no grad batch size
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
        if pooling == 'cls':
            logits = logits[:, 0]
        elif pooling == 'mean':
            logits = logits.to(torch.float64)
            logits = torch.einsum("ijk,ij->ik", logits, attention_mask.to(logits.dtype))
            logits.div_(attention_mask.sum(dim=1, keepdim=True))
        elif pooling == 'global-mean':
            logits = logits[:, :model.config.num_global_token]
            logits = logits.mean(dim=1)
        else:
            raise ValueError(f'pooling: {pooling}')
        if is_dist():
            logits = logits.contiguous()
            from .utils import GatherLayer
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
            self._custom_metrics = {}
        assert not model.training
        assert return_outputs is True
        ignore_keys = ['cik', 'date', 'sic', 'stock', 'token_length', 'naics']
        if 'labels' in inputs:
            inputs.pop('labels')
        model_inputs = {k: v for (k, v) in inputs.items() if k not in ignore_keys}
        attention_mask = model_inputs['attention_mask']
        output = model(**model_inputs)
        logits = output.logits.to(torch.float64)
        pooling = self.x_args.pooling
        if pooling == 'cls':
            logits = logits[:, 0]
        elif pooling == 'mean':
            logits = torch.einsum("ijk,ij->ik", logits, attention_mask.to(logits.dtype))
            logits.div_(attention_mask.sum(dim=1, keepdim=True))
        elif pooling == 'global-mean':
            logits = logits[:, :model.config.num_global_token]
            logits = logits.mean(dim=1)
        else:
            raise ValueError(f'pooling: {pooling}')

        if 'ics_metrics' not in self._custom_metrics:
            self._custom_metrics['ics_metrics'] = IcsMetrics(self.args.output_dir)
        self._custom_metrics['ics_metrics'].update(
            inputs['cik'], logits, inputs['token_length'], inputs['date'],
            inputs['sic'], inputs['naics'], inputs.get('stock')
        )
        placeholder = torch.tensor([0.], device=logits.device)
        return (placeholder, placeholder) if return_outputs else placeholder

