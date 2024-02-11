import numpy as np
from typing import Dict
from torch import Tensor
from transformers.trainer_utils import EvalPrediction

def compute_maskedlm_metrics(eval_prediction: EvalPrediction) -> Dict[str, Tensor]:
    labels = eval_prediction.label_ids
    masks = labels != -100
    predictions = eval_prediction.predictions
    labels = labels[masks]
    acc = np.mean((predictions == labels).astype(np.float64))
    return {'acc': acc}
