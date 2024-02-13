import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from fics import eval_main, EvalArguments, CustomDatasetName, CustomModelType
import torch

def test_eval():
    eval_main(EvalArguments(CustomModelType.lbert_no_head, dataset=[CustomDatasetName.tenk_eval_mini],
                            eval_dataset_sample=100, eval_batch_size=1))
    torch.cuda.empty_cache()
    eval_main(EvalArguments(CustomModelType.lbert_no_head, dataset=[CustomDatasetName.tenk_eval_mini],
                            eval_dataset_sample=100, eval_batch_size=4))
    torch.cuda.empty_cache()
    eval_main(EvalArguments(CustomModelType.lbert_no_head, dataset=[CustomDatasetName.tenk_eval_mini],
                            eval_dataset_sample=100, lbert_num_global_token=16, eval_batch_size=4))

if __name__ == '__main__':
    test_eval()
