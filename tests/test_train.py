import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from fics import train_main, TrainArguments, CustomDatasetName, CustomModelType
import torch

if __name__ == '__main__':
    train_main(TrainArguments(CustomModelType.roberta_base, dataset=[CustomDatasetName.tenk_pretrained_mini],
                              train_dataset_sample=5, eval_steps=5))
    torch.cuda.empty_cache()
    train_main(TrainArguments(CustomModelType.lbert, dataset=[CustomDatasetName.tenk_pretrained_mini],
                              train_dataset_sample=100, eval_steps=5))
