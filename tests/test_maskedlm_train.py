import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from fics import train_main, TrainArguments, CustomDatasetName, CustomModelType
import torch

def test_lbert_train():
    train_main(TrainArguments(CustomModelType.lbert, dataset=[CustomDatasetName.tenk_pretrained_mini],
                              train_dataset_sample=100, eval_steps=5, lbert_num_global_token=16,
                              batch_size=2))

if __name__ == '__main__':
    test_lbert_train()
