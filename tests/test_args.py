
from fics import TrainArguments, EvalArguments, CustomDatasetName, TaskType, CustomModelType

if __name__ == '__main__':
    train_args = TrainArguments(model_type=CustomModelType.lbert, 
                                dataset=CustomDatasetName.tenk_pretrained_mini)
    print(f'train_args: {train_args}')
    eval_args = EvalArguments(model_type=CustomModelType.lbert_no_head, dataset=CustomDatasetName.tenk_eval_mini)
    print(f'eval_args: {eval_args}')
