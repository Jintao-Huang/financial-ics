import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from fics import app_ui_main, EvalArguments, CustomDatasetName, CustomModelType

def test_app():
    eval_args = EvalArguments(model_type=CustomModelType.lbert_no_head, dataset=CustomDatasetName.tenk_demo,
                              eval_dataset_sample=100)
    app_ui_main(eval_args)

if __name__ == '__main__':
    test_app()
