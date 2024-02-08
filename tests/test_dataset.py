from fics import CustomDatasetName, get_dataset

if __name__ == '__main__':
    # for test
    train_dataset, val_dataset = get_dataset([CustomDatasetName.tenk_pretrained_mini], 0.01, 42)
    print(f'train_dataset: {train_dataset}')
    print(f'val_dataset: {val_dataset}')
    eval_dataset, _ = get_dataset([CustomDatasetName.tenk_eval])
    print(f'eval_dataset: {eval_dataset}')
    demo_dataset, _ = get_dataset([CustomDatasetName.tenk_demo])
    print(f'demo_dataset: {demo_dataset}')
