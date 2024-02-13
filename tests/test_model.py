from fics import get_model_tokenizer, CustomModelType

if __name__ == '__main__':
    # for test
    model, tokenizer = get_model_tokenizer(CustomModelType.roberta_base_no_head)
    print(f'roberta-base-no-head: {model}')
    model, tokenizer = get_model_tokenizer(CustomModelType.longformer)
    print(f'longformer: {model}')
    model, tokenizer = get_model_tokenizer(CustomModelType.lbert_no_head)
    print(f'lbert-no-head: {model}')
