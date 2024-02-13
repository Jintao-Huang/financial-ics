from transformers import AutoConfig
from torch import dtype as Dtype
from swift.llm import register_model, get_model_tokenizer_from_repo
from .lbert import LongBertForMaskedLM, set_lbert_config
from transformers.models.roberta import RobertaForMaskedLM
from transformers.models.longformer import LongformerForMaskedLM

from typing import Any, Dict
from torch.nn import Identity

class CustomModelType:
    roberta_base = 'roberta-base'
    roberta_base_no_head = 'roberta-base-no-head'
    lbert = 'lbert'
    lbert_no_head = 'lbert-no-head'
    longformer = 'longformer'
    longformer_no_head = 'longformer-no-head'

register_roberta_base_kwargs = {
    'model_id_or_path': 'AI-ModelScope/roberta-base',
    'ignore_file_pattern': [r'.+\.msgpack$', r'.+\.bin$', r'.+\.ot$', r'.+\.h5$'],
    'max_length': 512,
}
register_longformer_kwargs = {
    'model_id_or_path': 'huangjintao/longformer-base-4096',
    'ignore_file_pattern': [r'.+\.ot$', r'.+\.h5$'],
    'max_length': 4096,
}

register_lbert_kwargs = {
    'model_id_or_path': 'AI-ModelScope/roberta-base',
    'ignore_file_pattern': [r'.+\.msgpack$', r'.+\.bin$', r'.+\.ot$', r'.+\.h5$'],
    'max_length': 131072,
}


@register_model(CustomModelType.roberta_base, **register_roberta_base_kwargs)
@register_model(CustomModelType.roberta_base_no_head, **register_roberta_base_kwargs,
                function_kwargs={'remove_lm_head': True})
def get_model_tokenizer_roberta(*args, **kwargs):
    if 'automodel_class' not in kwargs:
        kwargs['automodel_class'] = RobertaForMaskedLM
    remove_lm_head = kwargs.pop('remove_lm_head', False)
    model, tokenizer = get_model_tokenizer_from_repo(*args, **kwargs)
    if model is not None and remove_lm_head:
        assert hasattr(model, 'lm_head')
        model.lm_head = Identity()
    return model, tokenizer

@register_model(CustomModelType.longformer, **register_longformer_kwargs)
@register_model(CustomModelType.longformer_no_head, **register_longformer_kwargs,
                function_kwargs={'remove_lm_head': True})
def get_model_tokenizer_longformer(*args, **kwargs):
    kwargs['automodel_class'] = LongformerForMaskedLM
    return get_model_tokenizer_roberta(*args, **kwargs)


@register_model(CustomModelType.lbert, **register_lbert_kwargs)
@register_model(CustomModelType.lbert_no_head, **register_lbert_kwargs,
                function_kwargs={'remove_lm_head': True})
def get_lbert_model_tokenizer(model_dir: str,
                                torch_dtype: Dtype,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    kwargs['automodel_class'] = LongBertForMaskedLM
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config_kwargs = {}
    args = kwargs.get('args')
    if args is not None:
        config_kwargs['max_position_embeddings'] = args.lbert_max_position_embeddings
        config_kwargs['window_size'] = args.lbert_window_size
        config_kwargs['num_global_token'] = args.lbert_num_global_token
    set_lbert_config(model_config, **config_kwargs)
    model, tokenizer = get_model_tokenizer_roberta(model_dir, torch_dtype, model_kwargs, load_model, 
                                                   model_config=model_config, **kwargs)
    tokenizer.padding_to = model_config.window_size
    tokenizer.num_global_token = model_config.num_global_token
    return model, tokenizer
