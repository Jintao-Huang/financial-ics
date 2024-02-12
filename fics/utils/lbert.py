from transformers.models.roberta import (
    RobertaPreTrainedModel, RobertaForCausalLM, RobertaConfig
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead, RobertaModel, RobertaEmbeddings, RobertaPooler, RobertaEncoder, RobertaLayer,
    RobertaIntermediate, RobertaOutput, RobertaAttention, RobertaSelfAttention, RobertaSelfOutput,
    create_position_ids_from_input_ids
)
from transformers.utils import logging
from typing import Optional, Tuple, List, Union
import torch.nn as nn
import torch
import math
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, apply_rotary_pos_emb
)

logger = logging.get_logger(__name__)


def set_lbert_config(config: RobertaConfig, **kwargs) -> None:
    config.position_embedding_type = 'rotary'
    config.max_position_embeddings = kwargs.get('max_position_embeddings', 131072)
    config.rope_scaling = {
        "factor": 1,
        "type": "linear"
    }
    config.rope_theta = 10000.
    config.window_size = kwargs.get('window_size', 512)
    config.num_global_token = kwargs.get('num_global_token', 1)


class LBertEmbeddings(RobertaEmbeddings):
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.global_token_embedding = nn.Embedding(config.num_global_token, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        self.padding_idx = config.pad_token_id
        self.config = config


    def forward(
        self, *args, **kwargs
    ):
        embeddings = super().forward(*args, **kwargs)
        device = embeddings.device
        num_global_token = self.config.num_global_token
        pos_ids = torch.arange(num_global_token, device=device)[None].repeat(embeddings.shape[0], 1)
        inpus_global_embeds = self.global_token_embedding(pos_ids)
        embeddings = torch.concat([inpus_global_embeds, embeddings], dim=1)
        return embeddings

def shift_x_reverse(x: torch.Tensor, window_size: int):
    N, H, LdW, window_size, E = x.shape
    x = x.reshape((N, H, LdW * window_size, E))
    x = torch.concat([x[:, :H//2], x[:, H//2:].roll(window_size // 2, 2)], dim=1)

    return x

def shift_x(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x = x.contiguous()
    N, H, L, E = x.shape
    x = torch.concat([x[:, :H//2], x[:, H//2:].roll(-window_size // 2, 2)], dim=1)  # shift half of head
    return x.reshape((N, H, L // window_size, window_size, E))

def shift_attention_mask(attention_mask: torch.Tensor, window_size: int, H: int) -> torch.Tensor:
    N, _, _, L  = attention_mask.shape
    attention_mask = attention_mask.repeat(1, H, 1, 1)
    attention_mask = torch.concat([attention_mask[:, :H//2], 
                     attention_mask[:, H//2:].roll(-window_size // 2, 3)], dim=1)
    attention_mask = attention_mask.reshape((N, H, L // window_size, 1, window_size))
    return attention_mask

class LBertSelfAttention(RobertaSelfAttention):

    def __init__(self, config, position_embedding_type=None):
        super(RobertaSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.config = config
        if config.position_embedding_type == 'rotary':
            self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                self.attention_head_size,
                max_position_embeddings=config.max_position_embeddings,
                scaling_factor=config.rope_scaling['factor'],
                base=config.rope_theta,
            )

    def global_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                    attention_mask: torch.Tensor) -> torch.Tensor:

        num_global_token = self.config.num_global_token
        global_q = q[:, :, :num_global_token]
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores = torch.matmul(global_q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        global_output = torch.matmul(attention_probs, v)
        return global_output


    def block_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        config = self.config
        num_global_token = config.num_global_token
        block_q = q[:, :, num_global_token:]
        block_k = k[:, :, num_global_token:]
        block_v = v[:, :, num_global_token:]

        block_q = shift_x(block_q, config.window_size)
        block_k = shift_x(block_k, config.window_size)
        block_v = shift_x(block_v, config.window_size)
        global_k = k[:, :, None, :num_global_token].repeat((1, 1, block_k.shape[2], 1, 1))
        global_v = v[:, :, None, :num_global_token].repeat((1, 1, block_v.shape[2], 1, 1))
        block_k = torch.concat([global_k, block_k], dim=3)
        block_v = torch.concat([global_v, block_v], dim=3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(block_q, block_k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            block_attn_mask = attention_mask[..., num_global_token:]
            block_attn_mask = shift_attention_mask(block_attn_mask, config.window_size, self.num_attention_heads)
            _0 = block_attn_mask.new_zeros(*block_attn_mask.shape[:4], num_global_token)
            block_attn_mask = torch.concat([_0, block_attn_mask], dim=-1)
            H = attention_scores.shape[1]
            attention_scores = attention_scores + block_attn_mask
            # avoid attention mix
            attention_scores[:, H//2:, -1, config.window_size//2:, 1:config.window_size//2+1] = torch.finfo(attention_scores.dtype).min
            attention_scores[:, H//2:, -1, :config.window_size//2, config.window_size//2+1:] = torch.finfo(attention_scores.dtype).min
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, block_v)
        context_layer = shift_x_reverse(context_layer, config.window_size)
        return context_layer


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        if hasattr(self, 'rotary_emb'):
            kv_seq_len = key_layer.shape[-2]
            cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            position_ids = torch.arange(0, kv_seq_len, dtype=torch.long, device=query_layer.device)[None]
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)

        global_output = self.global_attn(query_layer, key_layer, value_layer, attention_mask)
        block_output = self.block_attn(query_layer, key_layer, value_layer, attention_mask)
        context_layer = torch.concat([global_output, block_output], dim=2)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)

        return outputs

class LBertAttention(RobertaAttention):
    def __init__(self, config, position_embedding_type=None):
        super(RobertaAttention, self).__init__()
        self.self = LBertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()


class LBertLayer(RobertaLayer):
    def __init__(self, config):
        super(RobertaLayer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LBertAttention(config)
        self.is_decoder = False
        self.add_cross_attention = False
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = LBertAttention(config, position_embedding_type=None)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)


class LBertEncoder(RobertaEncoder):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([LBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False


class LBertModel(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super(RobertaModel, self).__init__(config)
        self.config = config

        self.embeddings = LBertEmbeddings(config)
        self.encoder = LBertEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args, **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        global_attention_mask = torch.ones((self.config.num_global_token,), device=input_ids.device
                                            )[None].repeat(input_ids.shape[0], 1)
        attention_mask = torch.concat([global_attention_mask, attention_mask], dim=1)
        output = super().forward(input_ids, attention_mask, *args, **kwargs)
        output.last_hidden_state = output.last_hidden_state[:,  self.config.num_global_token:]
        return output

class LBertForMaskedLM(RobertaForCausalLM):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super(RobertaForCausalLM, self).__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.roberta = LBertModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()


if __name__ == '__main__':
    # for test
    config = RobertaConfig()
    set_lbert_config(config)
    model = LBertForMaskedLM(config)
    print(model)
