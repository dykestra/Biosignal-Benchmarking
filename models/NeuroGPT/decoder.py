#!/usr/bin/env python3

from typing import Dict
import warnings
import torch
from transformers import GPT2Config, GPT2Model
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

class GPTModel(torch.nn.Module):
    def __init__(
        self,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        embed_dim: int = 768,
        intermediate_dim_factor: int = 4,
        n_positions: int = 512,
        hidden_activation: str = 'gelu',
        dropout: float = 0.1,
        **kwargs
        ) -> None:
        super().__init__()
        self.name = 'GPT'
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.embed_dim = embed_dim
        self.intermediate_dim_factor = intermediate_dim_factor
        self.n_positions = n_positions
        self.hidden_activation = hidden_activation
        self.dropout_resid = dropout
        self.dropout_attn = dropout
        self.dropout_embd = dropout
        self.mse_loss = torch.nn.MSELoss()
        self.bxe_loss = torch.nn.BCEWithLogitsLoss() 
        self.config = GPT2Config(
            vocab_size=1,
            n_positions=self.n_positions,
            n_embd=self.embed_dim,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            n_inner=self.embed_dim * self.intermediate_dim_factor,
            resid_pdrop=self.dropout_resid,
            attn_pdrop=self.dropout_attn,
            embd_pdrop=self.dropout_embd,
            activation_function=self.hidden_activation
        )
        self.transformer = GPT2Model(config=self.config)
        self.is_decoding_mode = False
        self.decoding_head = None
        self.num_decoding_classes = None
        self.pooler_layer = None
        self.add_pooler_layer()

    def switch_decoding_mode(
        self,
        is_decoding_mode: bool=False,
        num_decoding_classes: int=None
        ) -> None:
        self.is_decoding_mode = is_decoding_mode
        if self.is_decoding_mode:
            if self.pooler_layer is None:
                self.add_pooler_layer()
            self.add_decoding_head(num_decoding_classes=num_decoding_classes)
        else:
            self.decoding_head = None

    def add_pooler_layer(self):
        if self.pooler_layer is not None:
            warnings.warn(
                    'Warning: overwriting existing pooler layer'
                )
        self.pooler_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.embed_dim,
                out_features=self.embed_dim
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(self.dropout_resid)
        )

    def add_decoding_head(
        self,
        num_decoding_classes: int
        ) -> None:
        if self.decoding_head is not None:
            if self.num_decoding_classes == num_decoding_classes:
                warnings.warn(
                    'Warning: not overwriting decoding head, as '
                    f'{num_decoding_classes}-class decoding head exists.'
                )
                return None
            else:
                warnings.warn(
                    f'Warning: overwriting existing {num_decoding_classes}-class decoding head.'
                )
        self.num_decoding_classes = num_decoding_classes
        self.decoding_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, self.num_decoding_classes)
        )
        return None
    
    def decode(
        self,
        outputs: torch.tensor,
        attention_mask: torch.tensor,
        ) -> Dict[str, torch.tensor]:
        assert self.is_decoding_mode, 'GPTModel must be in decoding_mode.'
        assert self.pooler_layer is not None, 'pooler_layer head must be added.'
        assert self.decoding_head is not None, 'decoding head must be added.'
        batch_size = outputs.size()[0]
        sequence_lengths = attention_mask.sum(dim=1)-1
        decoding_outputs = {
            'pooler_outputs': self.pooler_layer(
                outputs[torch.arange(batch_size, device=outputs.device), sequence_lengths]
            )
        }
        decoding_outputs['decoding_logits'] = self.decoding_head(decoding_outputs['pooler_outputs'])
        return decoding_outputs

    def forward(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict[str, torch.tensor]:
        transformer_outputs = self.transformer.forward(
            inputs_embeds=batch['inputs_embeds'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids', None),
            return_dict=True
        )
        outputs = {'outputs': transformer_outputs['last_hidden_state']}

        if not self.is_decoding_mode:
            return outputs

        outputs.update(
            self.decode(
                outputs=outputs['outputs'],
                attention_mask=batch['attention_mask']
            )
        )
        return outputs


class PretrainedGPT2(GPTModel):
    
    def __init__(
        self,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.name = 'PretrainedGPT2'
        self.config = GPT2Config()
        self.n_positions = self.config.n_positions
        self.embed_dim = self.config.n_embd
        self.num_hidden_layers = self.config.n_layer
        self.num_attention_heads = self.config.n_head
        self.intermediate_dim_factor = 4
        self.dropout_resid = self.config.resid_pdrop
        self.dropout_attn = self.config.attn_pdrop
        self.dropout_embd = self.config.embd_pdrop
        self.hidden_activation = self.config.activation_function
        self.transformer = GPT2Model.from_pretrained("gpt2")

# ============== make_decoder.py =============================

def make_decoder(
    architecture: str='GPT',
    num_hidden_layers: int = 4,
    embed_dim: int = 768,
    output_dim: int = 1024,
    num_attention_heads: int = 12,
    intermediate_dim_factor: int=4,
    n_positions: int = 512,
    hidden_activation: str='gelu_new',
    dropout: float = 0.1
    ) -> torch.nn.Module:
    """
    Make a decoder object.
    
    The decoder contains the core
    model architecture used for learning.

    Args:
    -----
    architecture: str
        The model architecture to use.
        One of: 'GPT', 'BERT', 'NetBERT', autoencoder',
        'PretrainedGPT', 'PretrainedBERT', 'LinearBaseline'.
    num_hidden_layers: int
        The number of hidden layers of the model.
        Does not apply to 'PretrainedGPT', 'PretrainedBERT', 
        'LinearBaseline'. 
        For 'autoencoder', num_hidden_layers represents 
        the number of hidden layers of the encoder and decoder
        model.
    embed_dim: int
        The dimension of the used embedding space (see src.embedder).
    output_dim: int
        The dimension of the output projection (needs to match
        in_dim of src.embedder for upstream learning).
    num_attention_heads: int
        The number of attention heads of transformer models. Does
        not apply to any other model architecture as well as the
        'PretrainedGPT' and 'PretrainedBERT' architectures.
    intermediate_dim_factor: int
        Scales feed-forward transformer layer dimension relative to '
        embed_dim: intermediate_dim_factor * embed_dim
    n_positions: int
        The maximum number of sequence elements that
        the model can handle (in sequence elements).
    hidden_activation: str
        Type of hidden activation of transformer layers
        One of 'gelu', 'gelu_new', 'relu', 'silu'.
        Does not apply to non-transformer models.
    dropout: float
        Dropout ratio for attendion heads and residual layers
        of transofmer models and between LSTM layers of 
        encoder / decoder parts of autoencoder models. 

    Core methods:
    -----
    forward(batch: Dict):
        Forward pass of the model, generates Dict containing
        predicted output seqeuences, given input batch
        (as generated by src.embedder.prep_batch).
    decode(outputs: Dict):
        Make decoding prediction, given outputs generated by
        caling forward().    
    switch_decoding_mode(is_decoding_mode: bool):
        Switch model to decoding mode (is_decoding_mode=True).
        Relevant for adaptation of pre-trained models
        to downstream decoding tasks.
    """

    kwargs = {
        "num_hidden_layers": num_hidden_layers,
        "embed_dim": embed_dim,
        "output_dim": output_dim,
        "num_attention_heads": num_attention_heads,
        "intermediate_dim_factor": intermediate_dim_factor,
        "n_positions": n_positions,
        "hidden_activation": hidden_activation,
        "dropout": dropout
    }

    if architecture == 'GPT':
        return GPTModel(**kwargs)
    elif architecture == 'PretrainedGPT2':
        return PretrainedGPT2(**kwargs)

    else:
        raise ValueError(f'{architecture}-architecture unkown.')
    
# ===================== unembedder.py ==================================


class DeconvNet(nn.Module):
    def __init__(self, n_filters_time=40, n_channels=22, filter_time_length=25, stride_avg_pool=15, pool_time_length=75):
        super(DeconvNet, self).__init__()
        # To reverse AvgPool2d
        self.depool = nn.Sequential(Rearrange("b seq d_model -> b d_model 1 seq"),
                                    nn.Upsample(size=(1, 476), mode='nearest'))
        #nn.ConvTranspose2d(n_filters_time, n_filters_time, kernel_size=(1, pool_time_length), stride=(1, stride_avg_pool))
        self.deconv1 = nn.ConvTranspose2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1))
        self.deconv2 = nn.ConvTranspose2d(n_filters_time, 1, (1, filter_time_length), (1, 1))
        
    def forward(self, x):
        x = self.depool(x)
        x = self.deconv1(x)
        x = nn.ELU()(x)  # We're keeping ELU activation.
        x = self.deconv2(x)
        return {'outputs': x.squeeze()}


class UnEmbedder(torch.nn.Module):
    """
    Unmebedding model; used to project predicted 
    output sequences of src.decoder back to input 
    space during upstream learning.
    
    Args
    ----
    embed_dim: int
        Dimension of the embedding space.
    out_dim: int
        Dimension of the output space.
    num_hidden_layers: int
        Number of hidden layers of projection model.
        If >1, all hidden layers except for the last
        are activated with Gelu activation.
    dropout: float
        Dropout ratio for the projection model.

    Core methods
    ----
    forward(inputs, **kwargs)
        Projection of input to output space.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        out_dim: int = 1024,
        num_hidden_layers: int = 1,
        dropout: int = 0.1,
        ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        layer_stack = []
        for _ in range(self.num_hidden_layers-1):
            layer_stack.extend(
                [
                    torch.nn.Linear(
                        in_features=self.embed_dim,
                        out_features=self.embed_dim
                    ),
                    torch.nn.LayerNorm(self.embed_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(p=self.dropout)
                ]
            )
        layer_stack.extend(
            [
                torch.nn.Linear(
                    in_features=self.embed_dim,
                    out_features=self.out_dim
                )
            ]
        )
        self.model = torch.nn.Sequential(*layer_stack)

    def stack_inputs(
        self,
        tensor
        ) -> torch.tensor:
        
        return rearrange(
            tensor=tensor,
            pattern='b s e -> (b s) e'
        )

    def unstack_inputs(
        self,
        tensor,
        b
        ) -> torch.tensor:
        
        return rearrange(
            tensor=tensor,
            pattern='(b s) e -> b s e',
            b=b
        )

    def forward(
        self,
        inputs,
        **kwargs
        ) -> torch.tensor:
        inputs_stacked = self.stack_inputs(tensor=inputs)
        
        return {
            'outputs': self.unstack_inputs(
                tensor=self.model(inputs_stacked),
                b=inputs.size()[0]
            )
        }


def make_unembedder(
    embed_dim: int = 768,
    out_dim: int = 1024,
    num_hidden_layers: int = 1,
    dropout: int = 0.1
    ) -> torch.nn.Module:
    """
    Creates a UnEmbedder object.

    Args
    ----
    embed_dim: int
        Dimension of the embedding space.
    out_dim: int
        Dimension of the output space.
    num_hidden_layers: int
        Number of hidden layers of projection model.
        If >1, all hidden layers except for the last
        are activated with Gelu activation.
    dropout: float
        Dropout ratio for the projection model.

    Core methods
    ----
    forward(inputs, **kwargs)
        Projection of input to output space.
    """
    return UnEmbedder(
        embed_dim=embed_dim,
        out_dim=out_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout
    )