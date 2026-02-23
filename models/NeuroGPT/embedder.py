import pdb
from typing import Dict
import torch
from einops import rearrange

# =============== embedder/base.py ===========================

class EmbeddingModel(torch.nn.Module):

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 768,
        num_hidden_layers: int = 1,
        dropout: int = 0.1,
        ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        layer_stack = []
        for _ in range(self.num_hidden_layers-1):
            layer_stack.extend(
                [
                    torch.nn.Linear(
                        in_features=self.in_dim,
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
                    in_features=self.embed_dim if self.num_hidden_layers>1 else self.in_dim,
                    out_features=self.embed_dim
                ),
                torch.nn.LayerNorm(self.embed_dim),
                torch.nn.Dropout(p=self.dropout)
            ]
        )
        self.model = torch.nn.Sequential(*layer_stack)

    def _stack_inputs(
        self,
        tensor
        ) -> torch.tensor:
        
        return rearrange(
            tensor=tensor,
            pattern='b s e -> (b s) e'
        )

    def _unstack_inputs(
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
        inputs_stacked = self._stack_inputs(tensor=inputs)
        
        return self._unstack_inputs(
            tensor=self.model(inputs_stacked),
            b=inputs.size()[0]
        )


class BaseEmbedder(torch.nn.Module):
    def __init__(self,
        in_dim: int = 1024,
        embed_dim: int = 768,
        num_hidden_layers: int = 1,
        dropout: float = 0.1,
        **kwargs
        ) -> None:
        super().__init__()
        self.name = 'BaseEmbedder'
        self.training_style = 'base'
        self._root_training_style = 'base'
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.xe_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.bxe_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.l2_loss = torch.nn.MSELoss(reduction='mean') # for L2 loss
        # self.huber_loss = torch.nn.HuberLoss(reduction='mean', delta=1.0) # for Huber loss
        
        self.embed_model = EmbeddingModel(
            in_dim=self.in_dim,
            embed_dim=self.embed_dim,
            num_hidden_layers=self.num_hidden_layers,
            dropout=self.dropout
        )
        self.is_decoding_mode = False

    def switch_decoding_mode(self, is_decoding_mode: bool=False) -> None:
        self.is_decoding_mode = is_decoding_mode
        
        if self.is_decoding_mode:
            self.training_style = 'decoding'
        else:
            self.training_style = self._root_training_style
    
    @staticmethod
    def _pad_tensor_left_by_n(
        tensor,
        n,
        pad_value
        ) -> torch.tensor:
        filling = torch.ones(
            (
                tensor.size()[0],
                n,
                *tensor.size()[2:]
            ),
            device=tensor.device
        ) * pad_value
        
        return torch.cat(
            [
                filling,
                tensor
            ],
            dim=1
        ).to(torch.long)

    @staticmethod
    def _round_to_precision(
        x: torch.tensor,
        precision: float,
        ) -> torch.tensor:
        return torch.round(x / precision) * precision


    def embed_inputs(
        self,
        inputs: torch.tensor
        ) -> torch.tensor:
        return self.embed_model(inputs)
    
    def forward(
        self,
        batch: Dict[str, torch.tensor]
        ) -> torch.tensor:
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        
        if self.in_dim == self.embed_dim:
            inputs_embeds = batch[inputs_key]
        else:
            inputs_embeds = self.embed_inputs(inputs=batch[inputs_key])
        
        return inputs_embeds

    def decoding_loss(
        self,
        decoding_logits,
        labels,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        # pdb.set_trace()
        return {
            'decoding_loss': self.xe_loss(
                input=decoding_logits,
                target=labels.to(dtype=torch.long)
            )
        }
    
    def reconstruction_loss(
        self,
        input,
        target,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        
        return {
            'reconstruction_loss': self.l2_loss(
                input=input,
                target=target
            )
        }

    def prep_batch(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict:
        batch_out = {}
        
        for key in batch:
            
            if (
                torch.is_tensor(batch[key])
                and key != 'labels'
            ):
                batch_out[key] = batch[key].to(torch.float)
            
            elif key == 'labels':
                batch_out[key] = batch['labels'].to(torch.int)

            else:
                batch_out[key] = torch.clone(batch[key])
        
        # dummy copy of inputs to be used in forward pass
        batch_out['inputs_embeds'] = torch.clone(batch_out['inputs'])
        
        return batch_out

    def _root_loss(
        self,
        inputs,
        outputs,
        attention_mask,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        attention_mask = torch.unsqueeze(attention_mask, -1).repeat(1,1,self.in_dim)
        
        return  self.reconstruction_loss(
            input=torch.masked_select(outputs, attention_mask.to(torch.bool)),
            target=torch.masked_select(inputs, attention_mask.to(torch.bool))
        )

    def loss(
        self,
        batch,
        outputs
        ) -> Dict[str, torch.tensor]:

        if self.is_decoding_mode:
            losses = self.decoding_loss(
                **batch,
                **outputs
            )
        
        else:
            losses = self._root_loss(
                **batch,
                **outputs
            )

        if 'loss' not in losses:
            losses['loss'] = sum(losses.values())

        return losses

# ===================== csm.py ================================ #

class CSMEmbedder(BaseEmbedder):
    
    def __init__(
        self,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.name = 'CSMEmbedder'
        self.training_style = 'CSM'
        assert self.training_style in {'CSM', 'decoding'}, f'{self.training_style} not supported'
        self._root_training_style = 'CSM'
        ##=========
        self.in_dim_for_mask = self.in_dim
        self.msk_embed = torch.nn.Parameter(
            torch.empty(
                size=(1, 1, self.in_dim_for_mask)
            )
        )
        self.cls_embed = torch.nn.Parameter(
            torch.empty(
                size=(1, 1, self.in_dim_for_mask)
            )
        )
        self._embeds = [
            self.msk_embed,
            self.cls_embed
        ]
        self._init_embeds()

    def _init_embeds(self):
        
        for embed in self._embeds:
            torch.nn.init.normal_(
                tensor=embed,
                mean=0.0,
                std=1.0,
            )

    def prep_batch(
        self,
        batch: Dict[str, torch.tensor],
        ) -> Dict[str, torch.tensor]:
        batch_out = dict(batch)
        labels =  torch.clone(batch['labels']) if 'labels' in batch else None

        if self.training_style != 'decoding':
            return self.mask_inputs(batch=batch_out)

        batch_out =  self.add_cls_embed(batch=batch_out)
        
        if labels is not None:
            batch_out['labels'] = labels
        
        return batch_out
        
    def mask_inputs(
        self,
        batch: Dict[str, torch.tensor],
        ) -> Dict[str, torch.tensor]:
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        assert inputs_key in batch, f'{inputs_key} not found in batch'
        input_shape = batch[inputs_key].size()
        device = batch[inputs_key].device
        masking_i = torch.cat(
            [
                torch.randint(
                    low=1, # at least one seq value before mask!
                    high=sum(batch['attention_mask'][i]==1), # high is exclusive, so this accounts for 0-indexing
                    size=(1,),
                    device=device
                )
                for i in range(input_shape[0])
            ],
            dim=0
        )
        print("masking id", masking_i)
        modelling_mask = torch.zeros_like(
            batch[inputs_key],
            device=device
        )
        modelling_mask[torch.arange(input_shape[0]), masking_i] = 1
        batch['modelling_mask'] = modelling_mask.to(torch.long)
        batch['masked_inputs'] = torch.masked_select(
            input=batch[inputs_key],
            mask=batch['modelling_mask'].to(torch.bool)
        ).detach().clone()
        batch['inputs_embeds'] = torch.where(
            batch['modelling_mask']==1,
            self.msk_embed.repeat(
                input_shape[0],
                input_shape[1],
                1
            ),
            batch[inputs_key].to(torch.float)
        )
        batch['attention_mask'] = torch.cat(
            [
                torch.cat(
                    (
                        torch.ones(
                            (
                                1,
                                i+1 # to account for 0-indexing in python
                            ),
                            device=device
                        ),
                        torch.zeros(
                            (
                                1,
                                input_shape[1]-i-1 # to account for 0-indexing in python
                            ),
                            device=device
                        )
                    ),
                    dim = 1
                )
                for i in masking_i
            ],
            dim = 0
        ).to(torch.long)
        # re-mask inputs
        attention_mask_expanded = torch.unsqueeze(
            batch['attention_mask'],
            dim=2
        ).repeat(
            1,
            1,
            self.in_dim_for_mask
        )
        batch["inputs_embeds"] = torch.where(
            attention_mask_expanded == 1,
            batch['inputs_embeds'],
            torch.zeros_like(batch['inputs_embeds'])
        )

        return batch

    def add_cls_embed(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict[str, torch.tensor]:
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        assert inputs_key in batch, f'{inputs_key} not found in batch'
        batch_size = batch[inputs_key].size()[0]
        sequence_lengths = batch['attention_mask'].sum(dim=1)
        inputs_embeds = []
        
        if 't_rs' in batch:
            t_rs = []
        
        for i in range(len(sequence_lengths)):
            inputs_embeds.append(
                torch.cat(
                    [
                        batch[inputs_key][i, :sequence_lengths[i], :],
                        self.cls_embed[0],
                        batch[inputs_key][i, sequence_lengths[i]:, :]
                    ],
                    dim=0
                )
            )
            
            if 't_rs' in batch:
                t_rs.append(
                    torch.cat(
                        [
                            batch['t_rs'][i, :sequence_lengths[i]],
                            torch.ones(1, device=batch['t_rs'].device) * -1,
                            batch['t_rs'][i, sequence_lengths[i]:]
                        ],
                        dim=0
                    )
                )

        batch['inputs_embeds'] = torch.stack(
            inputs_embeds,
            dim=0
        )

        if 't_rs' in batch:
            batch['t_rs'] = torch.stack(
                t_rs,
                dim=0
            )

        if 'token_type_ids' in batch:
            batch['token_type_ids'] = self._pad_tensor_left_by_n(
                tensor=batch['token_type_ids'],
                n=1,
                pad_value=0
            )

        if 'modelling_mask' in batch:
            batch['modelling_mask'] = self._pad_tensor_left_by_n(
                tensor=batch['modelling_mask'],
                n=1,
                pad_value=0
            )

        if 'attention_mask' in batch:
            batch['attention_mask'] = self._pad_tensor_left_by_n(
                tensor=batch['attention_mask'],
                n=1,
                pad_value=1
            )
        return batch

    def masking_loss(
        self,
        masked_inputs,
        outputs,
        modelling_mask
        ) -> Dict[str, torch.tensor]:
        
        return {
            'masking_loss': self.reconstruction_loss(
                input=torch.masked_select(outputs, modelling_mask.to(torch.bool)),
                target=masked_inputs
            )['reconstruction_loss']
        }

    def _root_loss(
        self,
        masked_inputs,
        outputs,
        modelling_mask,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        
        return self.masking_loss(
            masked_inputs=masked_inputs,
            outputs=outputs,
            modelling_mask=modelling_mask
        )
    
# ============================ make.py ================================

def make_embedder(
    architecture: str='GPT',
    training_style: str='CSM',
    in_dim: int=1024,
    embed_dim: int=768,
    num_hidden_layers: int=1,
    dropout: float=0.1,
    n_positions: int=512
    ) -> torch.nn.Module:
    """
    Make an embedder object.
    
    The embedder is used to prepare an input batch 
    (as generated by src.batcher) for training and 
    compute the model's training loss, given the 
    specified training style.

    Args:
    -----
    architecture: str
        The model architecture to use.
        One of: 'GPT', 'BERT', 'NetBERT', autoencoder',
        'PretrainedGPT', 'PretrainedBERT', 'LinearBaseline'.
    training_style: str
        The used training style (ie., framework).
        One of: 'BERT', 'CSM', 'NetBERT', 'autoencoder',
        'decoding'.
    in_dim: int
        The input dimension (ie., # networks) of the
        parcelated BOLD data.
    embed_dim: int
        The dimension of the used embedding space.
    num_hidden_layers: int
        The number of hidden layers of the embedding
        model. If more than one layers are used, all
        layers except the last one are activated through
        Gelu activation (see src.base.EmbeddingModel).
    dropout: float
        Dropout rate used emebdding model.
    n_positions: int
        The maximum number of sequence elements that
        the model can handle (in sequence elements).

    Core methods: 
    -----
    prep_batch(batch):  
        Makes all training-style specific edits of input batch 
        (as generated by src.batcher); 
        i.e., projection of input BOLD sequences into an 
        embedding space (as defined by embed_dim) 
        and addition of all training-style specific tokens to 
        the input data 
    
    loss(batch, outputs):
        Compute the training-style specific loss,
        given batch (as generated by prep_batch) and 
        the the full model's (see src.model) output 
        (as generated by model.forward) 

    switch_decoding_mode(is_decoding_mode):
        Switch the embedder to decoding mode (is_decoding_mode=True).
        This function is needed to adapt a pre-trained model
        to a downstream decoding task.
    """

    kwargs = {
        "in_dim": in_dim,
        "embed_dim": embed_dim,
        "num_hidden_layers": num_hidden_layers,
        "dropout": dropout,
        "n_positions": n_positions
    }

    embedder = CSMEmbedder(**kwargs)
    
    return embedder