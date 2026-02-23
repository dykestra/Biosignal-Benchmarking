import pdb
import random
import os
import math
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from functools import partial
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn

from . import EEGPT_mcae

def seed_torch(seed=99):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch()

torch.set_float32_matmul_precision('medium' )

# EEGPT hparams
MAX_LR = 4e-4

def mask_channels(ch_names):
    # mask out channels
    chmask = np.isin(ch_names, list(EEGPT_mcae.CHANNEL_DICT.keys()))
    ch_names = ch_names[chmask]
    return ch_names, chmask

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

def create_1d_absolute_sin_cos_embedding(pos_len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    # i矩阵
    i_matrix = torch.arange(dim//2, dtype=torch.float)
    i_matrix /= dim / 2
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    # pos矩阵
    pos_vec = torch.arange(pos_len).to(torch.long)
    # 矩阵相乘，pos变成列向量，i_matrix变成行向量
    out = pos_vec[:, None] @ i_matrix[None, :]
    # 奇/偶数列
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    # 赋值
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb

def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics,
                threshold=threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results

def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

class EEGPTModule(pl.LightningModule):

    def __init__(self, ch_names, num_class, input_length, ckpt_path, train_head_only):
        super().__init__()    
        self.ch_names, self.ch_mask = mask_channels(ch_names)
        self.chans_num = len(self.ch_names)
        self.num_class = num_class
        self.is_binary = (self.num_class < 3)
        self.input_length = input_length

        self.max_lr = MAX_LR
        self.steps_per_epoch = 0
        self.max_epochs = 0

        self.train_head_only=train_head_only

        # init model
        target_encoder = EEGPT_mcae.EEGTransformer(
            img_size=[self.chans_num, input_length],
            patch_size=32*2,
            # patch_stride = 32,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        self.target_encoder = target_encoder
        self.chans_id       = target_encoder.prepare_chan_ids(self.ch_names)
        
        # -- load checkpoint
        if ckpt_path is None:
            ckpt_path = "models/EEGPT/ckpt/eegpt_mcae_58chs_4s_large4E.ckpt"
        pretrain_ckpt = torch.load(ckpt_path, weights_only=False)
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
                
        self.target_encoder.load_state_dict(target_encoder_stat)

        self.linear_probe1   = LinearWithConstraint(2048, 64, max_norm=1)
        self.drop            = torch.nn.Dropout(p=0.50)        
        self.decoder         = torch.nn.TransformerDecoder(
                                    decoder_layer=torch.nn.TransformerDecoderLayer(64, 4, 64*4, activation=torch.nn.functional.gelu, batch_first=False),
                                    num_layers=4
                                )
        self.cls_token =        torch.nn.Parameter(torch.rand(1,1,64)*0.001, requires_grad=True)
        self.linear_probe2   = LinearWithConstraint(64, self.num_class, max_norm=0.25)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
    
        self.running_scores = {"train":[], "valid":[]}
        self.results = {'train_accuracy':[], 'train_bacc':[], 'val_accuracy':[], 'val_bacc':[]}
        self.is_sanity = True
        
    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_training_params(self, steps_per_epoch, max_epochs):
        """ Set hyperparameters for optimizers """
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs=max_epochs

    def forward(self, x):
        # pdb.set_trace()
        x = x[:, self.ch_mask, :] 
        # x = x / 1000 # convert to mV
        B, C, T = x.shape

        x = temporal_interpolation(x, self.input_length)
        z = self.target_encoder(x, self.chans_id.to(x))
        
        h = z.flatten(2)
        
        h = self.linear_probe1(self.drop(h))
        pos = create_1d_absolute_sin_cos_embedding(h.shape[1], dim=64)
        h = h + pos.repeat((h.shape[0], 1, 1)).to(h)
        
        h = torch.cat([self.cls_token.repeat((h.shape[0], 1, 1)).to(h.device), h], dim=1)
        h = h.transpose(0,1)
        h = self.decoder(h, h)[0,:,:]
        
        h = self.linear_probe2(h)
        return x, h
    
    # TRAINING
    def on_train_epoch_start(self):
        self.running_scores['train']=[]
        return super().on_train_epoch_start()
    def on_train_epoch_end(self):
        label = torch.cat([l for l,_ in self.running_scores['train']], dim=0)
        y_score = torch.cat([y for _,y in self.running_scores['train']], dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, self.is_binary)
        
        self.results['train_accuracy'].append(results['accuracy'])
        self.results['train_bacc'].append(results['balanced_accuracy'])
        
        return super().on_train_epoch_end()
    def training_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)

        if self.is_binary:
            y_score = torch.softmax(logit, dim=-1)[:,1]
        else:
            y_score = logit
            
        self.running_scores['train'].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

        return loss
        
    # VALIDATION
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label = torch.cat([l for l,_ in self.running_scores['valid']], dim=0)
        y_score = torch.cat([y for _,y in self.running_scores['valid']], dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy"] 
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, self.is_binary)
        
        self.results['val_accuracy'].append(results['accuracy'])
        self.results['val_bacc'].append(results['balanced_accuracy'])
        
        return super().on_validation_epoch_end()
    def validation_step(self, batch, batch_idx):
        # pdb.set_trace()
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)

        if self.is_binary:
            y_score = torch.softmax(logit, dim=-1)[:,1]
        else:
            y_score = logit

        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        
        return loss
    
    def configure_optimizers(self):
        trainable_parameters = (list(self.linear_probe1.parameters()) +
                                list(self.linear_probe2.parameters()) +
                                [self.cls_token] + list(self.decoder.parameters()))
        
        if self.train_head_only:
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        else:
            trainable_parameters += list(self.target_encoder.parameters()) 

        optimizer = torch.optim.AdamW(trainable_parameters, weight_decay=0.01)
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr, steps_per_epoch=self.steps_per_epoch, epochs=self.max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )