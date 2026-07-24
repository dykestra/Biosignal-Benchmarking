import pdb
import os
import sys
import torch
import warnings
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, BinaryRecall, BinarySpecificity, BinaryAUROC, MulticlassAUROC, MulticlassRecall, MulticlassConfusionMatrix
from torchmetrics import Metric
from torcheval.metrics import MulticlassAUPRC, BinaryAUPRC

# NeuroRVQ config (copied from flags_NeuroRVQ.yml and utils/functional.py)
# EEG_size= 1600
n_code= 8192
use_for_pretraining= False
in_chans_second_stage= 1
out_chans_second_stage= 8 # 8 (base), 16 (large), 32 (huge)
depth_second_stage= 12 # 12 (base), 24 (large), 48 (huge)
num_heads_second_stage= 10 # 10 (base), 16 (large), 16 (huge)
mlp_ratio_second_stage= 4.
qkv_bias_second_stage= True
drop_rate_second_stage= 0.
attn_drop_rate_second_stage= 0.
drop_path_rate_second_stage= 0.
init_values_second_stage= 1.e-5 # 0.1 (base), 1.e-5 (large), 1.e-6 (huge)
init_scale_second_stage= 0.001

lr = 5e-4
layer_decay = 0.975
weight_decay = 1e-2
amp_dtype=torch.bfloat16
warmup_epochs = 4

head_type = 'flatten'


class BinaryBalancedAccuracy(Metric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.recall = BinaryRecall(threshold=threshold)
        self.spec = BinarySpecificity(threshold=threshold)

    def update(self, preds, target):
        self.recall.update(preds, target)
        self.spec.update(preds, target)

    def compute(self):
        return (self.recall.compute() + self.spec.compute()) / 2
    
    def reset(self):
        self.recall.reset()
        self.spec.reset()

def get_args(modality):
    if modality == 'eeg':
        return {
            'patch_size': 200,
            'n_patches': 256,
            'embed_dim_second_stage': 200
        }
    elif modality == 'ecg':
        return {
            'patch_size': 40,
            'n_patches': 600,
            'embed_dim_second_stage': 40
        }
    else:
        print(f"Unknown modality {modality}")
    return {}

def get_ch_names(modality):
    if modality == 'eeg':
        return np.array([b'a1', b'a2', b'af3', b'af4', b'af7', b'af8', b'afz', b'c1', b'c2',
        b'c3', b'c4', b'c5', b'c6', b'ccp1', b'ccp2', b'ccp3', b'ccp4',
        b'ccp5', b'ccp6', b'ccp7', b'ccp8', b'cfc1', b'cfc2', b'cfc3',
        b'cfc4', b'cfc5', b'cfc6', b'cfc7', b'cfc8', b'cp1', b'cp2',
        b'cp3', b'cp4', b'cp5', b'cp6', b'cpz', b'cz', b'eog', b'f1',
        b'f10', b'f2', b'f3', b'f4', b'f5', b'f6', b'f7', b'f8', b'f9',
        b'fc1', b'fc2', b'fc3', b'fc4', b'fc5', b'fc6', b'fcz', b'fp1',
        b'fp2', b'fpz', b'ft7', b'ft8', b'fz', b'iz', b'loc', b'o1', b'o2',
        b'oz', b'p08', b'p1', b'p10', b'p2', b'p3', b'p4', b'p5', b'p6',
        b'p7', b'p8', b'p9', b'po1', b'po10', b'po2', b'po3', b'po4',
        b'po7', b'po8', b'po9', b'poz', b'pz', b'roc', b'sp1', b'sp2',
        b't1', b't10', b't2', b't3', b't4', b't5', b't6', b't7', b't8',
        b't9', b'tp10', b'tp7', b'tp8', b'tp9'])
    elif modality == 'ecg':
        return np.array([b'avf', b'avl', b'avr', b'i', b'ii', b'iii', b'v1', b'v2', b'v3',
       b'v4', b'v5', b'v6', b'vx', b'vy', b'vz'])
    else:
        print(f"Unknown modality {modality}")
        return []

def create_embedding_ix(n_time, max_n_patches, ch_names_sample, ch_names_global):
    """Creates temporal and spatial embedding indices for a sample with given regular shape.
    Args:
        n_time: Int. Number of patches along the time dimension
        max_n_patches: The maximum number of patches, for aligning the current time-point to the right.
        ch_names_sample (n_channels_sample,): The specific channel names of the sample
        ch_names_global (n_channels_global): The reference channel names of the model
    Returns:
        temp_embed_ix (1, n_patches): tensor
        spat_embed_ix (1, n_patches): tensor
    """

    # Temporal embedding ix
    temp_embed_ix = torch.arange(max_n_patches - n_time, max_n_patches)
    temp_embed_ix = temp_embed_ix.repeat(len(ch_names_sample))
    temp_embed_ix = temp_embed_ix.reshape(1, -1)

    # Spatial embedding ix
    spat_embed_ix = torch.tensor([np.where(ch_names_global == c)[0][0] for c in ch_names_sample])
    spat_embed_ix = torch.repeat_interleave(spat_embed_ix, n_time)
    spat_embed_ix = spat_embed_ix.reshape(1, -1)

    return temp_embed_ix, spat_embed_ix


def get_class_weights(y, n_cls, alpha=1.5):
    y = torch.as_tensor(y, dtype=torch.long)
    counts = torch.bincount(y, minlength=n_cls).float()

    # missing classes get weight 0
    weights = torch.zeros_like(counts)
    mask = counts > 0
    weights[mask] = 1.0 / (counts[mask] ** alpha)

    weights = weights / weights.sum()
    weights = weights * n_cls
    return weights.cuda()

class NeuroRVQModule():
    def __init__(self, sample_length, chnames, n_out, ckpt_path, modality, train_head_only):
        args = get_args(modality)
        self.patch_size = args['patch_size']
        self.n_patches = args['n_patches']

        self.n_time = sample_length // self.patch_size
        chnames = np.array([c.lower().encode() for c in chnames])
        self.ch_names_global = get_ch_names(modality)
        self.chmask = np.isin(chnames, self.ch_names_global)
        self.chnames = chnames[self.chmask]
        self.n_out = n_out

        if modality == "ecg":
            from NeuroRVQ_ECG import NeuroRVQFM
        else:
            from NeuroRVQ_EEG import NeuroRVQFM # choose EEG by default

        self.model = NeuroRVQFM(n_patches=self.n_patches,
                                    patch_size=self.patch_size,
                                    in_chans=in_chans_second_stage, out_chans=out_chans_second_stage,
                                    num_classes=0,
                                    embed_dim=args['embed_dim_second_stage'],
                                    depth=depth_second_stage,
                                    num_heads=num_heads_second_stage,
                                    mlp_ratio=mlp_ratio_second_stage, qkv_bias=qkv_bias_second_stage,
                                    qk_norm=partial(nn.LayerNorm, eps=1e-6), drop_rate=drop_rate_second_stage,
                                    attn_drop_rate=attn_drop_rate_second_stage,
                                    drop_path_rate=drop_path_rate_second_stage,
                                    init_values=init_values_second_stage,
                                    init_scale=init_scale_second_stage,
                                    n_global_electrodes=len(self.ch_names_global),
                                    use_as_encoder=True, vocab_size=n_code,
                                    use_for_pretraining=use_for_pretraining,
                                    cls_head=head_type)
        
        if ckpt_path is None:
            if modality == 'eeg':
                ckpt_path = "models/NeuroRVQm/ckpt/second_stage_model.pt"
            elif modality == 'ecg':
                ckpt_path = "models/NeuroRVQm/ckpt/ecg_second_stage_model.pt"

        model_state_dict = torch.load(ckpt_path)
        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
        print(f"Missing keys: {missing_keys},\nUnexpected keys: {unexpected_keys}")

        self.train_head_only = train_head_only
        self.criterion = F.cross_entropy if self.n_out > 2 else F.binary_cross_entropy_with_logits

        if self.n_out > 2:
            self.metrics = {
                    'accuracy' : MulticlassAccuracy(num_classes=self.n_out, average="micro"),
                    'bacc' : MulticlassRecall(num_classes=self.n_out, average='macro'),
                    'auroc' : MulticlassAUROC(num_classes=self.n_out),
                    'auprc' : MulticlassAUPRC(num_classes=self.n_out),
                }
        else:
            self.metrics = {
                'accuracy' : BinaryAccuracy(),
                'bacc' : BinaryBalancedAccuracy(),
                'auroc' : BinaryAUROC(),
                'auprc' : BinaryAUPRC()
            }

        self.results = {}
        for m in self.metrics.keys():
            self.results[f'train_{m}'] = []
            self.results[f'val_{m}'] = []

        
    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        d_out = self.n_out if self.n_out > 2 else 1
        self.model.reset_classifier(d_out, num_channels=len(self.chnames), n_time=self.n_time)
        self.model.cuda()
        # Set model parameter groups with layer_decay on the learning rate

        if self.train_head_only:
            for name, param in self.model.named_parameters():
                if 'head.' in name or 'fc_norm.' in name:
                    continue
                else:
                    param.requires_grad = False

        param_groups = {}
        for i_m, (p_name, param) in enumerate(self.model.named_parameters()):  # model layers
            if not param.requires_grad:
                continue
            if ('head.' in p_name) or ('fc_norm.' in p_name): # normal lr for classification head
                param_groups[p_name] = {'params': [param],
                                        'weight_decay': weight_decay,
                                        'lr': lr}
            else:
                param_groups[p_name] = {'params': [param],
                                        'weight_decay': weight_decay,
                                        'lr': lr * layer_decay ** (
                                                len(list(self.model.named_parameters())) - i_m)}

        # Optimizer and lr_scheduler
        optimizer = torch.optim.AdamW(list(param_groups.values()))
        n_batches_train = int(np.ceil(len(train_dataset) / batch_size))
        if epochs < warmup_epochs + 1 :
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, end_factor=1,
                                                        total_iters=epochs * n_batches_train)
        else:
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, end_factor=1,
                                                            total_iters=warmup_epochs * n_batches_train)
            scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-1,
                                                            total_iters=(epochs-warmup_epochs) * n_batches_train)
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2],
                                                                    milestones=[warmup_epochs * n_batches_train])
        warnings.filterwarnings('ignore', category=UserWarning, module='torch.optim.lr_scheduler')
        # Prepare automatic mixed precision training
        scaler = torch.amp.GradScaler('cuda')

        y_train = [ys for _,ys in train_dataset]
        y_val = [ys for _,ys in validation_dataset]
        y = y_train + y_val
        class_weights = get_class_weights(y, self.n_out, alpha=1)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        temp_embed_ix, spat_embed_ix = create_embedding_ix(self.n_time, self.n_patches,
                                                                            self.chnames, self.ch_names_global)
        
        # Loop over epochs
        for i_epoch in range(epochs):
            print(f"Epoch {i_epoch}")
            # Loop over training batches
            self.model.train()
            for name, metric in self.metrics.items():
                metric.reset()
            for x_b, y_b in tqdm(train_dataloader):
                x_b = x_b[:,self.chmask,:]
                n, c, t = x_b.shape
                x_b = x_b.reshape(n, c, self.n_time, self.patch_size).float().cuda()
                y_b = y_b.long() if self.n_out > 2 else y_b.float()
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    optimizer.zero_grad(set_to_none=True)
                    p, _ = self.model(x_b, temp_embed_ix, spat_embed_ix)
                    p = p.squeeze(-1)  # remove class dim if binary task
                    loss_weight = class_weights if p.ndim == 2 else class_weights[y_b.long()]
                    loss = self.criterion(p, y_b.cuda(), weight=loss_weight)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                # Collect class predictions to compute metrics on the full epoch
                p = p.detach().cpu().float()
                p = p if p.ndim == 2 else torch.round(torch.sigmoid(p))
                for name, metric in self.metrics.items():
                    metric.update(p, y_b)

            for name, metric in self.metrics.items():
                score = metric.compute()
                macro = score.mean()
                value = macro.detach().cpu().item()
                self.results[f'train_{name}'].append(value)
                print(f"Training {name} = {value}")

            # Loop over validation batches
            self.model.eval()
            confusion_matrix = MulticlassConfusionMatrix(num_classes=self.n_out).cuda()
            for name, metric in self.metrics.items():
                metric.reset()
            for x_b, y_b in tqdm(val_dataloader):
                x_b = x_b[:,self.chmask,:]
                n, c, t = x_b.shape
                x_b = x_b.reshape(n, c, self.n_time, self.patch_size).float().cuda()
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    p, _ = self.model(x_b, temp_embed_ix, spat_embed_ix)
                    p = p.squeeze(-1)  # remove class dim if binary task        

                if self.n_out > 2:
                    confusion_matrix.update(p.argmax(dim=1), y_b.cuda())

                # Collect class predictions to compute metrics on the full epoch
                p = p.detach().cpu().float()
                p = p if p.ndim == 2 else torch.round(torch.sigmoid(p))
                for name, metric in self.metrics.items():
                    metric.update(p, y_b)

            torch.set_printoptions(threshold=float("inf"), linewidth=200)
            print(confusion_matrix.compute())
            for name, metric in self.metrics.items():
                score = metric.compute()
                macro = score.mean()
                value = macro.detach().cpu().item()
                self.results[f'val_{name}'].append(value)
                print(f"Validation {name} = {value}")