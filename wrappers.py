'''
Wrapper classes of foundation model modules for use in main benchmarking script
'''
import math
import numpy as np
import torch
from skorch.callbacks import LRScheduler
from abc import ABC, abstractmethod
from skorch.helper import predefined_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class FinetuningWrapper(ABC):
    """
    Wrapper class for initializing model, fitting and evaluating on benchmark data, and storing results
    """
    def __init__(self):
        self.model = None
        self.results = {}

    @abstractmethod
    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        print("fit function not implemented")

    def size(self):
        """ Returns number of trainable parameters in model """
        if self.model is None:
            print("model not initialised")
        else:
            return self.model.size()

class EEGNetv1Wrapper(FinetuningWrapper):
    def __init__(self, n_chans, sfreq, n_times, n_outputs):
        super().__init__()
        from braindecode.models import EEGNetv1
        self.model = EEGNetv1(
            n_chans=n_chans, 
            n_times=n_times, 
            input_window_seconds=n_times//sfreq, 
            n_outputs=n_outputs
            )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        net = get_braindecode_net(
            self.model, 
            batch_size=batch_size, 
            train_split=predefined_split(validation_dataset)
            )
        net.fit(train_dataset, y=None, epochs=epochs)
        self.results['train_accuracy'] = net.history[:, 'train_accuracy']
        self.results['val_accuracy'] = net.history[:, 'valid_accuracy']
        self.results['train_bacc'] = net.history[:, 'train_balanced_accuracy']
        self.results['val_bacc'] = net.history[:, 'valid_balanced_accuracy']

    def size(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

class EEGInceptionWrapper(FinetuningWrapper):
    def __init__(self, n_chans, sfreq, n_times, n_outputs):
        super().__init__()
        from braindecode.models import EEGInception
        self.model = EEGInception(
            n_chans=n_chans, 
            n_times=n_times, 
            input_window_seconds=n_times//sfreq, 
            n_outputs=n_outputs,
            sfreq=sfreq
        )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        net = get_braindecode_net(
            self.model, 
            batch_size=batch_size, 
            train_split=predefined_split(validation_dataset)
            )
        net.fit(train_dataset, y=None, epochs=epochs)
        self.results['train_accuracy'] = net.history[:, 'train_accuracy']
        self.results['val_accuracy'] = net.history[:, 'valid_accuracy']
        self.results['train_bacc'] = net.history[:, 'train_balanced_accuracy']
        self.results['val_bacc'] = net.history[:, 'valid_balanced_accuracy']

    def size(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class LaBraMWrapper(FinetuningWrapper):
    def __init__(self, ch_names, sfreq, n_outputs, ckpt_path, train_head_only):
        super().__init__()
        from models import LaBraMModule
        self.model  = LaBraMModule(
            ch_names=ch_names,
            sfreq=sfreq, 
            n_outputs=n_outputs, 
            ckpt_path=ckpt_path, 
            train_head_only=train_head_only
            )
    
    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        self.model.fit(train_dataset, validation_dataset, batch_size, epochs)
        self.results = self.model.results


class EEGPTWrapper(FinetuningWrapper):
    def __init__(self, ch_names, n_outputs, n_times, ckpt_path, train_head_only):
        super().__init__()
        from models import EEGPTModule
        self.model = EEGPTModule(
            ch_names=ch_names,
            num_class=n_outputs, 
            input_length=n_times, 
            ckpt_path=ckpt_path, 
            train_head_only=train_head_only
            )
        
    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        trainer = pl.Trainer(accelerator='cuda',
                        precision='16-mixed',
                        max_epochs=epochs, 
                        min_epochs=epochs,
                        logger=None,
                        enable_checkpointing=False)    
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=8, shuffle=False)  
        
        steps_per_epoch = math.ceil(len(train_loader))
        self.model.set_training_params(steps_per_epoch, max_epochs=epochs)
        trainer.fit(self.model, train_loader, validation_loader)
        self.results = self.model.results


class NeuroGPTWrapper(FinetuningWrapper):
    def __init__(self, ch_names, n_outputs, encoder_only=False, ckpt_path=None, train_head_only=False):
        super().__init__()
        from models import NeuroGPTModule
        self.model = NeuroGPTModule(
            ch_names=ch_names, 
            n_outputs=n_outputs, 
            ckpt_path=ckpt_path, 
            encoder_only=encoder_only,
            train_head_only=train_head_only)

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        self.model.fit(train_dataset, validation_dataset, batch_size=batch_size, epochs=epochs)
        self.results = self.model.results
        self.results['train_accuracy'] = np.zeros_like(self.results['val_accuracy'])
        self.results['train_bacc'] = np.zeros_like(self.results['val_bacc'])


class CBraModWrapper(FinetuningWrapper):
    def __init__(self, ch_names, n_times, sfreq, n_outputs, ckpt_path, train_head_only):
        super().__init__()
        from models import CBraModModule
        self.model = CBraModModule(
            ch_names=ch_names, 
            n_times=n_times, 
            sfreq=sfreq, 
            n_outputs=n_outputs, 
            ckpt_path=ckpt_path, 
            train_head_only=train_head_only)

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        self.model.fit(train_dataset, validation_dataset, batch_size, epochs)
        self.results = self.model.results


class BIOTWrapper(FinetuningWrapper):
    def __init__(self, ch_names, n_outputs, ckpt_path, train_head_only):
        super().__init__()
        from models import BIOTModule
        self.model = BIOTModule(
            ch_names=ch_names,
            n_outputs=n_outputs, 
            ckpt_path=ckpt_path, 
            train_head_only=train_head_only
            )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        trainer = pl.Trainer(accelerator='cuda',
                        max_epochs=epochs, 
                        min_epochs=epochs,
                        logger=None,
                        enable_checkpointing=False,
                        num_sanity_val_steps=0,
                        benchmark=True)
        trainer.fit(self.model, train_dataloader, val_dataloader)
        self.results = self.model.results


class MIRepNetWrapper(FinetuningWrapper):
    def __init__(self, ch_names, n_outputs, sbj_ids, ckpt_path, train_head_only):
        super().__init__()
        from models import MIRepNetModule
        self.model = MIRepNetModule(
            ch_names=ch_names,
            n_output=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
        self.sbj_ids = sbj_ids

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        self.model.fit(train_dataset, validation_dataset, self.sbj_ids, batch_size, epochs)
        self.results = self.model.results


class LUNAWrapper(FinetuningWrapper):
    def __init__(self, ch_names, n_outputs, ckpt_path, train_head_only):
        super().__init__()
        from models import LUNAModule
        self.model = LUNAModule(
            ch_names=ch_names,
            n_out=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        self.model.fit(train_dataset, validation_dataset, batch_size, epochs)
        self.results = self.model.results


class REVEWrapper(FinetuningWrapper):
    def __init__(self, ch_names, sfreq, n_outputs, n_time, ckpt_path, train_head_only):
        from models import REVEModule
        self.model = REVEModule(
            ch_names=ch_names,
            sfreq=sfreq,
            n_outputs=n_outputs,
            n_times=n_time,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        trainer = pl.Trainer(accelerator='cuda',
                        precision='16-mixed',
                        max_epochs=epochs, 
                        min_epochs=epochs,
                        logger=None,
                        enable_checkpointing=False)    
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=8, shuffle=False)  
        
        # steps_per_epoch = math.ceil(len(train_loader))
        # self.model.set_training_params(steps_per_epoch, max_epochs=epochs)
        trainer.fit(self.model, train_loader, validation_loader)
        self.results = self.model.results



class HuBERTECGWrapper(FinetuningWrapper):
    def __init__(self, n_outputs, ckpt_path, train_head_only):
        super().__init__()
        from models import HuBERTECGModule
        self.model = HuBERTECGModule(
            n_outputs=n_outputs, 
            ckpt_path=ckpt_path, 
            train_head_only=train_head_only
            )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        self.model.fit(train_dataset, validation_dataset, batch_size, epochs)
        self.results = self.model.results


class ECGFounderWrapper(FinetuningWrapper):
    def __init__(self, n_outputs, ckpt_path, train_head_only):
        super().__init__()
        from models import ECGFounderModule
        self.model = ECGFounderModule(
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    
    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        self.model.fit(train_dataset, validation_dataset, batch_size, epochs)
        self.results = self.model.results


class ECG_FMWrapper(FinetuningWrapper):
    def __init__(self, n_outputs, ckpt_path, train_head_only):
        super().__init__()
        from models import ECG_FMModule
        self.model = ECG_FMModule(
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        self.model.fit(train_dataset, validation_dataset, batch_size, epochs)
        self.results = self.model.results


class NeuroRVQWrapper(FinetuningWrapper):
    def __init__(self, n_time, ch_names, n_outputs, ckpt_path, modality, train_head_only):
        super().__init__()
        from models import NeuroRVQModule
        self.model = NeuroRVQModule(
            sample_length=n_time, 
            chnames=ch_names, 
            n_out=n_outputs, 
            ckpt_path=ckpt_path, 
            modality=modality, 
            train_head_only=train_head_only
            )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        self.model.fit(train_dataset, validation_dataset, batch_size, epochs)
        self.results = self.model.results



def get_braindecode_net(model, lr=0.001, weight_decay=0, n_epochs=100, batch_size=64, train_split=None):
    from braindecode import EEGClassifier
    return EEGClassifier(
        model,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        train_split=train_split,
        callbacks=["accuracy", "balanced_accuracy",
                   ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1))],
        device="cuda",
        max_epochs=n_epochs
    )