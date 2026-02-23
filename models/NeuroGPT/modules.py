import pdb
from typing import Dict

from . import model
from . import trainer
from . import batcher

from torch import manual_seed
manual_seed(1234)

CONFIG = {
  "parcellation_dim": 1024,
  "embedding_dim": 1024,
  "num_hidden_layers_embedding_model": 1,
  "freeze_embedder": False,
  "num_hidden_layers_unembedding_model": 1,
  "freeze_unembedder": False,
  "architecture": "GPT",
  "num_hidden_layers": 6,
  "num_attention_heads": 16,
  "intermediate_dim_factor": 4,
  "hidden_activation": "gelu_new",
  "freeze_decoder": False,
  "freeze_decoder_without_pooler_heads": False,
  "training_style": "decoding",
  "decoding_target": None,
  "optim": "adamw_hf",
  "learning_rate": 0.0001,
  "warmup_ratio": 0.01,
  "weight_decay": 0.1,
  "adam_beta_1": 0.9,
  "adam_beta_2": 0.999,
  "adam_epsilon": 1e-08,
  "max_grad_norm": 1.0,
  "lr_scheduler": "linear",
  "dropout": 0.1,
  "log_dir": "results",
  "log_every_n_steps": 1000,
  "run_name": "dst-0",
  "fp16": True,
  "deepspeed": None,
  "local_rank": 0,
  "num_workers": 0,
  "plot_model_graph": False,
  "smoke_test": False,
  "bold_dummy_mode": False,
  "do_train": True,
  "n_positions": 512,
  "chunk_len": 500,
  "num_chunks": 2,
  "chunk_ovlp": 250,
  "sampling_rate": 250,
  "fold_i": 0,
  "use_encoder": True,
  "do_normalization": True,
  "filter_time_length": 25,
  "pool_time_length": 75,
  "stride_avg_pool": 15,
  "n_filters_time": 40,
  "num_encoder_layers": 6,
  "freeze_encoder": False,
  "ft_only_encoder": False,
  "pretrained_model": ""
}

def model_init(params: Dict=None):
    model_config = dict(CONFIG)
    if params is not None:
        model_config |= params

    return model.make_model(model_config)

class NeuroGPTModule():
    def __init__(self, ch_names, n_outputs, ckpt_path, encoder_only=False, train_head_only=False):
        if encoder_only:
            CONFIG['ft_only_encoder'] = True
            CONFIG['freeze_decoder'] = True
            CONFIG['freeze_embedder'] = True
            CONFIG['freeze_unembedder'] = True
            
        if train_head_only:
            CONFIG['freeze_encoder'] = True
            CONFIG['freeze_embedder'] = True
            CONFIG['freeze_unembedder'] = True
            if not encoder_only:
                CONFIG['freeze_decoder_without_pooler_heads'] = True

        self.ch_names = ch_names
        CONFIG['num_decoding_classes'] = n_outputs
        
        if ckpt_path is None:
            ckpt_path = "models/NeuroGPT/ckpt/pytorch_model.bin"
        CONFIG['pretrained_model'] = ckpt_path

        self.results = {}

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        train_eeg, train_labels = train_dataset[:]
        val_eeg, val_labels = validation_dataset[:]
        _train_dataset = batcher.EEGDataset(train_eeg, train_labels, chnames=self.ch_names)
        _validation_dataset = batcher.EEGDataset(val_eeg, val_labels, chnames=self.ch_names)

        self.trainer = trainer.make_trainer(
            model_init=model_init,
            training_style=CONFIG["training_style"],
            run_name=CONFIG["run_name"],
            output_dir=".",
            train_dataset=_train_dataset,
            validation_dataset=_validation_dataset,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=CONFIG["num_workers"],
            optim=CONFIG["optim"],
            learning_rate=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
            adam_beta1=CONFIG["adam_beta_1"],
            adam_beta2=CONFIG["adam_beta_2"],
            adam_epsilon=CONFIG["adam_epsilon"],
            max_grad_norm=CONFIG["max_grad_norm"],
            lr_scheduler_type=CONFIG["lr_scheduler"],
            warmup_ratio=CONFIG["warmup_ratio"],
            num_train_epochs=epochs,
            save_strategy="no",
            seed=1234,
            fp16=CONFIG["fp16"],
            deepspeed=CONFIG["deepspeed"],
        )
        self.trainer.train()

        # TODO: implement train metrics
        self.results['val_accuracy'] = [x['eval_accuracy'] for x in self.trainer.state.log_history if 'eval_accuracy' in x]
        self.results['val_bacc'] = [x['eval_bacc'] for x in self.trainer.state.log_history if 'eval_accuracy' in x]

    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in model_init().parameters() if p.requires_grad)
