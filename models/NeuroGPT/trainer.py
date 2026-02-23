#!/usr/bin/env python3
import pdb
from typing import Dict, List, Optional, Tuple
# from apex import amp
import torch
from transformers import Trainer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from transformers import TrainingArguments
from transformers.trainer_callback import (
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    seed_worker
)

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class Trainer(Trainer):
    def __init__(
        self,
        is_deepspeed: bool = False,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.name = "Trainer"
        self.is_deepspeed = is_deepspeed
        self.mat1 = None
        self.mat2 = None
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        if isinstance(train_dataset, torch.utils.data.IterableDataset):

            print("iterable dataset")
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )

        train_sampler = self._get_train_sampler()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        return train_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                test_dataset = IterableDatasetShard(
                    test_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def prediction_step(
        self,
        model,
        batch,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List[str]] = None
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        batch = self._move_batch_to_device(batch=batch)
        
        with torch.no_grad():
            (loss, outputs) = self.compute_loss(
                model=model,
                batch=batch,
                return_outputs=True
            )

        if not prediction_loss_only and 'labels' in batch:
            return (loss, outputs['decoding_logits'], batch['labels'])
        
        else:
            return (loss, outputs, None)

    def compute_loss(
        self,
        model,
        batch,
        return_outputs=False,
        **kwargs
        ):
        batch = self._move_batch_to_device(batch=batch)

        if isinstance(
            model,
            (
                torch.nn.DataParallel, 
                torch.nn.parallel.DistributedDataParallel
            )
        ) or self.is_deepspeed:
            (losses, outputs) = model.module.compute_loss(
                batch=batch,
                return_outputs=True
            )
        
        else:
            (losses, outputs) = model.compute_loss(
                batch=batch,
                return_outputs=True
            )
        
        loss = losses['loss'] if 'loss' in losses.keys() else sum(losses.values())
        
        return (loss, outputs) if return_outputs else loss

    def _move_batch_to_device(
        self,
        batch
        ) -> Dict[str, torch.tensor]:
        batch = self._prepare_inputs(batch)
        
        if "labels" in batch:
            batch["labels"] = batch["labels"].to(torch.long).to(batch["inputs"].device)
        
        return self._prepare_inputs(batch)

# ============== trainer/make.py ==========================


def _cat_data_collator(features: List) -> Dict[str, torch.tensor]:

    if not isinstance(features[0], dict):
        features = [vars(f) for f in features] 

    return {
        k: torch.cat(
            [
                f[k]
                for f in features
            ]
        )
        for k in features[0].keys()
        if not k.startswith('__')
    }


def decoding_accuracy_metrics(eval_preds):
    preds, labels = eval_preds
    preds = preds.argmax(axis=-1)
    accuracy = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    return {
        "accuracy": round(accuracy, 3),
        "bacc": round(bacc, 3)
    }


def make_trainer(
    model_init,
    training_style,
    train_dataset,
    validation_dataset,
    do_train: bool = True,
    do_eval: bool = True,
    run_name: str = None,
    output_dir: str = None,
    overwrite_output_dir: bool = True,
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    optim: str='adamw_hf',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    adam_beta1: float=0.9,
    adam_beta2: float=0.999,
    adam_epsilon: float=1e-8,
    max_grad_norm: float=1.0,
    per_device_train_batch_size: int = 64,
    per_device_eval_batch_size: int = 64,
    dataloader_num_workers: int = 0,
    num_train_epochs: int = 1,
    lr_scheduler_type: str = 'linear',
    warmup_ratio: float = 0.01,
    evaluation_strategy: str = 'epoch',
    prediction_loss_only: bool = False,
    logging_strategy: str = 'epoch',
    save_strategy: str = 'steps',
    save_total_limit: int = 5,
    save_steps: int = 10000,
    eval_steps: int = None,
    logging_first_step: bool = True,
    greater_is_better: bool = True,
    seed: int = 1,
    fp16: bool = True,
    deepspeed: str = None,
    compute_metrics = None,
    **kwargs
    ) -> Trainer:
    """
    Make a Trainer object for training a model.
    Returns an instance of transformers.Trainer.
    
    See the HuggingFace transformers documentation for more details
    on input arguments:
    https://huggingface.co/transformers/main_classes/trainer.html

    Custom arguments:
    ---
    model_init: callable
        A callable that does not require any arguments and 
        returns model that is to be trained (see scripts.train.model_init)
    training_style: str
        The training style (ie., framework) to use.
        One of: 'BERT', 'CSM', 'NetBERT', 'autoencoder',
        'decoding'.
    train_dataset: src.batcher.dataset
        The training dataset, as generated by src.batcher.dataset
    validation_dataset: src.batcher.dataset
        The validation dataset, as generated by src.batcher.dataset

    Returns
    ----
    trainer: transformers.Trainer
    """
    trainer_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        do_train=do_train,
        do_eval=do_eval,
        overwrite_output_dir=overwrite_output_dir,
        prediction_loss_only=prediction_loss_only,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        optim=optim,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        greater_is_better=greater_is_better,
        save_steps=save_steps,
        logging_strategy=logging_strategy,
        logging_first_step=logging_first_step,
        evaluation_strategy=evaluation_strategy,
        seed=seed,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        deepspeed=deepspeed,
        report_to=["none"],
        **kwargs
    )

    data_collator = _cat_data_collator
    is_deepspeed = deepspeed is not None
    compute_metrics = decoding_accuracy_metrics if training_style=='decoding' and compute_metrics is None else compute_metrics

    trainer = Trainer(
        args=trainer_args,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        is_deepspeed=is_deepspeed
    )

    return trainer