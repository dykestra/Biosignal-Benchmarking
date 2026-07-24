import pdb
import os
import numpy as np
import json
import argparse
import skorch
from sklearn import model_selection
from torch.utils.data import Subset

from data import load_benchmark
from utils import get_logger, get_model, get_subdir

def perform_benchmarking(benchmarks, metrics, args):
    '''
    Performs full finetuning on all benchmarks for <n_epochs> with <n_splits> folds
    Returns validation metrics
    '''
    logger = get_logger(args)
    results = {}
    for (benchmark, n_outputs) in benchmarks:
        # Load data
        b = load_benchmark(benchmark, args.data_root, get_subdir(args.model_name), args.apply_car, n_outputs)
        X, sbj_id, y, ch_names = b.get_data()

        dataset = skorch.dataset.Dataset(X, y)
        sbj_id_unique = np.sort(np.unique(sbj_id))
        n, c, t = X.shape
        
        results[benchmark] = {}
        all_train_metrics = {m:[] for m in metrics}
        all_val_metrics = {m:[] for m in metrics}
        
        if args.predefined_splits:
            splits = b.get_splits(args.n_splits)
        else:
            kf = model_selection.KFold(n_splits=args.n_splits, shuffle=True, random_state=99)
            splits = kf.split(sbj_id_unique)

        for i_fold, (sbj_id_train, sbj_id_val) in enumerate(splits):
            if args.n_folds and i_fold >= args.n_folds:
                break
            print(f"FOLD {i_fold}...")
            # Split data by subjects
            ix_train = np.arange(n)[np.isin(sbj_id, sbj_id_unique[sbj_id_train])]
            ix_val = np.arange(n)[np.isin(sbj_id, sbj_id_unique[sbj_id_val])]

            if args.force_balanced_classes:
                ix_train = b.sample_balanced_set(ix_train, i_fold)

            train_dataset = Subset(dataset, ix_train[:args.end_idx])
            validation_dataset = Subset(dataset, ix_val[:args.end_idx])
            
            # Make model
            model = get_model(
                args.model_name, 
                n_chans=c, 
                ch_names=ch_names,
                sfreq=args.fs,
                n_times=t,
                n_outputs=n_outputs,
                sbj_ids=sbj_id,
                encoder_only=args.encoder_only,
                ckpt_path=args.ckpt_path,
                modality=args.modality,
                train_head_only=args.train_head_only
                )
            print(f"No. Trainable Parameters: {model.size()}")
            
            # Finetune model
            model.fit(
                train_dataset,
                validation_dataset,
                batch_size=args.batch_size,
                epochs=args.n_epochs
            )

            # Log fold results (per epoch)
            for m in metrics:
                fold_train_metrics = model.results[f'train_{m}']
                fold_val_metrics = model.results[f'val_{m}']
                all_train_metrics[m].append(fold_train_metrics)
                all_val_metrics[m].append(fold_val_metrics)
                for i in range(args.n_epochs):
                    logger.report_scalar(title=f"{args.model_name} {benchmark} {m}", series=f'train {i_fold}',
                            value=fold_train_metrics[i], iteration=i)
                    logger.report_scalar(title=f"{args.model_name} {benchmark} {m}", series=f'val {i_fold}',
                            value=fold_val_metrics[i], iteration=i)

        # Log mean across folds
        for m in metrics:
            for i in range(args.n_epochs):
                fold_mean_train = np.mean(np.array(all_train_metrics[m])[:,i])
                fold_mean_val = np.mean(np.array(all_val_metrics[m])[:,i])
                logger.report_scalar(title=f"{args.model_name} {benchmark} MEAN {m}", series=f'train',
                            value=fold_mean_train, iteration=i)
                logger.report_scalar(title=f"{args.model_name} {benchmark} MEAN {m}", series=f'val',
                            value=fold_mean_val, iteration=i)

        # Save to results
        print(f"{args.model_name} | {benchmark}")
        for m in metrics:
            mean_val = np.mean(np.array(all_val_metrics[m])[:,-1])
            print(f"{m} : {mean_val}")

            results[benchmark][f'{m}_mean'] = mean_val.item()
            results[benchmark][f'{m}_fold_vals'] = [v.item() for v in np.array(all_val_metrics[m])[:,-1]]

    return results

def perform_finetuning(benchmarks, metrics, args):
    '''
    Performs full finetuning on benchmarks using all data for training (no folds, no validation set)
    Saves finetuned model, no metrics returned
    '''
    logger = get_logger(args)
    results = {}
    for (benchmark, n_outputs) in benchmarks:
        # Load data
        b = load_benchmark(benchmark, args.data_root, get_subdir(args.model_name), args.apply_car, n_outputs)
        X, sbj_id, y, ch_names = b.get_data()

        n, c, t = X.shape
        dataset = skorch.dataset.Dataset(X[:args.end_idx], y[:args.end_idx])
        dummy_val = skorch.dataset.Dataset(np.array([X[0]]), np.array([y[0]]))  

        # Make model
        model = get_model(
            args.model_name, 
            n_chans=c, 
            ch_names=ch_names,
            sfreq=args.fs,
            n_times=t,
            n_outputs=n_outputs,
            sbj_ids=sbj_id,
            encoder_only=args.encoder_only,
            ckpt_path=args.ckpt_path,
            modality=args.modality,
            train_head_only=args.train_head_only
            )
        print(f"No. Trainable Parameters: {model.size()}")
        
        # Finetune model
        model.fit(
            dataset,
            dummy_val,
            batch_size=args.batch_size,
            epochs=args.n_epochs
        )

        # Log training results (per epoch)
        for m in metrics:
            results = model.results[f'train_{m}']
            for i in range(args.n_epochs):
                logger.report_scalar(title=f"{args.model_name} {benchmark} {m}", series=f'train',
                        value=results[i], iteration=i)

        # Save model
        out_path = f'{args.ex_id}_{benchmark}.pt'
        model.save_model(os.path.join(args.output_dir, out_path))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-splits', default=10, type=int, help="number of cross-fold validation splits")
    parser.add_argument('--predefined-splits', default=False, action="store_true", help="use preset fold splits")
    parser.add_argument('--n-folds', type=int, default=None, help="set only if number of folds is different to number of splits")
    parser.add_argument('--data-root', type=str, help="path to training data directory")
    parser.add_argument('--batch-size', default=64, type=int, help="batch size during training")
    parser.add_argument('--n-epochs', default=20, type=int, help="number of training epochs")
    parser.add_argument('--fs', default=200, type=int, help="sample frequency of eeg data (Hz)")
    parser.add_argument('--output-dir', default='results', type=str, help="output directory to save results")
    parser.add_argument("--end-idx", default=-1, type=int, help="max samples per epoch, used for debugging")
    parser.add_argument('--ex-id', default=0, type=int, help="experiment number")
    parser.add_argument('--apply-car', default=False, action="store_true", help="apply common average referencing to data")
    parser.add_argument('--encoder-only', default=False, action="store_true", help="finetune only the encoder part of the model (NeuroGPT)")
    parser.add_argument('--ckpt-path', default=None, type=str, help="specify custom checkpoint path for pretrained weights")
    parser.add_argument('--logger', default='clearml', type=str, help="where to log results", choices=['clearml', 'csv'])
    parser.add_argument('--train-head-only', default=False, action="store_true", help="freeze foundation model and train classification head only")
    parser.add_argument('--force-balanced-classes', default=False, action="store_true", help="randomly sample data to get balanced classes")
    parser.add_argument('--mode', default='benchmark', type=str, help="run mode", choices=['benchmark', 'finetune'])
    parser.add_argument('--modality', default='eeg', type=str, help="biosignal modality", choices=['eeg', 'ecg'])
    parser.add_argument('--model-name', default='EEGNet', type=str, help="name of model to be fine-tuned", 
                        choices=["EEGNet", "EEGInception", "LaBraM", "EEGPT", "NeuroGPT", "CBraMod", "BIOT", "MIRepNet", "HuBERTECG", "ECGFounder",
                                  "ECG-FM", "LUNA", "REVE", "NeuroRVQ"])
    args = parser.parse_args()

    # Select datasets (benchmark, n_classes)
    benchmarks = [
        ("High Gamma", 4),
        ("KU ERP", 2),
        ("Pavlov memory", 2),
        ("Sleep EDF", 6),
        ("Physionet MI", 2),
        # ("PTB-XL", 43)
    ]

    # Select evaluation metrics 
    # NOTE: metrics not included in this list will need to be implemented in the module for each model
    metrics = [
            "accuracy",
            "bacc",
            # "auroc",
            # "auprc"
        ]

    # Make output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Perform task
    if args.mode == 'benchmark':
        results = perform_benchmarking(benchmarks, metrics, args)
        
        # Print and save results
        print("="*30)
        print("Finetuning Complete!")
        print(f"Final Validation Results:")
        for b,_ in benchmarks:
            print(f"{b} | {[(m, results[b][f'{m}_mean']) for m in metrics]}")
        print("="*30)

        with open(os.path.join(args.output_dir, f'results_{args.ex_id}.txt'), 'w') as f: 
            f.write(json.dumps(results))
    elif args.mode == 'finetune':
        perform_finetuning(benchmarks, metrics, args)



        
