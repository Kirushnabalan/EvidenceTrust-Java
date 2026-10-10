from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.semantic.dataset import MegaVulSemanticDataset
from src.semantic.model import SemanticVulnerabilityModel



MODEL_NAME = 'microsoft/codebert-base'
TRAIN_FILE = PROJECT_ROOT / 'dataset' / 'processed' / 'model_inputs' / 'train.jsonl'
VALIDATION_FILE = PROJECT_ROOT / 'dataset' / 'processed' / 'model_inputs' / 'validation.jsonl'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'semantic'
DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def calculate_class_weights(dataset: MegaVulSemanticDataset) -> torch.Tensor:
    counts = dataset.class_counts()
    count_0 = int(counts[0])
    count_1 = int(counts[1])
    if count_0 <= 0:
        raise ValueError('Training data contains no non-vulnerable samples.')
    if count_1 <= 0:
        raise ValueError('Training data contains no vulnerable samples.')
    total = count_0 + count_1
    number_of_classes = 2
    weight_0 = total / (number_of_classes * count_0)
    weight_1 = total / (number_of_classes * count_1)

    return torch.tensor([weight_0, weight_1], dtype=torch.float32)


def make_stratified_subset(dataset: MegaVulSemanticDataset, max_samples: int, seed: int) -> Subset:
    if max_samples <= 1:
        raise ValueError('max_samples must be greater than 1.')
    if max_samples >= len(dataset):
        return Subset(dataset, list(range(len(dataset))))
    negative_indices: List[int] = []
    positive_indices: List[int] = []


    for index, record in enumerate(dataset.records):
        label = int(record['is_vul'])
        if label == 0:
            negative_indices.append(index)
        elif label == 1:
            positive_indices.append(index)
        else:
            raise ValueError(f'Unexpected label: {label}')

        
    if not negative_indices:
        raise ValueError('No class-0 records available.')
    if not positive_indices:
        raise ValueError('No class-1 records available.')

    
    rng = random.Random(seed)
    rng.shuffle(negative_indices)
    rng.shuffle(positive_indices)
    positive_target = max(1, max_samples // 4)
    positive_target = min(positive_target, len(positive_indices))
    negative_target = max_samples - positive_target
    negative_target = min(negative_target, len(negative_indices))
    selected = positive_indices[:positive_target] + negative_indices[:negative_target]


    if len(selected) < max_samples:
        already_selected = set(selected)
        remaining_candidates = negative_indices + positive_indices
        for index in remaining_candidates:
            if index in already_selected:
                continue
            selected.append(index)
            already_selected.add(index)
            if len(selected) >= max_samples:
                break
    rng.shuffle(selected)
    return Subset(dataset, selected)

def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float=0.5) -> Dict[str, object]:

    if labels.ndim != 1:
        raise ValueError('labels must be one-dimensional.')
    if probabilities.ndim != 1:
        raise ValueError('probabilities must be one-dimensional.')
    if len(labels) != len(probabilities):
        raise ValueError('labels and probabilities have different lengths.')

    
    predictions = (probabilities >= threshold).astype(np.int64)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    mcc = matthews_corrcoef(labels, predictions)
    pr_auc = average_precision_score(labels, probabilities)

    if len(np.unique(labels)) == 2:
        roc_auc = roc_auc_score(labels, probabilities)
    else:
        roc_auc = float('nan')

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {'threshold': float(threshold), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'mcc': float(mcc), 'pr_auc': float(pr_auc), 'roc_auc': float(roc_auc), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def encoder_is_frozen(model: SemanticVulnerabilityModel) -> bool:

    return not any((parameter.requires_grad for parameter in model.encoder.parameters()))

def create_train_loader(dataset, batch_size: int, num_workers: int, seed: int, epoch: int, pin_memory: bool) -> DataLoader:

    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator, pin_memory=pin_memory)

def train_one_epoch(model: SemanticVulnerabilityModel, loader: DataLoader, optimizer: AdamW, loss_function: nn.Module, device: torch.device, gradient_accumulation_steps: int) -> float:
    
    model.train()
    if encoder_is_frozen(model):
        model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    processed_batches = 0
    number_of_batches = len(loader)
   
    if number_of_batches <= 0:
        raise RuntimeError('Training loader has zero batches.')
    remainder = number_of_batches % gradient_accumulation_steps
    if remainder > 0:
        final_group_start = number_of_batches - remainder + 1
    else:
        final_group_start = number_of_batches + 1
   
    for batch_index, batch in enumerate(loader, start=1):
        
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        chunk_mask = batch['chunk_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, chunk_mask=chunk_mask)
        logits = outputs['logits']
        loss = loss_function(logits, labels)
        
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite training loss at batch {batch_index}: {loss.item()}')
        raw_loss = float(loss.detach().cpu().item())
        total_loss += raw_loss
        processed_batches += 1
        
        if remainder > 0 and batch_index >= final_group_start:
            accumulation_divisor = remainder
        else:
            accumulation_divisor = gradient_accumulation_steps
        
        scaled_loss = loss / accumulation_divisor
        scaled_loss.backward()
        in_regular_group = batch_index < final_group_start
        regular_group_complete = in_regular_group and batch_index % gradient_accumulation_steps == 0
        final_batch = batch_index == number_of_batches
        
        if regular_group_complete or final_batch:
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        if batch_index == 1 or batch_index % 10 == 0 or final_batch:
            print(f'  batch {batch_index:,}/{number_of_batches:,} | loss={raw_loss:.6f}')
    
    if processed_batches <= 0:
        raise RuntimeError('No training batches were processed.')
    return total_loss / processed_batches

@torch.no_grad()
def evaluate(model: SemanticVulnerabilityModel, loader: DataLoader, loss_function: nn.Module, device: torch.device) -> Tuple[float, Dict[str, object]]:
   
    model.eval()
    total_loss = 0.0
    processed_batches = 0
    all_labels: List[int] = []
    all_probabilities: List[float] = []
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        chunk_mask = batch['chunk_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, chunk_mask=chunk_mask)
        logits = outputs['logits']
        loss = loss_function(logits, labels)
        
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite validation loss encountered.')
        total_loss += float(loss.detach().cpu().item())
        processed_batches += 1
        probabilities = model.probabilities_from_logits(logits)
        vulnerable_probabilities = probabilities[:, 1]
        all_labels.extend(labels.detach().cpu().tolist())
        all_probabilities.extend(vulnerable_probabilities.detach().cpu().tolist())
    
    if processed_batches <= 0:
        raise RuntimeError('Validation loader produced zero batches.')
    labels_array = np.asarray(all_labels, dtype=np.int64)
    probability_array = np.asarray(all_probabilities, dtype=np.float64)
    metrics = calculate_metrics(labels=labels_array, probabilities=probability_array, threshold=0.5)
    mean_loss = total_loss / processed_batches
    return (mean_loss, metrics)

def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / (path.name + '.tmp')
    torch.save(payload, temporary_path)
    temporary_path.replace(path)

def save_best_checkpoint(path: Path, model: SemanticVulnerabilityModel, epoch: int, best_pr_auc: float, metrics: Dict[str, object], training_config: Dict[str, object]) -> None:
   
    payload = {'checkpoint_type': 'best_model', 'epoch': int(epoch), 'model_name': MODEL_NAME, 'model_state_dict': model.state_dict(), 'best_pr_auc': float(best_pr_auc), 'validation_metrics': metrics, 'training_config': training_config}
    atomic_torch_save(payload, path)

def save_resume_checkpoint(path: Path, model: SemanticVulnerabilityModel, optimizer: AdamW, epoch: int, best_pr_auc: float, metrics: Dict[str, object], history: List[Dict[str, object]], training_config: Dict[str, object]) -> None:
    
    payload = {'checkpoint_type': 'resume', 'epoch': int(epoch), 'model_name': MODEL_NAME, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_pr_auc': float(best_pr_auc), 'validation_metrics': metrics, 'history': history, 'training_config': training_config}
    atomic_torch_save(payload, path)

def load_resume_checkpoint(checkpoint_path: Path, model: SemanticVulnerabilityModel, optimizer: AdamW, device: torch.device, current_freeze_encoder: bool) -> Tuple[int, float, List[Dict[str, object]]]:
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    print('Loading resume checkpoint:')
    print(checkpoint_path)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_type = checkpoint.get('checkpoint_type')
    
    if checkpoint_type not in {None, 'resume'}:
        raise ValueError('The supplied checkpoint is not a resume checkpoint.')
    
    if 'optimizer_state_dict' not in checkpoint:
        raise ValueError('Resume checkpoint does not contain optimizer state.')
    checkpoint_model_name = checkpoint.get('model_name', MODEL_NAME)
    
    if checkpoint_model_name != MODEL_NAME:
        raise ValueError('Checkpoint model differs from the configured model.')
    previous_config = checkpoint.get('training_config', {})
    previous_freeze_encoder = bool(previous_config.get('freeze_encoder', current_freeze_encoder))
    
    if previous_freeze_encoder != current_freeze_encoder:
        raise ValueError('Cannot resume with a different --freeze-encoder setting.')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    completed_epoch = int(checkpoint['epoch'])
    
    best_pr_auc = float(checkpoint.get('best_pr_auc', float('-inf')))
    
    history = list(checkpoint.get('history', []))
    
    next_epoch = completed_epoch + 1
    
    print('Checkpoint loaded.')
    print('Completed epoch:', completed_epoch)
    print('Next epoch:', next_epoch)
    print('Best PR-AUC:', f'{best_pr_auc:.6f}')
    
    return (next_epoch, best_pr_auc, history)

def save_json(path: Path, data) -> None:
    
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / (path.name + '.tmp')
    
    with temporary_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=True)
    temporary_path.replace(path)

def print_metrics(metrics: Dict[str, object]) -> None:
    
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  MCC:       {metrics['mcc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print('  Confusion matrix:')
    print(f"    TN={metrics['tn']} FP={metrics['fp']}")
    print(f"    FN={metrics['fn']} TP={metrics['tp']}")

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description='Train EvidenceTrust semantic vulnerability detector.')
    parser.add_argument('--epochs', type=int, default=1, help='Total number of epochs. When resuming, this remains the target total epoch count.')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-05)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--smoke', action='store_true', help='Use small class-aware train and validation subsets.')
    parser.add_argument('--smoke-train-size', type=int, default=32)
    parser.add_argument('--smoke-validation-size', type=int, default=32)
    parser.add_argument('--freeze-encoder', action='store_true', help='Freeze CodeBERT and train only the function-level classification head.')
    parser.add_argument('--resume', type=str, default=None, help='Path to latest_checkpoint.pt.')
    parser.add_argument('--run-name', type=str, default=None, help='Name of the directory under outputs/semantic/.')
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    if args.epochs <= 0:
        raise ValueError('epochs must be positive.')
    if args.batch_size <= 0:
        raise ValueError('batch-size must be positive.')
    if args.gradient_accumulation <= 0:
        raise ValueError('gradient-accumulation must be positive.')
    if args.learning_rate <= 0:
        raise ValueError('learning-rate must be positive.')
    if args.weight_decay < 0:
        raise ValueError('weight-decay cannot be negative.')
    if args.num_workers < 0:
        raise ValueError('num-workers cannot be negative.')
    if args.smoke:
        if args.smoke_train_size <= 1:
            raise ValueError('smoke-train-size must be > 1.')
        if args.smoke_validation_size <= 1:
            raise ValueError('smoke-validation-size must be > 1.')
    set_seed(args.seed)
    
    device = get_device()
    
    pin_memory = device.type == 'cuda'
    
    print('-- SEMANTIC TRAINING --')
    print('Device:', device)
    print('PyTorch:', torch.__version__)
    print('CPU threads:', torch.get_num_threads())
    print('Model:', MODEL_NAME)
    print('Smoke mode:', args.smoke)
    print('Freeze encoder:', args.freeze_encoder)
    print()
    
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print('Loading datasets...')
    train_dataset = MegaVulSemanticDataset(TRAIN_FILE, tokenizer)
    validation_dataset = MegaVulSemanticDataset(VALIDATION_FILE, tokenizer)
    
    print('Train records:', len(train_dataset))
    print('Train classes:', train_dataset.class_counts())
    print('Validation records:', len(validation_dataset))
    print('Validation classes:', validation_dataset.class_counts())
    class_weights_cpu = calculate_class_weights(train_dataset)
    print()
    
    print('Class weights:')
    print('  non-vulnerable:', f'{class_weights_cpu[0].item():.6f}')
    print('  vulnerable:', f'{class_weights_cpu[1].item():.6f}')
    if args.smoke:
        active_train_dataset = make_stratified_subset(train_dataset, args.smoke_train_size, args.seed)
        active_validation_dataset = make_stratified_subset(validation_dataset, args.smoke_validation_size, args.seed + 1)
    else:
        active_train_dataset = train_dataset
        active_validation_dataset = validation_dataset
    print()
    
    print('Active training samples:', len(active_train_dataset))
    print('Active validation samples:', len(active_validation_dataset))
    validation_loader = DataLoader(active_validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    print()
    
    print('Loading semantic model...')
    model = SemanticVulnerabilityModel(model_name=MODEL_NAME)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        model.encoder.eval()
        print('CodeBERT encoder frozen.')
    model.to(device)
    total_parameters = sum((parameter.numel() for parameter in model.parameters()))
    trainable_parameters = sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    print('Total parameters:', f'{total_parameters:,}')
    print('Trainable parameters:', f'{trainable_parameters:,}')
    if trainable_parameters <= 0:
        raise RuntimeError('Model has no trainable parameters.')
    class_weights = class_weights_cpu.to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    if args.run_name:
        run_name = args.run_name
    elif args.resume:
        run_name = Path(args.resume).resolve().parent.name
    elif args.smoke:
        run_name = 'smoke_test'
    elif args.freeze_encoder:
        run_name = 'frozen_encoder'
    else:
        run_name = 'full_finetune'
    
    output_directory = OUTPUT_ROOT / run_name
    output_directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_directory / 'tokenizer')
    training_config = {'model_name': MODEL_NAME, 'epochs': args.epochs, 'batch_size': args.batch_size, 'gradient_accumulation': args.gradient_accumulation, 'effective_batch_size': args.batch_size * args.gradient_accumulation, 'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay, 'seed': args.seed, 'num_workers': args.num_workers, 'freeze_encoder': args.freeze_encoder, 'smoke': args.smoke, 'train_samples': len(active_train_dataset), 'validation_samples': len(active_validation_dataset), 'class_weight_non_vulnerable': float(class_weights_cpu[0].item()), 'class_weight_vulnerable': float(class_weights_cpu[1].item()), 'model_selection_metric': 'validation_pr_auc', 'decision_threshold': 0.5, 'test_used': False}
    save_json(output_directory / 'training_config.json', training_config)
    start_epoch = 1
    best_pr_auc = float('-inf')
    history: List[Dict[str, object]] = []
    
    if args.resume:
        start_epoch, best_pr_auc, history = load_resume_checkpoint(checkpoint_path=Path(args.resume), model=model, optimizer=optimizer, device=device, current_freeze_encoder=args.freeze_encoder)
    if start_epoch > args.epochs:
        raise ValueError('Checkpoint already completed the requested total number of epochs.')
    
    training_start = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        
        print()
        print('=' * 60)
        print(f'EPOCH {epoch}/{args.epochs}')
        print('=' * 60)
        epoch_start = time.time()
        train_loader = create_train_loader(dataset=active_train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed, epoch=epoch, pin_memory=pin_memory)
        train_loss = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, loss_function=loss_function, device=device, gradient_accumulation_steps=args.gradient_accumulation)
        validation_loss, metrics = evaluate(model=model, loader=validation_loader, loss_function=loss_function, device=device)
        epoch_seconds = time.time() - epoch_start
        
        print()
        print('=== EPOCH RESULT ===')
        print(f'Train loss:      {train_loss:.6f}')
        print(f'Validation loss: {validation_loss:.6f}')
        print_metrics(metrics)
        print(f'Epoch time: {epoch_seconds:.1f} seconds')
        epoch_record = {'epoch': int(epoch), 'train_loss': float(train_loss), 'validation_loss': float(validation_loss), **metrics, 'epoch_seconds': float(epoch_seconds)}
        history.append(epoch_record)
        current_pr_auc = float(metrics['pr_auc'])
        
        if current_pr_auc > best_pr_auc:
            best_pr_auc = current_pr_auc
            print(f'New best validation PR-AUC: {best_pr_auc:.4f}')
            save_best_checkpoint(path=output_directory / 'best_model.pt', model=model, epoch=epoch, best_pr_auc=best_pr_auc, metrics=metrics, training_config=training_config)
            save_json(output_directory / 'best_validation_metrics.json', metrics)
        save_resume_checkpoint(path=output_directory / 'latest_checkpoint.pt', model=model, optimizer=optimizer, epoch=epoch, best_pr_auc=best_pr_auc, metrics=metrics, history=history, training_config=training_config)
        save_json(output_directory / 'history.json', history)
        save_json(output_directory / 'latest_validation_metrics.json', metrics)
        print('Latest resume checkpoint saved.')
    
    
    total_seconds = time.time() - training_start
    print()
    print('=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f'Best validation PR-AUC: {best_pr_auc:.4f}')
    print(f'Total runtime: {total_seconds:.1f} seconds')
    print('Output directory:')
    print(output_directory)
    print()
    print('Best model:')
    print(output_directory / 'best_model.pt')
    print()
    print('Resume checkpoint:')
    print(output_directory / 'latest_checkpoint.pt')



if __name__ == '__main__':
    main()
