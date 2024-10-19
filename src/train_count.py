# File: ./scripts/run_count.py

import argparse
import os
import datetime
import json
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.auto import tqdm
import wandb

from src.match_count import GenerateMatchCountDataset, MatchCountVectorDataset, STRING_FUNCTION_MAPPING
from src.utils import get_optimizer, dump_dataset_to_csv, pprint_model_predictions, MultiLabelCausalTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Train Counting Task Model with CoT and Filler Tokens")
    
    # Dataset and Data Paths
    parser.add_argument('-dn', '--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('-de', '--data_extension', type=str, required=True, help='Data extension or path')
    
    # Training Parameters
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('-ma', '--mask_type', type=str, default='Final', choices=['Final', 'P'], help='Mask type for dataset')
    
    # Model Parameters
    parser.add_argument('-m', '--model_name', type=str, default='bert-base-uncased', help='Base model name')
    parser.add_argument('-cc', '--config_file', type=str, required=True, help='Path to model configuration JSON')
    
    # Optimization Parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam optimizer beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam optimizer beta2')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--lr_decay_on', action='store_true', help='Enable learning rate decay')
    
    # Other Parameters
    parser.add_argument('--no_wdb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--checkpoint', action='store_true', help='Enable checkpointing')
    parser.add_argument('--mpt', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--accumulation_factor', type=int, default=4, help='Gradient accumulation factor')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping based on accuracy')
    
    # Counting Task Specific Parameters
    parser.add_argument('--length', type=int, default=20, help='Length of the sequence')
    parser.add_argument('--target', type=int, default=5, help='Element to count in the sequence')
    parser.add_argument('--mod', type=int, default=10, help='Modulus for transformations')
    parser.add_argument('--transform', type=str, default='random', choices=['random', 'identity'], help='Type of transformation')
    parser.add_argument('--cot_rate', type=float, default=0.5, help='Rate of CoT inclusion')
    parser.add_argument('--true_instance_rate', type=float, default=0.7, help='Rate of true instances in data')
    parser.add_argument('--corruption_rate', type=float, default=4/3, help='Corruption rate for data generation')
    parser.add_argument('--filler_type', type=str, default='fillers', choices=['basic', 'fillers'], help='Type of filler tokens in CoT')
    
    args = parser.parse_args()
    
    # Initialize Weights & Biases
    if not args.no_wdb:
        wandb.init(project='counting_task', config=vars(args))
    
    # Generate Dataset
    print("Generating dataset...")
    GenerateMatchCountDataset(
        name=args.dataset_name,
        train_samples=10000,  # Adjust as needed or make it an argument
        test_samples=1000,    # Adjust as needed or make it an argument
        length=args.length,
        target=args.target,
        mod=args.mod,
        true_instance_rate=args.true_instance_rate,
        cot_rate=args.cot_rate,
        corruption_rate=args.corruption_rate,
        transform=args.transform,
        filler_to_string=STRING_FUNCTION_MAPPING[args.filler_type],
        cot_to_string=STRING_FUNCTION_MAPPING['basic'],  # or 'fillers'
        data_path='./data/'
    )
    
    # Load Dataset
    print("Loading dataset...")
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    train_file = os.path.join('./data/', f"{args.dataset_name}_trainset_{today_str}.csv")
    test_file = os.path.join('./data/', f"{args.dataset_name}_testset_{today_str}.csv")
    
    train_data = pd.read_csv(train_file, header=None, names=['text'])
    test_data = pd.read_csv(test_file, header=None, names=['text'])
    
    # Initialize Vector Dataset
    print("Vectorizing dataset...")
    train_dataset = MatchCountVectorDataset(train_data, length=args.length, target=args.target, mod=args.mod, mask=args.mask_type)
    test_dataset = MatchCountVectorDataset(test_data, length=args.length, target=args.target, mod=args.mod, mask=args.mask_type)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize Model
    print("Initializing model...")
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.mod)
    model = MultiLabelCausalTransformer(base_model, input_dim=train_dataset.input_dim, output_dim=args.mod)
    model.to("cuda")
    
    # Load Model Configuration if necessary
    # Example: Load pretrained weights or specific configurations
    # model.load_pretrained_weights(...)
    
    # Initialize Optimizer and Scheduler
    print("Setting up optimizer and scheduler...")
    optimizer, decay_scheduler, scaler = get_optimizer(
        optim="adam",
        lr_decay_on=args.lr_decay_on,
        weight_decay=args.weight_decay,
        mpt=args.mpt,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        model=model,
        tot_opt_steps=args.epochs * len(train_loader)
    )
    
    # Define Loss Function
    loss_func = torch.nn.MSELoss()  # For regression tasks; change if necessary
    
    # Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f'###### EPOCH {epoch+1}/{args.epochs} ######')
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            inputs = batch['input_ids'].to("cuda")
            labels = batch['labels'].to("cuda").float()
            
            if args.mpt:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.accumulation_factor == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if args.lr_decay_on:
                        decay_scheduler.step()
                    optimizer.zero_grad()
            else:
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                loss.backward()
                if (batch_idx + 1) % args.accumulation_factor == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if args.lr_decay_on:
                        decay_scheduler.step()
                    optimizer.zero_grad()
            
            epoch_loss += loss.item()
            # Example accuracy calculation; adjust based on task
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).float().sum()
            accuracy = correct / labels.numel()
            epoch_accuracy += accuracy.item()
            
            # Logging
            if not args.no_wdb:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_accuracy': accuracy.item(),
                    'epoch': epoch + 1
                })
            
            # Evaluation
            if (batch_idx + 1) % 100 == 0:
                eval_metrics = evaluate(model, eval_loader, loss_func)
                print(f"Batch {batch_idx+1}: Train Loss={loss.item():.4f}, Train Acc={accuracy.item():.4f}")
                print(f"Evaluation: {eval_metrics}")
                if not args.no_wdb:
                    wandb.log(eval_metrics)
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_accuracy = epoch_accuracy / len(train_loader)
        print(f"Epoch {epoch+1} Summary: Loss={avg_epoch_loss:.4f}, Accuracy={avg_epoch_accuracy:.4f}")
        
        # Save Checkpoints
        if args.checkpoint:
            checkpoint_dir = os.path.join('./output_dir/', f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}-{args.dataset_name}-checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved at {checkpoint_dir}")
        
        # Early Stopping
        if args.early_stop and avg_epoch_accuracy > 0.995:
            print("Early stopping triggered due to high accuracy.")
            break
    
    # Final Model Save
    final_model_dir = os.path.join('./output_dir/', f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}-{args.dataset_name}-checkpoint-final")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    print(f"Final model saved at {final_model_dir}")
    
    # Finish Weights & Biases
    if not args.no_wdb:
        wandb.finish()

def evaluate(model, eval_loader, loss_func):
    """
    Evaluation loop for the Counting Task.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_elements = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            inputs = batch['input_ids'].to("cuda")
            labels = batch['labels'].to("cuda").float()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).float().sum()
            total_correct += correct.item()
            total_elements += labels.numel()
    
    avg_loss = total_loss / len(eval_loader)
    accuracy = total_correct / total_elements
    metrics = {
        'eval_loss': avg_loss,
        'eval_accuracy': accuracy
    }
    return metrics

if __name__ == "__main__":
    main()

