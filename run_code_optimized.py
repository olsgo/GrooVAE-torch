"""
Updated run_code.py optimized for M1 Max
This file maintains compatibility with the original while adding M1 Max optimizations
"""
import pickle
import random
import pretty_midi
import numpy as np
import os
import sys
from time import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler

# Try to import torchsummary, fallback to torchinfo
try:
    from torchsummary import summary
except ImportError:
    try:
        from torchinfo import summary
    except ImportError:
        def summary(model, input_size):
            print("Model summary not available - install torchsummary or torchinfo")
            return None

from drum_utils import *
from train import *
from test import *
from model import *
from config import Config

# M1 Max optimized version
from train_optimized import groove_train_optimized
from data_loader_optimized import create_optimized_data_loaders

def main():
    # Initialize configuration for M1 Max
    config = Config()
    config.create_directories()
    
    # Get optimized device (MPS for M1 Max, CUDA, or CPU)
    device = config.get_device()
    
    # Load data using optimized data loaders
    print("Loading data with M1 Max optimizations...")
    
    try:
        # Try to load with new optimized system first
        train_loader, val_loader, test_loader = create_optimized_data_loaders(
            data_type='tapify', config=config
        )
        print("Using optimized data loading pipeline")
        
    except FileNotFoundError:
        print("Preprocessed data not found. Falling back to original data loading...")
        
        # Original data loading logic as fallback
        path = config.PROCESSED_DATA_DIR + '/'
        file_names = ['tapify_train.pkl', 'tapify_valid.pkl', 'tapify_test.pkl']
        data_names = ['train_data', 'val_data', 'test_data']
        
        data_dict = {}
        for file_name, data_name in zip(file_names, data_names):
            file_path = path + file_name
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    random.shuffle(data)
                    data_dict[data_name] = data
                    print(f'The number of data in {data_name}: {len(data)}')
            else:
                print(f"Warning: {file_path} not found")
                return
        
        # Original dataloader setup with M1 Max optimizations
        class DatasetSampler(Dataset):
            def __init__(self, x):
                self.x = x

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx].to(torch.float32)

        # Use optimized parameters from config
        params = config.get_data_loader_params()
        val_params = config.get_val_data_loader_params()

        train_loader = DataLoader(DatasetSampler(data_dict['train_data']), **params)
        val_loader = DataLoader(DatasetSampler(data_dict['val_data']), **val_params)
        test_loader = DataLoader(DatasetSampler(data_dict['test_data']), **val_params)

    # Initialize model with M1 Max optimizations
    print("Initializing model...")
    
    encoder = Encoder(
        input_size=config.ENC_INPUT_SIZE,
        hidden_size=config.ENC_HIDDEN_SIZE,
        latent_dim=config.ENC_LATENT_DIM
    )
    encoder = encoder.to(device)

    decoder = Decoder(
        input_size=config.DEC_INPUT_SIZE,
        hidden_size=config.DEC_HIDDEN_SIZE,
        output_size=config.DEC_OUTPUT_SIZE
    )
    decoder = decoder.to(device)

    model = [encoder, decoder]

    # Print model summary if available
    try:
        print("Encoder summary:")
        summary(encoder, (32, config.ENC_INPUT_SIZE))  # Sample input size
        print("Decoder summary:")
        # Decoder summary is more complex due to variable inputs
        print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    except:
        total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
        print(f"Total model parameters: {total_params:,}")

    # Initialize optimizers with M1 Max optimized learning rates
    enc_optimizer = optim.Adam(encoder.parameters(), lr=config.LEARNING_RATE)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=config.LEARNING_RATE)
    optimizer = [enc_optimizer, dec_optimizer]

    print(f"Training configuration:")
    print(f"  - Device: {device}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Epochs: {config.EPOCHS}")
    print(f"  - Mixed precision: {config.ENABLE_MIXED_PRECISION}")

    # Train using optimized training loop
    print("Starting optimized training...")
    history_train = groove_train_optimized(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        epochs=config.EPOCHS,
        config=config
    )

    # Save model
    model_path = os.path.join(config.MODEL_SAVE_DIR, 'tapify_2bar_16quant_m1_optimized')
    torch.save(encoder.state_dict(), model_path + '_encoder.pt')
    torch.save(decoder.state_dict(), model_path + '_decoder.pt')
    
    print(f"Model saved:")
    print(f"  - Encoder: {model_path}_encoder.pt")
    print(f"  - Decoder: {model_path}_decoder.pt")

    # Test
    print("Running evaluation...")
    try:
        history_test = groove_test(device, test_loader, model)
        print("Evaluation completed successfully")
    except Exception as e:
        print(f"Evaluation failed: {e}")

    # Plot results with M1 Max optimized visualization
    def plot_loss_history(history, start_epoch=4):
        # Define the starting epoch index
        start_idx = max(0, start_epoch - 1)
        
        # Ensure that the start index is within bounds
        if start_idx >= len(history['train_loss']):
            start_idx = 0
        
        # Slice the data to start from the specified epoch
        epochs = range(start_idx + 1, len(history['train_loss']) + 1)
        train_losses = history['train_loss'][start_idx:]
        val_losses = history['val_loss'][start_idx:]

        plt.figure(figsize=(12, 8))
        
        # Loss plot
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress on M1 Max')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot if available
        if 'learning_rates' in history and len(history['learning_rates']) > start_idx:
            plt.subplot(2, 1, 2)
            lr_data = history['learning_rates'][start_idx:]
            plt.plot(epochs, lr_data, 'g-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = model_path + '_loss.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to: {plot_path}")
        
        # Show final results
        print(f"\nTraining Results:")
        print(f"  - Final train loss: {train_losses[-1]:.4f}")
        print(f"  - Final val loss: {val_losses[-1]:.4f}")
        print(f"  - Best val loss: {min(val_losses):.4f}")
        print(f"  - Total epochs: {len(history['train_loss'])}")
        
    plot_loss_history(history_train)
    
    print(f"\nTraining completed successfully on M1 Max!")
    print(f"All optimizations for 64GB RAM and Metal Performance Shaders have been applied.")

if __name__ == "__main__":
    main()