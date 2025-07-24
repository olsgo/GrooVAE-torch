#!/usr/bin/env python3
"""
GrooVAE-torch: H100 SXM Optimized Training Script for RunPod

This script provides optimized training for GrooVAE on H100 SXM GPUs,
including mixed precision, torch.compile, and optimized data loading.
"""

import torch
import torch.optim as optim
import argparse
import logging
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from config import Config
from model import Encoder, Decoder
from train_optimized import groove_train_optimized, setup_logging
from data_loader import setup_data_loaders, verify_data_format

def setup_device():
    """Setup and verify H100 device"""
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    
    logger.info(f"GPU Device: {gpu_name}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    
    # Check if we're on H100
    if 'H100' in gpu_name:
        logger.info("H100 detected - enabling all optimizations")
        Config.USE_MIXED_PRECISION = True
        Config.USE_TORCH_COMPILE = True
    else:
        logger.warning(f"Not running on H100 (detected: {gpu_name})")
        logger.warning("Some optimizations may not be effective")
    
    # Set memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"Memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    return device

def create_models(device):
    """Create and initialize models"""
    logger = logging.getLogger(__name__)
    
    # Create encoder
    encoder = Encoder(
        input_size=Config.ENCODER_INPUT_SIZE,
        hidden_size=Config.ENCODER_HIDDEN_SIZE,
        latent_dim=Config.ENCODER_LATENT_DIM
    )
    
    # Create decoder
    decoder = Decoder(
        input_size=Config.DECODER_INPUT_SIZE,
        hidden_size=Config.DECODER_HIDDEN_SIZE,
        output_size=Config.DECODER_OUTPUT_SIZE,
        num_layers=Config.DECODER_NUM_LAYERS
    )
    
    # Move to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Count parameters
    enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_params = enc_params + dec_params
    
    logger.info(f"Model created:")
    logger.info(f"  Encoder parameters: {enc_params:,}")
    logger.info(f"  Decoder parameters: {dec_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")
    
    return encoder, decoder

def create_optimizers(encoder, decoder):
    """Create optimizers with H100-optimized settings"""
    logger = logging.getLogger(__name__)
    
    # Use AdamW for better performance on H100
    enc_optimizer = optim.AdamW(
        encoder.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=1e-4,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    dec_optimizer = optim.AdamW(
        decoder.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=1e-4,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    logger.info(f"Optimizers created with learning rate: {Config.LEARNING_RATE}")
    
    return enc_optimizer, dec_optimizer

def save_final_models(encoder, decoder, history):
    """Save final trained models"""
    logger = logging.getLogger(__name__)
    
    # Save encoder
    encoder_path = Config.get_model_save_path('encoder.pt')
    torch.save(encoder.state_dict(), encoder_path)
    
    # Save decoder
    decoder_path = Config.get_model_save_path('decoder.pt')
    torch.save(decoder.state_dict(), decoder_path)
    
    # Save training history
    history_path = Config.get_model_save_path('training_history.pt')
    torch.save(history, history_path)
    
    logger.info(f"Models saved:")
    logger.info(f"  Encoder: {encoder_path}")
    logger.info(f"  Decoder: {decoder_path}")
    logger.info(f"  History: {history_path}")

def plot_training_history(history):
    """Plot and save training history"""
    logger = logging.getLogger(__name__)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot detailed training losses
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_recon_loss'], 'g-', label='Reconstruction Loss')
    plt.plot(epochs, history['train_kl_loss'], 'm-', label='KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Components')
    plt.legend()
    plt.grid(True)
    
    # Plot log scale
    plt.subplot(1, 3, 3)
    plt.semilogy(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.semilogy(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss (Log Scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Config.get_model_save_path('training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Training history plot saved: {plot_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='GrooVAE H100 Optimized Training')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                        help=f'Number of training epochs (default: {Config.EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                        help=f'Batch size (default: {Config.BATCH_SIZE})')
    parser.add_argument('--learning-rate', type=float, default=Config.LEARNING_RATE,
                        help=f'Learning rate (default: {Config.LEARNING_RATE})')
    parser.add_argument('--kl-weight', type=float, default=Config.KL_WEIGHT,
                        help=f'KL divergence weight (default: {Config.KL_WEIGHT})')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile optimization')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.learning_rate
    Config.KL_WEIGHT = args.kl_weight
    
    if args.no_mixed_precision:
        Config.USE_MIXED_PRECISION = False
    if args.no_compile:
        Config.USE_TORCH_COMPILE = False
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting GrooVAE H100 Optimized Training")
    logger.info(f"Configuration: {vars(Config)}")
    
    try:
        # Setup device
        device = setup_device()
        
        # Setup data loaders
        train_loader, val_loader, test_loader = setup_data_loaders()
        
        # Verify data format
        verify_data_format(train_loader, expected_shape=(32, 54))
        
        # Create models
        encoder, decoder = create_models(device)
        model = [encoder, decoder]
        
        # Create optimizers
        enc_optimizer, dec_optimizer = create_optimizers(encoder, decoder)
        optimizer = [enc_optimizer, dec_optimizer]
        
        # Train model
        logger.info("Starting training...")
        history = groove_train_optimized(
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            epochs=Config.EPOCHS,
            kl_weight=Config.KL_WEIGHT
        )
        
        # Save models and results
        save_final_models(encoder, decoder, history)
        plot_training_history(history)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()