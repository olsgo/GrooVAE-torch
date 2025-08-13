"""
Optimized main training script for M1 Max
"""
import os
import sys
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model import Encoder, Decoder
from train_optimized import groove_train_optimized, save_checkpoint, load_checkpoint
from data_loader_optimized import create_optimized_data_loaders, benchmark_data_loading
from test import groove_test


def setup_model(config, device):
    """
    Initialize model with M1 Max optimizations
    """
    print("Initializing model...")
    
    # Create encoder
    encoder = Encoder(
        input_size=config.ENC_INPUT_SIZE,
        hidden_size=config.ENC_HIDDEN_SIZE,
        latent_dim=config.ENC_LATENT_DIM
    )
    
    # Create decoder
    decoder = Decoder(
        input_size=config.DEC_INPUT_SIZE,
        hidden_size=config.DEC_HIDDEN_SIZE,
        output_size=config.DEC_OUTPUT_SIZE
    )
    
    # Move to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Print model info
    total_params_enc = sum(p.numel() for p in encoder.parameters())
    total_params_dec = sum(p.numel() for p in decoder.parameters())
    total_params = total_params_enc + total_params_dec
    
    print(f"Model Summary:")
    print(f"  - Encoder parameters: {total_params_enc:,}")
    print(f"  - Decoder parameters: {total_params_dec:,}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Device: {device}")
    
    return encoder, decoder


def setup_optimizers(encoder, decoder, config):
    """
    Setup optimizers with M1 Max optimized learning rates
    """
    # Use higher learning rate for larger batch sizes (linear scaling rule)
    lr = config.LEARNING_RATE
    
    enc_optimizer = optim.Adam(
        encoder.parameters(), 
        lr=lr,
        weight_decay=1e-5,  # Small weight decay for regularization
        eps=1e-8,
        amsgrad=True  # More stable for larger batches
    )
    
    dec_optimizer = optim.Adam(
        decoder.parameters(), 
        lr=lr,
        weight_decay=1e-5,
        eps=1e-8,
        amsgrad=True
    )
    
    print(f"Optimizers initialized with lr={lr}")
    
    return enc_optimizer, dec_optimizer


def plot_training_history(history, save_path):
    """
    Plot and save training history
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'learning_rates' in history:
        ax2.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved to {save_path}")


def check_disk_space(min_gb=10):
    """Check available disk space before training"""
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    print(f"💾 Available disk space: {free_gb:.1f} GB")
    
    if free_gb < min_gb:
        print(f"⚠️  WARNING: Low disk space ({free_gb:.1f} GB < {min_gb} GB)")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Training aborted due to low disk space")
            exit(1)
    return free_gb

def main():
    parser = argparse.ArgumentParser(description='GrooVAE Training Optimized for M1 Max')
    parser.add_argument('--data-type', type=str, default='tapify', 
                       choices=['tapify', 'humanize'],
                       help='Type of data to use for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run data loading benchmark')
    parser.add_argument('--no-train', action='store_true',
                       help='Skip training (useful for testing setup)')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Override config with command line arguments
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        config.BATCH_SIZE_VAL = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.epochs:
        config.EPOCHS = args.epochs
    
    # Create necessary directories
    config.create_directories()
    
    # Setup device
    device = config.get_device()
    
    print(f"Starting GrooVAE training optimized for M1 Max")
    print(f"Configuration:")
    print(f"  - Data type: {args.data_type}")
    print(f"  - Epochs: {config.EPOCHS}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Device: {device}")
    print(f"  - Mixed precision: {config.ENABLE_MIXED_PRECISION}")
    
    # Load data
    print("\nLoading data...")
    try:
        train_loader, val_loader, test_loader = create_optimized_data_loaders(
            data_type=args.data_type, config=config
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the preprocessed data files exist in the data/processed directory.")
        print("Run the preprocessing scripts first:")
        print("  - python humanize_preprocess.py")
        print("  - python tapify_preprocess.py")
        return
    
    # Benchmark data loading if requested
    if args.benchmark:
        print("\nRunning data loading benchmark...")
        benchmark_data_loading(train_loader, device, num_batches=20)
        if args.no_train:
            return
    
    # Setup model
    encoder, decoder = setup_model(config, device)
    model = [encoder, decoder]
    
    # Setup optimizers
    enc_optimizer, dec_optimizer = setup_optimizers(encoder, decoder, config)
    optimizer = [enc_optimizer, dec_optimizer]
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(args.resume, encoder, decoder, 
                                       enc_optimizer, dec_optimizer)
        print(f"Resumed from epoch {start_epoch}")
    
    if args.no_train:
        print("Setup complete. Skipping training as requested.")
        return
    
    # Training
    print(f"\nStarting training...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        history = groove_train_optimized(
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            epochs=config.EPOCHS,
            config=config
        )
        
        # Save final model
        model_save_path = os.path.join(
            config.MODEL_SAVE_DIR, 
            f'{args.data_type}_model_{timestamp}'
        )
        
        torch.save(encoder.state_dict(), f'{model_save_path}_encoder.pth')
        torch.save(decoder.state_dict(), f'{model_save_path}_decoder.pth')
        
        print(f"\nFinal model saved:")
        print(f"  - Encoder: {model_save_path}_encoder.pth")
        print(f"  - Decoder: {model_save_path}_decoder.pth")
        
        # Plot training history
        plot_path = os.path.join(config.MODEL_SAVE_DIR, f'{args.data_type}_training_history_{timestamp}.png')
        plot_training_history(history, plot_path)
        
        # Test the model
        print(f"\nRunning final evaluation...")
        try:
            test_history = groove_test(device, test_loader, model)
            print(f"Test completed successfully")
        except Exception as e:
            print(f"Test evaluation failed: {e}")
        
        print(f"\nTraining completed successfully!")
        print(f"Best validation loss: {min(history['val_loss']):.4f}")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        # Save emergency checkpoint
        emergency_path = os.path.join(config.MODEL_SAVE_DIR, 'emergency_checkpoint.pth')
        save_checkpoint(encoder, decoder, enc_optimizer, dec_optimizer, 
                       0, float('inf'), config, is_best=False)
        print(f"Emergency checkpoint saved to {emergency_path}")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Add this after config setup but before training starts
    print(f"\nChecking system resources...")
    initial_disk_space = check_disk_space(min_gb=20)  # Require 20GB minimum
    
    main()