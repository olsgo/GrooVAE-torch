import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import argparse
from pathlib import Path

from model import Encoder, Decoder
from data_loader_optimized import create_optimized_data_loaders
from train_optimized import groove_train_optimized
from config import Config

def find_pretrained_models(model_dir="model"):
    """Find available pre-trained model files"""
    model_path = Path(model_dir)
    encoder_files = list(model_path.glob("*encoder*.pt")) + list(model_path.glob("*encoder*.pth"))
    decoder_files = list(model_path.glob("*decoder*.pt")) + list(model_path.glob("*decoder*.pth"))
    
    models = {}
    for encoder_file in encoder_files:
        base_name = encoder_file.stem.replace('_encoder', '')
        decoder_file = None
        
        # Look for corresponding decoder
        for decoder in decoder_files:
            if base_name in decoder.stem:
                decoder_file = decoder
                break
        
        if decoder_file:
            models[base_name] = {
                'encoder': str(encoder_file),
                'decoder': str(decoder_file)
            }
    
    return models

def load_pretrained_model(encoder_path, decoder_path, device, config):
    """Load pre-trained encoder and decoder models"""
    print(f"Loading pre-trained models:")
    print(f"  Encoder: {encoder_path}")
    print(f"  Decoder: {decoder_path}")
    
    # Initialize models with config parameters
    encoder = Encoder(
        input_size=config.ENC_INPUT_SIZE,
        hidden_size=config.ENC_HIDDEN_SIZE,
        latent_dim=config.ENC_LATENT_DIM
    ).to(device)
    
    decoder = Decoder(
        input_size=config.DEC_INPUT_SIZE,
        hidden_size=config.DEC_HIDDEN_SIZE,
        output_size=config.DEC_OUTPUT_SIZE
    ).to(device)
    
    # Load pre-trained weights
    try:
        encoder_state = torch.load(encoder_path, map_location=device)
        decoder_state = torch.load(decoder_path, map_location=device)
        
        encoder.load_state_dict(encoder_state)
        decoder.load_state_dict(decoder_state)
        
        print("✓ Pre-trained models loaded successfully!")
        return encoder, decoder
        
    except Exception as e:
        print(f"Error loading pre-trained models: {e}")
        print("Initializing models with random weights instead.")
        return encoder, decoder

def setup_model_for_finetune(encoder, decoder, freeze_encoder=False, freeze_decoder_layers=0):
    """Setup model for fine-tuning with optional layer freezing"""
    if freeze_encoder:
        print("Freezing encoder parameters")
        for param in encoder.parameters():
            param.requires_grad = False
    
    if freeze_decoder_layers > 0:
        print(f"Freezing first {freeze_decoder_layers} decoder layers")
        # Freeze LSTM layers
        for i in range(min(freeze_decoder_layers, getattr(decoder, 'num_layers', 2))):
            for name, param in decoder.named_parameters():
                if f'lstm.weight_ih_l{i}' in name or f'lstm.weight_hh_l{i}' in name or \
                   f'lstm.bias_ih_l{i}' in name or f'lstm.bias_hh_l{i}' in name:
                    param.requires_grad = False
    
    return encoder, decoder

def setup_optimizers_for_finetune(encoder, decoder, config, lr_factor=0.1):
    """Setup optimizers with reduced learning rate for fine-tuning"""
    base_lr = config.LEARNING_RATE
    finetune_lr = base_lr * lr_factor
    
    print(f"Using fine-tuning learning rate: {finetune_lr} (base: {base_lr}, factor: {lr_factor})")
    
    # Separate parameters that require gradients
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    decoder_params = [p for p in decoder.parameters() if p.requires_grad]
    
    enc_optimizer = None
    dec_optimizer = None
    
    if encoder_params:
        enc_optimizer = optim.Adam(encoder_params, lr=finetune_lr)
    if decoder_params:
        dec_optimizer = optim.Adam(decoder_params, lr=finetune_lr)
    
    return enc_optimizer, dec_optimizer

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GrooVAE model')
    parser.add_argument('--model-name', type=str, help='Specific model to use (e.g., "1st_humanize")')
    parser.add_argument('--list-models', action='store_true', help='List available pre-trained models')
    parser.add_argument('--data-type', type=str, default='humanize', choices=['humanize', 'tapify'],
                       help='Type of data to use for fine-tuning')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for fine-tuning')
    parser.add_argument('--lr-factor', type=float, default=0.1, help='Learning rate reduction factor')
    parser.add_argument('--freeze-encoder', action='store_true', help='Freeze encoder during fine-tuning')
    parser.add_argument('--freeze-decoder-layers', type=int, default=0, help='Number of decoder layers to freeze')
    parser.add_argument('--save-dir', type=str, default='saved_models', help='Directory to save fine-tuned models')
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    
    # Find available models
    available_models = find_pretrained_models()
    
    if args.list_models:
        print("Available pre-trained models:")
        if available_models:
            for name, paths in available_models.items():
                print(f"  {name}:")
                print(f"    Encoder: {paths['encoder']}")
                print(f"    Decoder: {paths['decoder']}")
        else:
            print("  No pre-trained models found in model/ directory")
            print("  Make sure you have downloaded the models and they follow the naming convention:")
            print("    *encoder*.pt and *decoder*.pt")
        return
    
    if not available_models:
        print("No pre-trained models found. Please check the model/ directory.")
        return
    
    # Select model
    if args.model_name:
        if args.model_name not in available_models:
            print(f"Model '{args.model_name}' not found. Available models: {list(available_models.keys())}")
            return
        selected_model = args.model_name
    else:
        # Use the first available model
        selected_model = list(available_models.keys())[0]
        print(f"Using model: {selected_model}")
    
    # Setup device
    device = config.get_device()
    print(f"Using device: {device}")
    
    # Load pre-trained model
    model_paths = available_models[selected_model]
    encoder, decoder = load_pretrained_model(
        model_paths['encoder'], 
        model_paths['decoder'], 
        device,
        config
    )
    
    # Setup for fine-tuning
    encoder, decoder = setup_model_for_finetune(
        encoder, decoder, 
        freeze_encoder=args.freeze_encoder,
        freeze_decoder_layers=args.freeze_decoder_layers
    )
    
    # Setup optimizers
    enc_optimizer, dec_optimizer = setup_optimizers_for_finetune(encoder, decoder, config, args.lr_factor)
    
    # Load data using the correct function
    print(f"Loading {args.data_type} data...")
    train_loader, val_loader, test_loader = create_optimized_data_loaders(data_type=args.data_type, config=config)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Fine-tune the model
    print(f"Starting fine-tuning for {args.epochs} epochs...")
    
    # Around line 196, replace the existing try block with:
    try:
        result = groove_train_optimized(
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            model=(encoder, decoder),
            optimizer=(enc_optimizer, dec_optimizer),
            epochs=args.epochs,
            config=config
        )
        # Handle multiple return values safely
        if isinstance(result, tuple) and len(result) >= 2:
            train_losses, valid_losses = result[0], result[1]
        else:
            train_losses, valid_losses = [], []
            print("Warning: Unexpected return format from training function")
        
        print("\n✓ Fine-tuning completed successfully!")
        print(f"Final training loss: {train_losses[-1]:.4f}")
        print(f"Final validation loss: {valid_losses[-1]:.4f}")
        
        # Save fine-tuned models
        model_name = f"{selected_model}_finetuned_{args.data_type}"
        encoder_path = save_dir / f"{model_name}_encoder.pt"
        decoder_path = save_dir / f"{model_name}_decoder.pt"
        
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)
        
        print(f"Models saved:")
        print(f"  Encoder: {encoder_path}")
        print(f"  Decoder: {decoder_path}")
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()