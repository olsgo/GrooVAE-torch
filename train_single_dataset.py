import torch
import torch.nn as nn
import os
import argparse
from datetime import datetime
from glob import glob
import matplotlib.pyplot as plt

# Dynamically discover available datasets
processed_data_dir = 'data/processed'
def get_available_datasets(data_dir):
    available_datasets = set()
    train_files = glob.glob(os.path.join(data_dir, '*_train.pkl'))
    for f in train_files:
        basename = os.path.basename(f)
        dataset_name = basename.replace('_train.pkl', '')
        # Verify that test and validation sets also exist
        test_file = os.path.join(data_dir, f'{dataset_name}_test.pkl')
        valid_file = os.path.join(data_dir, f'{dataset_name}_valid.pkl')
        if os.path.exists(test_file) and os.path.exists(valid_file):
            available_datasets.add(dataset_name)
    if not available_datasets:
        print(f"Error: No datasets found in {processed_data_dir}. Please run preprocessing scripts first.")
        exit()
    return sorted(list(available_datasets))

def main():
    available_datasets = get_available_datasets(processed_data_dir)
    parser = argparse.ArgumentParser(description='GrooVAE: Generate music scores with Variational Auto-encoder')
    parser.add_argument('--model_name', type=str, default=None, required=False,
                        help='name of the model to be saved. If not provided, a timestamp-based name will be generated.')
    parser.add_argument('--dataset', type=str, default=None, required=True,
                        choices=available_datasets,
                        help='dataset to be used for training the model.')
    parser.add_argument('--config', type=str, default='config_optimized', help='config file name')
    parser.add_argument('--data_dir', type=str, default=processed_data_dir, 
                        help='directory containing the processed dataset files')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size in config.')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate in config.')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs in config.')
    parser.add_argument('--description', type=str, default=None, help='A brief description of the training run.')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint to resume training from.')

    args = parser.parse_args()

    # Use the specified or default data directory
    processed_data_dir = args.data_dir

    # Check if the chosen dataset is valid
    if args.dataset not in get_available_datasets(processed_data_dir):
        print(f"Error: Dataset '{args.dataset}' not found in '{processed_data_dir}'.")
        print(f"Available datasets: {', '.join(get_available_datasets(processed_data_dir))}")
        exit()

    # import config file
    config_module = __import__(args.config)
    config = config_module.ConfigOptimized()

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
    
    # Check disk space
    check_disk_space(min_gb=20)
    
    print(f"🎵 GrooVAE Training - Optimized for Nuance Capture")
    print(f"{'='*60}")
    print(f"📊 Dataset: {args.dataset.upper()}")
    print(f"🏷️  Model name: {args.model_name}")
    print(f"📝 Description: {args.description or 'None'}")
    print(f"🔧 Configuration:")
    print(f"   - Epochs: {config.EPOCHS}")
    print(f"   - Batch size: {config.BATCH_SIZE}")
    print(f"   - Learning rate: {config.LEARNING_RATE}")
    print(f"   - Model architecture: Enc({config.ENC_HIDDEN_SIZE}→{config.ENC_LATENT_DIM}), Dec({config.DEC_HIDDEN_SIZE})")
    print(f"   - Device: {device}")
    print(f"   - Mixed precision: {config.ENABLE_MIXED_PRECISION}")
    print(f"   - KL weight: {config.KL_WEIGHT} (optimized for reconstruction)")
    print(f"   - Teacher forcing: {config.TEACHER_FORCING_RATIO} (enhanced)")
    
    # Load dataset
    print(f"\n📁 Loading {args.dataset} dataset...")
    try:
        train_loader, val_loader, test_loader = create_optimized_data_loaders(
            data_type=args.dataset, config=config
        )
        print(f"✅ Dataset loaded successfully")
        print(f"   - Training batches: {len(train_loader)}")
        print(f"   - Validation batches: {len(val_loader)}")
        print(f"   - Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"❌ Failed to load {args.dataset} dataset: {e}")
        print("\nMake sure the preprocessed data files exist:")
        print(f"   - data/processed/{args.dataset}_train.pkl")
        print(f"   - data/processed/{args.dataset}_valid.pkl")
        print(f"   - data/processed/{args.dataset}_test.pkl")
        return
    
    # Initialize model
    print(f"\n🧠 Initializing model...")
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
    
    # Print model info
    total_params_enc = sum(p.numel() for p in encoder.parameters())
    total_params_dec = sum(p.numel() for p in decoder.parameters())
    total_params = total_params_enc + total_params_dec
    
    print(f"📈 Model Summary:")
    print(f"   - Encoder parameters: {total_params_enc:,}")
    print(f"   - Decoder parameters: {total_params_dec:,}")
    print(f"   - Total parameters: {total_params:,}")
    
    # Setup optimizers
    enc_optimizer = torch.optim.Adam(
        encoder.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=1e-5,
        amsgrad=True
    )
    
    dec_optimizer = torch.optim.Adam(
        decoder.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=1e-5,
        amsgrad=True
    )
    
    model = [encoder, decoder]
    optimizer = [enc_optimizer, dec_optimizer]
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            enc_optimizer.load_state_dict(checkpoint['enc_optimizer_state_dict'])
            dec_optimizer.load_state_dict(checkpoint['dec_optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"✅ Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return
    
    # Training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n🚀 Starting training...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
        
        # Save final model with custom name
        model_save_path = os.path.join(
            config.MODEL_SAVE_DIR, 
            f'{args.model_name}_{timestamp}.pth'
        )
        
        # Save complete model state
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'enc_optimizer_state_dict': enc_optimizer.state_dict(),
            'dec_optimizer_state_dict': dec_optimizer.state_dict(),
            'epoch': config.EPOCHS,
            'dataset': args.dataset,
            'model_name': args.model_name,
            'description': args.description,
            'history': history,
            'config': {
                'BATCH_SIZE': config.BATCH_SIZE,
                'LEARNING_RATE': config.LEARNING_RATE,
                'ENC_HIDDEN_SIZE': config.ENC_HIDDEN_SIZE,
                'ENC_LATENT_DIM': config.ENC_LATENT_DIM,
                'DEC_HIDDEN_SIZE': config.DEC_HIDDEN_SIZE,
                'KL_WEIGHT': config.KL_WEIGHT,
                'TEACHER_FORCING_RATIO': config.TEACHER_FORCING_RATIO
            },
            'total_parameters': total_params,
            'timestamp': timestamp
        }, model_save_path)
        
        print(f"\n✅ Training completed successfully!")
        print(f"💾 Model saved to: {model_save_path}")
        print(f"📈 Training Results:")
        print(f"   - Best validation loss: {min(history['val_loss']):.6f}")
        print(f"   - Final training loss: {history['train_loss'][-1]:.6f}")
        print(f"   - Final validation loss: {history['val_loss'][-1]:.6f}")
        
        # Plot training history
        plot_path = os.path.join(
            config.MODEL_SAVE_DIR, 
            f'{args.model_name}_{timestamp}_history.png'
        )
        plot_training_history(history, plot_path)
        
        # Create a summary file
        summary_path = os.path.join(
            config.MODEL_SAVE_DIR,
            f'{args.model_name}_{timestamp}_summary.txt'
        )
        
        with open(summary_path, 'w') as f:
            f.write(f"GrooVAE Training Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Model Name: {args.model_name}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Description: {args.description}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"  Epochs: {config.EPOCHS}\n")
            f.write(f"  Batch Size: {config.BATCH_SIZE}\n")
            f.write(f"  Learning Rate: {config.LEARNING_RATE}\n")
            f.write(f"  Model Architecture: Enc({config.ENC_HIDDEN_SIZE}→{config.ENC_LATENT_DIM}), Dec({config.DEC_HIDDEN_SIZE})\n")
            f.write(f"  Total Parameters: {total_params:,}\n")
            f.write(f"  KL Weight: {config.KL_WEIGHT}\n")
            f.write(f"  Teacher Forcing: {config.TEACHER_FORCING_RATIO}\n")
            f.write(f"\nResults:\n")
            f.write(f"  Best Validation Loss: {min(history['val_loss']):.6f}\n")
            f.write(f"  Final Training Loss: {history['train_loss'][-1]:.6f}\n")
            f.write(f"  Final Validation Loss: {history['val_loss'][-1]:.6f}\n")
        
        print(f"📄 Training summary saved to: {summary_path}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        # Save emergency checkpoint
        emergency_path = os.path.join(
            config.MODEL_SAVE_DIR, 
            f'{args.model_name}_emergency_{timestamp}.pth'
        )
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'enc_optimizer_state_dict': enc_optimizer.state_dict(),
            'dec_optimizer_state_dict': dec_optimizer.state_dict(),
            'epoch': 0,  # Unknown epoch
            'dataset': args.dataset,
            'model_name': args.model_name,
            'description': f"Emergency save - {args.description}",
            'timestamp': timestamp
        }, emergency_path)
        print(f"💾 Emergency checkpoint saved to: {emergency_path}")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()