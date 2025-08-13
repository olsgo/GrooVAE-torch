import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from time import time
import os
from config import Config

def groove_train_optimized(device, train_loader, val_loader, model, optimizer, epochs=100, config=None):
    """
    Optimized training function for M1 Max with:
    - Mixed precision training
    - Gradient accumulation
    - Better scheduling
    - Memory optimization
    """
    if config is None:
        config = Config()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    encoder, decoder = model
    enc_optimizer, dec_optimizer = optimizer
    
    # Initialize mixed precision scaler for M1 Max
    use_amp = config.ENABLE_MIXED_PRECISION and device.type == 'mps'
    scaler = GradScaler() if use_amp else None
    
    # Learning rate schedulers optimized for larger batches
    if config.LR_SCHEDULER == 'cosine':
        enc_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            enc_optimizer, epochs, eta_min=config.LR_MIN
        )
        dec_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            dec_optimizer, epochs, eta_min=config.LR_MIN
        )
    elif config.LR_SCHEDULER == 'step':
        enc_scheduler = optim.lr_scheduler.StepLR(
            enc_optimizer, step_size=epochs//3, gamma=0.1
        )
        dec_scheduler = optim.lr_scheduler.StepLR(
            dec_optimizer, step_size=epochs//3, gamma=0.1
        )
    else:
        enc_scheduler = optim.lr_scheduler.ExponentialLR(
            enc_optimizer, gamma=0.95
        )
        dec_scheduler = optim.lr_scheduler.ExponentialLR(
            dec_optimizer, gamma=0.95
        )

    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # KL annealing - gradually increase KL weight
        if hasattr(config, 'KL_WARMUP_EPOCHS'):
            kl_weight = min(config.KL_WEIGHT, epoch * config.KL_WEIGHT / config.KL_WARMUP_EPOCHS)
        else:
            kl_weight = config.KL_WEIGHT
        
        # Add disk space check every 10 epochs
        if epoch % 10 == 0:
            import shutil
            total, used, free = shutil.disk_usage(config.MODEL_SAVE_DIR)
            free_gb = free / (1024**3)
            print(f"💾 Disk space check (Epoch {epoch}): {free_gb:.1f}GB available")
            
            if free_gb < 5:
                print(f"🚨 STOPPING: Critical disk space ({free_gb:.1f}GB)")
                break
        
        start_time = time()
        
        train_loss = 0
        val_loss = 0
        
        # Training phase with optimizations
        encoder.train()
        decoder.train()
        
        # Progress bar with memory info
        train_pbar = tqdm(enumerate(train_loader), 
                         total=len(train_loader),
                         desc=f"Train Epoch {epoch}")
        
        for batch_idx, batch in train_pbar:
            batch = batch.to(device, non_blocking=True)
            batch_size = batch.size(0)
            seq_len = batch.size(1)
            
            x_train = batch[:, :, :27]   
            x_train_target = batch[:, :, 27:]  
            
            # Zero gradients
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast():
                    # Encoder
                    z, x_train_mu, x_train_std = encoder(x_train)
                    
                    # Decoder
                    output, output_hits, output_velocities, output_offsets = decoder(
                        z, seq_len, target=x_train_target, 
                        teacher_forcing_ratio=config.TEACHER_FORCING_RATIO
                    )
                    
                    # Loss calculation
                    reconstruction_loss = decoder.compute_loss(
                        x_train_target, output_hits, output_velocities, output_offsets
                    )
                    kl_loss = -0.5 * torch.sum(
                        1 + x_train_std - x_train_mu.pow(2) - x_train_std.exp()
                    ) / x_train_mu.size(0)
                    loss = reconstruction_loss + kl_weight * kl_loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(enc_optimizer)
                scaler.unscale_(dec_optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.MAX_GRAD_NORM)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.MAX_GRAD_NORM)
                
                # Optimizer step
                scaler.step(enc_optimizer)
                scaler.step(dec_optimizer)
                scaler.update()
                
            else:
                # Standard forward pass for CPU or non-AMP training
                # Encoder
                z, x_train_mu, x_train_std = encoder(x_train)
                
                # Decoder
                output, output_hits, output_velocities, output_offsets = decoder(
                    z, seq_len, target=x_train_target, 
                    teacher_forcing_ratio=config.TEACHER_FORCING_RATIO
                )
                
                # Loss calculation
                reconstruction_loss = decoder.compute_loss(
                    x_train_target, output_hits, output_velocities, output_offsets
                )
                kl_loss = -0.5 * torch.sum(
                    1 + x_train_std - x_train_mu.pow(2) - x_train_std.exp()
                ) / x_train_mu.size(0)
                loss = reconstruction_loss + kl_weight * kl_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.MAX_GRAD_NORM)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.MAX_GRAD_NORM)
                
                # Optimizer step
                enc_optimizer.step()
                dec_optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            if batch_idx % config.LOG_INTERVAL == 0:
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'rec_loss': f'{reconstruction_loss.item():.4f}',
                    'kl_loss': f'{kl_loss.item():.4f}',
                    'kl_weight': f'{kl_weight:.4f}'
                })
        
        # Update schedulers
        enc_scheduler.step()
        dec_scheduler.step()
        
        train_loss = train_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        history['learning_rates'].append(enc_scheduler.get_last_lr()[0])

        # Validation phase
        encoder.eval()
        decoder.eval()
        
        val_pbar = tqdm(enumerate(val_loader), 
                       total=len(val_loader),
                       desc=f"Val Epoch {epoch}")
        
        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                batch = batch.to(device, non_blocking=True)
                batch_size = batch.size(0)
                seq_len = batch.size(1)
                
                x_val = batch[:, :, :27]  
                x_val_target = batch[:, :, 27:]  
                
                if use_amp:
                    with autocast():
                        # Forward pass
                        z, x_val_mu, x_val_std = encoder(x_val)
                        output, output_hits, output_velocities, output_offsets = decoder(
                            z, seq_len, target=None, teacher_forcing_ratio=0.0
                        )
                        
                        # Loss calculation
                        reconstruction_loss = decoder.compute_loss(
                            x_val_target, output_hits, output_velocities, output_offsets
                        )
                        kl_loss = -0.5 * torch.sum(
                            1 + x_val_std - x_val_mu.pow(2) - x_val_std.exp()
                        ) / x_val_mu.size(0)
                        loss = reconstruction_loss + kl_weight * kl_loss
                else:
                    # Forward pass
                    z, x_val_mu, x_val_std = encoder(x_val)
                    output, output_hits, output_velocities, output_offsets = decoder(
                        z, seq_len, target=None, teacher_forcing_ratio=0.0
                    )
                    
                    # Loss calculation
                    reconstruction_loss = decoder.compute_loss(
                        x_val_target, output_hits, output_velocities, output_offsets
                    )
                    kl_loss = -0.5 * torch.sum(
                        1 + x_val_std - x_val_mu.pow(2) - x_val_std.exp()
                    ) / x_val_mu.size(0)
                    loss = reconstruction_loss + kl_weight * kl_loss
                
                val_loss += loss.item()
                
                # Update validation progress bar
                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        val_loss = val_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Print epoch summary
        elapsed_time = time() - start_time
        print(f'Epoch {epoch} ({elapsed_time:.2f}s) - '
              f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, '
              f'lr: {enc_scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if config.SAVE_BEST_MODEL and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(encoder, decoder, enc_optimizer, dec_optimizer, 
                          epoch, val_loss, config, is_best=True)
            print(f"New best model saved with val_loss: {val_loss:.4f}")
        
        # Save checkpoint every N epochs
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(encoder, decoder, enc_optimizer, dec_optimizer, 
                          epoch, val_loss, config, is_best=False)
    
    return history

def save_checkpoint(encoder, decoder, enc_optimizer, dec_optimizer, epoch, val_loss, config, is_best=False):
    "Save model checkpoint with enhanced size and disk monitoring"
    import shutil
    
    # Check available disk space before saving
    total, used, free = shutil.disk_usage(config.MODEL_SAVE_DIR)
    free_gb = free / (1024**3)
    
    if free_gb < 5:  # Less than 5GB available
        print(f"🚨 CRITICAL: Very low disk space ({free_gb:.1f} GB)")
        print("Skipping checkpoint to prevent disk full")
        return
    elif free_gb < 10:  # Less than 10GB available
        print(f"⚠️  WARNING: Low disk space ({free_gb:.1f} GB)")
    
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        # Remove optimizer states to reduce size
        'val_loss': val_loss,
    }
    
    if is_best:
        filename = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')
    else:
        filename = os.path.join(config.MODEL_SAVE_DIR, f'checkpoint_epoch_{epoch}.pth')
    
    # Save and monitor size
    torch.save(checkpoint, filename)
    file_size = os.path.getsize(filename) / (1024*1024)  # MB
    print(f"💾 Checkpoint saved: {file_size:.1f}MB - {filename}")
    print(f"💾 Remaining disk space: {free_gb:.1f}GB")
    
    # Alert if file is suspiciously large
    if file_size > 100:  # Alert if > 100MB
        print(f"⚠️  WARNING: Checkpoint file is unusually large ({file_size:.1f}MB)")

def load_checkpoint(checkpoint_path, encoder, decoder, enc_optimizer=None, dec_optimizer=None):
    "Load model checkpoint"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    if enc_optimizer is not None:
        enc_optimizer.load_state_dict(checkpoint['enc_optimizer_state_dict'])
    if dec_optimizer is not None:
        dec_optimizer.load_state_dict(checkpoint['dec_optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['val_loss']


def emergency_cleanup(config):
    """Emergency cleanup of old checkpoints when disk space is critical"""
    import glob
    import os
    
    checkpoint_pattern = os.path.join(config.MODEL_SAVE_DIR, "checkpoint_epoch_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if len(checkpoints) > 3:  # Keep only 3 most recent
        checkpoints.sort(key=os.path.getmtime)
        old_checkpoints = checkpoints[:-3]
        
        for checkpoint in old_checkpoints:
            try:
                size_mb = os.path.getsize(checkpoint) / (1024*1024)
                os.remove(checkpoint)
                print(f"🗑️  Removed old checkpoint: {checkpoint} ({size_mb:.1f}MB)")
            except Exception as e:
                print(f"Failed to remove {checkpoint}: {e}")