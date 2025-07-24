import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import logging
from config import Config

def setup_logging():
    """Setup logging for training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.OUTPUT_PATH / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def groove_train_optimized(device, train_loader, val_loader, model, optimizer, 
                          epochs=100, kl_weight=0.001):
    """
    Optimized training function for H100 SXM
    """
    logger = setup_logging()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
    }
    
    encoder, decoder = model
    enc_optimizer, dec_optimizer = optimizer
    
    # Learning rate schedulers
    enc_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        enc_optimizer, epochs, eta_min=1e-6
    )
    dec_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        dec_optimizer, epochs, eta_min=1e-6
    )
    
    # Mixed precision scaler for H100
    scaler = GradScaler() if Config.USE_MIXED_PRECISION else None
    
    # Compile models for H100 optimization
    if Config.USE_TORCH_COMPILE:
        try:
            encoder = torch.compile(encoder, mode='max-autotune')
            decoder = torch.compile(decoder, mode='max-autotune')
            logger.info("Models compiled with torch.compile for H100 optimization")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}, continuing without compilation")
    
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Mixed precision: {Config.USE_MIXED_PRECISION}")
    logger.info(f"Batch size: {Config.BATCH_SIZE}")
    logger.info(f"Device: {device}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Training phase
        train_loss, train_recon_loss, train_kl_loss = train_epoch(
            encoder, decoder, train_loader, enc_optimizer, dec_optimizer,
            device, kl_weight, scaler, epoch, logger
        )
        
        # Validation phase
        val_loss = validate_epoch(
            encoder, decoder, val_loader, device, kl_weight, epoch, logger
        )
        
        # Update schedulers
        enc_scheduler.step()
        dec_scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon_loss'].append(train_recon_loss)
        history['train_kl_loss'].append(train_kl_loss)
        
        # Log epoch results
        epoch_time = time.time() - start_time
        logger.info(
            f'Epoch {epoch:3d}/{epochs} ({epoch_time:.1f}s) - '
            f'Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, '
            f'KL: {train_kl_loss:.4f}) - Val Loss: {val_loss:.4f} - '
            f'LR: {enc_scheduler.get_last_lr()[0]:.6f}'
        )
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(encoder, decoder, epoch, history)
            logger.info(f"Checkpoint saved at epoch {epoch}")

    return history

def train_epoch(encoder, decoder, train_loader, enc_optimizer, dec_optimizer,
                device, kl_weight, scaler, epoch, logger):
    """Training for one epoch with H100 optimizations"""
    encoder.train()
    decoder.train()
    
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
        batch = batch.to(device, non_blocking=True)
        batch_size = batch.size(0)
        seq_len = batch.size(1)
        
        x_train = batch[:, :, :27]   
        x_train_target = batch[:, :, 27:]  
        
        # Zero gradients
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        
        if Config.USE_MIXED_PRECISION and scaler is not None:
            # Mixed precision forward pass
            with autocast():
                # Encoder
                z, x_train_mu, x_train_std = encoder(x_train)
                
                # Decoder
                output, output_hits, output_velocities, output_offsets = decoder(
                    z, seq_len, target=x_train_target, teacher_forcing_ratio=0.5
                )
                
                # Loss calculation
                reconstruction_loss = decoder.compute_loss(
                    x_train_target, output_hits, output_velocities, output_offsets
                )
                kl_loss = -0.5 * torch.sum(
                    1 + x_train_std - x_train_mu.pow(2) - x_train_std.exp()
                ) / x_train_mu.size(0)
                loss = reconstruction_loss + kl_weight * kl_loss
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(enc_optimizer)
            scaler.step(dec_optimizer)
            scaler.update()
        else:
            # Standard precision
            # Encoder
            z, x_train_mu, x_train_std = encoder(x_train)
            
            # Decoder
            output, output_hits, output_velocities, output_offsets = decoder(
                z, seq_len, target=x_train_target, teacher_forcing_ratio=0.5
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
            enc_optimizer.step()
            dec_optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += reconstruction_loss.item()
        total_kl_loss += kl_loss.item()
    
    return (total_loss / num_batches, 
            total_recon_loss / num_batches, 
            total_kl_loss / num_batches)

def validate_epoch(encoder, decoder, val_loader, device, kl_weight, epoch, logger):
    """Validation for one epoch"""
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Val Epoch {epoch}")):
            batch = batch.to(device, non_blocking=True)
            batch_size = batch.size(0)
            seq_len = batch.size(1)
            
            x_val = batch[:, :, :27]  
            x_val_target = batch[:, :, 27:]  
            
            if Config.USE_MIXED_PRECISION:
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
            
            total_loss += loss.item()
    
    return total_loss / num_batches

def save_checkpoint(encoder, decoder, epoch, history):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'history': history
    }
    
    checkpoint_path = Config.get_model_save_path(f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save latest checkpoint
    latest_path = Config.get_model_save_path('latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)