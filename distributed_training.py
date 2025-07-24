"""
Distributed training support for GrooVAE on multiple H100s
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import logging
from typing import Tuple, List

def setup_distributed(rank: int, world_size: int, port: str = "12355"):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def create_ddp_model(model, device_id: int):
    """Wrap model for distributed training"""
    model = model.to(device_id)
    return DDP(model, device_ids=[device_id])

def create_distributed_dataloader(dataset, batch_size: int, rank: int, world_size: int, 
                                num_workers: int = 8, shuffle: bool = True):
    """Create distributed dataloader"""
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=shuffle
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    
    return dataloader, sampler

def reduce_loss(loss: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reduce loss across all processes"""
    with torch.no_grad():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= world_size
    return loss

class DistributedGrooVAETrainer:
    """Distributed trainer for GrooVAE"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logger for distributed training"""
        logger = logging.getLogger(f'GrooVAE_rank_{self.rank}')
        if self.rank == 0:  # Only log from rank 0
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        return logger
    
    def train_epoch_distributed(self, encoder, decoder, train_loader, optimizers, 
                               scaler, kl_weight: float, epoch: int):
        """Train one epoch with distributed training"""
        encoder.train()
        decoder.train()
        
        train_loader.sampler.set_epoch(epoch)  # Important for proper shuffling
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = len(train_loader)
        
        enc_optimizer, dec_optimizer = optimizers
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device, non_blocking=True)
            
            x_train = batch[:, :, :27]
            x_train_target = batch[:, :, 27:]
            
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                z, x_train_mu, x_train_std = encoder(x_train)
                output, output_hits, output_velocities, output_offsets = decoder(
                    z, x_train.size(1), target=x_train_target, teacher_forcing_ratio=0.5
                )
                
                reconstruction_loss = decoder.module.compute_loss(
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
            
            # Reduce losses for logging
            loss_reduced = reduce_loss(loss.clone().detach(), self.world_size)
            recon_loss_reduced = reduce_loss(reconstruction_loss.clone().detach(), self.world_size)
            kl_loss_reduced = reduce_loss(kl_loss.clone().detach(), self.world_size)
            
            total_loss += loss_reduced.item()
            total_recon_loss += recon_loss_reduced.item()
            total_kl_loss += kl_loss_reduced.item()
        
        return (total_loss / num_batches, 
                total_recon_loss / num_batches, 
                total_kl_loss / num_batches)
    
    def save_checkpoint_distributed(self, encoder, decoder, epoch: int, history: dict):
        """Save checkpoint from rank 0 only"""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.module.state_dict(),  # Remove DDP wrapper
                'decoder_state_dict': decoder.module.state_dict(),  # Remove DDP wrapper
                'history': history
            }
            
            from config import Config
            checkpoint_path = Config.get_model_save_path(f'distributed_checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved at epoch {epoch}")

def run_distributed_training(rank: int, world_size: int, config_overrides: dict = None):
    """Main function for distributed training"""
    try:
        # Setup distributed
        setup_distributed(rank, world_size)
        
        # Import here to avoid issues with distributed initialization
        from config import Config
        from model import Encoder, Decoder
        from data_loader import setup_data_loaders
        from memory_utils import setup_h100_memory_optimization
        
        # Apply config overrides
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(Config, key, value)
        
        # Setup memory optimization
        setup_h100_memory_optimization()
        
        # Create trainer
        trainer = DistributedGrooVAETrainer(rank, world_size)
        
        # Setup data loaders with distributed sampling
        train_loader, val_loader, test_loader = setup_data_loaders()
        
        # Recreate train_loader with distributed sampler
        from data_loader import GrooVAEDataset
        from data_loader import load_data
        
        train_data = load_data(Config.get_data_file_path(Config.TRAIN_FILE))
        train_dataset = GrooVAEDataset(train_data)
        
        train_loader, train_sampler = create_distributed_dataloader(
            train_dataset, 
            Config.BATCH_SIZE // world_size,  # Scale batch size
            rank, 
            world_size,
            Config.NUM_WORKERS
        )
        
        # Create models
        encoder = Encoder(
            Config.ENCODER_INPUT_SIZE,
            Config.ENCODER_HIDDEN_SIZE,
            Config.ENCODER_LATENT_DIM
        )
        decoder = Decoder(
            Config.DECODER_INPUT_SIZE,
            Config.DECODER_HIDDEN_SIZE,
            Config.DECODER_OUTPUT_SIZE
        )
        
        # Wrap models for distributed training
        encoder = create_ddp_model(encoder, rank)
        decoder = create_ddp_model(decoder, rank)
        
        # Create optimizers
        enc_optimizer = torch.optim.AdamW(encoder.parameters(), lr=Config.LEARNING_RATE)
        dec_optimizer = torch.optim.AdamW(decoder.parameters(), lr=Config.LEARNING_RATE)
        optimizers = [enc_optimizer, dec_optimizer]
        
        # Create scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        # Training loop
        history = {'train_loss': [], 'train_recon_loss': [], 'train_kl_loss': []}
        
        for epoch in range(1, Config.EPOCHS + 1):
            train_loss, train_recon_loss, train_kl_loss = trainer.train_epoch_distributed(
                encoder, decoder, train_loader, optimizers, scaler, Config.KL_WEIGHT, epoch
            )
            
            history['train_loss'].append(train_loss)
            history['train_recon_loss'].append(train_recon_loss)
            history['train_kl_loss'].append(train_kl_loss)
            
            if rank == 0:
                trainer.logger.info(
                    f'Epoch {epoch:3d}/{Config.EPOCHS} - '
                    f'Train Loss: {train_loss:.4f} '
                    f'(Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f})'
                )
            
            # Save checkpoint
            if epoch % 10 == 0:
                trainer.save_checkpoint_distributed(encoder, decoder, epoch, history)
        
        # Final save
        trainer.save_checkpoint_distributed(encoder, decoder, Config.EPOCHS, history)
        
    except Exception as e:
        if rank == 0:
            logging.error(f"Distributed training failed: {e}")
        raise
    finally:
        cleanup_distributed()

def launch_distributed_training(world_size: int = None, config_overrides: dict = None):
    """Launch distributed training"""
    if world_size is None:
        world_size = torch.cuda.device_count()
    
    if world_size < 2:
        raise ValueError("Distributed training requires at least 2 GPUs")
    
    print(f"Launching distributed training on {world_size} GPUs")
    
    mp.spawn(
        run_distributed_training,
        args=(world_size, config_overrides),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # Example usage
    launch_distributed_training(world_size=2)