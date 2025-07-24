import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import logging
from pathlib import Path
from config import Config

class GrooVAEDataset(Dataset):
    """Optimized dataset class for GrooVAE"""
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample.to(torch.float32)

def load_data(file_path, shuffle=True):
    """Load and optionally shuffle data from pickle file"""
    logger = logging.getLogger(__name__)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if shuffle:
            random.shuffle(data)
            
        logger.info(f'Loaded {len(data)} samples from {file_path}')
        return data
    
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        logger.info("Please ensure data files are in the correct location:")
        logger.info(f"Expected path: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def create_optimized_dataloader(dataset, batch_size=None, shuffle=True, num_workers=None):
    """Create optimized DataLoader for H100"""
    
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS
    
    # H100-optimized DataLoader parameters
    dataloader_params = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': Config.PIN_MEMORY,
        'persistent_workers': True if num_workers > 0 else False,
        'prefetch_factor': Config.PREFETCH_FACTOR if num_workers > 0 else 2,
        'drop_last': True,  # For consistent batch sizes
    }
    
    return DataLoader(dataset, **dataloader_params)

def setup_data_loaders():
    """Setup train, validation, and test data loaders"""
    logger = logging.getLogger(__name__)
    logger.info("Setting up data loaders...")
    
    # Load data files
    train_data = load_data(Config.get_data_file_path(Config.TRAIN_FILE))
    val_data = load_data(Config.get_data_file_path(Config.VAL_FILE))
    test_data = load_data(Config.get_data_file_path(Config.TEST_FILE))
    
    # Create datasets
    train_dataset = GrooVAEDataset(train_data)
    val_dataset = GrooVAEDataset(val_data)
    test_dataset = GrooVAEDataset(test_data)
    
    # Create data loaders
    train_loader = create_optimized_dataloader(train_dataset, shuffle=True)
    val_loader = create_optimized_dataloader(val_dataset, shuffle=False)
    test_loader = create_optimized_dataloader(test_dataset, shuffle=False)
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    logger.info(f"  Batch size: {Config.BATCH_SIZE}")
    logger.info(f"  Num workers: {Config.NUM_WORKERS}")
    
    return train_loader, val_loader, test_loader

def verify_data_format(data_loader, expected_shape=None):
    """Verify data format and shape"""
    logger = logging.getLogger(__name__)
    
    for batch in data_loader:
        logger.info(f"Batch shape: {batch.shape}")
        logger.info(f"Batch dtype: {batch.dtype}")
        logger.info(f"Batch device: {batch.device}")
        
        if expected_shape:
            if batch.shape[1:] != expected_shape:
                logger.warning(f"Expected shape {expected_shape}, got {batch.shape[1:]}")
        
        # Check for NaN or inf values
        if torch.isnan(batch).any():
            logger.warning("Found NaN values in batch")
        if torch.isinf(batch).any():
            logger.warning("Found inf values in batch")
            
        break  # Only check first batch
    
    logger.info("Data format verification complete")