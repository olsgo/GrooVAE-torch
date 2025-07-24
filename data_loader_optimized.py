"""
Optimized data loading and preprocessing for M1 Max
"""
import pickle
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing as mp
from config import Config


class OptimizedDatasetSampler(Dataset):
    """
    Optimized dataset for M1 Max with:
    - Memory efficient loading
    - Proper tensor types
    - Data caching
    """
    def __init__(self, data, device_type='mps'):
        self.data = data
        self.device_type = device_type
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Ensure proper tensor type for M1 Max
        if isinstance(item, torch.Tensor):
            return item.to(torch.float32)
        else:
            return torch.tensor(item, dtype=torch.float32)


class DataLoaderManager:
    """
    Manages data loading optimized for M1 Max architecture
    """
    def __init__(self, config=None):
        self.config = config or Config()
        self.cache = {}
        
    def load_data_file(self, filepath, cache_key=None):
        """
        Load data file with caching for M1 Max's large memory
        """
        if cache_key and cache_key in self.cache:
            print(f"Loading {cache_key} from cache")
            return self.cache[cache_key]
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        print(f"Loading data from {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Shuffle data for better training
        random.shuffle(data)
        
        # Cache in memory if we have enough RAM (M1 Max has 64GB)
        if cache_key:
            self.cache[cache_key] = data
            print(f"Cached {cache_key} in memory ({len(data)} samples)")
        
        return data
    
    def create_data_loaders(self, data_type='tapify'):
        """
        Create optimized data loaders for M1 Max
        
        Args:
            data_type: 'tapify' or 'humanize'
        """
        if data_type not in self.config.TRAIN_FILES:
            raise ValueError(f"Unknown data type: {data_type}")
        
        file_names = self.config.TRAIN_FILES[data_type]
        data_names = ['train_data', 'val_data', 'test_data']
        
        datasets = {}
        
        for file_name, data_name in zip(file_names, data_names):
            filepath = os.path.join(self.config.PROCESSED_DATA_DIR, file_name)
            
            # Load with caching
            data = self.load_data_file(filepath, cache_key=f"{data_type}_{data_name}")
            
            print(f'Loaded {data_name}: {len(data)} samples')
            
            # Create dataset
            datasets[data_name] = OptimizedDatasetSampler(data)
        
        # Create data loaders with M1 Max optimized parameters
        train_params = self.config.get_data_loader_params()
        val_params = self.config.get_val_data_loader_params()
        
        train_loader = DataLoader(datasets['train_data'], **train_params)
        val_loader = DataLoader(datasets['val_data'], **val_params)
        test_loader = DataLoader(datasets['test_data'], **val_params)
        
        print(f"Created data loaders:")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        print(f"  - Batch size (train/val): {train_params['batch_size']}/{val_params['batch_size']}")
        print(f"  - Num workers: {train_params['num_workers']}")
        
        return train_loader, val_loader, test_loader
    
    def preprocess_parallel(self, file_paths, output_dir, data_type='tapify', num_workers=None):
        """
        Parallel preprocessing optimized for M1 Max's multiple cores
        """
        if num_workers is None:
            # Use M1 Max's performance cores efficiently
            num_workers = min(8, mp.cpu_count())
        
        print(f"Starting parallel preprocessing with {num_workers} workers")
        
        # This would be used for preprocessing MIDI files
        # Implementation depends on the specific preprocessing pipeline
        pass
    
    def get_memory_usage(self):
        """
        Get current memory usage of cached data
        """
        total_size = 0
        for key, data in self.cache.items():
            if isinstance(data, list):
                # Estimate size of tensor list
                if data and hasattr(data[0], 'numel'):
                    size_per_item = data[0].numel() * 4  # float32 = 4 bytes
                    total_size += len(data) * size_per_item
            
        total_size_mb = total_size / (1024 * 1024)
        print(f"Data cache memory usage: {total_size_mb:.1f} MB")
        return total_size_mb
    
    def clear_cache(self):
        """Clear data cache to free memory"""
        self.cache.clear()
        print("Data cache cleared")


def create_optimized_data_loaders(data_type='tapify', config=None):
    """
    Convenience function to create optimized data loaders
    """
    if config is None:
        config = Config()
    
    # Ensure directories exist
    config.create_directories()
    
    # Create data loader manager
    manager = DataLoaderManager(config)
    
    # Create and return data loaders
    return manager.create_data_loaders(data_type)


def benchmark_data_loading(data_loader, device, num_batches=10):
    """
    Benchmark data loading performance on M1 Max
    """
    print(f"Benchmarking data loading performance...")
    
    import time
    times = []
    
    for i, batch in enumerate(data_loader):
        start_time = time.time()
        
        # Move to device (simulating training)
        batch = batch.to(device, non_blocking=True)
        
        # Simulate some processing
        _ = batch.mean()
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        if i >= num_batches - 1:
            break
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Data loading benchmark results:")
    print(f"  - Average time per batch: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"  - Batches per second: {1/avg_time:.2f}")
    print(f"  - Device: {device}")
    
    return avg_time, std_time