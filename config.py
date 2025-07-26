"""
Configuration for GrooVAE-torch optimized for M1 Max
"""
import os
import torch

class Config:
    # Data paths - make them relative or configurable
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MIDI_DATA_DIR = os.path.join(DATA_DIR, 'midi_data', 'groove')
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
    
    # Device configuration optimized for M1 Max
    @staticmethod
    def get_device():
        """Get the best available device with M1 Max optimizations"""
        if torch.backends.mps.is_available():
            # M1 Max Metal Performance Shaders
            device = torch.device('mps')
            print(f"Using MPS (Metal Performance Shaders) on M1 Max")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device
    
    # M1 Max optimized hyperparameters
    # Reduced batch sizes to handle large datasets
    BATCH_SIZE = 256  # Further reduced from 512
    BATCH_SIZE_VAL = 256  # Keep validation batch size consistent
    
    # Data loading optimized for M1 Max
    # M1 Max has 10 performance cores, optimize worker count
    NUM_WORKERS = 8  # Optimal for M1 Max (leave 2 cores for system)
    PIN_MEMORY = True  # Works well with unified memory
    PREFETCH_FACTOR = 4  # Increased for better throughput
    
    # Model architecture
    ENC_INPUT_SIZE = 27
    ENC_HIDDEN_SIZE = 512
    ENC_LATENT_DIM = 256
    DEC_INPUT_SIZE = 256
    DEC_HIDDEN_SIZE = 256
    DEC_OUTPUT_SIZE = 27
    
    # Training parameters optimized for larger batches
    LEARNING_RATE = 2e-3  # Increased for larger batch size
    EPOCHS = 100
    KL_WEIGHT = 0.001
    TEACHER_FORCING_RATIO = 0.5
    
    # M1 Max specific optimizations
    ENABLE_MIXED_PRECISION = True  # Use automatic mixed precision
    GRADIENT_ACCUMULATION_STEPS = 1  # Can increase if needed
    MAX_GRAD_NORM = 1.0  # Gradient clipping for stability
    
    # Learning rate scheduling
    LR_SCHEDULER = 'cosine'  # Options: 'cosine', 'step', 'exponential'
    LR_MIN = 1e-6
    
    # Data file names
    @classmethod
    def get_available_datasets(cls):
        """Dynamically discover available datasets from processed directory"""
        import glob
        
        if not os.path.exists(cls.PROCESSED_DATA_DIR):
            return {}
        
        # Find all *_train.pkl files
        train_files = glob.glob(os.path.join(cls.PROCESSED_DATA_DIR, '*_train.pkl'))
        
        datasets = {}
        for train_file in train_files:
            # Extract dataset name (remove _train.pkl suffix)
            basename = os.path.basename(train_file)
            dataset_name = basename.replace('_train.pkl', '')
            
            # Check if corresponding valid and test files exist
            valid_file = os.path.join(cls.PROCESSED_DATA_DIR, f'{dataset_name}_valid.pkl')
            test_file = os.path.join(cls.PROCESSED_DATA_DIR, f'{dataset_name}_test.pkl')
            
            if os.path.exists(valid_file) and os.path.exists(test_file):
                datasets[dataset_name] = [
                    f'{dataset_name}_train.pkl',
                    f'{dataset_name}_valid.pkl', 
                    f'{dataset_name}_test.pkl'
                ]
        
        return datasets
    
    # Replace the hardcoded TRAIN_FILES with dynamic discovery
    @property
    def TRAIN_FILES(self):
        return self.get_available_datasets()
    
    # Model saving
    SAVE_EVERY_N_EPOCHS = 10
    SAVE_BEST_MODEL = True
    
    # Monitoring
    LOG_INTERVAL = 10  # Log every N batches
    ENABLE_TENSORBOARD = False  # Set to True if tensorboard is installed
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_SAVE_DIR, exist_ok=True)
    
    @classmethod
    def get_data_loader_params(cls):
        """Get optimized data loader parameters for M1 Max"""
        return {
            'batch_size': cls.BATCH_SIZE,
            'shuffle': True,
            'pin_memory': cls.PIN_MEMORY,
            'num_workers': cls.NUM_WORKERS,
            'prefetch_factor': cls.PREFETCH_FACTOR,
            'persistent_workers': True,  # Reuse workers across epochs
        }
    
    @classmethod
    def get_val_data_loader_params(cls):
        """Get validation data loader parameters"""
        params = cls.get_data_loader_params()
        params['batch_size'] = cls.BATCH_SIZE_VAL
        params['shuffle'] = False
        return params