import os
from pathlib import Path

class Config:
    """Configuration class for GrooVAE training"""
    
    # Environment
    RUNPOD = os.getenv('RUNPOD', 'false').lower() == 'true'
    
    # Paths
    if RUNPOD:
        BASE_PATH = Path('/workspace')
        DATA_PATH = BASE_PATH / 'data' / 'data_processed'
        MODEL_PATH = BASE_PATH / 'models'
        OUTPUT_PATH = BASE_PATH / 'outputs'
    else:
        BASE_PATH = Path(os.getcwd())
        DATA_PATH = BASE_PATH / 'data'
        MODEL_PATH = BASE_PATH / 'model'
        OUTPUT_PATH = BASE_PATH / 'outputs'
    
    # Create directories if they don't exist
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Model hyperparameters
    ENCODER_INPUT_SIZE = 27
    ENCODER_HIDDEN_SIZE = 512
    ENCODER_LATENT_DIM = 256
    
    DECODER_INPUT_SIZE = 256
    DECODER_HIDDEN_SIZE = 256
    DECODER_OUTPUT_SIZE = 27
    DECODER_NUM_LAYERS = 2
    
    # Training hyperparameters
    BATCH_SIZE = 512  # Optimized for H100
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    KL_WEIGHT = 0.001
    
    # H100 optimizations
    USE_MIXED_PRECISION = True
    USE_TORCH_COMPILE = True
    NUM_WORKERS = 8  # Optimized for H100 memory bandwidth
    PIN_MEMORY = True
    PREFETCH_FACTOR = 4
    
    # Data files
    TRAIN_FILE = 'tapify_train.pkl'
    VAL_FILE = 'tapify_valid.pkl'
    TEST_FILE = 'tapify_test.pkl'
    
    # Model naming
    MODEL_NAME = 'groovae_h100_optimized'
    
    @classmethod
    def get_data_file_path(cls, filename):
        return cls.DATA_PATH / filename
    
    @classmethod
    def get_model_save_path(cls, suffix=''):
        if suffix:
            return cls.MODEL_PATH / f"{cls.MODEL_NAME}_{suffix}"
        return cls.MODEL_PATH / cls.MODEL_NAME