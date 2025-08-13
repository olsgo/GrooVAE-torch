from config import Config

class ConfigOptimized(Config):
    "Magenta-aligned configuration for realistic drum pattern generation"
    
    # Maximize batch size for your 64GB RAM
    BATCH_SIZE = 1024
    BATCH_SIZE_VAL = 1024
    
    # MAGENTA-ALIGNED MODEL ARCHITECTURE
    # Reduced dimensions for better stability and generalization
    ENC_HIDDEN_SIZE = 512   # Magenta recommendation: 512 (was 1024)
    ENC_LATENT_DIM = 256    # Magenta recommendation: 256 (was 512)
    DEC_INPUT_SIZE = 256    # MUST match ENC_LATENT_DIM!
    DEC_HIDDEN_SIZE = 256   # Magenta recommendation: 256 (was 512)
    
    # CRITICAL: VARIATIONAL INFORMATION BOTTLENECK
    # This is the most important change for realistic patterns
    KL_WEIGHT = 0.2         # Magenta recommendation: 0.2 (was 0.0005)
    
    # Adjusted learning rate for smaller model
    LEARNING_RATE = 5e-4    # Reduced for better stability with smaller architecture
    TEACHER_FORCING_RATIO = 0.5  # Magenta standard (reduced from 0.7)
    
    # Enhanced training stability
    MAX_GRAD_NORM = 0.5
    LR_SCHEDULER = 'cosine'
    LR_MIN = 1e-7
    
    # Checkpointing
    SAVE_EVERY_N_EPOCHS = 10
    EPOCHS = 100