#!/usr/bin/env python3
"""
Comprehensive verification script for GrooVAE H100 setup
Tests all components without requiring actual data files
"""

import sys
import importlib
import torch
from pathlib import Path

def test_imports():
    """Test all module imports"""
    print("🔍 Testing module imports...")
    
    modules_to_test = [
        'config',
        'model', 
        'train_optimized',
        'train_h100',
        'data_loader',
        'memory_utils',
        'distributed_training',
        'monitor_training'
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"  ⚠️ {module_name}: {e}")
    
    return len(failed_imports) == 0

def test_torch_features():
    """Test PyTorch H100 features"""
    print("\n🔍 Testing PyTorch H100 features...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("  ❌ CUDA not available")
        return False
    
    print(f"  ✅ CUDA available: {torch.version.cuda}")
    print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Check mixed precision
    has_amp = hasattr(torch.cuda.amp, 'autocast')
    print(f"  {'✅' if has_amp else '❌'} Mixed Precision (AMP): {has_amp}")
    
    # Check torch.compile
    has_compile = hasattr(torch, 'compile')
    print(f"  {'✅' if has_compile else '❌'} torch.compile: {has_compile}")
    
    # Check TF32
    tf32_enabled = torch.backends.cuda.matmul.allow_tf32
    print(f"  {'✅' if tf32_enabled else '⚠️'} TF32 enabled: {tf32_enabled}")
    
    # Test basic tensor operations
    try:
        x = torch.randn(100, 100, device='cuda')
        y = torch.mm(x, x.T)
        print("  ✅ Basic CUDA operations working")
    except Exception as e:
        print(f"  ❌ CUDA operations failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation and basic operations"""
    print("\n🔍 Testing model creation...")
    
    try:
        from model import Encoder, Decoder
        from config import Config
        
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
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        print("  ✅ Models created successfully")
        
        # Test forward pass
        batch_size = 4
        seq_len = 32
        input_data = torch.randn(batch_size, seq_len, 27, device=device)
        
        with torch.no_grad():
            z, mu, std = encoder(input_data)
            output, output_hits, output_velocities, output_offsets = decoder(
                z, seq_len
            )
        
        print("  ✅ Forward pass working")
        print(f"    Input shape: {input_data.shape}")
        print(f"    Latent shape: {z.shape}")
        print(f"    Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False

def test_config():
    """Test configuration system"""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import Config
        
        # Check required attributes
        required_attrs = [
            'ENCODER_INPUT_SIZE', 'ENCODER_HIDDEN_SIZE', 'ENCODER_LATENT_DIM',
            'DECODER_INPUT_SIZE', 'DECODER_HIDDEN_SIZE', 'DECODER_OUTPUT_SIZE',
            'BATCH_SIZE', 'LEARNING_RATE', 'EPOCHS', 'KL_WEIGHT',
            'USE_MIXED_PRECISION', 'USE_TORCH_COMPILE', 'NUM_WORKERS'
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(Config, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            print(f"  ❌ Missing config attributes: {missing_attrs}")
            return False
        
        print("  ✅ Configuration complete")
        print(f"    Batch size: {Config.BATCH_SIZE}")
        print(f"    Mixed precision: {Config.USE_MIXED_PRECISION}")
        print(f"    torch.compile: {Config.USE_TORCH_COMPILE}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_memory_utils():
    """Test memory optimization utilities"""
    print("\n🔍 Testing memory utilities...")
    
    try:
        from memory_utils import setup_h100_memory_optimization, log_memory_usage, cleanup_memory
        
        setup_h100_memory_optimization()
        print("  ✅ Memory optimization setup")
        
        log_memory_usage("test")
        print("  ✅ Memory logging")
        
        cleanup_memory()
        print("  ✅ Memory cleanup")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory utilities test failed: {e}")
        return False

def test_data_loader():
    """Test data loader creation (without actual data)"""
    print("\n🔍 Testing data loader...")
    
    try:
        from data_loader import GrooVAEDataset, create_optimized_dataloader
        import torch
        
        # Create dummy data
        dummy_data = [torch.randn(32, 54) for _ in range(10)]
        dataset = GrooVAEDataset(dummy_data)
        
        print("  ✅ Dataset creation")
        
        # Create dataloader
        dataloader = create_optimized_dataloader(dataset, batch_size=4, shuffle=True)
        
        print("  ✅ DataLoader creation")
        print(f"    Batch size: {dataloader.batch_size}")
        print(f"    Num workers: {dataloader.num_workers}")
        print(f"    Pin memory: {dataloader.pin_memory}")
        
        # Test iteration
        for batch in dataloader:
            print(f"    Sample batch shape: {batch.shape}")
            break
        
        print("  ✅ DataLoader iteration")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data loader test failed: {e}")
        return False

def test_file_structure():
    """Test file structure and permissions"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        'config.py',
        'model.py',
        'train_h100.py',
        'train_optimized.py',
        'data_loader.py',
        'memory_utils.py',
        'distributed_training.py',
        'monitor_training.py',
        'requirements.txt',
        'runpod_setup.sh',
        'RUNPOD_SETUP.md',
        '.gitignore'
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = Path(file_name)
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            print(f"  ✅ {file_name}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    # Check executable permissions
    executable_files = ['train_h100.py', 'monitor_training.py', 'runpod_setup.sh']
    for file_name in executable_files:
        file_path = Path(file_name)
        if file_path.exists() and not file_path.stat().st_mode & 0o111:
            print(f"  ⚠️ {file_name} not executable")
    
    print("  ✅ File structure complete")
    return True

def main():
    """Run all verification tests"""
    print("🚀 GrooVAE H100 Setup Verification")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("PyTorch Features", test_torch_features),
        ("Model Creation", test_model_creation),
        ("Memory Utils", test_memory_utils),
        ("Data Loader", test_data_loader),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"\n❌ {test_name} test failed")
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
    
    print(f"\n{'='*50}")
    print(f"Verification Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! GrooVAE H100 setup is ready.")
        
        # Additional setup recommendations
        print("\n📋 Next Steps:")
        print("1. Upload your data files to data/data_processed/")
        print("2. Run: python train_h100.py")
        print("3. Monitor with: python monitor_training.py")
        print("4. Check logs: tail -f outputs/training.log")
        
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)