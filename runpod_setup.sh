#!/bin/bash

# GrooVAE-torch H100 Setup Script for RunPod
# This script automates the setup process for training on RunPod H100 instances

set -e  # Exit on any error

echo "🚀 GrooVAE-torch H100 Setup for RunPod"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if we're on RunPod
print_header "Checking Environment"
if [ "$RUNPOD" = "true" ]; then
    print_status "RunPod environment detected"
else
    print_warning "RUNPOD environment variable not set"
    export RUNPOD=true
fi

# Set working directory
cd /workspace

# Check CUDA availability
print_header "Verifying CUDA Setup"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    print_status "CUDA devices detected"
else
    print_error "nvidia-smi not found. Please ensure you're on a GPU instance."
    exit 1
fi

# Check Python and PyTorch
print_header "Checking PyTorch Installation"
python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
"

# Clone repository if not exists
print_header "Setting up GrooVAE-torch Repository"
if [ ! -d "GrooVAE-torch" ]; then
    print_status "Cloning repository..."
    git clone https://github.com/olsgo/GrooVAE-torch.git
else
    print_status "Repository already exists, updating..."
    cd GrooVAE-torch
    git pull
    cd ..
fi

cd GrooVAE-torch

# Install dependencies
print_header "Installing Dependencies"
print_status "Installing Python packages..."
pip install -r requirements.txt

# Verify H100 specific features
print_header "Verifying H100 Optimizations"
python3 -c "
import torch
print('Checking H100 optimization support...')
print(f'Mixed Precision (AMP): {hasattr(torch.cuda.amp, \"autocast\")}')
print(f'TF32 Support: {torch.backends.cuda.matmul.allow_tf32}')
print(f'torch.compile Available: {hasattr(torch, \"compile\")}')

# Check if H100
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'
if 'H100' in gpu_name:
    print('✅ H100 GPU detected - all optimizations will be enabled')
else:
    print(f'⚠️  GPU: {gpu_name} - H100 optimizations may not be fully effective')
"

# Create necessary directories
print_header "Creating Directory Structure"
mkdir -p data/data_processed
mkdir -p model
mkdir -p outputs
print_status "Directory structure created"

# Set up environment variables for optimal performance
print_header "Configuring CUDA Environment"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Add to bashrc for persistence
echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" >> ~/.bashrc
echo "export CUDA_LAUNCH_BLOCKING=0" >> ~/.bashrc
echo "export TORCH_CUDNN_V8_API_ENABLED=1" >> ~/.bashrc
echo "export RUNPOD=true" >> ~/.bashrc

print_status "Environment variables configured"

# Create a quick test script
print_header "Creating Verification Script"
cat > verify_setup.py << 'EOF'
#!/usr/bin/env python3
"""Quick verification script for GrooVAE H100 setup"""

import torch
import sys
from pathlib import Path

def main():
    print("🔍 GrooVAE H100 Setup Verification")
    print("=" * 40)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU: {gpu_name}")
    
    # Check H100 specific features
    h100_detected = 'H100' in gpu_name
    print(f"{'✅' if h100_detected else '⚠️'} H100 Detection: {h100_detected}")
    
    # Check memory
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU Memory: {memory_gb:.1f} GB")
    
    # Check PyTorch features
    print(f"✅ Mixed Precision: {hasattr(torch.cuda.amp, 'autocast')}")
    print(f"✅ torch.compile: {hasattr(torch, 'compile')}")
    print(f"✅ TF32: {torch.backends.cuda.matmul.allow_tf32}")
    
    # Check data directories
    data_path = Path("data/data_processed")
    print(f"{'✅' if data_path.exists() else '❌'} Data directory: {data_path}")
    
    # Quick tensor operation test
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.mm(x, x.T)
        print("✅ CUDA tensor operations working")
    except Exception as e:
        print(f"❌ CUDA tensor test failed: {e}")
        return False
    
    print("\n🎉 Setup verification complete!")
    
    if not h100_detected:
        print("\n⚠️  Note: H100 not detected. Performance optimizations may be limited.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x verify_setup.py

# Run verification
print_header "Running Setup Verification"
python3 verify_setup.py

# Print completion message
print_header "Setup Complete!"
print_status "GrooVAE-torch is ready for H100 training on RunPod"
echo ""
echo "Next steps:"
echo "1. Upload your data files to: /workspace/GrooVAE-torch/data/data_processed/"
echo "   Required files:"
echo "   - tapify_train.pkl"
echo "   - tapify_valid.pkl"
echo "   - tapify_test.pkl"
echo ""
echo "2. Start training:"
echo "   cd /workspace/GrooVAE-torch"
echo "   python train_h100.py"
echo ""
echo "3. Monitor training:"
echo "   tail -f outputs/training.log"
echo "   nvidia-smi -l 1"
echo ""
echo "For detailed instructions, see: RUNPOD_SETUP.md"

# Create a convenient alias
echo "alias groovae-train='cd /workspace/GrooVAE-torch && python train_h100.py'" >> ~/.bashrc
echo "alias groovae-monitor='tail -f /workspace/GrooVAE-torch/outputs/training.log'" >> ~/.bashrc
echo "alias groovae-gpu='nvidia-smi -l 1'" >> ~/.bashrc

print_status "Convenient aliases added to ~/.bashrc"
print_status "Use 'groovae-train', 'groovae-monitor', 'groovae-gpu' for quick access"

echo ""
echo "🎉 Setup completed successfully!"