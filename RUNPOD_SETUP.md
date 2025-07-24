# GrooVAE-torch: H100 SXM Setup for RunPod

## Quick Start on RunPod

### 1. Create RunPod Instance

1. Go to [RunPod.io](https://runpod.io) and sign in
2. Click "Deploy" and select "GPU Pod"
3. Choose an H100 SXM instance:
   - **Recommended**: H100 SXM 80GB
   - **Template**: PyTorch 2.1+ (or use custom template below)
   - **Container Disk**: At least 50GB
   - **Volume**: 100GB+ for data storage

### 2. Recommended RunPod Template

Use this custom template for optimal setup:

```
Template Name: GrooVAE-H100
Container Image: runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04
Container Disk: 50GB
Environment Variables:
  - RUNPOD=true
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  - CUDA_LAUNCH_BLOCKING=0
Exposed Ports: 8888, 6006
```

### 3. Setup Commands (Run in Terminal)

```bash
# Clone the repository
cd /workspace
git clone https://github.com/olsgo/GrooVAE-torch.git
cd GrooVAE-torch

# Install dependencies
pip install -r requirements.txt

# Verify H100 setup
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'CUDA: {torch.version.cuda}'); print(f'Available: {torch.cuda.is_available()}')"
```

### 4. Data Preparation

Place your data files in the correct location:

```bash
# Create data directory
mkdir -p /workspace/data/data_processed

# Upload your pickle files to:
# - /workspace/data/data_processed/tapify_train.pkl
# - /workspace/data/data_processed/tapify_valid.pkl  
# - /workspace/data/data_processed/tapify_test.pkl
```

### 5. Start Training

```bash
# Basic training with H100 optimizations
python train_h100.py

# Advanced options
python train_h100.py --epochs 200 --batch-size 1024 --learning-rate 2e-3

# Monitor training (in another terminal)
tail -f outputs/training.log
```

### 6. Monitor Training Progress

```bash
# Check GPU utilization
nvidia-smi -l 1

# View training logs
tail -f /workspace/GrooVAE-torch/outputs/training.log

# Optional: Start TensorBoard (if implemented)
tensorboard --logdir=/workspace/GrooVAE-torch/outputs --host=0.0.0.0 --port=6006
```

## H100 Optimizations Included

### Hardware Optimizations
- ✅ Mixed precision training (FP16/BF16)
- ✅ `torch.compile()` with max-autotune mode
- ✅ TensorFloat-32 (TF32) enabled
- ✅ Optimized CUDA memory management
- ✅ High-bandwidth data loading

### Software Optimizations
- ✅ AdamW optimizer with optimized hyperparameters
- ✅ Gradient scaling for numerical stability
- ✅ Persistent workers for data loading
- ✅ Pin memory for faster GPU transfers
- ✅ Prefetch factor for pipeline optimization

### Memory Optimizations
- ✅ Gradient checkpointing (when needed)
- ✅ Optimized batch sizes for H100's 80GB memory
- ✅ Non-blocking data transfers
- ✅ CUDA memory pool optimization

## Performance Expectations

On H100 SXM 80GB, you should expect:
- **Training Speed**: ~2-3x faster than A100
- **Memory Usage**: Up to 80GB for large batch sizes
- **Batch Size**: Up to 1024+ depending on sequence length
- **Mixed Precision**: ~40% speedup with minimal accuracy loss

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size
   python train_h100.py --batch-size 256
   ```

2. **CUDA Version Mismatch**
   ```bash
   # Check versions
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Data Loading Slow**
   ```bash
   # Reduce num_workers if CPU limited
   # Edit config.py: NUM_WORKERS = 4
   ```

4. **Compilation Errors**
   ```bash
   # Disable torch.compile if issues
   python train_h100.py --no-compile
   ```

## File Structure

```
/workspace/GrooVAE-torch/
├── config.py              # Configuration settings
├── train_h100.py          # Main training script
├── train_optimized.py     # Optimized training functions
├── data_loader.py         # Optimized data loading
├── model.py               # Model definitions
├── requirements.txt       # Dependencies
├── runpod_setup.sh       # Automated setup script
└── outputs/              # Training outputs
    ├── training.log      # Training logs
    ├── models/          # Saved models
    └── plots/           # Training plots
```

## Cost Optimization Tips

1. **Use Spot Instances**: Save 50-80% on costs
2. **Auto-pause**: Set idle timeout to avoid charges
3. **Batch Jobs**: Run multiple experiments in sequence
4. **Volume Storage**: Use persistent volumes for data
5. **Monitor Usage**: Track GPU utilization

## Support

For issues specific to this H100 optimization:
1. Check the training logs in `outputs/training.log`
2. Verify GPU compatibility with the verification script
3. Review RunPod documentation for instance setup
4. Check CUDA and PyTorch compatibility

## Advanced Features

### Distributed Training (Multi-H100)
If using multiple H100s:
```bash
# Will be implemented in future version
torchrun --nproc_per_node=2 train_h100.py
```

### Custom Data Formats
Edit `data_loader.py` to support your specific data format.

### Hyperparameter Tuning
Use the configuration system in `config.py` for systematic tuning.