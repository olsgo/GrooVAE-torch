"""
Memory optimization utilities for H100 SXM training
"""

import torch
import gc
import logging
from typing import Optional, Dict, Any
import psutil
import os

class MemoryOptimizer:
    """Memory optimization utility for H100 training"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.initial_memory = self._get_gpu_memory()
        
    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'reserved': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,
                'max_reserved': torch.cuda.max_memory_reserved() / 1024**3,
            }
        return {}
    
    def _get_cpu_memory(self) -> Dict[str, float]:
        """Get current CPU memory usage"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024**3,  # Resident Set Size
            'vms': memory_info.vms / 1024**3,  # Virtual Memory Size
            'percent': process.memory_percent(),
        }
    
    def log_memory_usage(self, stage: str = ""):
        """Log current memory usage"""
        gpu_mem = self._get_gpu_memory()
        cpu_mem = self._get_cpu_memory()
        
        if gpu_mem:
            self.logger.info(
                f"Memory usage {stage}: "
                f"GPU Allocated: {gpu_mem['allocated']:.2f}GB, "
                f"Reserved: {gpu_mem['reserved']:.2f}GB, "
                f"CPU: {cpu_mem['rss']:.2f}GB ({cpu_mem['percent']:.1f}%)"
            )
        else:
            self.logger.info(
                f"Memory usage {stage}: "
                f"CPU: {cpu_mem['rss']:.2f}GB ({cpu_mem['percent']:.1f}%)"
            )
    
    def cleanup_memory(self, aggressive: bool = False):
        """Clean up memory"""
        # Clear Python garbage
        gc.collect()
        
        if torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            if aggressive:
                # Force synchronization and cleanup
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
        
        self.logger.debug("Memory cleanup completed")
    
    def setup_memory_optimization(self):
        """Setup memory optimizations for H100"""
        if torch.cuda.is_available():
            # Enable memory pool optimization
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Enable CUDA graphs if supported
            try:
                torch.backends.cuda.enable_memory_efficient_attention()
                self.logger.info("CUDA memory efficient attention enabled")
            except:
                pass
            
            # Set CUDA memory fraction if needed (for multi-process)
            # torch.cuda.set_per_process_memory_fraction(0.9)
            
            self.logger.info("Memory optimizations applied")
    
    def get_optimal_batch_size(self, model, sample_input_shape, max_memory_gb=70):
        """Estimate optimal batch size for given memory limit"""
        if not torch.cuda.is_available():
            return 32  # Default fallback
        
        # Start with a small batch size and increase
        batch_size = 1
        optimal_batch_size = 1
        
        try:
            while batch_size <= 2048:  # Reasonable upper limit
                # Create dummy input
                dummy_input = torch.randn(batch_size, *sample_input_shape, device='cuda')
                
                # Clear memory
                self.cleanup_memory()
                
                try:
                    # Test forward pass
                    with torch.no_grad():
                        _ = model[0](dummy_input)  # Encoder
                    
                    # Check memory usage
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    
                    if memory_used < max_memory_gb:
                        optimal_batch_size = batch_size
                        batch_size *= 2
                    else:
                        break
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        break
                    else:
                        raise
                
                del dummy_input
                
        except Exception as e:
            self.logger.warning(f"Batch size estimation failed: {e}")
        
        finally:
            self.cleanup_memory()
        
        self.logger.info(f"Estimated optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def monitor_memory_during_training(self, enabled: bool = True):
        """Enable/disable memory monitoring during training"""
        if enabled and torch.cuda.is_available():
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            self.logger.info("Memory monitoring enabled")
    
    def get_memory_summary(self) -> str:
        """Get a detailed memory summary"""
        lines = ["Memory Summary:", "=" * 50]
        
        if torch.cuda.is_available():
            gpu_mem = self._get_gpu_memory()
            lines.extend([
                f"GPU Memory:",
                f"  Allocated: {gpu_mem['allocated']:.2f} GB",
                f"  Reserved:  {gpu_mem['reserved']:.2f} GB",
                f"  Max Allocated: {gpu_mem['max_allocated']:.2f} GB",
                f"  Max Reserved:  {gpu_mem['max_reserved']:.2f} GB",
            ])
        
        cpu_mem = self._get_cpu_memory()
        lines.extend([
            f"CPU Memory:",
            f"  RSS: {cpu_mem['rss']:.2f} GB",
            f"  VMS: {cpu_mem['vms']:.2f} GB",
            f"  Percent: {cpu_mem['percent']:.1f}%",
        ])
        
        return "\n".join(lines)

# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()

def setup_h100_memory_optimization():
    """Setup H100-specific memory optimizations"""
    memory_optimizer.setup_memory_optimization()
    memory_optimizer.log_memory_usage("initial")

def cleanup_memory():
    """Convenient cleanup function"""
    memory_optimizer.cleanup_memory()

def log_memory_usage(stage: str = ""):
    """Convenient memory logging function"""
    memory_optimizer.log_memory_usage(stage)

def get_memory_summary() -> str:
    """Get memory summary"""
    return memory_optimizer.get_memory_summary()