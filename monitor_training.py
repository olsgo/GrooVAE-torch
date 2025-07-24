#!/usr/bin/env python3
"""
Training monitoring utility for GrooVAE H100 training
"""

import time
import psutil
import GPUtil
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import threading
from datetime import datetime
import logging
from typing import Dict, List, Optional

class TrainingMonitor:
    """Monitor training progress and system resources"""
    
    def __init__(self, log_file: str = "outputs/training.log", 
                 output_dir: str = "outputs"):
        self.log_file = Path(log_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.monitoring = False
        self.monitor_thread = None
        
        # Data storage
        self.metrics = {
            'timestamps': [],
            'gpu_util': [],
            'gpu_memory': [],
            'cpu_util': [],
            'ram_util': [],
            'gpu_temp': [],
            'train_loss': [],
            'val_loss': []
        }
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger"""
        logger = logging.getLogger('TrainingMonitor')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def get_gpu_stats(self) -> Dict:
        """Get GPU statistics"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature
                }
        except Exception as e:
            self.logger.warning(f"Failed to get GPU stats: {e}")
        
        return {
            'utilization': 0,
            'memory_used': 0,
            'memory_total': 0,
            'memory_percent': 0,
            'temperature': 0
        }
    
    def get_cpu_stats(self) -> Dict:
        """Get CPU and RAM statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'ram_total_gb': psutil.virtual_memory().total / (1024**3)
        }
    
    def parse_training_log(self) -> Dict:
        """Parse training log for loss values"""
        training_data = {'epochs': [], 'train_loss': [], 'val_loss': []}
        
        if not self.log_file.exists():
            return training_data
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    # Look for epoch lines
                    if 'Epoch' in line and 'Train Loss:' in line:
                        try:
                            # Extract epoch number
                            epoch_start = line.find('Epoch') + 6
                            epoch_end = line.find('/', epoch_start)
                            epoch = int(line[epoch_start:epoch_end].strip())
                            
                            # Extract train loss
                            train_start = line.find('Train Loss:') + 12
                            train_end = line.find(' ', train_start)
                            if train_end == -1:
                                train_end = line.find('\n', train_start)
                            train_loss = float(line[train_start:train_end])
                            
                            # Extract val loss if present
                            val_loss = None
                            if 'Val Loss:' in line:
                                val_start = line.find('Val Loss:') + 10
                                val_end = line.find(' ', val_start)
                                if val_end == -1:
                                    val_end = line.find('\n', val_start)
                                val_loss = float(line[val_start:val_end])
                            
                            training_data['epochs'].append(epoch)
                            training_data['train_loss'].append(train_loss)
                            training_data['val_loss'].append(val_loss)
                            
                        except (ValueError, IndexError):
                            continue
                            
        except Exception as e:
            self.logger.warning(f"Failed to parse training log: {e}")
        
        return training_data
    
    def monitor_resources(self, interval: int = 5):
        """Monitor system resources continuously"""
        while self.monitoring:
            timestamp = datetime.now()
            
            # Get system stats
            gpu_stats = self.get_gpu_stats()
            cpu_stats = self.get_cpu_stats()
            
            # Store metrics
            self.metrics['timestamps'].append(timestamp)
            self.metrics['gpu_util'].append(gpu_stats['utilization'])
            self.metrics['gpu_memory'].append(gpu_stats['memory_percent'])
            self.metrics['cpu_util'].append(cpu_stats['cpu_percent'])
            self.metrics['ram_util'].append(cpu_stats['ram_percent'])
            self.metrics['gpu_temp'].append(gpu_stats['temperature'])
            
            # Log current stats
            self.logger.info(
                f"GPU: {gpu_stats['utilization']:.1f}% util, "
                f"{gpu_stats['memory_percent']:.1f}% mem, "
                f"{gpu_stats['temperature']:.0f}°C | "
                f"CPU: {cpu_stats['cpu_percent']:.1f}% | "
                f"RAM: {cpu_stats['ram_percent']:.1f}%"
            )
            
            time.sleep(interval)
    
    def start_monitoring(self, interval: int = 5):
        """Start monitoring in a separate thread"""
        if self.monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self.monitor_resources, 
            args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Resource monitoring stopped")
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        # Convert timestamps to strings for JSON serialization
        metrics_serializable = self.metrics.copy()
        metrics_serializable['timestamps'] = [
            ts.isoformat() for ts in self.metrics['timestamps']
        ]
        
        metrics_file = self.output_dir / 'monitoring_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
    
    def create_monitoring_plots(self):
        """Create monitoring plots"""
        if not self.metrics['timestamps']:
            self.logger.warning("No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        timestamps = self.metrics['timestamps']
        
        # GPU Utilization
        axes[0, 0].plot(timestamps, self.metrics['gpu_util'], 'b-', label='GPU Util')
        axes[0, 0].plot(timestamps, self.metrics['cpu_util'], 'r-', label='CPU Util')
        axes[0, 0].set_ylabel('Utilization (%)')
        axes[0, 0].set_title('GPU/CPU Utilization')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Memory Usage
        axes[0, 1].plot(timestamps, self.metrics['gpu_memory'], 'g-', label='GPU Memory')
        axes[0, 1].plot(timestamps, self.metrics['ram_util'], 'orange', label='RAM')
        axes[0, 1].set_ylabel('Memory Usage (%)')
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # GPU Temperature
        axes[1, 0].plot(timestamps, self.metrics['gpu_temp'], 'purple', label='GPU Temp')
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].set_title('GPU Temperature')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Training Loss (if available)
        training_data = self.parse_training_log()
        if training_data['epochs']:
            axes[1, 1].plot(training_data['epochs'], training_data['train_loss'], 
                           'b-', label='Train Loss')
            if any(x is not None for x in training_data['val_loss']):
                val_losses = [x for x in training_data['val_loss'] if x is not None]
                val_epochs = [training_data['epochs'][i] for i, x in enumerate(training_data['val_loss']) if x is not None]
                axes[1, 1].plot(val_epochs, val_losses, 'r-', label='Val Loss')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_title('Training Progress')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No training data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Training Progress')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'monitoring_dashboard.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Monitoring plots saved to {plot_file}")
    
    def generate_report(self):
        """Generate a monitoring report"""
        report = []
        report.append("GrooVAE Training Monitoring Report")
        report.append("=" * 40)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if self.metrics['timestamps']:
            start_time = self.metrics['timestamps'][0]
            end_time = self.metrics['timestamps'][-1]
            duration = end_time - start_time
            
            report.append(f"Monitoring Duration: {duration}")
            report.append(f"Data Points: {len(self.metrics['timestamps'])}")
            report.append("")
            
            # Resource usage summary
            if self.metrics['gpu_util']:
                report.append("Resource Usage Summary:")
                report.append(f"  GPU Utilization: {max(self.metrics['gpu_util']):.1f}% max, {sum(self.metrics['gpu_util'])/len(self.metrics['gpu_util']):.1f}% avg")
                report.append(f"  GPU Memory: {max(self.metrics['gpu_memory']):.1f}% max, {sum(self.metrics['gpu_memory'])/len(self.metrics['gpu_memory']):.1f}% avg")
                report.append(f"  CPU Utilization: {max(self.metrics['cpu_util']):.1f}% max, {sum(self.metrics['cpu_util'])/len(self.metrics['cpu_util']):.1f}% avg")
                report.append(f"  GPU Temperature: {max(self.metrics['gpu_temp']):.0f}°C max, {sum(self.metrics['gpu_temp'])/len(self.metrics['gpu_temp']):.0f}°C avg")
                report.append("")
        
        # Training progress
        training_data = self.parse_training_log()
        if training_data['epochs']:
            report.append("Training Progress:")
            report.append(f"  Epochs Completed: {max(training_data['epochs'])}")
            report.append(f"  Latest Train Loss: {training_data['train_loss'][-1]:.4f}")
            if training_data['val_loss'][-1] is not None:
                report.append(f"  Latest Val Loss: {training_data['val_loss'][-1]:.4f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / 'monitoring_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        self.logger.info(f"Report saved to {report_file}")

def main():
    """Main monitoring function"""
    parser = argparse.ArgumentParser(description='GrooVAE Training Monitor')
    parser.add_argument('--log-file', default='outputs/training.log',
                        help='Training log file to monitor')
    parser.add_argument('--output-dir', default='outputs',
                        help='Output directory for monitoring files')
    parser.add_argument('--interval', type=int, default=5,
                        help='Monitoring interval in seconds')
    parser.add_argument('--duration', type=int, default=None,
                        help='Monitoring duration in seconds (None for indefinite)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate plots from existing data')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_file, args.output_dir)
    
    if args.plot_only:
        print("Generating monitoring plots...")
        monitor.create_monitoring_plots()
        monitor.generate_report()
        return
    
    try:
        print(f"Starting training monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop monitoring and generate report")
        
        monitor.start_monitoring(args.interval)
        
        if args.duration:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        monitor.stop_monitoring()
        monitor.save_metrics()
        monitor.create_monitoring_plots()
        monitor.generate_report()

if __name__ == "__main__":
    main()