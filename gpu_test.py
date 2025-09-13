"""
GPU Performance Test and Monitoring for PPO Trading Agent
"""

import torch
import time
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_gpu_availability():
    """Check GPU availability and performance."""
    print("=" * 60)
    print("GPU PERFORMANCE CHECK")
    print("=" * 60)
    
    # Basic CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"  Multiprocessors: {gpu_props.multi_processor_count}")
            
        # Memory check
        torch.cuda.empty_cache()
        print(f"\nCurrent GPU memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
        
    else:
        print("\n⚠️  No CUDA GPU detected!")
        print("   For faster training, consider:")
        print("   1. Installing NVIDIA GPU drivers")
        print("   2. Installing CUDA toolkit")
        print("   3. Installing PyTorch with CUDA support")
        print("   4. Using Google Colab or cloud GPU instances")
    
    return torch.cuda.is_available()

def benchmark_device_performance():
    """Benchmark CPU vs GPU performance."""
    print("\n" + "=" * 60)
    print("DEVICE PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Test parameters
    batch_size = 1024
    input_size = 1500  # Same as trading environment observation space
    hidden_size = 256
    output_size = 3    # BUY, SELL, HOLD
    
    # Create test neural network
    class TestNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size),
                torch.nn.Softmax(dim=-1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Test data
    test_input = torch.randn(batch_size, input_size)
    
    # CPU benchmark
    print("Testing CPU performance...")
    model_cpu = TestNetwork()
    test_input_cpu = test_input.to('cpu')
    
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model_cpu(test_input_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU: {cpu_time:.3f} seconds for 100 forward passes")
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        print("Testing GPU performance...")
        model_gpu = TestNetwork().to('cuda')
        test_input_gpu = test_input.to('cuda')
        
        # Warm up GPU
        for _ in range(10):
            with torch.no_grad():
                _ = model_gpu(test_input_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = model_gpu(test_input_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"GPU: {gpu_time:.3f} seconds for 100 forward passes")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x faster on GPU")
        
        # Memory usage after benchmark
        print(f"\nGPU memory after benchmark:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    return torch.cuda.is_available()

def test_training_with_gpu():
    """Test actual PPO training with GPU."""
    print("\n" + "=" * 60)
    print("PPO TRAINING GPU TEST")
    print("=" * 60)
    
    try:
        from train import TradingTrainer
        from utils import ConfigManager
        
        # Check if config exists
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            print("Error: config.yaml not found. Run setup_windows.bat first.")
            return False
        
        # Load config and override for quick test
        config_manager = ConfigManager(config_path)
        config = config_manager.to_dict()
        config['training']['total_timesteps'] = 2000  # Quick test
        config['training']['eval_freq'] = 1000
        
        print("Starting quick PPO training test...")
        
        # Create trainer and measure initialization time
        start_time = time.time()
        trainer = TradingTrainer(config)
        init_time = time.time() - start_time
        print(f"Trainer initialization: {init_time:.2f} seconds")
        
        # Run training to get device info
        start_time = time.time()
        results = trainer.train()
        training_time = time.time() - start_time
        
        # Show device being used
        device = trainer.agent.device
        print(f"Training device: {device}")
        
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Final return: {results['final_metrics']['avg_return']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"Training test failed: {e}")
        return False

def main():
    """Run complete GPU performance analysis."""
    print("🚀 PPO Trading Agent - GPU Performance Analysis")
    
    # Check GPU availability
    has_gpu = check_gpu_availability()
    
    # Benchmark performance
    benchmark_device_performance()
    
    # Test actual training
    print("\nTesting with actual PPO training...")
    success = test_training_with_gpu()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if has_gpu:
        print("✅ GPU detected and available for training")
        print("✅ Faster training performance expected")
        if success:
            print("✅ PPO training completed successfully on GPU")
        else:
            print("⚠️  PPO training test failed")
    else:
        print("❌ No GPU available - training will use CPU")
        print("💡 Consider upgrading to GPU for faster training")
    
    print("\n🎯 Your PPO trading system is ready!")
    print("   Use 'python test_training.py' for quick tests")
    print("   Use 'python src/train.py' for full training")

if __name__ == "__main__":
    main()