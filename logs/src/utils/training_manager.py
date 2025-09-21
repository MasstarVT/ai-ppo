"""
Background training manager for non-blocking model training with progress tracking.
"""

import threading
import queue
import time
import json
import os
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TrainingProgress:
    """Class to track training progress and metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.current_episode = 0
        self.total_episodes = 0
        self.current_timestep = 0
        self.total_timesteps = 0
        self.episode_rewards = []
        self.episode_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.learning_rates = []
        self.evaluation_returns = []
        self.best_return = -np.inf
        self.training_active = False
        self.completion_percentage = 0.0
        self.estimated_time_remaining = 0
        self.current_lr = 0
        self.network_info = {}
        
    def update(self, **kwargs):
        """Update progress metrics."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Calculate completion percentage
        if self.total_timesteps > 0:
            self.completion_percentage = min(100.0, (self.current_timestep / self.total_timesteps) * 100)
        
        # Estimate time remaining
        elapsed_time = time.time() - self.start_time
        if self.completion_percentage > 0:
            total_estimated_time = elapsed_time / (self.completion_percentage / 100)
            self.estimated_time_remaining = total_estimated_time - elapsed_time
    
    def to_dict(self) -> Dict:
        """Convert progress to dictionary for JSON serialization."""
        return {
            'start_time': self.start_time,
            'current_episode': self.current_episode,
            'total_episodes': self.total_episodes,
            'current_timestep': self.current_timestep,
            'total_timesteps': self.total_timesteps,
            'episode_rewards': self.episode_rewards[-100:],  # Keep last 100
            'episode_losses': self.episode_losses[-100:],
            'policy_losses': self.policy_losses[-100:],
            'value_losses': self.value_losses[-100:],
            'learning_rates': self.learning_rates[-100:],
            'evaluation_returns': self.evaluation_returns,
            'best_return': self.best_return,
            'training_active': self.training_active,
            'completion_percentage': self.completion_percentage,
            'estimated_time_remaining': self.estimated_time_remaining,
            'current_lr': self.current_lr,
            'network_info': self.network_info
        }


class NetworkAnalyzer:
    """Analyze and visualize neural network architecture."""
    
    @staticmethod
    def analyze_model(model) -> Dict[str, Any]:
        """Analyze a PyTorch model and extract architecture information."""
        info = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'layers': [],
            'layer_types': {},
            'model_size_mb': 0,
            'architecture_summary': ""
        }
        
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            info['total_parameters'] = total_params
            info['trainable_parameters'] = trainable_params
            
            # Estimate model size in MB
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            info['model_size_mb'] = (param_size + buffer_size) / 1024 / 1024
            
            # Analyze layers
            layer_count = {}
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    layer_type = type(module).__name__
                    layer_count[layer_type] = layer_count.get(layer_type, 0) + 1
                    
                    layer_info = {
                        'name': name,
                        'type': layer_type,
                        'parameters': sum(p.numel() for p in module.parameters()),
                    }
                    
                    # Add specific info for different layer types
                    if isinstance(module, nn.Linear):
                        layer_info['input_features'] = module.in_features
                        layer_info['output_features'] = module.out_features
                    elif isinstance(module, nn.Conv2d):
                        layer_info['in_channels'] = module.in_channels
                        layer_info['out_channels'] = module.out_channels
                        layer_info['kernel_size'] = module.kernel_size
                    
                    info['layers'].append(layer_info)
            
            info['layer_types'] = layer_count
            
            # Create architecture summary
            summary_lines = []
            summary_lines.append(f"Total Parameters: {total_params:,}")
            summary_lines.append(f"Trainable Parameters: {trainable_params:,}")
            summary_lines.append(f"Model Size: {info['model_size_mb']:.2f} MB")
            summary_lines.append("\nLayer Distribution:")
            for layer_type, count in layer_count.items():
                summary_lines.append(f"  {layer_type}: {count}")
            
            info['architecture_summary'] = "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error analyzing model: {e}")
            info['error'] = str(e)
        
        return info


class BackgroundTrainingManager:
    """Manages background training with progress tracking."""
    
    def __init__(self, progress_file: str = "training_progress.json"):
        self.progress_file = progress_file
        self.progress = TrainingProgress()
        self.training_thread = None
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.trainer = None
        
    def start_training(self, trainer_class, config: Dict, **kwargs):
        """Start training in background thread."""
        if self.is_training():
            raise RuntimeError("Training is already in progress")
        
        # Reset progress
        self.progress = TrainingProgress()
        self.progress.training_active = True
        self.stop_event.clear()
        
        # Start training thread
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(trainer_class, config),
            kwargs=kwargs,
            daemon=True
        )
        self.training_thread.start()
        
        logger.info("Background training started")
    
    def stop_training(self):
        """Stop training gracefully."""
        if self.is_training():
            self.stop_event.set()
            if self.training_thread:
                self.training_thread.join(timeout=5.0)
            self.progress.training_active = False
            logger.info("Training stopped")
    
    def is_training(self) -> bool:
        """Check if training is currently active."""
        return (self.training_thread is not None and 
                self.training_thread.is_alive() and 
                self.progress.training_active)
    
    def get_progress(self) -> Dict:
        """Get current training progress."""
        # Update with any queued progress updates
        while not self.progress_queue.empty():
            try:
                update = self.progress_queue.get_nowait()
                self.progress.update(**update)
            except queue.Empty:
                break
        
        return self.progress.to_dict()
    
    def save_progress(self):
        """Save progress to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def load_progress(self) -> Optional[Dict]:
        """Load progress from file."""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    # Update progress object
                    for key, value in data.items():
                        if hasattr(self.progress, key):
                            setattr(self.progress, key, value)
                    return data
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
        return None
    
    def _training_worker(self, trainer_class, config: Dict, **kwargs):
        """Worker function that runs training in background."""
        try:
            # Create trainer instance
            self.trainer = trainer_class(config)
            
            # Add progress callback
            self.trainer.progress_callback = self._update_progress
            
            # Set total timesteps for progress tracking
            total_timesteps = config.get('training', {}).get('total_timesteps', 100000)
            self.progress.total_timesteps = total_timesteps
            
            # Analyze network architecture if agent is available
            if hasattr(self.trainer, 'agent') and self.trainer.agent:
                self._analyze_network()
            
            # Start training
            logger.info("Starting background training...")
            self.trainer.train(**kwargs)
            
            # Training completed successfully
            self.progress.training_active = False
            logger.info("Background training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in background training: {e}")
            self.progress.training_active = False
            self.progress_queue.put({'error': str(e)})
        
        finally:
            self.save_progress()
    
    def _update_progress(self, **kwargs):
        """Callback to update progress from training thread."""
        self.progress_queue.put(kwargs)
        
        # Check for stop signal
        if self.stop_event.is_set():
            raise KeyboardInterrupt("Training stopped by user")
    
    def _analyze_network(self):
        """Analyze the neural network architecture."""
        try:
            if hasattr(self.trainer, 'agent') and self.trainer.agent:
                agent = self.trainer.agent
                network_info = {}
                
                # Analyze policy network
                if hasattr(agent, 'policy_net'):
                    network_info['policy'] = NetworkAnalyzer.analyze_model(agent.policy_net)
                
                # Analyze value network
                if hasattr(agent, 'value_net'):
                    network_info['value'] = NetworkAnalyzer.analyze_model(agent.value_net)
                
                self.progress.network_info = network_info
                
        except Exception as e:
            logger.error(f"Error analyzing network: {e}")


# Global training manager instance
training_manager = BackgroundTrainingManager()