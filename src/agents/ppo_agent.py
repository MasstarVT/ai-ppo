"""
PPO (Proximal Policy Optimization) agent for trading.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import random

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """Policy network for the PPO agent."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, activation: str = 'tanh'):
        super(PolicyNetwork, self).__init__()
        
        self.activation_fn = getattr(torch, activation) if hasattr(torch, activation) else torch.tanh
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation_layer(activation)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation_layer(self, activation: str):
        """Get activation layer."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            return nn.Tanh()
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ValueNetwork(nn.Module):
    """Value network for the PPO agent."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str = 'tanh'):
        super(ValueNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation_layer(activation)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation_layer(self, activation: str):
        """Get activation layer."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            return nn.Tanh()
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PPOBuffer:
    """Experience buffer for PPO."""
    
    def __init__(self, buffer_size: int, obs_dim: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Buffers
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
    
    def store(self, obs: np.ndarray, action: int, reward: float, value: float, log_prob: float, done: bool):
        """Store a single step."""
        assert self.ptr < self.max_size
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0):
        """Calculate advantages and returns when a trajectory ends."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # Calculate GAE advantages
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = self._discount_cumsum(deltas, self.gamma * self.gae_lambda)
        
        # Calculate returns
        returns = advantages + values[:-1]
        
        # Store advantages and returns
        if not hasattr(self, 'advantages'):
            self.advantages = np.zeros(self.max_size, dtype=np.float32)
            self.returns = np.zeros(self.max_size, dtype=np.float32)
        
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        
        self.path_start_idx = self.ptr
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all buffer data."""
        assert self.ptr == self.max_size
        
        self.ptr = 0
        self.path_start_idx = 0
        
        # Normalize advantages
        advantages = self.advantages.copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        data = {
            'observations': torch.as_tensor(self.observations, dtype=torch.float32),
            'actions': torch.as_tensor(self.actions, dtype=torch.long),
            'returns': torch.as_tensor(self.returns, dtype=torch.float32),
            'advantages': torch.as_tensor(advantages, dtype=torch.float32),
            'log_probs': torch.as_tensor(self.log_probs, dtype=torch.float32)
        }
        
        return data
    
    def _discount_cumsum(self, x: np.ndarray, discount: float) -> np.ndarray:
        """Calculate discounted cumulative sum."""
        return np.array([np.sum(x[i:] * (discount ** np.arange(len(x[i:])))) for i in range(len(x))])


class PPOAgent:
    """PPO agent for trading."""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # PPO hyperparameters
        self.lr = config.get('ppo', {}).get('learning_rate', 3e-4)
        self.clip_range = config.get('ppo', {}).get('clip_range', 0.2)
        self.ent_coef = config.get('ppo', {}).get('ent_coef', 0.01)
        self.vf_coef = config.get('ppo', {}).get('vf_coef', 0.5)
        self.max_grad_norm = config.get('ppo', {}).get('max_grad_norm', 0.5)
        self.gamma = config.get('ppo', {}).get('gamma', 0.99)
        self.gae_lambda = config.get('ppo', {}).get('gae_lambda', 0.95)
        
        # Training parameters
        self.n_steps = config.get('ppo', {}).get('n_steps', 2048)
        self.batch_size = config.get('ppo', {}).get('batch_size', 64)
        self.n_epochs = config.get('ppo', {}).get('n_epochs', 10)
        
        # Network architecture
        network_config = config.get('network', {})
        policy_layers = network_config.get('policy_layers', [256, 256])
        value_layers = network_config.get('value_layers', [256, 256])
        activation = network_config.get('activation', 'tanh')
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enhanced GPU logging
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.info("Using CPU - consider installing CUDA for faster training")
        
        logger.info(f"PyTorch device: {self.device}")
        
        # Networks
        self.policy_net = PolicyNetwork(obs_dim, policy_layers, action_dim, activation).to(self.device)
        self.value_net = ValueNetwork(obs_dim, value_layers, activation).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        
        # Experience buffer
        self.buffer = PPOBuffer(self.n_steps, obs_dim, self.gamma, self.gae_lambda)
        
        # Training stats
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100)
        }
        
        logger.info(f"Initialized PPO agent with {sum(p.numel() for p in self.policy_net.parameters())} policy parameters")
        logger.info(f"Initialized PPO agent with {sum(p.numel() for p in self.value_net.parameters())} value parameters")
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """Get action from policy."""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # Get action probabilities and value
            action_probs = self.policy_net(obs_tensor)
            value = self.value_net(obs_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
            
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1))).squeeze(-1)
            
            return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_experience(self, obs: np.ndarray, action: int, reward: float, value: float, log_prob: float, done: bool):
        """Store experience in buffer."""
        self.buffer.store(obs, action, reward, value, log_prob, done)
    
    def finish_episode(self, last_obs: np.ndarray):
        """Finish episode and calculate advantages."""
        with torch.no_grad():
            last_value = self.value_net(torch.as_tensor(last_obs, dtype=torch.float32, device=self.device))
            last_value = last_value.cpu().numpy()[0]
        
        self.buffer.finish_path(last_value)
    
    def update(self) -> Dict[str, float]:
        """Update policy and value networks."""
        data = self.buffer.get()
        
        # Move to device with non_blocking for better GPU performance
        for key in data:
            data[key] = data[key].to(self.device, non_blocking=True)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        
        for epoch in range(self.n_epochs):
            # Create mini-batches on GPU if available
            batch_indices = torch.randperm(self.n_steps, device=self.device)
            
            for start_idx in range(0, self.n_steps, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.n_steps)
                batch_idx = batch_indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = data['observations'][batch_idx]
                batch_actions = data['actions'][batch_idx]
                batch_returns = data['returns'][batch_idx]
                batch_advantages = data['advantages'][batch_idx]
                batch_old_log_probs = data['log_probs'][batch_idx]
                
                # Forward pass
                action_probs = self.policy_net(batch_obs)
                values = self.value_net(batch_obs).squeeze()
                
                # Calculate policy loss
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(batch_actions)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Calculate entropy bonus
                entropy = action_dist.entropy().mean()
                
                # Total loss
                total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                # Statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                
                # KL divergence (approximate)
                kl = (batch_old_log_probs - new_log_probs).mean().item()
                total_kl += kl
        
        # Average losses
        n_updates = self.n_epochs * (self.n_steps // self.batch_size)
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy = total_entropy / n_updates
        avg_kl = total_kl / n_updates
        
        # Store training stats
        self.training_stats['policy_loss'].append(avg_policy_loss)
        self.training_stats['value_loss'].append(avg_value_loss)
        self.training_stats['entropy'].append(avg_entropy)
        self.training_stats['kl_divergence'].append(avg_kl)
        
        # Clear GPU cache periodically to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'kl_divergence': avg_kl
        }
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get recent training statistics."""
        return {
            'policy_loss': np.mean(self.training_stats['policy_loss']) if self.training_stats['policy_loss'] else 0,
            'value_loss': np.mean(self.training_stats['value_loss']) if self.training_stats['value_loss'] else 0,
            'entropy': np.mean(self.training_stats['entropy']) if self.training_stats['entropy'] else 0,
            'kl_divergence': np.mean(self.training_stats['kl_divergence']) if self.training_stats['kl_divergence'] else 0
        }