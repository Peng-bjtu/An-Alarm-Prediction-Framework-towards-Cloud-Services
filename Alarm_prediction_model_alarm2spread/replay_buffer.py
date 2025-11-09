import random
from collections import deque
import torch
from config import Config

class Experience:
    def __init__(self, state, action, reward, next_state, done, inference_path):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.inference_path = inference_path

class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.buffer = deque(maxlen=config.REPLAY_BUFFER_SIZE)
        
    def add(self, state, action, reward, next_state, done, inference_path):
        experience = Experience(state, action, reward, next_state, done, inference_path)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.stack([exp.state for exp in batch])
        actions = torch.stack([exp.action for exp in batch])
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp.next_state for exp in batch])
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.bool)
        inference_paths = [exp.inference_path for exp in batch]
        
        return states, actions, rewards, next_states, dones, inference_paths
    
    def __len__(self):
        return len(self.buffer)