import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_layer = nn.Linear(config.D_MODEL, config.HIDDEN_DIM)
        self.output_layer = nn.Linear(config.HIDDEN_DIM, 1)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        self.W3 = nn.Linear(config.D_MODEL, config.HIDDEN_DIM)
        self.b3 = nn.Parameter(torch.zeros(config.HIDDEN_DIM))
        
    def forward(self, state, action_space_embeddings):
        
        if len(action_space_embeddings) == 0:
            return torch.tensor([])
        
        state_features = F.relu(self.W3(state) + self.b3)  # [hidden_dim]
        state_features = self.dropout(state_features)
        
        action_scores = []
        for action_embed in action_space_embeddings:
            state_action = state_features * action_embed[:len(state_features)]
            score = self.output_layer(state_action.unsqueeze(0))  
            action_scores.append(score.squeeze())
        
        
        action_probs = F.softmax(torch.stack(action_scores), dim=0)
        return action_probs

class ValueNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        
        self.W4 = nn.Linear(config.D_MODEL, config.HIDDEN_DIM)
        self.b4 = nn.Parameter(torch.zeros(config.HIDDEN_DIM))
        self.W5 = nn.Linear(config.HIDDEN_DIM, 1)
        self.b5 = nn.Parameter(torch.zeros(1))
        
    def forward(self, state):
        hidden = F.relu(self.W4(state) + self.b4)
        value = self.W5(hidden) + self.b5
        return value

class LocalRewardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        
        self.W1 = nn.Linear(config.D_MODEL * 2, config.HIDDEN_DIM)
        self.b1 = nn.Parameter(torch.zeros(config.HIDDEN_DIM))
        self.W2 = nn.Linear(config.HIDDEN_DIM, 1)
        self.b2 = nn.Parameter(torch.zeros(1))
        
    def forward(self, state, action):
        
        state_action = torch.cat([state, action], dim=-1)
        
        
        hidden = torch.tanh(self.W1(state_action) + self.b1)
        hidden = torch.tanh(hidden)  
        D_output = torch.sigmoid(self.W2(hidden) + self.b2)
        
        return D_output
    
    def compute_reward(self, state, action):
        D_output = self.forward(state, action)
        reward = torch.log(D_output + 1e-8) - (1 - torch.log(D_output + 1e-8))
        return reward