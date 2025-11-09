import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class GlobalDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.gru = nn.GRU(config.EMBEDDING_DIM, config.HIDDEN_DIM, batch_first=True)

        self.W_g = nn.Linear(config.HIDDEN_DIM, 1)
        self.b_g = nn.Parameter(torch.zeros(1))
        
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
    def forward(self, path_embeddings):

        if len(path_embeddings.shape) == 2:
            path_embeddings = path_embeddings.unsqueeze(0)
            

        _, hidden = self.gru(path_embeddings)  # hidden: [1, batch_size, hidden_dim]
        hidden = hidden.squeeze(0)
        hidden = self.dropout(hidden)

        probability = torch.sigmoid(self.W_g(hidden) + self.b_g)
        return probability
    
    def compute_discriminator_loss(self, real_paths, generated_paths):

        real_probs = self.forward(real_paths)
        generated_probs = self.forward(generated_paths)
        

        real_loss = -torch.log(real_probs + 1e-8).mean()
        generated_loss = -torch.log(1 - generated_probs + 1e-8).mean()
        
        total_loss = real_loss + generated_loss
        return total_loss
    
    def compute_discriminator_reward(self, generated_paths):
   
        generated_probs = self.forward(generated_paths)
        reward = torch.log(generated_probs + 1e-8) - (1 - torch.log(generated_probs + 1e-8))
        return reward