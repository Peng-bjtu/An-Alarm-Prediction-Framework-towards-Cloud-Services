import torch
import torch.nn as nn
import numpy as np
from typing import List, Set

class TreeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
        
    def forward(self, node_features, children_features):
        if not children_features:
            h, c = self.lstm_cell(node_features)
            return h, c
        
        child_h = torch.mean(torch.stack([ch for ch, _ in children_features]), dim=0)
        child_c = torch.mean(torch.stack([cc for _, cc in children_features]), dim=0)
        
        combined_input = node_features + child_h
        h, c = self.lstm_cell(combined_input, (child_h, child_c))
        return h, c

class SemanticConsistencyChecker:
    def __init__(self, input_dim=64, hidden_dim=128):
        self.tree_lstm = TreeLSTM(input_dim, hidden_dim)
        self.similarity_metric = nn.CosineSimilarity(dim=0)
        
    def compute_subtree_representation(self, subtree):

        node_vectors = []
        for node in subtree.nodes:
            node_vec = self._node_to_vector(node)
            node_vectors.append(node_vec)
        
        if not node_vectors:
            return torch.zeros(self.hidden_dim)
            
        subtree_vector = torch.mean(torch.stack(node_vectors), dim=0)
        return subtree_vector
    
    def _node_to_vector(self, node):

        type_embedding = torch.tensor([hash(node.alarm_type) % 100] * 32, dtype=torch.float32)
        time_feature = torch.tensor([node.timestamp] * 32, dtype=torch.float32)
        return torch.cat([type_embedding, time_feature])
    
    def check_consistency(self, subtree_instances, threshold=0.7):
        if len(subtree_instances) <= 1:
            return True
            
        representations = []
        for instance in subtree_instances:
            repr_vec = self.compute_subtree_representation(instance)
            representations.append(repr_vec)
        
        total_similarity = 0
        count = 0
        for i in range(len(representations)):
            for j in range(i + 1, len(representations)):
                similarity = self.similarity_metric(representations[i], representations[j])
                total_similarity += similarity.item()
                count += 1
                
        if count == 0:
            return True
            
        avg_similarity = total_similarity / count
        return avg_similarity >= threshold