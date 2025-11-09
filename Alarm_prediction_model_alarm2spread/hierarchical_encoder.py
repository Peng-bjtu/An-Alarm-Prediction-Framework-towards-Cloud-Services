import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from config import Config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x, position):
        return x + self.pe[position:position+1]

class SequenceEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.positional_encoding = PositionalEncoding(d_model)
        
    def forward(self, path_nodes, node_embeddings):
        if len(path_nodes) == 0:
            return torch.zeros(1, self.d_model)
            
        seq_embeddings = []
        for i, node in enumerate(path_nodes):
            node_embed = node_embeddings.get(node, torch.zeros(self.d_model))
            pos_encoded = self.positional_encoding(node_embed.unsqueeze(0), i)
            seq_embeddings.append(pos_encoded)
        
        seq_matrix = torch.cat(seq_embeddings, dim=0).unsqueeze(1)  # [seq_len, 1, d_model]
        encoded_seq = self.transformer(seq_matrix)
        return encoded_seq.squeeze(1)

class GraphStructureEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            GraphAttentionLayer(d_model, nhead) for _ in range(num_layers)
        ])
        
    def forward(self, node_features, adjacency_matrix):
        features = node_features
        for layer in self.layers:
            features = layer(features, adjacency_matrix)
        return features

class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, features, adjacency_matrix):
        # 注意力mask
        attn_mask = (adjacency_matrix == 0)
        attn_output, _ = self.attention(features, features, features, attn_mask=attn_mask)
        features = self.norm1(features + attn_output)
        
        ff_output = self.linear2(F.relu(self.linear1(features)))
        features = self.norm2(features + ff_output)
        return features

class HierarchicalFusionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.sequence_encoder = SequenceEncoder(config.D_MODEL, config.NHEAD, config.NUM_LAYERS)
        self.graph_encoder = GraphStructureEncoder(config.D_MODEL, config.NHEAD, config.NUM_LAYERS)
        self.cross_attention = nn.MultiheadAttention(config.D_MODEL, config.NHEAD, batch_first=True)
        
    def forward(self, inference_path, topology_graph, node_embeddings):
     
        seq_features = self.sequence_encoder(inference_path, node_embeddings)
        
       
        graph_features, adjacency_matrix = self._encode_graph(topology_graph, inference_path[-1], node_embeddings)
        
       
        fused_features = self._cross_modal_fusion(seq_features, graph_features)
        
        return fused_features[-1] 
    
    def _encode_graph(self, graph, center_node, node_embeddings):
        
        subgraph_nodes = list(nx.ego_graph(graph, center_node, radius=2).nodes())
        
        if not subgraph_nodes:
            return torch.zeros(1, self.config.D_MODEL), torch.zeros(1, 1)
            
      
        node_to_idx = {node: i for i, node in enumerate(subgraph_nodes)}
        adjacency_matrix = torch.zeros(len(subgraph_nodes), len(subgraph_nodes))
        
        for i, node_i in enumerate(subgraph_nodes):
            for j, node_j in enumerate(subgraph_nodes):
                if graph.has_edge(node_i, node_j):
                    adjacency_matrix[i, j] = 1
        
       
        graph_embeddings = []
        for node in subgraph_nodes:
            embed = node_embeddings.get(node, torch.zeros(self.config.D_MODEL))
            graph_embeddings.append(embed)
        
        graph_matrix = torch.stack(graph_embeddings)  # [num_nodes, d_model]
        encoded_graph = self.graph_encoder(graph_matrix, adjacency_matrix)
        
        return encoded_graph, adjacency_matrix
    
    def _cross_modal_fusion(self, seq_features, graph_features):
       
        if len(seq_features) == 0 or len(graph_features) == 0:
            return torch.zeros(1, self.config.D_MODEL)
            
        fused_features, _ = self.cross_attention(
            seq_features.unsqueeze(0),  # [1, seq_len, d_model]
            graph_features.unsqueeze(0),  # [1, graph_len, d_model]
            graph_features.unsqueeze(0)
        )
        
        return fused_features.squeeze(0)