import torch
import networkx as nx
import numpy as np
import pickle
import random
from alarm2spread import Alarm2Spread
from config import Config

def load_knowledge_graph(graph_path):
    try:
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        print(f"ok")
        return graph
    except Exception as e:
        print(f"error")
        return None

def load_node_embeddings(embeddings_path):
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"embed ok")
        
        for node_id, embedding in embeddings.items():
            if isinstance(embedding, np.ndarray):
                embeddings[node_id] = torch.from_numpy(embedding).float()
            elif isinstance(embedding, torch.Tensor):
                embeddings[node_id] = embedding.float()
            else:
                embeddings[node_id] = torch.tensor(embedding, dtype=torch.float)
                
        return embeddings
    except Exception as e:
        print(f"embed false)
        return None

def load_real_paths(paths_path):
    try:
        with open(paths_path, 'rb') as f:
            paths = pickle.load(f)
        print(f"path ok")
        return paths
    except Exception as e:
        print(f"path false")
        return None

def main():
    graph_path = "knowledge_graph.pkl"  
    embeddings_path = "node_embeddings.pkl"  
    paths_path = "real_paths.pkl"  
    
    topology_graph = load_knowledge_graph(graph_path)
    node_embeddings = load_node_embeddings(embeddings_path)
    real_paths = load_real_paths(paths_path)
    
    config = Config()
    model = Alarm2Spread(topology_graph, node_embeddings, real_paths, config)
    
    num_epochs = 1000
    for epoch in range(num_epochs):
        total_policy_loss = 0
        total_value_loss = 0
        total_discriminator_loss = 0
        num_batches = 0
 
        for path in real_paths[:min(100, len(real_paths))]:  
            if len(path) < 2:
                continue
                
       
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                inference_path = path[:i+1]
                

                state = model.encoder(inference_path, topology_graph, node_embeddings)
                
          
                if next_node in node_embeddings:
                    action_embedding = node_embeddings[next_node]
                else:
                    action_embedding = torch.zeros(config.EMBEDDING_DIM)
                
        
                reward = model.compute_local_reward(state, action_embedding)
                
   
                next_inference_path = path[:i+2]
                next_state = model.encoder(next_inference_path, topology_graph, node_embeddings)
                

                done = (i == len(path) - 2)
                model.memory.add(state, action_embedding, reward, next_state, done, inference_path)
        

        policy_loss, value_loss, discriminator_loss = model.train_step()
        
        total_policy_loss += policy_loss
        total_value_loss += value_loss
        total_discriminator_loss += discriminator_loss
        num_batches += 1
        
        if epoch % 100 == 0:
            avg_policy_loss = total_policy_loss / max(num_batches, 1)
            avg_value_loss = total_value_loss / max(num_batches, 1)
            avg_discriminator_loss = total_discriminator_loss / max(num_batches, 1)
            
            print(f"Epoch {epoch}: Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}, "
                  f"Discriminator Loss: {avg_discriminator_loss:.4f}")
            

            total_policy_loss = 0
            total_value_loss = 0
            total_discriminator_loss = 0
            num_batches = 0
    

    model.save_model("alarm2spread_model.pth")
    print("model saved alarm2spread_model.pth")
    

    if topology_graph and len(topology_graph.nodes()) > 0:
        test_start_node = list(topology_graph.nodes())[0]
        predicted_path = model.predict_alarm_path(test_start_node)

if __name__ == "__main__":
    main()