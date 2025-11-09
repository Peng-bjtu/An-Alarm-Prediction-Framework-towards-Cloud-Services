import numpy as np
from collections import defaultdict

class DFSCode:
    def __init__(self):
        self.codes = []
        
    def add_edge_code(self, source_id, target_id, source_type, target_type, edge_type):
        self.codes.append((source_id, target_id, source_type, target_type, edge_type))
        
    def __lt__(self, other):
        min_len = min(len(self.codes), len(other.codes))
        for i in range(min_len):
            if self.codes[i] != other.codes[i]:
                return self.codes[i] < other.codes[i]
        return len(self.codes) < len(other.codes)
    
    def __repr__(self):
        return f"DFSCode({self.codes})"

def generate_dfs_code(subtree):
    if not subtree.nodes:
        return DFSCode()
    
    dfs_code = DFSCode()
    visited = set()
    stack = []
    
    start_node = min(subtree.nodes, key=lambda x: x.timestamp)
    stack.append(start_node)
    visited.add(start_node)
    
    node_to_id = {}
    next_id = 0
    node_to_id[start_node] = next_id
    
    while stack:
        current = stack.pop()
        current_id = node_to_id[current]
        
        outgoing_edges = [e for e in subtree.edges if e.source == current and e.target not in visited]
        
        outgoing_edges.sort(key=lambda e: (e.target.alarm_type, e.target.timestamp))
        
        for edge in outgoing_edges:
            target = edge.target
            if target not in node_to_id:
                next_id += 1
                node_to_id[target] = next_id
                
            target_id = node_to_id[target]
            dfs_code.add_edge_code(current_id, target_id, 
                                 current.alarm_type, target.alarm_type,
                                 "propagate")
            
            if target not in visited:
                visited.add(target)
                stack.append(target)
    
    return dfs_code