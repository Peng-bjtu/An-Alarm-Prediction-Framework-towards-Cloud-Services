from collections import defaultdict
from typing import List, Set, Dict
from graph_structures import AlarmNode, AlarmEdge, AlarmSubtree
from dfs_utils import generate_dfs_code, DFSCode
from semantic_consistency import SemanticConsistencyChecker

class FrequentSubtreeMiner:
    def __init__(self, min_support=0.1, semantic_threshold=0.7):
        self.min_support = min_support
        self.semantic_threshold = semantic_threshold
        self.semantic_checker = SemanticConsistencyChecker()
        
    def find_frequent_edges(self, alarm_graphs):
        edge_count = defaultdict(int)
        total_graphs = len(alarm_graphs)
        
        for graph in alarm_graphs:
            edges_in_graph = set()
            for edge in graph.edges:
                edge_key = (edge.source.alarm_id, edge.target.alarm_id, 
                          edge.source.alarm_type, edge.target.alarm_type)
                edges_in_graph.add(edge_key)
                
            for edge_key in edges_in_graph:
                edge_count[edge_key] += 1
                
        frequent_edges = []
        for edge_key, count in edge_count.items():
            support = count / total_graphs
            if support >= self.min_support:
                source_id, target_id, source_type, target_type = edge_key
                source_node = AlarmNode(source_id, source_type)
                target_node = AlarmNode(target_id, target_type)
                edge = AlarmEdge(source_node, target_node, frequency=count)
                frequent_edges.append(edge)
                
        return frequent_edges
    
    def find_forward_edges(self, subtree, alarm_graphs):
        forward_edges = set()
        
        for graph in alarm_graphs:
            if not self._contains_subtree(subtree, graph):
                continue
                
            subtree_instances = self._find_subtree_instances(subtree, graph)
            
            for instance in subtree_instances:
                for node in instance.nodes:
                    for edge in graph.edges:
                        if edge.source == node and edge.target not in instance.nodes:
                            forward_edges.add(edge)
                            

        frequent_forward_edges = []
        total_containing = sum(1 for g in alarm_graphs if self._contains_subtree(subtree, g))
        
        for edge in forward_edges:
            count = sum(1 for g in alarm_graphs 
                       if self._contains_subtree(subtree, g) and 
                       self._has_forward_edge(subtree, edge, g))
            
            support = count / total_containing if total_containing > 0 else 0
            if support >= self.min_support:
                frequent_edge = AlarmEdge(edge.source, edge.target, frequency=count)
                frequent_forward_edges.append(frequent_edge)
                
        return frequent_forward_edges
    
    def mine_frequent_subtrees(self, alarm_graphs):

        frequent_edges = self.find_frequent_edges(alarm_graphs)
        print(f"Found {len(frequent_edges)} frequent edges")
        

        frequent_edges_with_code = []
        for edge in frequent_edges:
            subtree = AlarmSubtree()
            subtree.add_edge(edge)
            dfs_code = generate_dfs_code(subtree)
            frequent_edges_with_code.append((dfs_code, edge.frequency, subtree))
            
        frequent_edges_with_code.sort(key=lambda x: (x[0], -x[1]))  
        
        Tr = set()  
        
        for dfs_code, freq, initial_subtree in frequent_edges_with_code:
            if initial_subtree not in Tr:
                Tr.add(initial_subtree)
                self._fst_expand(alarm_graphs, initial_subtree, Tr)
                
        return list(Tr)
    
    def _fst_expand(self, alarm_graphs, subtree, Tr):

        if not self._has_minimum_dfs_code(subtree):
            return
            
        forward_edges = self.find_forward_edges(subtree, alarm_graphs)
        
        for edge in forward_edges:
            new_subtree = AlarmSubtree(subtree.nodes.copy(), subtree.edges.copy())
            new_subtree.add_edge(edge)
            
            if (not self._is_isomorphic_to_any(new_subtree, Tr) and 
                self._check_semantic_consistency(new_subtree, alarm_graphs)):
                
                Tr.add(new_subtree)
                self._fst_expand(alarm_graphs, new_subtree, Tr)
    
    def _contains_subtree(self, subtree, graph):

        for subtree_edge in subtree.edges:
            edge_found = False
            for graph_edge in graph.edges:
                if (subtree_edge.source.alarm_id == graph_edge.source.alarm_id and
                    subtree_edge.target.alarm_id == graph_edge.target.alarm_id):
                    edge_found = True
                    break
            if not edge_found:
                return False
        return True
    
    def _find_subtree_instances(self, subtree, graph):

        instances = []
        instance = AlarmSubtree()
        
        for subtree_edge in subtree.edges:
            for graph_edge in graph.edges:
                if (subtree_edge.source.alarm_id == graph_edge.source.alarm_id and
                    subtree_edge.target.alarm_id == graph_edge.target.alarm_id):
                    instance.add_edge(graph_edge)
                    break
                    
        if len(instance.edges) == len(subtree.edges):
            instances.append(instance)
            
        return instances
    
    def _has_forward_edge(self, subtree, edge, graph):
        for graph_edge in graph.edges:
            if (graph_edge.source.alarm_id == edge.source.alarm_id and
                graph_edge.target.alarm_id == edge.target.alarm_id):
                return True
        return False
    
    def _has_minimum_dfs_code(self, subtree):

        return True
    
    def _is_isomorphic_to_any(self, subtree, subtree_set):

        for existing in subtree_set:
            if self._are_subtrees_isomorphic(subtree, existing):
                return True
        return False
    
    def _are_subtrees_isomorphic(self, subtree1, subtree2):

        if len(subtree1.nodes) != len(subtree2.nodes) or len(subtree1.edges) != len(subtree2.edges):
            return False
            

        code1 = generate_dfs_code(subtree1)
        code2 = generate_dfs_code(subtree2)
        return code1.codes == code2.codes
    
    def _check_semantic_consistency(self, subtree, alarm_graphs):

        instances = []
        for graph in alarm_graphs:
            graph_instances = self._find_subtree_instances(subtree, graph)
            instances.extend(graph_instances)
            
        return self.semantic_checker.check_consistency(instances, self.semantic_threshold)