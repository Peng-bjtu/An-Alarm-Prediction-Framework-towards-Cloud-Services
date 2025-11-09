import json
from frequent_subtree_miner import FrequentSubtreeMiner
from graph_structures import AlarmNode, AlarmEdge, AlarmSubtree

def load_alarm_graphs_from_file(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"false: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    alarm_graphs = []
    
    if isinstance(data, dict):
        graph = _parse_single_graph(data)
        alarm_graphs.append(graph)
    elif isinstance(data, list):
        for graph_data in data:
            graph = _parse_single_graph(graph_data)
            alarm_graphs.append(graph)
    else:
        raise ValueError("error)
    
    print(f"load")
    return alarm_graphs

def _parse_single_graph(graph_data):
    graph = AlarmSubtree()
    
 
    nodes_dict = {}  
    
    if 'nodes' in graph_data:
        for node_data in graph_data['nodes']:
            node_id = node_data.get('alarm_id', str(len(nodes_dict)))
            alarm_type = node_data.get('alarm_type', 'Unknown')
            properties = node_data.get('properties', {})
            
            node = AlarmNode(node_id, alarm_type, properties)
            nodes_dict[node_id] = node
            graph.nodes.add(node)
    
    if 'edges' in graph_data:
        for edge_data in graph_data['edges']:
            source_id = edge_data.get('source')
            target_id = edge_data.get('target')
            
            if source_id in nodes_dict and target_id in nodes_dict:
                source_node = nodes_dict[source_id]
                target_node = nodes_dict[target_id]
                
                frequency = edge_data.get('frequency', 1)
                edge_properties = edge_data.get('properties', {})
                
                edge = AlarmEdge(source_node, target_node, frequency, edge_properties)
                graph.edges.add(edge)
            else:
                print(f"error2")
    
    return graph



def main():

    alarm_graphs = load_alarm_graphs_from_file("data.json")
    alarm_graphs = create_sample_data()
    
    print(f"Loaded {len(alarm_graphs)} alarm graphs")
    

    miner = FrequentSubtreeMiner(min_support=0.3, semantic_threshold=0.7)
    
    frequent_subtrees = miner.mine_frequent_subtrees(alarm_graphs)
    
    print(f"Found {len(frequent_subtrees)} frequent subtrees:")
    for i, subtree in enumerate(frequent_subtrees):
        print(f"Subtree {i+1}: {subtree}")
        for edge in subtree.edges:
            print(f"  {edge}")

    output_file = "frequent_subtrees_result.json"
    save_results(frequent_subtrees, output_file)


def save_results(frequent_subtrees, output_path):

    results = []
    
    for subtree in frequent_subtrees:
        subtree_data = {
            "nodes": [],
            "edges": []
        }
        

        for node in subtree.nodes:
            node_data = {
                "alarm_id": node.alarm_id,
                "alarm_type": node.alarm_type,
                "properties": node.properties
            }
            subtree_data["nodes"].append(node_data)
        

        for edge in subtree.edges:
            edge_data = {
                "source": edge.source.alarm_id,
                "target": edge.target.alarm_id,
                "frequency": edge.frequency,
                "properties": edge.properties
            }
            subtree_data["edges"].append(edge_data)
        
        results.append(subtree_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()