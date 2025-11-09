class AlarmNode:
    def __init__(self, alarm_id, alarm_type, properties=None):
        self.alarm_id = alarm_id
        self.alarm_type = alarm_type
        self.properties = properties or {}
        self.timestamp = properties.get('time', 0) if properties else 0
        
    def __hash__(self):
        return hash(self.alarm_id)
    
    def __eq__(self, other):
        return self.alarm_id == other.alarm_id
    
    def __repr__(self):
        return f"Alarm({self.alarm_id}, {self.alarm_type})"

class AlarmEdge:
    def __init__(self, source, target, frequency=1, properties=None):
        self.source = source
        self.target = target
        self.frequency = frequency
        self.properties = properties or {}
        
    def __hash__(self):
        return hash((self.source, self.target))
    
    def __eq__(self, other):
        return self.source == other.source and self.target == other.target
    
    def __repr__(self):
        return f"Edge({self.source} -> {self.target}, freq:{self.frequency})"

class AlarmSubtree:
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes or set()
        self.edges = edges or set()
        self.dfs_code = None
        
    def add_edge(self, edge):
        self.edges.add(edge)
        self.nodes.add(edge.source)
        self.nodes.add(edge.target)
        
    def __hash__(self):
        return hash(tuple(sorted(node.alarm_id for node in self.nodes)))
    
    def __eq__(self, other):
        return self.nodes == other.nodes and self.edges == other.edges
    
    def __repr__(self):
        return f"Subtree(nodes:{len(self.nodes)}, edges:{len(self.edges)})"