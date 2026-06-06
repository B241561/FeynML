"""
DAG Builder Module
==================
Constructs and validates Directed Acyclic Graphs (DAGs) representing causal assumptions.

Key Concepts:
-------------
A DAG is a directed graph with no cycles. In causal inference, edges represent 
direct causal effects (X -> Y means X causes Y).
"""

import json

class DAG:
    """
    Representation of a Directed Acyclic Graph for causal modeling.
    """
    def __init__(self):
        self.nodes = {}  # name -> label
        self.edges = []  # list of (from, to)
        self.adj = {}    # name -> list of children
        self.rev_adj = {} # name -> list of parents

    def add_node(self, name, label=None):
        """Add a node to the DAG."""
        if name not in self.nodes:
            self.nodes[name] = label or name
            self.adj[name] = []
            self.rev_adj[name] = []

    def add_edge(self, from_node, to_node):
        """Add a directed edge between nodes."""
        self.add_node(from_node)
        self.add_node(to_node)
        if (from_node, to_node) not in self.edges:
            self.edges.append((from_node, to_node))
            self.adj[from_node].append(to_node)
            self.rev_adj[to_node].append(from_node)
        
        if self.has_cycle():
            # Rollback
            self.edges.remove((from_node, to_node))
            self.adj[from_node].remove(to_node)
            self.rev_adj[to_node].remove(from_node)
            raise ValueError(f"Adding edge {from_node} -> {to_node} would create a cycle.")

    def has_cycle(self):
        """Detect cycles using DFS."""
        visited = set()
        rec_stack = set()

        def is_cyclic_util(v):
            visited.add(v)
            rec_stack.add(v)
            for neighbor in self.adj.get(v, []):
                if neighbor not in visited:
                    if is_cyclic_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(v)
            return False

        for node in self.nodes:
            if node not in visited:
                if is_cyclic_util(node):
                    return True
        return False

    def get_parents(self, node):
        """Return list of parent nodes."""
        return self.rev_adj.get(node, [])

    def get_children(self, node):
        """Return list of child nodes."""
        return self.adj.get(node, [])

    def get_ancestors(self, node):
        """Return set of ancestor nodes."""
        ancestors = set()
        stack = self.get_parents(node)
        while stack:
            curr = stack.pop()
            if curr not in ancestors:
                ancestors.add(curr)
                stack.extend(self.get_parents(curr))
        return ancestors

    def all_paths(self, source, target, visited=None, path=None):
        """Find all directed paths between source and target."""
        if visited is None: visited = set()
        if path is None: path = []
        
        visited.add(source)
        path.append(source)
        
        results = []
        if source == target:
            results.append(list(path))
        else:
            for neighbor in self.adj.get(source, []):
                if neighbor not in visited:
                    results.extend(self.all_paths(neighbor, target, visited, path))
        
        path.pop()
        visited.remove(source)
        return results

    def backdoor_paths(self, X, Y):
        """
        Identify all backdoor paths between X and Y.
        A backdoor path is a non-directed path starting with an arrow into X.
        X <- ... -> Y.
        """
        # Backdoor paths are paths in the undirected version of the graph 
        # that start with an edge into X and end at Y.
        # This is a simplified version: look for common ancestors.
        anc_x = self.get_ancestors(X)
        anc_y = self.get_ancestors(Y)
        common = anc_x.intersection(anc_y)
        
        paths = []
        for c in common:
            # Path X <- ... <- C -> ... -> Y
            # For simplicity, we just return the common ancestors that create backdoor paths.
            paths.append(c)
        return paths

    def is_d_separated(self, X, Y, given_set):
        """
        Check if X and Y are d-separated given a set of nodes.
        (Placeholder for full d-separation algorithm)
        """
        # Implementation of d-separation is complex. 
        # Placeholder logic: check if given_set blocks all paths.
        return False

    def adjustment_set(self, X, Y):
        """Identify minimal adjustment set for causal query X -> Y."""
        # Simple backdoor criterion: adjust for all parents of X.
        return self.get_parents(X)

    def to_dict(self):
        """Return serializable dictionary representation."""
        return {
            "nodes": self.nodes,
            "edges": self.edges
        }

    def __repr__(self):
        return f"DAG(nodes={list(self.nodes.keys())}, edges={self.edges})"

def run_verification():
    """Run module verification."""
    print("--- DAG Builder Verification ---")
    dag = DAG()
    dag.add_edge("Z", "X")
    dag.add_edge("Z", "Y")
    dag.add_edge("X", "Y")
    
    print(f"DAG: {dag}")
    print(f"Parents of X: {dag.get_parents('X')}")
    print(f"Ancestors of Y: {dag.get_ancestors('Y')}")
    print(f"Paths X to Y: {dag.all_paths('X', 'Y')}")
    
    try:
        dag.add_edge("Y", "Z") # Should fail
    except ValueError as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    run_verification()
