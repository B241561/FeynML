"""
Confounder Detector Module
==========================
Identifies structural roles of nodes (confounder, collider, mediator) 
given a DAG and a causal query X -> Y.
"""

import os
import sys

# Import DAG
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dag_builder import DAG

def classify_node_role(dag, X, Y, Z):
    """
    Classify the role of node Z relative to the causal query X -> Y.
    """
    # 1. Confounder: Z -> X and Z -> Y
    parents_x = dag.get_parents(X)
    parents_y = dag.get_parents(Y)
    if Z in parents_x and Z in parents_y:
        return "CONFOUNDER"
    
    # 2. Mediator: X -> Z -> Y
    children_x = dag.get_children(X)
    parents_y = dag.get_parents(Y)
    if Z in children_x and Z in parents_y:
        return "MEDIATOR"
    
    # 3. Collider: X -> Z and Y -> Z
    children_x = dag.get_children(X)
    children_y = dag.get_children(Y)
    if Z in children_x and Z in children_y:
        return "COLLIDER"
    
    return "NONE"

def find_all_confounders(dag, X, Y):
    """Find all common causes of X and Y."""
    anc_x = dag.get_ancestors(X)
    anc_y = dag.get_ancestors(Y)
    return list(anc_x.intersection(anc_y))

def find_all_colliders(dag, X, Y):
    """Find all common effects of X and Y."""
    children_x = set(dag.get_children(X))
    children_y = set(dag.get_children(Y))
    return list(children_x.intersection(children_y))

def find_mediators(dag, X, Y):
    """Find nodes on directed paths from X to Y."""
    paths = dag.all_paths(X, Y)
    mediators = set()
    for path in paths:
        # Path is [X, M1, M2, ..., Y]
        for node in path[1:-1]:
            mediators.add(node)
    return list(mediators)

def should_adjust_for(dag, X, Y, Z):
    """
    Decide if we should adjust for Z in the causal query X -> Y.
    Rule: Adjust for confounders, NEVER adjust for colliders or mediators.
    """
    role = classify_node_role(dag, X, Y, Z)
    if role == "CONFOUNDER":
        return True, "Z is a confounder (common cause)."
    if role == "COLLIDER":
        return False, "Z is a collider. Adjusting would induce selection bias."
    if role == "MEDIATOR":
        return False, "Z is a mediator. Adjusting would block the direct effect."
    
    return False, "Z has no clear structural role needing adjustment."

def adjustment_warning(dag, X, Y, proposed_set):
    """Generate warnings for a proposed adjustment set."""
    warnings = []
    for Z in proposed_set:
        role = classify_node_role(dag, X, Y, Z)
        if role == "COLLIDER":
            warnings.append(f"WARNING: Adjusting for collider '{Z}' induces selection bias.")
        if role == "MEDIATOR":
            warnings.append(f"WARNING: Adjusting for mediator '{Z}' blocks direct causal effect.")
    return warnings

def run_verification():
    """Run module verification."""
    print("--- Confounder Detector Verification ---")
    dag = DAG()
    # Confounder
    dag.add_edge("Age", "Treatment")
    dag.add_edge("Age", "Outcome")
    # Mediator
    dag.add_edge("Treatment", "Biomarker")
    dag.add_edge("Biomarker", "Outcome")
    # Collider
    dag.add_edge("Treatment", "SideEffect")
    dag.add_edge("Outcome", "SideEffect")
    
    X, Y = "Treatment", "Outcome"
    print(f"Roles for X={X}, Y={Y}:")
    for node in ["Age", "Biomarker", "SideEffect"]:
        role = classify_node_role(dag, X, Y, node)
        should, reason = should_adjust_for(dag, X, Y, node)
        print(f"  {node}: {role} | Should adjust? {should} ({reason})")

if __name__ == "__main__":
    run_verification()
