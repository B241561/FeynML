"""
Causal Thinking Module
======================

Builds and analyzes causal DAGs (Directed Acyclic Graphs):
  • Confounder identification
  • Collider detection
  • Mediator analysis
  • Simpson's Paradox detection

Educational Focus:
  Causal thinking is essential for correct interpretation of ML results:
  1. Correlation ≠ causation
  2. Different variables play different roles in causal graphs
  3. Conditioning on the wrong variable can hide or create bias
  
  This module teaches DAG reasoning through concrete examples.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
import warnings

from engine.base_module import BaseModule

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class CausalGraphBuilder(BaseModule):
    """
    Builds and analyzes causal DAGs for reasoning about treatment effects.
    
    Educational Philosophy:
      DAGs are a language for causal reasoning. This module teaches:
      1. How to specify assumptions about causal structure
      2. What bias can arise from different conditioning strategies
      3. How to identify the right adjustment sets
    """

    def __init__(self, verbose=True):
        super().__init__(verbose=verbose)
        self.dag = None

    def build_dag(self, nodes: List[str], edges: List[Tuple[str, str]]):
        """
        Create a DAG from user-specified nodes and edges.
        
        Parameters
        ----------
        nodes : list of str
            Variable names
        edges : list of (from_node, to_node) tuples
            Causal directions
        
        Returns
        -------
        dict with keys:
          'dag' : networkx.DiGraph
          'is_acyclic' : bool
          'cycle' : list or None
          'n_nodes' : int
          'n_edges' : int
        
        Raises
        ------
        ValueError if edges contain cycles
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx required for DAG building")
        
        self._log(f"Building DAG with {len(nodes)} nodes and {len(edges)} edges")
        
        # Create directed graph
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)
        dag.add_edges_from(edges)
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(dag):
            cycles = list(nx.simple_cycles(dag))
            raise ValueError(f"DAG contains cycles: {cycles}. A DAG must be acyclic.")
        
        self.dag = dag
        
        self._log(f"DAG built successfully. Acyclic: {nx.is_directed_acyclic_graph(dag)}")
        
        return {
            'dag': dag,
            'is_acyclic': True,
            'cycle': None,
            'n_nodes': len(nodes),
            'n_edges': len(edges),
            'node_list': nodes,
            'edge_list': edges,
            'status': 'SUCCESS'
        }

    def identify_confounders(self, treatment: str, outcome: str):
        """
        Find all common causes of treatment and outcome (backdoor paths).
        
        WHY THIS MATTERS:
          Confounders are the #1 source of spurious associations:
          1. They cause both treatment and outcome
          2. Naive comparison of treatment groups is biased
          3. We must adjust for confounders in analysis
        
        Parameters
        ----------
        treatment : str
        outcome : str
        
        Returns
        -------
        dict with keys:
          'confounders' : list of str
              Common cause variables
          'explanation' : str
          'adjustment_recommendation' : str
        
        Raises
        ------
        ValueError if DAG not built or nodes not found
        """
        if self.dag is None:
            raise ValueError("DAG not built. Call build_dag() first.")
        
        if treatment not in self.dag or outcome not in self.dag:
            raise ValueError(f"Treatment or outcome not in DAG")
        
        self._log(f"Identifying confounders between '{treatment}' and '{outcome}'")
        
        # Common ancestors are potential confounders
        ancestors_of_treatment = nx.ancestors(self.dag, treatment)
        ancestors_of_outcome = nx.ancestors(self.dag, outcome)
        
        confounders = list(ancestors_of_treatment & ancestors_of_outcome)
        
        explanation = (
            f"Confounders for {treatment} → {outcome}:\n"
            f"  • Common causes: {confounders if confounders else 'None'}\n"
            f"  • These variables influence both treatment assignment and outcome\n"
        )
        
        adjustment_rec = (
            f"To estimate causal effect of {treatment} on {outcome}:\n"
            f"  • Adjust for (condition on): {', '.join(confounders) if confounders else 'none needed'}\n"
            f"  • OR use instrumental variables if confounders unobserved\n"
        )
        
        return {
            'confounders': confounders,
            'explanation': explanation,
            'adjustment_recommendation': adjustment_rec,
            'n_confounders': len(confounders),
            'status': 'SUCCESS'
        }

    def identify_colliders(self, treatment: str, outcome: str):
        """
        Find collider variables (caused by both treatment and outcome).
        
        WHY THIS MATTERS:
          Colliders are causal dead-ends. DO NOT condition on them:
          1. Conditioning on a collider OPENS a backdoor path
          2. Creates spurious association between treatment and outcome
          3. Common mistake: medical students' error, Berkson's paradox
        
        Parameters
        ----------
        treatment : str
        outcome : str
        
        Returns
        -------
        dict with keys:
          'colliders' : list of str
          'explanation' : str
          'warning' : str
        """
        if self.dag is None:
            raise ValueError("DAG not built. Call build_dag() first.")
        
        self._log(f"Identifying colliders for {treatment} and {outcome}")
        
        colliders = []
        
        # Check each node
        for node in self.dag.nodes():
            # Does node have incoming edges from both treatment and outcome?
            treatment_path = nx.has_path(self.dag, treatment, node)
            outcome_path = nx.has_path(self.dag, outcome, node)
            
            if treatment_path and outcome_path and node != treatment and node != outcome:
                colliders.append(node)
        
        explanation = (
            f"Colliders for {treatment} and {outcome}:\n"
            f"  • Variables caused by both: {colliders if colliders else 'None'}\n"
            f"  • These are NOT confounders—conditioning opens new bias\n"
        )
        
        warning = (
            f"⚠️  WARNING: Do NOT condition on colliders!\n"
            f"  • Conditioning on {colliders if colliders else 'none'} creates spurious association\n"
            f"  • This is Berkson's paradox / selection bias\n"
        ) if colliders else "No colliders detected."
        
        return {
            'colliders': colliders,
            'explanation': explanation,
            'warning': warning,
            'n_colliders': len(colliders),
            'status': 'SUCCESS'
        }

    def identify_mediators(self, treatment: str, outcome: str):
        """
        Find variables on the causal path from treatment to outcome.
        
        WHY THIS MATTERS:
          Mediators are on the causal pathway:
          1. Treatment → Mediator → Outcome
          2. If you condition on mediators, you block the effect
          3. Useful for understanding mechanisms (indirect effects)
        
        Parameters
        ----------
        treatment : str
        outcome : str
        
        Returns
        -------
        dict with keys:
          'mediators' : list of str
          'explanation' : str
          'usage_note' : str
        """
        if self.dag is None:
            raise ValueError("DAG not built. Call build_dag() first.")
        
        self._log(f"Identifying mediators between {treatment} and {outcome}")
        
        mediators = []
        
        # Find all paths from treatment to outcome
        try:
            all_paths = list(nx.all_simple_paths(self.dag, treatment, outcome))
            
            if all_paths:
                # Mediators are nodes on any path (excluding treatment and outcome)
                path_nodes = set()
                for path in all_paths:
                    path_nodes.update(path[1:-1])  # Exclude first (treatment) and last (outcome)
                
                mediators = list(path_nodes)
        except nx.NetworkXNoPath:
            mediators = []
        
        explanation = (
            f"Mediators for {treatment} → {outcome}:\n"
            f"  • Variables on causal path: {mediators if mediators else 'None'}\n"
            f"  • These transmit the effect from treatment to outcome\n"
        )
        
        usage_note = (
            f"Usage for mediation analysis:\n"
            f"  • Direct effect: adjust for mediators\n"
            f"  • Indirect effect: don't adjust for mediators\n"
            f"  • Total effect: don't adjust for mediators\n"
        )
        
        return {
            'mediators': mediators,
            'explanation': explanation,
            'usage_note': usage_note,
            'n_mediators': len(mediators),
            'status': 'SUCCESS'
        }

    def detect_simpsons_paradox(self, df, treatment_col, outcome_col, group_col):
        """
        Detect Simpson's Paradox: reversal of treatment effect by stratification.
        
        WHY THIS MATTERS:
          Simpson's Paradox is a classic statistical paradox:
          1. Treatment looks beneficial in aggregate
          2. But harmful in every subgroup (or vice versa)
          3. Caused by confounding stratification variable
          4. Dangerous: easy to misinterpret data
        
        Parameters
        ----------
        df : pandas.DataFrame
        treatment_col : str
            Binary treatment indicator (0/1)
        outcome_col : str
            Binary outcome (0/1) or continuous
        group_col : str
            Stratification variable
        
        Returns
        -------
        dict with keys:
          'paradox_detected' : bool
          'overall_effect' : float
          'stratified_effects' : dict
          'explanation' : str
          'recommendation' : str
        """
        self._log(f"Testing Simpson's Paradox: {treatment_col} → {outcome_col}, stratified by {group_col}")
        
        if treatment_col not in df.columns or outcome_col not in df.columns or group_col not in df.columns:
            raise ValueError("One or more columns not found")
        
        # Compute overall effect (e.g., difference in means)
        treated = df[df[treatment_col] == 1][outcome_col]
        control = df[df[treatment_col] == 0][outcome_col]
        
        if len(treated) == 0 or len(control) == 0:
            return {
                'paradox_detected': False,
                'overall_effect': None,
                'stratified_effects': {},
                'explanation': 'Insufficient treatment/control groups',
                'status': 'FAILED'
            }
        
        overall_effect = float(np.mean(treated) - np.mean(control))
        
        # Compute stratified effects
        stratified_effects = {}
        stratified_signs = []
        
        for group in df[group_col].unique():
            group_df = df[df[group_col] == group]
            
            treated_g = group_df[group_df[treatment_col] == 1][outcome_col]
            control_g = group_df[group_df[treatment_col] == 0][outcome_col]
            
            if len(treated_g) > 0 and len(control_g) > 0:
                effect_g = float(np.mean(treated_g) - np.mean(control_g))
                stratified_effects[str(group)] = effect_g
                stratified_signs.append(np.sign(effect_g))
        
        # Paradox if:
        # 1. Overall effect has one sign
        # 2. All (or most) stratified effects have opposite sign
        overall_sign = np.sign(overall_effect)
        paradox = (
            len(stratified_signs) > 1 and
            all(s == stratified_signs[0] for s in stratified_signs) and
            stratified_signs[0] != overall_sign
        )
        
        explanation = (
            f"Simpson's Paradox Analysis:\n"
            f"  • Overall effect: {overall_effect:.4f}\n"
            f"  • Stratified effects: {stratified_effects}\n"
            f"  • Paradox detected: {paradox}\n"
        )
        
        recommendation = (
            "WARNING: Simpson's Paradox likely!\n"
            f"  • Report stratified effects, NOT overall\n"
            f"  • {group_col} is a confounding variable\n"
            f"  • Control for {group_col} in analysis\n"
        ) if paradox else "No paradox detected."
        
        return {
            'paradox_detected': bool(paradox),
            'overall_effect': overall_effect,
            'stratified_effects': stratified_effects,
            'explanation': explanation,
            'recommendation': recommendation,
            'status': 'SUCCESS'
        }

    def run(self, **kwargs):
        """
        Placeholder run method (causal thinking is typically interactive).
        """
        return self._result(
            {'message': 'CausalGraphBuilder requires manual DAG specification via methods'},
            severity='NONE',
            module_name='CausalGraphBuilder'
        )
