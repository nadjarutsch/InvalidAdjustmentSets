import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations


def estimate_scm(data: np.ndarray, graph: nx.DiGraph) -> nx.DiGraph:
    """
    Estimate the Structural Causal Model (SCM) coefficients based on the provided DAG and normalized data, where the data is structured with variables as rows and samples as columns. Return the graph with edges labeled with the regression coefficients.

    Parameters:
        data (np.ndarray): Observational data where rows correspond to variables and columns to samples in the DAG.
        graph (nx.DiGraph): Directed acyclic graph representing the causal structure among the variables.

    Returns:
        nx.DiGraph: A new directed graph with the same structure as the input graph but with edges annotated with the estimated coefficients.
    """
    # Number of variables should match the number of nodes in the graph
    assert data.shape[0] == len(
        graph.nodes()), "The number of rows in data must match the number of nodes in the graph."

    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data.T).T  # Transpose for normalization and transpose back

    # Variables are assumed to be in the same order as the nodes in the graph for simplicity
    variable_names = list(graph.nodes())
    scm_graph = nx.DiGraph()

    # Initialize the graph with nodes
    scm_graph.add_nodes_from(variable_names)

    for i, var in enumerate(variable_names):
        # Identify parents of the variable in the DAG
        parents = list(graph.predecessors(var))
        if parents:
            # If there are parents, get the indices of the parents
            parent_indices = [variable_names.index(p) for p in parents]
            # Extract the normalized data for the parents
            X = normalized_data[parent_indices, :]
        else:
            # If no parents, no edges to add for this variable
            continue

        # Response variable from normalized data
        y = normalized_data[i, :]

        # Create the Linear Regression model without an intercept
        model = LinearRegression(fit_intercept=False)

        # Fit the model
        model.fit(X.T, y)  # Transpose X to match samples as columns

        # Extract coefficients and add edges with these coefficients as labels
        for j, coef in zip(parent_indices, model.coef_):
            parent = variable_names[j]
            # Add edge with coefficient as the label
            scm_graph.add_edge(parent, var, weight=coef)

    return scm_graph


def get_adjustment_set(data, graph, optimality):
    adjustment_candidates = prune_variables(graph)
    scm_graph = estimate_scm(data, graph)       
    
    potential_adj_sets = []
    sample_size = data.shape[1] # Number of columns/samples

    # Generating all combinations
    for r in range(0, len(adjustment_candidates) + 1):
        all_combinations = combinations(adjustment_candidates, r)
        potential_adj_sets.extend([set(combo) for combo in all_combinations])
        
    potential_adj_sets = prune_adj_sets(potential_adj_sets, graph)
    properties = []

    for adjustment_set in potential_adj_sets:
        bias, variability_ratio = get_bias_and_var_ratio(scm_graph, adjustment_set)
        variance = variability_ratio / (sample_size - len(adjustment_set) - 3)
        expected_mse = bias ** 2 + variance
        properties.append({
            'Adjustment set': adjustment_set,
            'Size': len(adjustment_set),
            'Variability ratio': variability_ratio,
            'Bias': bias,
            'Variance': variance,
            'MSE': expected_mse
        })

    # Find the adjustment set with the minimum MSE or Variance
    best_property = min(properties, key=lambda x: x[optimality])

    return best_property['Adjustment set'], best_property['Size'], best_property['Variability ratio'], best_property['Bias'], best_property['Variance'], best_property['MSE']


def prune_variables(graph):
    pruned_variables = list(graph.nodes())  # TODO: implement graphical criterion
    return pruned_variables


def prune_adj_sets(potential_adj_sets, graph):  # TODO: implement graphical criterion
    return potential_adj_sets


def get_bias_and_var_ratio(scm_graph, adjustment_set):
    # TODO: compute from graph

    direct_effect = scm_graph['A']['Y']['weight']
    conf_f1o1 = scm_graph['O1']['Y']['weight'] * scm_graph['F1']['O1']['weight'] * scm_graph['F1']['A']['weight']
    conf_f2o1 = scm_graph['O1']['Y']['weight'] * scm_graph['F2']['O1']['weight'] * scm_graph['F2']['A']['weight']
    conf_o2 = scm_graph['O2']['Y']['weight'] * scm_graph['O2']['A']['weight']

    coef_o1a = scm_graph['F1']['O1']['weight'] * scm_graph['F1']['A']['weight'] + scm_graph['F2']['O1']['weight'] * scm_graph['F2']['A']['weight']
    coef_f1y = scm_graph['F1']['O1']['weight'] * scm_graph['O1']['Y']['weight']
    coef_f2y = scm_graph['F2']['O1']['weight'] * scm_graph['O1']['Y']['weight']
    coef_f1a = scm_graph['F1']['A']['weight']
    coef_f2a = scm_graph['F2']['A']['weight']
    coef_o2a = scm_graph['O2']['A']['weight']
    coef_o1y = scm_graph['O1']['Y']['weight']
    coef_o2y = scm_graph['O2']['Y']['weight']

    if adjustment_set == set([]):
        treatment_var = 1
        outcome_var = 1 - direct_effect ** 2
        bias = (conf_f1o1 + conf_f2o1 + conf_o2) / treatment_var

    elif adjustment_set == set(["O1"]):
        treatment_var = 1 - coef_o1a ** 2
        outcome_var = 1 - coef_o1y ** 2 - direct_effect ** 2
        bias = conf_o2 / treatment_var

    elif adjustment_set == set(["O2"]):
        treatment_var = 1 - coef_o2a ** 2
        outcome_var = 1 - coef_o2y ** 2 - direct_effect ** 2
        bias = (conf_f1o1 + conf_f2o1) / treatment_var

    elif adjustment_set == set(["F1"]):
        treatment_var = 1 - coef_f1a ** 2
        outcome_var = 1 - coef_f1y ** 2 - direct_effect ** 2
        bias = (conf_f2o1 + conf_o2) / treatment_var

    elif adjustment_set == set(["F2"]):
        treatment_var = 1 - coef_f2a ** 2
        outcome_var = 1 - coef_f2y ** 2 - direct_effect ** 2
        bias = (conf_f1o1 + conf_o2) / treatment_var

    elif adjustment_set == set(["O1", "O2"]):
        treatment_var = 1 - coef_o1a ** 2 - coef_o2a ** 2
        outcome_var = 1 - coef_o1y ** 2 - coef_o2y ** 2 - direct_effect ** 2
        bias = 0

    elif adjustment_set == set(["O2", "F1"]):
        treatment_var = max(1 - coef_f1a ** 2 - coef_o2a ** 2, 1e-6)    # CAN NOT BE NEGATIVE
        outcome_var = 1 - coef_o2y ** 2 - coef_f1y ** 2 - direct_effect ** 2
        bias = conf_f2o1 / treatment_var

    elif adjustment_set == set(["O2", "F2"]):
        treatment_var = 1 - coef_f2a ** 2 - coef_o2a ** 2
        outcome_var = 1 - coef_o2y ** 2 - coef_f2y ** 2 - direct_effect ** 2
        bias = conf_f1o1 / treatment_var

    else:   # factually exclude other adjustment sets
        treatment_var = 0.001
        outcome_var = 1
        bias = 1000

    return bias, outcome_var / treatment_var