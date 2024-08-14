import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from itertools import combinations
from scipy.stats import bootstrap
from collections import defaultdict




def estimate_treatment_effect(data: np.ndarray, adjustment_set: list, variables: list) -> float:
    # Map variable names to indices if adjustment_set contains names
    variable_indices = {name: i for i, name in enumerate(variables)}
    adjustment_indices = [variable_indices[name] for name in adjustment_set]

    # Extracting Y and A from the data
    Y = data[variable_indices['Y'], :].T  # Assuming Y is the outcome variable
    A = data[variable_indices['A'], :].T  # Assuming 'A' is the treatment variable

    # Prepare covariates X
    if adjustment_indices:
        X = np.hstack([data[idx, :].reshape(-1, 1) for idx in adjustment_indices])
        X = np.column_stack((X, A))
    else:
        X = A.reshape(-1, 1)

    # Performing linear regression
    model = LinearRegression().fit(X, Y)

    # Returning the coefficient of A (treatment effect)
    return model.coef_[-1]


def get_adjustment_set(data, graph, optimality, n_bootstrap):
    adjustment_candidates = prune_variables(graph)
  #  scm_graph = estimate_scm(data, graph)
    
    potential_adj_sets = []
 #   sample_size = data.shape[1] # Number of columns/samples

    # Generating all combinations
    for r in range(0, len(adjustment_candidates) + 1):
        all_combinations = combinations(adjustment_candidates, r)
        potential_adj_sets.extend([set(combo) for combo in all_combinations])
        
    potential_adj_sets = prune_adj_sets(potential_adj_sets, graph)
    properties = []

    # TODO: remove manual
    potential_adj_sets = [set([]), set(["O1"]), set(["O2"]), set(["F1", "O2"]), set(["F2", "O2"]), set(["F1"]), set(["F2"])]
  #  potential_adj_sets = [set([]), set(["O1", "O2"]), set(["O1"]), set(["O2"]), set(["M1", "O2"]), set(["M1"])]

    est_error_var_outcome, rss_A = estimate_variance(data, graph, set(["O1", "O2"]))
    o_variance = est_error_var_outcome / rss_A

    properties.append({
        'Adjustment set': set(["O1", "O2"]),
        'Size': len(set(["O1", "O2"])),
        'Bias': 0,
        'Variance': o_variance,
        'MSE': o_variance,
        'RSS_A': rss_A,
        'est_error_var_outcome': est_error_var_outcome
    })

    # Store the adjustment sets that should remain
    filtered_adj_sets = [set(["O1", "O2"])]

    for adjustment_set in potential_adj_sets:
        est_error_var_outcome, rss_A = estimate_variance(data, graph, adjustment_set)
        variance = est_error_var_outcome / rss_A

        properties.append({
            'Adjustment set': adjustment_set,
            'Size': len(adjustment_set),
            'Variance': variance,
            'RSS_A': rss_A,
            'est_error_var_outcome': est_error_var_outcome
        })

        # Keep only those adjustment sets where the variance is not less than o_variance
        if variance < o_variance:
            filtered_adj_sets.append(adjustment_set)

    biases = estimate_bias(data, graph, filtered_adj_sets, ["O1", "O2"], n_bootstrap)

    # Adding bias to each entry in properties
    for prop in properties:
        adjustment_set = prop['Adjustment set']
        prop['Bias'] = biases[tuple(adjustment_set)]
        prop['MSE'] = prop['Bias'] ** 2 + prop['Variance']

    # Find the adjustment set with the minimum MSE or Variance
    best_property = min(properties, key=lambda x: x[optimality])

    return best_property['Adjustment set'], best_property['Size'], best_property['Bias'], best_property['Variance'], best_property['MSE'], best_property['RSS_A']


def estimate_variance(data, graph, adjustment_set):
    # Extract column indices from graph nodes
    nodes = list(graph.nodes())
    treatment_index = nodes.index("A")
    outcome_index = nodes.index("Y")

    # Define the adjustment variables
    adjustment_indices = [nodes.index(node) for node in adjustment_set]

    # Prepare data
    A = data[treatment_index, :].reshape(-1, 1)  # Treatment variable (transposed for samples as rows)
    Y = data[outcome_index, :].reshape(-1, 1)  # Outcome variable (transposed for samples as rows)

    if adjustment_indices:
        # Create the adjustment matrix X
        X_adjust = data[adjustment_indices, :].T  # Adjustment variables (transposed for samples as rows)

        # Fit linear regression model for the treatment
        reg_treatment = LinearRegression().fit(X_adjust, A)

        # Add the treatment variable A to the adjustment matrix X
        X_adjust_with_treatment = np.column_stack((X_adjust, A))

        # Fit linear regression model for the outcome
        reg_outcome = LinearRegression().fit(X_adjust_with_treatment, Y)

        # Calculate residuals
        residuals_treatment = A - reg_treatment.predict(X_adjust)
        residuals_outcome = Y - reg_outcome.predict(X_adjust_with_treatment)
    else:
        # If no adjustment variables, just use the mean
        residuals_treatment = A - np.mean(A)
        residuals_outcome = Y - np.mean(Y)

    # Calculate RSS
    rss_treatment = np.sum(residuals_treatment ** 2)
    rss_outcome = np.sum(residuals_outcome ** 2)

    # Calculate the estimated error variance for the outcome
    # If there are no adjustment variables, adjust degrees of freedom accordingly
    df = data.shape[1] - len(adjustment_set) - 1
    est_error_var_outcome = rss_outcome / df

    return est_error_var_outcome, rss_treatment


# Function to estimate the bias of the adjustment set
def estimate_bias(data, graph, adjustment_sets, unbiased_set, n_bootstrap=1000):
    # Create a list of variable names in the order they appear in data
    variables = list(graph.nodes())  # Assuming the order in the graph matches the order in the data

    biases = defaultdict(list)

    for _ in range(n_bootstrap):
        # Resample the data with replacement
        bootstrap_sample = resample(data.T).T

        # Estimate the treatment effect using the unbiased set
        unbiased_treatment_effect = estimate_treatment_effect(bootstrap_sample, unbiased_set, variables)

        for adj_set in adjustment_sets:
            # Estimate the treatment effect using the adjustment set
            adj_treatment_effect = estimate_treatment_effect(bootstrap_sample, adj_set, variables)

            # Calculate the bias
            bias = adj_treatment_effect - unbiased_treatment_effect
            biases[tuple(adj_set)].append(bias)

    # Calculate the mean bias for each key
    mean_biases = {key: np.mean(value) for key, value in biases.items()}
    return mean_biases

'''def estimate_bias(data, graph, adjustment_set, unbiased_set, n_bootstrap=5000):
    # Create a list of variable names in the order they appear in data
    variables = list(graph.nodes())  # Assuming the order in the graph matches the order in the data

    biases = []

    for _ in range(n_bootstrap):
        # Resample the data with replacement
        bootstrap_sample = resample(data.T).T

        # Estimate the treatment effect using the adjustment set
        adj_treatment_effect = estimate_treatment_effect(bootstrap_sample, adjustment_set, variables)

        # Estimate the treatment effect using the unbiased set
        unbiased_treatment_effect = estimate_treatment_effect(bootstrap_sample, unbiased_set, variables)

        # Calculate the bias
        bias = adj_treatment_effect - unbiased_treatment_effect
        biases.append(bias)

    # Calculate the mean bias over all bootstrap samples
    mean_bias = np.mean(biases)
    return mean_bias'''


def prune_variables(graph):
    pruned_variables = list(graph.nodes())  # TODO: implement graphical criterion
    return pruned_variables


def prune_adj_sets(potential_adj_sets, graph):  # TODO: implement graphical criterion
    return potential_adj_sets

