import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from itertools import combinations
from collections import defaultdict


def estimate_treatment_effect(data: np.ndarray, adjustment_set: list[str], variables: list[str]) -> float:
    """
    Estimates the average treatment effect of the treatment A on the outcome Y with ordinary least-squares
    (OLS), conditional on the adjustment set.

    Parameters:
    - data: Data array of shape (len(variables), sample_size).
    - adjustment_set: List of variables that are included in the adjustment set.
    - variables: List of all variables, order corresponding to the indices in the data array.

    Returns:
    - float: estimated average treatment effect
    """
    # map variable names to indices 
    variable_indices = {name: i for i, name in enumerate(variables)}
    adjustment_indices = [variable_indices[name] for name in adjustment_set]

    Y = data[variable_indices['Y'], :].T  # outcome variable
    A = data[variable_indices['A'], :].T  # treatment variable

    # prepare covariates X
    if adjustment_indices:
        X = np.hstack([data[idx, :].reshape(-1, 1) for idx in adjustment_indices])
        X = np.column_stack((X, A))
    else:
        X = A.reshape(-1, 1)

    # estimate ATE
    model = LinearRegression().fit(X, Y)
    
    return model.coef_[-1]


def get_adjustment_set(
    graph_id: str,
    data: np.ndarray,
    graph: nx.DiGraph,
    optimality: str,
    n_bootstrap: int,
    variables: list[str]
) -> tuple[set[str], float | None, float, float, float]:
    """
    Selects the optimal adjustment set based on the specified optimality criterion (Variance or MSE).

    Parameters:
    - graph_id: Identifier for the causal graph, determines candidate adjustment sets.
    - data: Data array of shape (len(variables), sample_size).
    - graph: NetworkX graph representing the causal structure.
    - optimality: Selection criterion ("Variance" or "MSE").
    - n_bootstrap: Number of bootstrap replications for bias estimation.
    - variables: List of all variable names, consistent with the data ordering.

    Returns:
    - tuple: (best adjustment set, estimated bias, estimated variance, estimated MSE, treatment RSS)
    """  
    potential_adj_sets = []
    properties = []

    # search space excluding asymptotically optimal adjustment set O
    if graph_id == "m1":
        potential_adj_sets = [set([]), set(["O1"]), set(["O2"]), set(["W1", "O2"]), set(["W2", "O2"]), set(["W1"]), set(["W2"]), set(["W1", "W2"])]
    elif graph_id == "m2":
        potential_adj_sets = [set([]), set(["O1"]), set(["O2"]), set(["C1", "O2"]), set(["C1"])]
    else:
        raise ValueError(f"Unknown graph_id: {graph_id}")

    # estimate variance of the asymptotically optimal adjustment set
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

    # store adjustment sets for which variance is less or equal than the variance of the 
    # asymptotically optimal adjustment set (o_variance)
    filtered_adj_sets = [set(["O1", "O2"])]

    for adjustment_set in potential_adj_sets:
        # estimate variance of potential adjustment set
        est_error_var_outcome, rss_A = estimate_variance(data, graph, adjustment_set)
        variance = est_error_var_outcome / rss_A

        # keep only adjustment sets with smaller variance
        if variance < o_variance:
            properties.append({
                'Adjustment set': adjustment_set,
                'Bias': None,
                'MSE': None,
                'Variance': variance,
                'RSS_A': rss_A,
                'est_error_var_outcome': est_error_var_outcome
            })

            filtered_adj_sets.append(adjustment_set)

    # bias estimation not needed for variance-optimal adjustment set
    if optimality != "Variance":
        # estimate bias for each remaining adjustment set
        biases = estimate_bias(data, graph, filtered_adj_sets, ["O1", "O2"], n_bootstrap)

        for prop in properties:
            adjustment_set = prop['Adjustment set']
            prop['Bias'] = biases[tuple(adjustment_set)]
            prop['MSE'] = prop['Bias'] ** 2 + prop['Variance']

    # choose adjustment set with the minimum MSE or Variance
    best_set = min(properties, key=lambda x: x[optimality])

    return best_set['Adjustment set'], best_set['Bias'], best_set['Variance'], best_set['MSE'], best_set['RSS_A']


def estimate_variance(
    data: np.ndarray,
    graph: nx.DiGraph,
    adjustment_set: set[str]
) -> tuple[float, float]:
    """
    Estimates the error variance of the outcome and the residual sum of squares (RSS) of the treatment,
    given a specific adjustment set. From these values, the variance of the treatment effect estimator 
    can be estimated.

    Parameters:
    - data: Data array of shape (len(variables), sample_size).
    - graph: NetworkX graph representing the causal structure.
    - adjustment_set: Set of variables to condition on.

    Returns:
    - tuple: (estimated error variance of outcome, residual sum of squares of treatment)
    """
    # extract data indices from graph nodes
    nodes = list(graph.nodes())
    treatment_index = nodes.index("A")
    outcome_index = nodes.index("Y")
    adjustment_indices = [nodes.index(node) for node in adjustment_set]

    # prepare data
    A = data[treatment_index, :].reshape(-1, 1)  # treatment variable (transposed for samples as rows)
    Y = data[outcome_index, :].reshape(-1, 1)  # outcome variable (transposed for samples as rows)

    if adjustment_indices:
        # regress the treatment on the adjustment set
        X_adjust = data[adjustment_indices, :].T
        reg_treatment = LinearRegression().fit(X_adjust, A)

        # regress the outcome on the treatment and the adjustment set
        X_adjust_with_treatment = np.column_stack((X_adjust, A))
        reg_outcome = LinearRegression().fit(X_adjust_with_treatment, Y)

        # calculate residuals
        residuals_treatment = A - reg_treatment.predict(X_adjust)
        residuals_outcome = Y - reg_outcome.predict(X_adjust_with_treatment)
    else:
        # if the adjustment set is the empty set, we use the mean
        residuals_treatment = A - np.mean(A)
        residuals_outcome = Y - np.mean(Y)

    # calculate the treatment RSS
    rss_treatment = np.sum(residuals_treatment ** 2)
    
    # estimate the error variance for the outcome from the RSS
    rss_outcome = np.sum(residuals_outcome ** 2)
    df = data.shape[1] - len(adjustment_set) - 1
    est_error_var_outcome = rss_outcome / df

    return est_error_var_outcome, rss_treatment


def estimate_bias(
    data: np.ndarray,
    graph: nx.DiGraph,
    adjustment_sets: list[set[str]],
    unbiased_set: list[str],
    n_bootstrap: int = 1000
) -> dict[tuple[str, ...], float]:
    """
    Estimates the bias of each candidate adjustment set by comparing treatment effect estimates
    to the effect estimated using the unbiased set, based on bootstrapping.

    Parameters:
    - data: Data array of shape (len(variables), sample_size).
    - graph: NetworkX graph representing the causal structure.
    - adjustment_sets: List of candidate adjustment sets to evaluate.
    - unbiased_set: Adjustment set assumed to give unbiased estimate of the treatment effect.
    - n_bootstrap: Number of bootstrap replications.

    Returns:
    - dict: Mapping from adjustment set (as tuple) to estimated bias.
    """
    # create a list of variable names in the order they appear in the data
    variables = list(graph.nodes())  

    # keep track of estimated biases
    biases = defaultdict(list)

    # bootstrap
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(data.T).T

        # estimate the ATE using the unbiased set
        unbiased_treatment_effect = estimate_treatment_effect(bootstrap_sample, unbiased_set, variables)

        for adj_set in adjustment_sets:
            # estimate the ATE using the (potentially biased) adjustment set
            adj_treatment_effect = estimate_treatment_effect(bootstrap_sample, adj_set, variables)

            bias = adj_treatment_effect - unbiased_treatment_effect
            biases[tuple(adj_set)].append(bias)

    # calculate the mean bias for each adjustment set
    mean_biases = {key: np.mean(value) for key, value in biases.items()}
    return mean_biases
