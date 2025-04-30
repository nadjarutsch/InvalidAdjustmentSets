import numpy as np
import sympy as sp
import hydra
from omegaconf import DictConfig
import networkx as nx
import json
import os
import uuid

from adjustment_sets import get_adjustment_set, estimate_treatment_effect


def extract_graph(graph_config: DictConfig) -> nx.DiGraph:
    """
    Extracts a directed networkx DiGraph from the provided configuration.

    Parameters:
        graph_config (DictConfig): The configuration object containing variables and structural equations.

    Returns:
        nx.DiGraph: A directed graph representing the causal relationships among variables.
    """
    graph = nx.DiGraph()

    # add nodes
    for variable in graph_config.variables:
        graph.add_node(variable.name)

    # add edges
    for edge in graph_config.edges:
        effect = edge.effect
        equation_vars = [var.name for var in graph_config.variables if var.name in edge.equation]
        for var in equation_vars:
            graph.add_edge(var, effect)

    return graph


def generate_data(variables: list, cfg: DictConfig, seed: int) -> np.ndarray:

    equations = cfg.graph.get('edges', [])
    sample_size = cfg.sample_size

    data = {}

    # generate Gaussian noise epsilon_i for all variables V_i in V
    np.random.seed(seed)
    for var in variables:
        data[var] = np.random.normal(loc=0, scale=1, size=sample_size)

    for eq in equations:
        # generate function from each structural equation, important: order of equations in config must be causal order
        effect = eq['effect']
        expression = sp.sympify(eq['equation'], locals={var: sp.symbols(var) for var in variables})
        expr_variables = [str(var) for var in expression.free_symbols]
        func = sp.lambdify([sp.symbols(var) for var in expr_variables], expression, 'numpy')
        
        # compute the values of the structural equation and add to the Gaussian noise
        computed_values = func(*[data[var] for var in expr_variables])
        data[effect] += computed_values

    data_array = np.array([data[var] for var in variables])

    return data_array


@hydra.main(config_path='./config', config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Run experiments based on a Hydra configuration.

    For each seed, this function generates data and estimates the treatment effect, either with a fixed adjustment set
    or by estimating the MSE-optimal or variance-optimal adjustment set. The results are saved to a JSON file.

    Parameters:
        cfg (DictConfig): The Hydra configuration object containing experimental settings, including the number of seeds,
                        the sample size, the graph, whether to estimate the adjustment set, a fixed adjustment set (only
                        used if not estimated), and the optimality criterion.
    """

    # extract variable names and adjustment set from Hydra config
    variables = [var['name'] for var in cfg.graph['variables']]
    results = []
    adjustment_set = cfg.adjustment_set

    # generate unique filename
    output_dir = os.getcwd()
    unique_id = str(uuid.uuid4())
    filename = os.path.join(output_dir, f'results_{cfg.graph.id}_{cfg.sample_size}_estimated_{cfg.estimate_adjustment_set}_{cfg.optimality}_optimality_{unique_id}.json')

    for seed in range(0, cfg.n_seeds):
        data = generate_data(variables, cfg, seed)

        # estimate MSE-optimal or variance-optimal adjustment set
        if cfg.estimate_adjustment_set:
            graph = extract_graph(cfg.graph)
            adjustment_set, bias_est, variance_est, mse_est, rss_A = get_adjustment_set(cfg.graph.id, data, graph, cfg.optimality, cfg.n_bootstrap, variables)
        else:
            # for a fixed adjustment set, we do not need to estimate bias and variance
            bias_est, variance_est, rss_A, mse_est = None, None, None, None

        treatment_effect_est = estimate_treatment_effect(data, adjustment_set, variables)

        result = {
            "Seed": seed,
            "Treatment Effect": treatment_effect_est,
            "Adjustment Set": list(adjustment_set),
            "Estimated": cfg.estimate_adjustment_set,   # whether the adjustment set was estimated or not
            "Sample size": cfg.sample_size,
            "Bias_est": bias_est,
            "Variance_est": variance_est,
            "MSE_est": mse_est,
            "RSS_A": rss_A,
            "Optimality": cfg.optimality,   # optimality criterion used
        }

        results.append(result)
       
        # save results to JSON file
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"File saved successfully: {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
