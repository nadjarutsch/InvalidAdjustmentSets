import wandb
import numpy as np
import sympy as sp
import hydra
from omegaconf import DictConfig, OmegaConf
import networkx as nx
import json
#from tqdm import tqdm
import os

from sklearn.linear_model import LinearRegression
from adjustment_sets import get_adjustment_set, estimate_treatment_effect

def extract_graph(graph_config: DictConfig) -> nx.DiGraph:
    """
    Extracts a directed networkx DiGraph from the provided configuration.

    Parameters:
        graph_config (DictConfig): The configuration object containing variables and edges information.

    Returns:
        nx.DiGraph: A directed graph representing the causal relationships among variables.
    """
    graph = nx.DiGraph()

    # Add nodes to the graph
    for variable in graph_config.variables:
        graph.add_node(variable.name)

    # Add edges to the graph
    for edge in graph_config.edges:
        effect = edge.effect
        equation_vars = [var.name for var in graph_config.variables if var.name in edge.equation]
        for var in equation_vars:
            graph.add_edge(var, effect)

    return graph


def extract_coefficients(equation: str, variables: list) -> list:
    """
    Extracts coefficients and their associated variables from a symbolic equation.

    Parameters:
    equation (str): The equation in string format.
    variables (list): List of variable names involved in the equation.

    Returns:
    list: A list of tuples, each containing a variable name and its coefficient in the equation.
    """
    # Convert the equation string to a sympy expression
    expr = sp.sympify(equation)

    # List to hold tuples of variable names and their coefficients
    coefficients = []

    # Iterate through each term in the expression
    for term in expr.as_ordered_terms():
        # For each variable, extract its coefficient and add to the list
        for var in variables:
            symbol = sp.Symbol(var)
            coef = term.coeff(symbol)
            if coef != 0:
                coefficients.append((var, float(coef.evalf())))

    return coefficients


def generate_data(variables: list, cfg: DictConfig, seed: int) -> np.ndarray:

    equations = cfg.graph.get('edges', [])
    sample_size = cfg.sample_size

    data = {}

    # Generate standard normal values for all variables first
    np.random.seed(seed)
    for var in variables:
        data[var] = np.random.normal(loc=0, scale=1, size=sample_size)

    for eq in equations:
        effect = eq['effect']
        expression = sp.sympify(eq['equation'], locals={var: sp.symbols(var) for var in variables})
        # Lambdify your expression
        expr_variables = [str(var) for var in expression.free_symbols]
        # Lambdify your expression, but only for the variables used in this specific expression
        func = sp.lambdify([sp.symbols(var) for var in expr_variables], expression, 'numpy')
        computed_values = func(*[data[var] for var in expr_variables])
        coefficients_list = extract_coefficients(eq['equation'], variables)

        # Only sum the squares of the coefficients, ignoring the variable names
        var_coefficients = sum(coef ** 2 for _, coef in coefficients_list)

        # Calculate the variance of the noise to be added
        noise_variance = 1 - var_coefficients

        # Ensure noise_variance is positive
        noise_variance = max(noise_variance, 0)

        # Add noise to the computed values
        noise = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=sample_size)
        data[effect] = computed_values + noise

    # Convert data dictionary to numpy array with variables in rows
    data_array = np.array([data[var] for var in variables])

    return data_array


@hydra.main(config_path='./config', config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Main function to run experiments based on a Hydra configuration.

    For each seed specified in the configuration, this function initializes a Weights & Biases (wandb) run,
    generates data, estimates the treatment effect, and logs the seed and estimated treatment effect to wandb.

    Parameters:
    cfg (DictConfig): The Hydra configuration object containing experimental settings, including the number of seeds,
                      wandb project and entity information, and other necessary configuration details.
    """

    # Extract variable names and adjustment set from the configuration
    variables = [var['name'] for var in cfg.graph['variables']]
    results = []
    adjustment_set = cfg.adjustment_set

    # Iterate over the number of seeds specified in the configuration
    for seed in range(0, cfg.n_seeds):
        # Generate data based on the current configuration and seed
        data = generate_data(variables, cfg, seed)

        # Find adjustment set from the known graph and given data
        if cfg.estimate_adjustment_set:
            graph = extract_graph(cfg.graph)
            adjustment_set, size, bias, variance, mse, rss_A = get_adjustment_set(data, graph, cfg.optimality)
        else:
            # For a fixed adjustment set, we do not need to evaluate the estimation of bias and variance
            bias, variance, size, rss_A = None, None, len(adjustment_set), None

        # Estimate the treatment effect using the generated data and specified adjustment set
        treatment_effect = estimate_treatment_effect(data, adjustment_set, variables)

        result = {
            "Seed": seed,
            "Treatment Effect": treatment_effect,
            "Adjustment Set": list(adjustment_set),
            "Estimated": cfg.estimate_adjustment_set,
            "Sample size": cfg.sample_size,
            "Bias_est": bias,
            "Variance_est": variance,
            "RSS_A": rss_A
        }

        if cfg.wandb.enabled:
            # Initialize wandb run for the current seed with the experiment configuration
            wandb.init(project=cfg.wandb.project,
                       entity=cfg.wandb.entity,
                       config=OmegaConf.to_container(cfg, resolve=True),
                       reinit=True)

            # Set the current seed in wandb configuration for reproducibility and tracking
            wandb.config.update({"Seed": seed})
            # Overwrite the adjustment set in the wandb configuration
            wandb.config.update({"adjustment_set": adjustment_set}, allow_val_change=True)

            # Log the estimated treatment effect as a summary metric for the current run
            wandb.run.summary["Estimated Treatment Effect"] = treatment_effect

            if cfg.estimate_adjustment_set:
                wandb.run.summary["Size"] = size
                wandb.run.summary["Estimated bias"] = bias
                wandb.run.summary["Estimated variance"] = variance

            # Finish the current wandb run before proceeding to the next seed
            wandb.finish()

        results.append(result)

    # Check if scratch directory is available
    scratch_dir = '/scratch-local/nrutsch'
    if os.path.exists(scratch_dir) and os.access(scratch_dir, os.W_OK):
        output_dir = scratch_dir
    else:
        output_dir = os.getcwd()
    print(f"Saving files in {output_dir}")

    # Define the filename with the path in the chosen directory
    filename = os.path.join(output_dir, f'results_{cfg.sample_size}_estimated_{cfg.estimate_adjustment_set}.json')

    # Save results to a JSON file
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"File saved successfully: {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
