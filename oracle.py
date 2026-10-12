import numpy as np
import sympy as sp
import hydra
from omegaconf import DictConfig
from itertools import combinations


def population_covariance(graph_config: DictConfig) -> np.ndarray:
    """
    Computes the covariance matrix of the variables in a linear structural causal model with independent
    noise terms of variance 1.

    Parameters:
    - graph_config (DictConfig): The configuration object containing variables and structural equations.

    Returns:
    - np.ndarray: Covariance matrix of shape (len(variables), len(variables)), in the order of the config.
    """
    variables = [var.name for var in graph_config.variables]
    symbols = {var: sp.symbols(var) for var in variables}

    # matrix of structural coefficients, with B[i, j] the coefficient of variable j in the equation of i
    B = np.zeros((len(variables), len(variables)))
    for eq in graph_config.edges:
        expression = sp.sympify(eq.equation, locals=symbols)
        for j, var in enumerate(variables):
            B[variables.index(eq.effect), j] = float(expression.coeff(symbols[var]))

    # V = B V + noise, such that V = (I - B)^{-1} noise
    transform = np.linalg.inv(np.eye(len(variables)) - B)

    return transform @ transform.T


def true_mse(
    covariance: np.ndarray,
    adjustment_set: list[str],
    variables: list[str],
    treatment_effect: float,
    sample_size: int
) -> tuple[float, float, float]:
    """
    Computes the bias, the asymptotic variance and the MSE of the OLS treatment effect estimator for
    an adjustment set K in a linear Gaussian model: MSE = bias^2 + aVar / (n - |K| - 3).

    Parameters:
    - covariance: Covariance matrix of the variables.
    - adjustment_set: Variables that are included in the adjustment set.
    - variables: List of all variables, order corresponding to the indices of the covariance matrix.
    - treatment_effect: True average treatment effect.
    - sample_size: Sample size n.

    Returns:
    - tuple: (bias, asymptotic variance, MSE), where the MSE is infinite if n <= |K| + 3
    """
    # regressors: treatment first, then the adjustment set
    x = [variables.index('A')] + [variables.index(name) for name in adjustment_set]
    y = variables.index('Y')

    precision = np.linalg.inv(covariance[np.ix_(x, x)])
    coefficients = precision @ covariance[x, y]

    # bias of the coefficient of the treatment
    bias = coefficients[0] - treatment_effect

    # asymptotic variance sigma_yy.ak / sigma_aa.k
    residual_var_outcome = covariance[y, y] - covariance[x, y] @ coefficients
    asymptotic_variance = residual_var_outcome * precision[0, 0]

    df = sample_size - len(adjustment_set) - 3
    mse = bias ** 2 + asymptotic_variance / df if df > 0 else np.inf

    return bias, asymptotic_variance, mse


def get_oracle_set(graph_config: DictConfig, sample_size: int) -> tuple[list[str], float]:
    """
    Finds the MSE-optimal adjustment set O_n among all subsets of the covariates, using the true model
    parameters.

    Parameters:
    - graph_config (DictConfig): The configuration object containing variables and structural equations.
    - sample_size (int): Sample size n.

    Returns:
    - tuple: (MSE-optimal adjustment set, its MSE)
    """
    variables = [var.name for var in graph_config.variables]
    covariates = [var for var in variables if var not in ('A', 'Y')]
    covariance = population_covariance(graph_config)

    # search over all subsets of the covariates
    subsets = [list(subset) for size in range(len(covariates) + 1) for subset in combinations(covariates, size)]
    mses = [true_mse(covariance, subset, variables, graph_config.treatment_effect, sample_size)[2]
            for subset in subsets]
    best = int(np.argmin(mses))

    return subsets[best], mses[best]


@hydra.main(config_path='./config', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Prints the MSE-optimal adjustment set O_n and its MSE for each sample size of the graph config.

    Parameters:
        cfg (DictConfig): The Hydra configuration object containing the graph.
    """
    for sample_size in cfg.graph.sample_sizes:
        oracle_set, mse = get_oracle_set(cfg.graph, sample_size)
        print(f"n = {sample_size}: O_n = {{{', '.join(oracle_set)}}}, MSE = {mse:.6f}")


if __name__ == "__main__":
    main()
