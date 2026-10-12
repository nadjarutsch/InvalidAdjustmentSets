import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import brentq
from scipy.special import ndtr, log_ndtr, roots_hermitenorm
from multiprocessing import Pool
import json
import math
import os

from oracle import true_mse

# M3: O1 and O2 are noise terms, A = O1 + O2 + noise, Y = A + O1 + 0.15 O2 + noise
M3_VARIABLES = ["A", "Y", "O1", "O2"]
M3_COEFFICIENTS = np.array([
    [0, 0, 1, 1],       # A
    [1, 0, 1, 0.15],    # Y
    [0, 0, 0, 0],       # O1
    [0, 0, 0, 0],       # O2
])
M3_TREATMENT_EFFECT = 1.0
M3_SMALL_SET = ["O1"]       # invalid, small bias and lower asymptotic variance
M3_FULL_SET = ["O1", "O2"]  # valid adjustment set O


def transform_noise(z: np.ndarray, noise: dict) -> np.ndarray:
    """
    Transforms standard normal draws into noise terms with mean zero and variance one from another
    distribution.

    Parameters:
    - z: Array of independent standard normal draws.
    - noise: Noise distribution, with the key 'family' ("gaussian", "uniform", "exponential" or
        "sinh_arcsinh") and for the sinh-arcsinh family the parameters 'epsilon' and 'delta'.

    Returns:
    - np.ndarray: Noise terms of the same shape as z.
    """
    if noise['family'] == "gaussian":
        return z
    if noise['family'] == "uniform":
        # U(-sqrt(3), sqrt(3))
        return math.sqrt(12) * (ndtr(z) - 0.5)
    if noise['family'] == "exponential":
        # E - 1 with E ~ Exp(1)
        return -log_ndtr(-z) - 1
    if noise['family'] == "sinh_arcsinh":
        epsilon, delta = noise['epsilon'], noise['delta']
        if epsilon == 0 and delta == 1:
            return z

        # population mean and standard deviation by Gauss-Hermite quadrature
        nodes, weights = roots_hermitenorm(160)
        weights = weights / math.sqrt(2 * math.pi)
        values = np.sinh((np.arcsinh(nodes) + epsilon) / delta)
        mean = np.dot(weights, values)
        sd = math.sqrt(np.dot(weights, values ** 2) - mean ** 2)

        return (np.sinh((np.arcsinh(z) + epsilon) / delta) - mean) / sd
    raise ValueError(f"Unknown noise family: {noise['family']}")


def estimate_treatment_effects(data: np.ndarray, adjustment_set: list[str], variables: list[str]) -> np.ndarray:
    """
    Estimates the average treatment effect with OLS (including an intercept) on many datasets at once.

    Parameters:
    - data: Data array of shape (number of datasets, sample_size, len(variables)).
    - adjustment_set: Variables that are included in the adjustment set.
    - variables: List of all variables, order corresponding to the last axis of the data array.

    Returns:
    - np.ndarray: Estimated average treatment effect for each dataset.
    """
    Y = data[:, :, variables.index('Y')]
    X = data[:, :, [variables.index(name) for name in adjustment_set] + [variables.index('A')]]

    # center within each dataset to account for the intercept
    Y = Y - Y.mean(axis=1, keepdims=True)
    X = X - X.mean(axis=1, keepdims=True)

    # solve the normal equations, the treatment is the last covariate
    gram = np.einsum('rnp,rnq->rpq', X, X)
    coefficients = np.linalg.solve(gram, np.einsum('rnp,rn->rp', X, Y)[..., None])[..., 0]

    return coefficients[:, -1]


def gaussian_mse(adjustment_set: list[str], sample_size: float) -> float:
    """
    Computes the exact MSE of the OLS treatment effect estimator in M3 under Gaussian noise.

    Parameters:
    - adjustment_set: Variables that are included in the adjustment set.
    - sample_size: Sample size n.

    Returns:
    - float: MSE under Gaussian noise
    """
    transform = np.linalg.inv(np.eye(len(M3_VARIABLES)) - M3_COEFFICIENTS)
    covariance = transform @ transform.T

    return true_mse(covariance, adjustment_set, M3_VARIABLES, M3_TREATMENT_EFFECT, sample_size)[2]


def gaussian_threshold() -> float:
    """
    Computes the sample size at which the MSEs of {O1} and {O1, O2} cross in M3 under Gaussian noise.

    Returns:
    - float: Gaussian sample size threshold
    """
    return brentq(lambda n: gaussian_mse(M3_SMALL_SET, n) - gaussian_mse(M3_FULL_SET, n), 6, 1000, xtol=1e-12)


def estimate_mse_difference(noise: dict, sample_size: int, n_replications: int, seed: int) -> dict:
    """
    Estimates the MSEs of {O1} and {O1, O2} in M3 and their difference for a given noise distribution.

    Each non-Gaussian dataset is coupled with a Gaussian dataset generated from the same standard normal
    draws. The simulated Gaussian MSE is used as a control variate: it is subtracted from the simulated
    non-Gaussian MSE and the exact Gaussian MSE is added.

    Parameters:
    - noise: Noise distribution, see transform_noise.
    - sample_size: Sample size n.
    - n_replications: Number of simulated datasets.
    - seed: Random seed.

    Returns:
    - dict: Sample size, number of replications, MSE of both sets, MSE difference ({O1} minus {O1, O2})
        and the standard error of the difference.
    """
    transform = np.linalg.inv(np.eye(len(M3_VARIABLES)) - M3_COEFFICIENTS)
    corrections = {"small": [], "full": []}

    # simulate in chunks of 500 datasets
    for start in range(0, n_replications, 500):
        count = min(500, n_replications - start)
        rng = np.random.default_rng(np.random.SeedSequence([seed, sample_size, start]))
        z = rng.standard_normal((count, sample_size, len(M3_VARIABLES)))

        # coupled Gaussian and non-Gaussian datasets
        gaussian_data = z @ transform.T
        data = transform_noise(z, noise) @ transform.T

        for name, adjustment_set in [("small", M3_SMALL_SET), ("full", M3_FULL_SET)]:
            gaussian_error = (estimate_treatment_effects(gaussian_data, adjustment_set, M3_VARIABLES) - M3_TREATMENT_EFFECT) ** 2
            error = (estimate_treatment_effects(data, adjustment_set, M3_VARIABLES) - M3_TREATMENT_EFFECT) ** 2
            corrections[name].append(error - gaussian_error)

    correction_small = np.concatenate(corrections["small"])
    correction_full = np.concatenate(corrections["full"])
    correction_difference = correction_small - correction_full

    mse_small = gaussian_mse(M3_SMALL_SET, sample_size) + correction_small.mean()
    mse_full = gaussian_mse(M3_FULL_SET, sample_size) + correction_full.mean()

    return {
        "n": sample_size,
        "replications": n_replications,
        "mse_small": mse_small,
        "mse_full": mse_full,
        "difference": gaussian_mse(M3_SMALL_SET, sample_size) - gaussian_mse(M3_FULL_SET, sample_size) + correction_difference.mean(),
        "difference_se": correction_difference.std(ddof=1) / math.sqrt(n_replications),
    }


def get_sign_changes(rows: list[dict]) -> list[tuple[dict, dict]]:
    """
    Finds the pairs of consecutive evaluated sample sizes between which the estimated MSE difference
    changes sign.

    Parameters:
    - rows: Results of estimate_mse_difference for different sample sizes.

    Returns:
    - list: Pairs of results (smaller sample size, larger sample size) with a sign change.
    """
    rows = sorted(rows, key=lambda row: row['n'])

    return [(left, right) for left, right in zip(rows, rows[1:])
            if left['difference'] == 0 or left['difference'] * right['difference'] < 0]


def interpolate_threshold(left: dict, right: dict) -> tuple[float, float]:
    """
    Estimates the sample size threshold by linear interpolation between two sample sizes at which the
    estimated MSE difference changes sign, with a standard error from the delta method.

    Parameters:
    - left: Result of estimate_mse_difference at the smaller sample size.
    - right: Result of estimate_mse_difference at the larger sample size.

    Returns:
    - tuple: (estimated threshold, standard error)
    """
    d0, d1 = left['difference'], right['difference']
    step = right['n'] - left['n']
    threshold = left['n'] - d0 * step / (d1 - d0)

    # the two differences are simulated independently
    derivative_d0 = -step * d1 / (d1 - d0) ** 2
    derivative_d1 = step * d0 / (d1 - d0) ** 2
    se = math.sqrt((derivative_d0 * left['difference_se']) ** 2 + (derivative_d1 * right['difference_se']) ** 2)

    return threshold, se


def estimate_threshold(noise: dict, seed: int, n_replications: int, max_replications: int) -> dict:
    """
    Estimates the sample size threshold between {O1} and {O1, O2} in M3 for a given noise distribution
    and the largest relative MSE disadvantage from using the Gaussian threshold instead.

    The MSE difference is first estimated on a coarse grid of sample sizes and then at every integer
    between sample sizes with a sign change. The number of replications near the threshold is doubled
    (up to max_replications) until there is a single sign change, both adjacent differences are
    significantly different from zero and 1.96 times the standard error of the threshold is at most three.

    Parameters:
    - noise: Noise distribution, see transform_noise.
    - seed: Random seed.
    - n_replications: Initial number of replications per sample size.
    - max_replications: Maximum number of replications per sample size.

    Returns:
    - dict: Estimated threshold with standard error, relative shift from the Gaussian threshold, largest
        relative MSE disadvantage, number of replications near the threshold and whether the threshold is
        resolved (single sign change and 1.96 times the standard error at most three).
    """
    n_gaussian = gaussian_threshold()
    results = {}

    def evaluate(sample_size, replications):
        if sample_size not in results or results[sample_size]['replications'] < replications:
            results[sample_size] = estimate_mse_difference(noise, sample_size, replications, seed)

    # coarse grid of sample sizes, expanded if it contains no sign change
    for n in sorted(set(range(50, 151, 10)) | {math.floor(n_gaussian), math.ceil(n_gaussian)}):
        evaluate(n, n_replications)
    for _ in range(2):
        if get_sign_changes(list(results.values())):
            break
        lowest, highest = min(results), max(results)
        for n in list(range(max(6, lowest - 30), lowest, 10)) + list(range(highest + 10, highest + 31, 10)):
            evaluate(n, n_replications)

    # every integer sample size between sample sizes with a sign change
    for left, right in get_sign_changes(list(results.values())):
        for n in range(left['n'], right['n'] + 1):
            evaluate(n, n_replications)

    # increase the number of replications near the threshold
    replications = n_replications
    sign_changes = get_sign_changes(list(results.values()))
    while replications < max_replications:
        if len(sign_changes) == 1:
            left, right = sign_changes[0]
            threshold, se = interpolate_threshold(left, right)
            if (abs(left['difference']) > 1.96 * left['difference_se']
                    and abs(right['difference']) > 1.96 * right['difference_se']
                    and 1.96 * se <= 3):
                break

        replications = min(max_replications, 2 * replications)
        local = {math.floor(n_gaussian), math.ceil(n_gaussian)}
        for left, right in sign_changes:
            local.update(range(max(6, left['n'] - 2), right['n'] + 3))
        if not sign_changes:
            nearest = min(results.values(), key=lambda row: abs(row['difference']))
            local.update(range(max(6, nearest['n'] - 5), nearest['n'] + 6))
        for n in sorted(local):
            evaluate(n, replications)
        sign_changes = get_sign_changes(list(results.values()))

    summary = {**noise, "replications": replications, "resolved": False}
    if len(sign_changes) != 1:
        return summary

    threshold, se = interpolate_threshold(*sign_changes[0])

    # relative MSE disadvantage of the set chosen by the Gaussian threshold, at the evaluated integer
    # sample sizes between the Gaussian and the estimated threshold
    lower, upper = sorted((n_gaussian, threshold))
    disadvantages = [0.0]
    for n in sorted(results):
        if not math.ceil(lower) <= n <= math.floor(upper):
            continue
        row = results[n]
        chosen = row['mse_small'] if n < n_gaussian else row['mse_full']
        best = min(row['mse_small'], row['mse_full'])
        disadvantages.append((chosen - best) / best)

    summary.update({
        "threshold": threshold,
        "threshold_se": se,
        "relative_shift": (threshold - n_gaussian) / n_gaussian,
        "max_relative_mse_disadvantage": max(disadvantages),
        "resolved": bool(1.96 * se <= 3),
    })

    return summary


def run_m3_condition(args: tuple[dict, int, list[list[int]]]) -> dict:
    """
    Estimates the sample size threshold in M3 for one noise distribution, rerunning the estimation with
    more replications as long as the threshold is not resolved.

    Parameters:
    - args: Tuple of the noise distribution, the random seed and the list of (initial, maximum) numbers
        of replications.

    Returns:
    - dict: Result of estimate_threshold for the last run.
    """
    noise, seed, replication_stages = args

    for n_replications, max_replications in replication_stages:
        summary = estimate_threshold(noise, seed, n_replications, max_replications)
        if summary['resolved']:
            break

    return summary


def run_m3(cfg: DictConfig) -> dict:
    """
    Runs the sample size threshold experiment in M3 on a grid of sinh-arcsinh noise distributions and for
    uniform and centered exponential noise.

    Parameters:
    - cfg (DictConfig): Hydra configuration with the grid, the numbers of replications and the seed.

    Returns:
    - dict: Gaussian threshold and the results for all noise distributions.
    """
    epsilons = np.linspace(cfg.m3.epsilon_min, cfg.m3.epsilon_max, cfg.m3.n_epsilon)
    deltas = np.linspace(cfg.m3.delta_min, cfg.m3.delta_max, cfg.m3.n_delta)

    conditions = [{"family": "sinh_arcsinh", "epsilon": float(epsilon), "delta": float(delta)}
                  for epsilon in epsilons for delta in deltas]
    conditions += [{"family": "uniform"}, {"family": "exponential"}]

    # separate random seed for each noise distribution
    stages = OmegaConf.to_container(cfg.m3.replication_stages)
    tasks = [(noise, cfg.m3.seed + 100000 * index, stages) for index, noise in enumerate(conditions)]

    # one numerical thread per process, the noise distributions are run in parallel
    for var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[var] = "1"

    with Pool(cfg.n_jobs) as pool:
        results = pool.map(run_m3_condition, tasks, chunksize=1)

    return {"gaussian_threshold": gaussian_threshold(), "results": results}


def run_m4(cfg: DictConfig) -> dict:
    """
    Estimates the MSEs of the adjustment sets {W1} and {O1} in M4 with uniform noise for W1 and Y and
    centered Gamma noise for O1 and A, using conditional Monte Carlo.

    The model is W1 = noise, O1 = b W1 + noise, A = noise, Y = A + b O1 + noise. Writing the treatment
    noise as (G - shape) / sqrt(shape) with Gamma draws G, the sum S of the Gamma draws is independent
    of the proportions P = G / S. The expectation over S, the outcome noise and (for {W1}) the noise of
    O1 is calculated analytically, such that only P and the covariates are simulated.

    Parameters:
    - cfg (DictConfig): Hydra configuration with the sample size, the Gamma shape, the coefficient b,
        the number of replications and the seed.

    Returns:
    - dict: Estimated MSEs, their difference ({W1} minus {O1}) and the standard error of the difference.
    """
    n, shape, b = cfg.m4.sample_size, cfg.m4.gamma_shape, cfg.m4.coefficient
    rng = np.random.default_rng(cfg.m4.seed)

    # shape * E(1 / S^2), with S ~ Gamma(n * shape, 1)
    factor = shape / ((n * shape - 1) * (n * shape - 2))
    values = []

    # simulate in chunks of 20000 datasets
    for start in range(0, cfg.m4.n_replications, 20000):
        count = min(20000, cfg.m4.n_replications - start)

        # proportions of the Gamma draws of the treatment (Dirichlet distributed)
        gamma_A = rng.gamma(shape, 1, (count, n))
        P = gamma_A / gamma_A.sum(axis=1, keepdims=True)
        P = P - 1 / n

        # covariates
        W1 = rng.uniform(-np.sqrt(3), np.sqrt(3), (count, n))
        O1 = b * W1 + (rng.gamma(shape, 1, (count, n)) - shape) / np.sqrt(shape)
        W1 = W1 - W1.mean(axis=1, keepdims=True)
        O1 = O1 - O1.mean(axis=1, keepdims=True)

        # residual sum of squares of P after regressing on each covariate and an intercept
        PP = np.einsum('ij,ij->i', P, P)
        rss_W1 = PP - np.einsum('ij,ij->i', P, W1) ** 2 / np.einsum('ij,ij->i', W1, W1)
        rss_O1 = PP - np.einsum('ij,ij->i', P, O1) ** 2 / np.einsum('ij,ij->i', O1, O1)

        # adjusting for {W1} leaves the noise of O1 in the outcome, with variance b^2
        values.append(factor * np.column_stack(((1 + b ** 2) / rss_W1, 1 / rss_O1)))

    values = np.concatenate(values)
    differences = values[:, 0] - values[:, 1]

    return {
        "mse_W1": values[:, 0].mean(),
        "mse_O1": values[:, 1].mean(),
        "difference": differences.mean(),
        "difference_se": differences.std(ddof=1) / np.sqrt(len(differences)),
    }


@hydra.main(config_path='./config', config_name='non_gaussian', version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Runs the experiments with non-Gaussian noise based on a Hydra configuration: either the sensitivity
    of the sample size criterion in M3 or the counterexample to the graphical criteria in M4. The results
    are saved to a JSON file.

    Parameters:
        cfg (DictConfig): The Hydra configuration object containing the experiment ("m3" or "m4") and its settings.
    """
    if cfg.experiment == "m3":
        results = run_m3(cfg)
    elif cfg.experiment == "m4":
        results = run_m4(cfg)
    else:
        raise ValueError(f"Unknown experiment: {cfg.experiment}")

    filename = os.path.join(os.getcwd(), f'non_gaussian_{cfg.experiment}.json')
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"File saved successfully: {filename}")


if __name__ == "__main__":
    main()
