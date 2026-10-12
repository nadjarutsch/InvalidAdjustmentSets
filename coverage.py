import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from scipy.stats import t as student_t
from multiprocessing import Pool
import json
import os
import uuid

from adjustment_sets import get_adjustment_set, estimate_treatment_effect, estimate_variance
from main import generate_data


def regression_interval(
    data: np.ndarray,
    adjustment_set: list[str],
    variables: list[str],
    confidence_level: float
) -> tuple[float, float]:
    """
    Computes the usual regression confidence interval for the treatment effect, treating the adjustment
    set as fixed.

    Parameters:
    - data: Data array of shape (len(variables), sample_size).
    - adjustment_set: Variables that are included in the adjustment set.
    - variables: List of all variables, order corresponding to the indices in the data array.
    - confidence_level: Intended coverage of the interval.

    Returns:
    - tuple: (lower boundary, upper boundary)
    """
    treatment_effect = estimate_treatment_effect(data, adjustment_set, variables)
    est_error_var_outcome, rss_A = estimate_variance(data, adjustment_set, variables)

    # quantile of the t-distribution with n - |K| - 2 degrees of freedom
    df = data.shape[1] - len(adjustment_set) - 2
    quantile = student_t.ppf(1 - (1 - confidence_level) / 2, df)
    half_length = quantile * np.sqrt(est_error_var_outcome / rss_A)

    return treatment_effect - half_length, treatment_effect + half_length


def percentile_interval(estimates: list[float], confidence_level: float) -> tuple[float, float]:
    """
    Computes the bootstrap percentile interval from treatment effect estimates on bootstrap samples.

    Parameters:
    - estimates: Treatment effect estimates, one per bootstrap sample.
    - confidence_level: Intended coverage of the interval.

    Returns:
    - tuple: (lower boundary, upper boundary)
    """
    alpha = 1 - confidence_level
    lower, upper = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return lower, upper


def run_seed(args: tuple[dict, int]) -> dict:
    """
    Generates one dataset and computes regression and bootstrap confidence intervals, based on the
    selected adjustment set and on the reference set O.

    For the bootstrap interval of the selected set, the whole selection procedure (including bias
    estimation) is repeated on every bootstrap sample.

    Parameters:
    - args: Tuple of the configuration (as dictionary) and the random seed.

    Returns:
    - dict: Selected adjustment set and the boundaries of the four intervals.
    """
    cfg, seed = args
    cfg = OmegaConf.create(cfg)

    variables = [var['name'] for var in cfg.graph['variables']]
    candidates = [list(adj_set) for adj_set in cfg.graph.candidates]
    valid_sets = [list(adj_set) for adj_set in cfg.graph.valid_sets]
    reference_set = list(cfg.graph.reference_set)

    # select the adjustment set on the full dataset, with the same random stream as in main.py
    data = generate_data(variables, cfg, seed)
    adjustment_set = get_adjustment_set(
        data, variables, candidates, reference_set, valid_sets, cfg.optimality, cfg.n_bootstrap)[0]

    # separate random streams for the bootstrap samples and for each selection within the bootstrap
    outer_rng = np.random.RandomState(np.random.SeedSequence([seed, 1]).generate_state(1)[0])
    estimates_reference = []
    estimates_selected = []

    for b in range(cfg.n_bootstrap_outer):
        indices = outer_rng.randint(0, cfg.sample_size, size=cfg.sample_size)
        bootstrap_sample = data[:, indices]
        estimates_reference.append(estimate_treatment_effect(bootstrap_sample, reference_set, variables))

        # repeat selection and estimation on the bootstrap sample
        np.random.seed(np.random.SeedSequence([seed, 2, b]).generate_state(1)[0])
        bootstrap_set = get_adjustment_set(
            bootstrap_sample, variables, candidates, reference_set, valid_sets, cfg.optimality, cfg.n_bootstrap)[0]
        estimates_selected.append(estimate_treatment_effect(bootstrap_sample, bootstrap_set, variables))

    return {
        "Seed": seed,
        "Adjustment Set": adjustment_set,
        "Selected Regression": regression_interval(data, adjustment_set, variables, cfg.confidence_level),
        "Selected Bootstrap": percentile_interval(estimates_selected, cfg.confidence_level),
        "Reference Regression": regression_interval(data, reference_set, variables, cfg.confidence_level),
        "Reference Bootstrap": percentile_interval(estimates_reference, cfg.confidence_level),
    }


def summarize(results: list[dict], treatment_effect: float) -> dict:
    """
    Computes the empirical coverage and the mean length of each interval, with standard errors.

    Parameters:
    - results: Results of all seeds, as returned by run_seed.
    - treatment_effect: True average treatment effect.

    Returns:
    - dict: Mapping from interval name to coverage, mean length and their standard errors.
    """
    summary = {}

    for name in ["Selected Regression", "Selected Bootstrap", "Reference Regression", "Reference Bootstrap"]:
        intervals = np.array([result[name] for result in results])
        covered = (intervals[:, 0] <= treatment_effect) & (treatment_effect <= intervals[:, 1])
        lengths = intervals[:, 1] - intervals[:, 0]

        summary[name] = {
            "Coverage": covered.mean(),
            "Coverage SE": np.sqrt(covered.mean() * (1 - covered.mean()) / len(results)),
            "Mean length": lengths.mean(),
            "Mean length SE": lengths.std(ddof=1) / np.sqrt(len(results)),
        }

    return summary


@hydra.main(config_path='./config', config_name='coverage', version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Runs the confidence interval experiments based on a Hydra configuration.

    For each seed, this function generates data, selects an adjustment set and computes regression and
    bootstrap confidence intervals for the selected set and the reference set O. The intervals and a summary of
    their coverage and length are saved to a JSON file.

    Parameters:
        cfg (DictConfig): The Hydra configuration object containing experimental settings, including the number of seeds,
                        the sample size, the graph, the numbers of bootstrap samples, the confidence level, the
                        optimality criterion, and the number of parallel processes.
    """

    # generate unique filename
    output_dir = os.getcwd()
    unique_id = str(uuid.uuid4())
    filename = os.path.join(output_dir, f'coverage_{cfg.graph.id}_{cfg.sample_size}_{cfg.optimality}_optimality_{unique_id}.json')

    # one numerical thread per process, the seeds are run in parallel
    for var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[var] = "1"

    tasks = [(OmegaConf.to_container(cfg), seed) for seed in range(0, cfg.n_seeds)]
    results = []

    with Pool(cfg.n_jobs) as pool:
        for result in pool.imap(run_seed, tasks):
            results.append(result)

            # save results to JSON file
            with open(filename, 'w') as f:
                json.dump({"summary": summarize(results, cfg.graph.treatment_effect), "per_seed": results}, f, indent=4)

    print(json.dumps(summarize(results, cfg.graph.treatment_effect), indent=4))
    print(f"File saved successfully: {filename}")


if __name__ == "__main__":
    main()
