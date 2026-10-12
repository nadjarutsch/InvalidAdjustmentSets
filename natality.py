import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from itertools import combinations
import json
import os

from adjustment_sets import get_adjustment_set, estimate_treatment_effect, bootstrap_treatment_effects

# body mass index takes the role of the treatment A and birthweight the role of the outcome Y
VARIABLES = ['A', 'Y', 'age', 'height', 'education_2', 'education_3', 'education_4', 'previous_live_births']

# covariates, the three education indicators are always included or excluded together
COVARIATE_GROUPS = [['age'], ['height'], ['education_2', 'education_3', 'education_4'], ['previous_live_births']]

# the eight education categories of the birth certificate are combined into four groups: (1) less than
# high school, (2) high school graduate, (3) some college or associate degree, (4) bachelor's degree or higher
EDUCATION_GROUPS = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 4}


def encode(records: np.ndarray, columns: list[str]) -> np.ndarray:
    """
    Converts natality records to a data array for the regressions. Education is represented by three
    indicator variables, with the first group as the reference category. Pre-pregnancy smoking is not
    used as a covariate.

    Parameters:
    - records: Array of shape (number of records, len(columns)), as saved by prepare_natality.py.
    - columns: Names of the columns of the records.

    Returns:
    - np.ndarray: Data array of shape (len(VARIABLES), number of records).
    """
    records = records.astype(float)
    column = {name: records[:, columns.index(name)] for name in columns}
    education = np.array([EDUCATION_GROUPS[int(category)] for category in column['education']])

    return np.array([
        column['bmi'], column['birthweight'], column['age'], column['height'],
        education == 2, education == 3, education == 4, column['previous_live_births'],
    ], dtype=float)


def estimate_reference_coefficient(data: np.ndarray, adjustment_set: list[str]) -> tuple[float, float]:
    """
    Estimates the coefficient of body mass index in the regression on the reference sample, with a
    heteroskedasticity-robust (HC3) standard error.

    Parameters:
    - data: Data array of shape (len(VARIABLES), sample_size).
    - adjustment_set: Covariates of the regression.

    Returns:
    - tuple: (coefficient, standard error)
    """
    indices = [VARIABLES.index(name) for name in adjustment_set] + [VARIABLES.index('A')]
    Y = data[VARIABLES.index('Y'), :]

    # design matrix with intercept, covariates are centered for numerical stability
    X = data[indices, :].T
    X = np.column_stack((np.ones(len(Y)), X - X.mean(axis=0)))

    inverse = np.linalg.inv(X.T @ X)
    coefficients = inverse @ X.T @ Y
    residuals = Y - X @ coefficients

    # HC3: residuals are divided by one minus the leverage
    leverage = np.sum((X @ inverse) * X, axis=1)
    influence = (X @ inverse[:, -1]) * residuals / (1 - leverage)

    return coefficients[-1], np.sqrt(influence @ influence)


@hydra.main(config_path='./config', config_name='natality', version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Runs the data application based on a Hydra configuration.

    The coefficient of body mass index in the regression with all covariates is estimated on the reference
    sample. For each sample size, small datasets are drawn from the training sample and the coefficient is
    estimated with all covariates, with the estimated MSE-optimal set and with the set of smallest estimated
    variance. The estimates and the mean squared deviations from the reference coefficient are saved to a
    JSON file.

    Parameters:
        cfg (DictConfig): The Hydra configuration object containing the path of the prepared data, the sample
                        sizes, the number of datasets, the number of bootstrap samples and the random seed.
    """
    sample = np.load(to_absolute_path(cfg.sample_path))
    columns = list(sample['columns'])
    reference_data = encode(sample['reference'], columns)
    training_data = encode(sample['training'], columns)

    # all combinations of the four covariates; the full set Z is the reference set and the only set
    # with estimated bias zero
    candidates = [sum(groups, []) for size in range(len(COVARIATE_GROUPS) + 1)
                  for groups in combinations(COVARIATE_GROUPS, size)]
    full_set = candidates[-1]

    reference_coefficient, reference_se = estimate_reference_coefficient(reference_data, full_set)
    results = []

    for sample_size in cfg.sample_sizes:
        for dataset in range(cfg.n_datasets):
            # draw a dataset from the training sample without replacement
            rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, sample_size, dataset, 0]))
            data = training_data[:, rng.choice(training_data.shape[1], size=sample_size, replace=False)]

            # skip datasets for which the regression with all covariates is rank-deficient
            _, full_rank = bootstrap_treatment_effects(data, full_set, VARIABLES, np.arange(sample_size)[None, :])
            if not full_rank[0]:
                continue

            # estimated MSE-optimal set, with a separate random stream for the bootstrap
            rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, sample_size, dataset, 1]))
            mse_set = get_adjustment_set(data, VARIABLES, candidates, full_set, [full_set], "MSE", cfg.n_bootstrap, rng=rng)[0]

            # set with the smallest estimated variance
            variance_set = get_adjustment_set(data, VARIABLES, candidates, full_set, [full_set], "Variance")[0]

            results.append({
                "Sample size": sample_size,
                "Dataset": dataset,
                "Full": estimate_treatment_effect(data, full_set, VARIABLES),
                "MSE": estimate_treatment_effect(data, mse_set, VARIABLES),
                "Variance": estimate_treatment_effect(data, variance_set, VARIABLES),
                "MSE Adjustment Set": mse_set,
                "Variance Adjustment Set": variance_set,
            })

    # mean and standard deviation of the squared deviations from the reference coefficient
    summary = []
    for sample_size in cfg.sample_sizes:
        rows = [result for result in results if result["Sample size"] == sample_size]
        row = {"Sample size": sample_size, "Datasets": len(rows)}

        # also for the reference coefficient shifted by 1.96 standard errors
        for label, shift in [("", 0), (" (reference - 1.96 SE)", -1.96), (" (reference + 1.96 SE)", 1.96)]:
            for method in ["Full", "MSE", "Variance"]:
                squared_errors = (np.array([result[method] for result in rows]) - reference_coefficient - shift * reference_se) ** 2
                row[f"{method} MSE{label}"] = squared_errors.mean()
                if shift == 0:
                    row[f"{method} SD"] = squared_errors.std(ddof=1)
        summary.append(row)

    output = {
        "reference_coefficient": reference_coefficient,
        "reference_se": reference_se,
        "summary": summary,
        "per_dataset": results,
    }

    filename = os.path.join(os.getcwd(), 'natality_results.json')
    with open(filename, 'w') as f:
        json.dump(output, f, indent=4)

    print(json.dumps(summary, indent=4))
    print(f"File saved successfully: {filename}")


if __name__ == "__main__":
    main()
