import numpy as np
from sklearn.linear_model import LinearRegression


def fit_outcome_regression(
    data: np.ndarray,
    adjustment_set: list[str],
    variables: list[str]
) -> tuple[float, bool]:
    """
    Regresses the outcome Y on the treatment A and the adjustment set with ordinary least-squares (OLS),
    including an intercept.

    Parameters:
    - data: Data array of shape (len(variables), sample_size).
    - adjustment_set: Variables that are included in the adjustment set.
    - variables: List of all variables, order corresponding to the indices in the data array.

    Returns:
    - tuple: (coefficient of the treatment, whether the regression design has full column rank)
    """
    # map variable names to indices
    variable_indices = {name: i for i, name in enumerate(variables)}
    adjustment_indices = [variable_indices[name] for name in sorted(adjustment_set)]

    Y = data[variable_indices['Y'], :]  # outcome variable
    A = data[variable_indices['A'], :]  # treatment variable

    # covariates X, with the treatment in the last column
    X = np.column_stack((data[adjustment_indices, :].T, A))
    model = LinearRegression().fit(X, Y)

    # the rank refers to the centered covariates, the intercept is fitted separately
    return model.coef_[-1], model.rank_ == X.shape[1]


def estimate_treatment_effect(data: np.ndarray, adjustment_set: list[str], variables: list[str]) -> float:
    """
    Estimates the average treatment effect of the treatment A on the outcome Y with ordinary least-squares
    (OLS), conditional on the adjustment set.

    Parameters:
    - data: Data array of shape (len(variables), sample_size).
    - adjustment_set: Variables that are included in the adjustment set.
    - variables: List of all variables, order corresponding to the indices in the data array.

    Returns:
    - float: estimated average treatment effect
    """
    treatment_effect, _ = fit_outcome_regression(data, adjustment_set, variables)

    return treatment_effect


def estimate_variance(
    data: np.ndarray,
    adjustment_set: list[str],
    variables: list[str]
) -> tuple[float, float]:
    """
    Estimates the error variance of the outcome and the residual sum of squares (RSS) of the treatment,
    given a specific adjustment set. Their ratio RSS_y.ak / ((n - |K| - 2) RSS_a.k) is the usual OLS
    estimate of the variance of the treatment effect estimator. Both regressions include an intercept.

    Parameters:
    - data: Data array of shape (len(variables), sample_size).
    - adjustment_set: Variables to condition on.
    - variables: List of all variables, order corresponding to the indices in the data array.

    Returns:
    - tuple: (estimated error variance of outcome, residual sum of squares of treatment)
    """
    adjustment_indices = [variables.index(name) for name in sorted(adjustment_set)]

    A = data[variables.index('A'), :]  # treatment variable
    Y = data[variables.index('Y'), :]  # outcome variable
    X_adjust = data[adjustment_indices, :].T

    if adjustment_indices:
        # regress the treatment on the adjustment set
        reg_treatment = LinearRegression().fit(X_adjust, A)
        residuals_treatment = A - reg_treatment.predict(X_adjust)
    else:
        # if the adjustment set is the empty set, we use the mean
        residuals_treatment = A - np.mean(A)

    # regress the outcome on the treatment and the adjustment set
    X_adjust_with_treatment = np.column_stack((X_adjust, A))
    reg_outcome = LinearRegression().fit(X_adjust_with_treatment, Y)
    residuals_outcome = Y - reg_outcome.predict(X_adjust_with_treatment)

    # calculate the treatment RSS
    rss_treatment = np.sum(residuals_treatment ** 2)

    # estimate the error variance for the outcome from the RSS; the degrees of freedom account for
    # the coefficients of the adjustment variables, the treatment and the intercept
    rss_outcome = np.sum(residuals_outcome ** 2)
    df = data.shape[1] - len(adjustment_indices) - 2
    est_error_var_outcome = rss_outcome / df

    return est_error_var_outcome, rss_treatment


def estimate_bias(
    data: np.ndarray,
    adjustment_sets: list[list[str]],
    reference_set: list[str],
    variables: list[str],
    n_bootstrap: int = 1000,
    max_proposals: int = 100000
) -> dict[tuple[str, ...], float]:
    """
    Estimates the bias of each candidate adjustment set with a paired bootstrap: on every bootstrap
    sample, the treatment effect estimate of the candidate is compared to the estimate using the
    reference set, which is assumed to be unbiased.

    Bootstrap samples can have rank-deficient regression designs because observations are repeated.
    A proposed sample is only accepted if the designs of the reference set and of all candidates have
    full column rank. Otherwise it is discarded for all sets and a new sample is drawn, until
    n_bootstrap samples are accepted.

    Parameters:
    - data: Data array of shape (len(variables), sample_size).
    - adjustment_sets: List of candidate adjustment sets to evaluate.
    - reference_set: Adjustment set assumed to give an unbiased estimate of the treatment effect.
    - variables: List of all variables, order corresponding to the indices in the data array.
    - n_bootstrap: Number of accepted bootstrap samples.
    - max_proposals: Maximum number of proposed bootstrap samples.

    Returns:
    - dict: Mapping from adjustment set (as sorted tuple) to estimated bias.
    """
    sample_size = data.shape[1]
    reference_set = tuple(sorted(reference_set))
    adjustment_sets = [tuple(sorted(adj_set)) for adj_set in adjustment_sets]

    # keep track of the differences to the reference estimate
    differences = {adj_set: [] for adj_set in adjustment_sets}
    n_accepted = 0
    n_proposed = 0

    while n_accepted < n_bootstrap:
        if n_proposed == max_proposals:
            raise RuntimeError(f"Only {n_accepted} of {n_bootstrap} bootstrap samples had full rank "
                               f"after {max_proposals} proposals.")

        # draw observations with replacement, shared by all adjustment sets
        indices = np.random.randint(0, sample_size, size=sample_size)
        bootstrap_sample = data[:, indices]
        n_proposed += 1

        # estimate the ATE using the reference set and each (potentially biased) adjustment set
        fits = [fit_outcome_regression(bootstrap_sample, adj_set, variables)
                for adj_set in [reference_set] + adjustment_sets]

        # discard the sample for all sets if any regression design is rank-deficient
        if not all(full_rank for _, full_rank in fits):
            continue

        reference_treatment_effect = fits[0][0]
        for adj_set, (adj_treatment_effect, _) in zip(adjustment_sets, fits[1:]):
            differences[adj_set].append(adj_treatment_effect - reference_treatment_effect)
        n_accepted += 1

    # calculate the mean difference for each adjustment set
    return {adj_set: np.mean(values) for adj_set, values in differences.items()}


def get_adjustment_set(
    data: np.ndarray,
    variables: list[str],
    candidates: list[list[str]],
    reference_set: list[str],
    valid_sets: list[list[str]],
    optimality: str,
    n_bootstrap: int = 1000
) -> tuple[list[str], float | None, float, float | None, float]:
    """
    Selects the adjustment set with the smallest estimated MSE or the smallest estimated variance.

    The candidates are first screened against the reference set: a candidate is only kept if its
    estimated variance is smaller than the estimated variance of the reference set. For the MSE, the
    bias of valid adjustment sets is set to zero and the bias of the remaining invalid sets is
    estimated by bootstrapping. The reference set is selected in case of ties.

    Parameters:
    - data: Data array of shape (len(variables), sample_size).
    - variables: List of all variable names, consistent with the data ordering.
    - candidates: Candidate adjustment sets (search space).
    - reference_set: Valid adjustment set used as the reference, e.g. the asymptotically optimal set O.
    - valid_sets: Candidate adjustment sets that are valid according to the causal graph.
    - optimality: Selection criterion ("Variance" or "MSE").
    - n_bootstrap: Number of accepted bootstrap samples for bias estimation.

    Returns:
    - tuple: (best adjustment set, estimated bias, estimated variance, estimated MSE, treatment RSS)
    """
    if optimality not in ("MSE", "Variance"):
        raise ValueError(f"Unknown optimality: {optimality}")

    sample_size = data.shape[1]
    reference_set = tuple(sorted(reference_set))
    valid_sets = {tuple(sorted(adj_set)) for adj_set in valid_sets} | {reference_set}

    # estimate the variance of the reference set, which is unbiased
    est_error_var_outcome, rss_A = estimate_variance(data, reference_set, variables)
    reference_variance = est_error_var_outcome / rss_A

    # the reference set comes first, such that it is selected in case of ties
    properties = [{
        'Adjustment set': reference_set,
        'Bias': 0.0,
        'Variance': reference_variance,
        'MSE': reference_variance,
        'RSS_A': rss_A
    }]

    for adjustment_set in candidates:
        adjustment_set = tuple(sorted(adjustment_set))

        # skip the reference set and sets without finite MSE at this sample size
        if adjustment_set == reference_set or sample_size <= len(adjustment_set) + 3:
            continue

        est_error_var_outcome, rss_A = estimate_variance(data, adjustment_set, variables)
        variance = est_error_var_outcome / rss_A

        # keep only adjustment sets with smaller variance than the reference set
        if variance < reference_variance:
            valid = adjustment_set in valid_sets
            properties.append({
                'Adjustment set': adjustment_set,
                'Bias': 0.0 if valid else None,
                'Variance': variance,
                'MSE': variance if valid else None,
                'RSS_A': rss_A
            })

    # bias estimation is only needed for MSE-optimality and invalid adjustment sets
    invalid_properties = [prop for prop in properties if prop['Adjustment set'] not in valid_sets]
    if optimality == "MSE" and invalid_properties:
        biases = estimate_bias(data, [prop['Adjustment set'] for prop in invalid_properties],
                               reference_set, variables, n_bootstrap)

        for prop in invalid_properties:
            prop['Bias'] = biases[prop['Adjustment set']]
            prop['MSE'] = prop['Bias'] ** 2 + prop['Variance']

    # choose adjustment set with the minimum MSE or Variance
    best_set = min(properties, key=lambda x: x[optimality])

    return list(best_set['Adjustment set']), best_set['Bias'], best_set['Variance'], best_set['MSE'], best_set['RSS_A']
