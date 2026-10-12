import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import glob
import json
import os

from oracle import get_oracle_set

PALETTE = sns.color_palette('bright').as_hex()

# colors of the adjustment sets, shared by all figures
SET_COLORS = {'O1': PALETTE[1], 'O1 O2': PALETTE[2], 'W2': PALETTE[0], 'C1': PALETTE[0], 'W1': PALETTE[3],
              'Empty': PALETTE[7], 'O2': PALETTE[8], 'O2 W1': PALETTE[5], 'O2 W2': PALETTE[6]}


def set_name(adjustment_set: list[str]) -> str:
    """
    Converts an adjustment set to a name that does not depend on the order of its variables.

    Parameters:
    - adjustment_set: Variables that are included in the adjustment set.

    Returns:
    - str: Sorted variable names separated by spaces, or "Empty" for the empty set.
    """
    return ' '.join(sorted(adjustment_set)) if len(adjustment_set) > 0 else "Empty"


def set_label(name: str) -> str:
    """
    Converts the name of an adjustment set to a label for the figures.

    Parameters:
    - name: Name of the adjustment set, as returned by set_name.

    Returns:
    - str: Label in LaTeX math notation.
    """
    if name == "Empty":
        return r'$\varnothing$'
    if name == "O1 O2":
        return '$O$'

    return r'$\{' + ','.join(f'{var[0]}_{{{var[1:]}}}' for var in name.split(' ')) + r'\}$'


def load_results(results_dir: str) -> pd.DataFrame:
    """
    Loads all result files of main.py in a directory and computes the squared errors.

    Parameters:
    - results_dir: Directory with the result files, searched recursively.

    Returns:
    - pd.DataFrame: One row per graph, sample size, seed and method. The column 'Method' is "MSE" or
        "Variance" for estimated adjustment sets and "Fixed" otherwise.
    """
    files = glob.glob(os.path.join(results_dir, '**', 'results_*.json'), recursive=True)
    df = pd.concat([pd.read_json(f) for f in files], ignore_index=True)

    # true treatment effect of each graph
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'graph')
    treatment_effects = {graph: OmegaConf.load(os.path.join(config_dir, f'{graph}.yaml')).treatment_effect
                         for graph in df['Graph'].unique()}

    df['Adjustment Set'] = df['Adjustment Set'].apply(set_name)
    df['Method'] = np.where(df['Estimated'], df['Optimality'], 'Fixed')
    df['Squared Error'] = (df['Treatment Effect'] - df['Graph'].map(treatment_effects)) ** 2

    return df


def get_method_results(df: pd.DataFrame, graph_config: DictConfig) -> pd.DataFrame:
    """
    Collects the squared errors of the compared methods for one graph: adjustment for the optimal set O,
    for the estimated MSE-optimal set, for the set with the smallest estimated variance and for the oracle
    MSE-optimal set O_n.

    Parameters:
    - df: Results as returned by load_results.
    - graph_config: The configuration object of the graph.

    Returns:
    - pd.DataFrame: Rows of df with an additional column 'Procedure'.
    """
    df = df[df['Graph'] == graph_config.id]
    reference = set_name(graph_config.reference_set)
    fixed = df[df['Method'] == 'Fixed']
    parts = [
        fixed[fixed['Adjustment Set'] == reference].assign(Procedure='O'),
        df[df['Method'] == 'MSE'].assign(Procedure='Estimated MSE'),
        df[df['Method'] == 'Variance'].assign(Procedure='Estimated variance'),
    ]

    # the oracle set depends on the sample size
    for sample_size in sorted(df['Sample size'].unique()):
        oracle = set_name(get_oracle_set(graph_config, int(sample_size))[0])
        rows = fixed[(fixed['Sample size'] == sample_size) & (fixed['Adjustment Set'] == oracle)]
        parts.append(rows.assign(Procedure='Oracle'))

    return pd.concat(parts, ignore_index=True)


def print_tables(results: pd.DataFrame, graph_id: str) -> None:
    """
    Prints the mean and the standard deviation of the squared errors for each procedure and sample size.

    Parameters:
    - results: Results as returned by get_method_results.
    - graph_id: Identifier of the graph.
    """
    table = results.groupby(['Sample size', 'Procedure'])['Squared Error'].agg(['mean', 'std', 'count']).unstack('Procedure')
    print(f"Squared errors for {graph_id}:")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


def plot_mse(results: pd.DataFrame, filename: str) -> None:
    """
    Plots the MSE against the sample size for adjustment for O, the estimated MSE-optimal set and the
    oracle set, with logarithmic scales on both axes.

    Parameters:
    - results: Results as returned by get_method_results.
    - filename: Path of the figure.
    """
    lines = [('Estimated MSE', r'Estimated $\hat O_n$', PALETTE[4], 'o', '-'),
             ('O', '$O$', PALETTE[2], 's', '-'),
             ('Oracle', '$O_n$ (oracle)', '#222222', '^', '--')]
    mse = results.groupby(['Procedure', 'Sample size'])['Squared Error'].mean()

    fig, ax = plt.subplots()
    for procedure, label, color, marker, linestyle in lines:
        if procedure in mse.index.get_level_values('Procedure'):
            ax.plot(mse[procedure].index, mse[procedure].values, color=color, marker=marker,
                    linestyle=linestyle, markersize=9, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([10, 50, 100, 500, 1000])
    ax.set_xticklabels(['10', '50', '100', '500', '1000'])
    ax.minorticks_off()
    ax.set_xlabel('Sample size (log scale)')
    ax.set_ylabel('MSE (log scale)')
    ax.legend(fontsize=22)

    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_selection(results: pd.DataFrame, sample_size: int, filename: str) -> None:
    """
    Plots how often each adjustment set is selected by the estimated MSE-optimal procedure at one sample
    size. Only sets that are selected at least once are shown.

    Parameters:
    - results: Results as returned by get_method_results.
    - sample_size: Sample size of the datasets.
    - filename: Path of the figure.
    """
    selected = results[(results['Procedure'] == 'Estimated MSE') & (results['Sample size'] == sample_size)]
    frequencies = 100 * selected['Adjustment Set'].value_counts(normalize=True)

    # order of the colors dictionary, with the optimal set O last
    order = ['Empty', 'O1', 'O2', 'W1', 'W2', 'C1', 'O2 W1', 'O2 W2', 'O1 O2']
    frequencies = frequencies[[name for name in order if name in frequencies.index]]

    fig, ax = plt.subplots(figsize=(1.3 * len(frequencies) + 1.2, 4.8))
    bars = ax.bar(range(len(frequencies)), frequencies.values, width=0.5, linewidth=0,
                  color=[SET_COLORS[name] for name in frequencies.index])
    for bar, frequency in zip(bars, frequencies.values):
        ax.text(bar.get_x() + bar.get_width() / 2, frequency + 1.5, f'{frequency:.1f}', ha='center',
                va='bottom', fontsize=19)

    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(frequencies)))
    ax.set_xticklabels([set_label(name) for name in frequencies.index], rotation=40, ha='right', rotation_mode='anchor')
    ax.grid(axis='x', visible=False)
    ax.set_xlabel('Adjustment set')
    ax.set_ylabel('Selection frequency (%)')

    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_example(df: pd.DataFrame, graph_config: DictConfig, example_config: DictConfig, filename: str) -> None:
    """
    Plots the RMSE of the treatment effect estimator against the sample size for fixed adjustment sets.
    Dashed lines mark the sample sizes at which the predicted MSE-optimal set among the shown sets
    changes, and the predicted set is given at the top of the plot.

    Parameters:
    - df: Results as returned by load_results.
    - graph_config: The configuration object of the graph.
    - example_config: Configuration with the shown adjustment sets and the range of sample sizes.
    - filename: Path of the figure.
    """
    names = [set_name(adj_set) for adj_set in example_config.adjustment_sets]
    n_min, n_max = example_config.sample_size_range

    fixed = df[(df['Graph'] == graph_config.id) & (df['Method'] == 'Fixed') & df['Adjustment Set'].isin(names)
               & df['Sample size'].between(n_min, n_max)]
    if fixed.empty:
        return
    rmse = np.sqrt(fixed.groupby(['Adjustment Set', 'Sample size'])['Squared Error'].mean())

    fig, ax = plt.subplots()
    for name in names:
        if name in rmse.index.get_level_values('Adjustment Set'):
            ax.plot(rmse[name].index, rmse[name].values, color=SET_COLORS[name], label=set_label(name))

    # predicted MSE-optimal set among the shown sets for every sample size
    sample_sizes = np.arange(n_min, n_max + 1)
    predicted = [set_name(get_oracle_set(graph_config, int(n), example_config.adjustment_sets)[0]) for n in sample_sizes]

    # mark changes of the predicted set and label the regions in between
    changes = [i for i in range(1, len(predicted)) if predicted[i] != predicted[i - 1]]
    for i in changes:
        ax.axvline(x=sample_sizes[i] - 0.5, color='gray', linestyle='--')
    boundaries = [0] + changes + [len(predicted)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        center = (sample_sizes[start] + sample_sizes[end - 1]) / 2
        ax.text(center, 1.02, set_label(predicted[start]), color=SET_COLORS[predicted[start]], ha='center',
                va='bottom', fontsize=20, transform=ax.get_xaxis_transform())

    ax.set_xlim(n_min, n_max)
    ax.set_ylim(0, None)
    ax.set_xlabel('Sample size')
    ax.set_ylabel('RMSE')
    ax.legend(title='Adjustment set', fontsize=20, title_fontsize=20)

    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_non_gaussian(non_gaussian_path: str, filename: str) -> None:
    """
    Plots the sensitivity of the sample size criterion to non-Gaussian noise in M3: the relative shift of
    the estimated sample size threshold from the Gaussian threshold (left) and the largest relative MSE
    disadvantage from using the Gaussian threshold (right), over the grid of sinh-arcsinh parameters.

    Parameters:
    - non_gaussian_path: Path of the results of non_gaussian.py (experiment=m3).
    - filename: Path of the figure.
    """
    with open(non_gaussian_path) as f:
        results = pd.DataFrame(json.load(f)['results'])
    grid = results[results['family'] == 'sinh_arcsinh']

    # rows are values of delta, columns are values of epsilon, both in percent
    shift = 100 * grid.pivot(index='delta', columns='epsilon', values='relative_shift')
    disadvantage = 100 * grid.pivot(index='delta', columns='epsilon', values='max_relative_mse_disadvantage')

    epsilons, deltas = shift.columns.values, shift.index.values
    step_epsilon, step_delta = epsilons[1] - epsilons[0], deltas[1] - deltas[0]
    extent = [epsilons[0] - step_epsilon / 2, epsilons[-1] + step_epsilon / 2,
              deltas[0] - step_delta / 2, deltas[-1] + step_delta / 2]

    # symmetric color scale for the shift, limited to the 98% quantile of the absolute values
    limit = np.quantile(np.abs(shift.values), 0.98)

    sns.set_theme(style='white', context='poster')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    image = axes[0].imshow(shift.values, origin='lower', aspect='auto', extent=extent, cmap='RdBu_r',
                           norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation='nearest')
    fig.colorbar(image, ax=axes[0], label='Threshold shift (%)')
    image = axes[1].imshow(disadvantage.values, origin='lower', aspect='auto', extent=extent, cmap='magma',
                           interpolation='nearest')
    fig.colorbar(image, ax=axes[1], label='Worst-case MSE loss (%)')
    for ax in axes:
        ax.set_xlabel(r'$\epsilon$')
        ax.set_ylabel(r'$\delta$')

    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


@hydra.main(config_path='./config', config_name='plots', version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Creates the figures and prints the tables of the simulation experiments based on a Hydra configuration.

    Parameters:
        cfg (DictConfig): The Hydra configuration object containing the paths of the results and the output
                        directory.
    """
    output_dir = to_absolute_path(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if cfg.results_dir is not None:
        df = load_results(to_absolute_path(cfg.results_dir))
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'graph')

        for graph_id in sorted(df['Graph'].unique()):
            graph_config = OmegaConf.load(os.path.join(config_dir, f'{graph_id}.yaml'))
            results = get_method_results(df, graph_config)
            print_tables(results, graph_id)

            sns.set_theme(style='darkgrid', palette='bright', context='poster')
            plot_mse(results, os.path.join(output_dir, f'{graph_id}_mse.pdf'))
            plot_selection(results, cfg.selection_sample_size, os.path.join(output_dir, f'{graph_id}_selection.pdf'))
            if graph_id in cfg.examples:
                plot_example(df, graph_config, cfg.examples[graph_id], os.path.join(output_dir, f'{graph_id}_example.pdf'))

    if cfg.non_gaussian_path is not None:
        plot_non_gaussian(to_absolute_path(cfg.non_gaussian_path), os.path.join(output_dir, 'non_gaussian_sensitivity.pdf'))

    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
