# Finite sample-optimal adjustment sets in linear Gaussian causal models

Code for the experiments in the paper "Finite sample-optimal adjustment sets in linear Gaussian causal models" by N. Rutsch, S. Magliacane and S. L. van der Pas.

## Setup

```sh
conda env create -f config/environment.yml
conda activate InvalidAdjustmentSets
```

All scripts are configured with [Hydra](https://hydra.cc). Settings can be changed in the files in `config/` or on the command line, e.g. `python main.py graph=m2 sample_size=100`. Results are saved to the Hydra output directory of each run (`outputs/<date>/<time>/`).

## Files

| File | Content |
| --- | --- |
| `adjustment_sets.py` | Selection of the adjustment set with the smallest estimated MSE or variance, including variance and bootstrap bias estimation. |
| `main.py` | Simulations for the models M1 and M2 with an estimated or a fixed adjustment set. |
| `oracle.py` | MSE-optimal adjustment set based on the true model parameters. |
| `coverage.py` | Confidence intervals after adjustment set selection. |
| `non_gaussian.py` | Experiments with non-Gaussian noise (models M3 and M4). |
| `prepare_natality.py`, `natality.py` | Application to the 2023 U.S. birth certificate data. |
| `plots.py` | Figures and tables of the simulation experiments. |
| `config/` | Hydra configurations. The models M1 and M2 are defined in `config/graph/`, together with the reduced search space obtained from the graphical criteria, the valid adjustment sets and the reference set O. |

## Reproducing the experiments

### Selection of adjustment sets (M1 and M2)

There are two different ways in which an MSE-optimal adjustment set appears in these experiments:

- The **oracle set** $O_n$ is the adjustment set with the smallest true MSE. It is computed from the true model parameters and does not use any data. It is not available in practice and only serves as a benchmark.
- The **estimated set** $\hat O_n$ is selected from data: the MSE of each candidate set is estimated on the simulated dataset, and the set with the smallest estimated MSE is chosen. This is the procedure proposed in the paper.

#### Oracle: MSE-optimal set from the true parameters

`oracle.py` computes the true MSE of every adjustment set from the structural coefficients in `config/graph/` and prints the oracle set $O_n$ for each sample size. No data is simulated.

```sh
python oracle.py graph=m1
python oracle.py graph=m2
```

#### Estimation: selecting the adjustment set from data

`main.py` simulates datasets and, for each dataset, selects an adjustment set from the reduced search space and estimates the treatment effect with it. The selection only uses the dataset and the causal graph, not the true parameters. The simulations are run separately for each model, sample size and selection criterion. For example, for M1 with n = 50:

```sh
# estimated MSE-optimal set
python main.py graph=m1 sample_size=50 optimality=MSE

# set with the smallest estimated variance
python main.py graph=m1 sample_size=50 optimality=Variance
```

#### Baselines: fixed adjustment sets

For comparison, `main.py` is run on the same datasets with a fixed adjustment set instead of a selected one: the asymptotically optimal set $O$, and the oracle set $O_n$ printed by `oracle.py` for that sample size (for M1 with n = 50, this is $\{O_1\}$).

```sh
python main.py graph=m1 sample_size=50 estimate_adjustment_set=false "adjustment_set=[O1,O2]"
python main.py graph=m1 sample_size=50 estimate_adjustment_set=false "adjustment_set=[O1]"
```

#### Tables and figures

The sample sizes used in the paper are listed in `config/graph/m1.yaml` and `config/graph/m2.yaml`. Several settings can be run with one command using Hydra's multirun option, e.g. `python main.py -m graph=m1 sample_size=10,20,30,40,50,100,500,1000 optimality=MSE,Variance`.

The tables and figures are created from all result files in a directory. They compare the MSE of the treatment effect estimate for the estimated set, the optimal set $O$ and the oracle set:

```sh
python plots.py results_dir=outputs
```

### Confidence intervals

```sh
python coverage.py graph=m1 sample_size=50
```

This is run for `graph=m1,m2` and `sample_size=50,100`. The number of parallel processes is set with `n_jobs`.

### Non-Gaussian noise

```sh
python non_gaussian.py experiment=m3   # sensitivity of the sample size criterion
python non_gaussian.py experiment=m4   # counterexample to the graphical criteria
python plots.py results_dir=null non_gaussian_path=<path to non_gaussian_m3.json>
```

### Data application

Download the 2023 natality public-use file `Nat2023us.zip` from the [National Center for Health Statistics](https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm) (direct link: <https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/Nat2023us.zip>) and save it to `data/Nat2023us.zip`. The use of the data is subject to the [data user agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html) of the National Center for Health Statistics.

```sh
python prepare_natality.py   # selects 200,000 eligible records, requires the unzip command
python natality.py
```
