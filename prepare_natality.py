import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import subprocess

# positions of the fields in the fixed-width records (one-based, inclusive), see the 2023 user guide
FIELDS = {
    'age_imputed': (73, 73), 'age': (75, 76),
    'education': (124, 124), 'education_reporting': (126, 126),
    'prior_living': (171, 172), 'prior_dead': (173, 174),
    'smoking': (253, 254), 'smoking_reporting': (265, 265),
    'height': (280, 281), 'height_reporting': (282, 282), 'bmi': (283, 286),
    'plurality': (454, 454), 'plurality_imputed': (456, 456), 'birthweight': (504, 507),
}

# published ranges of the variables, values outside are missing or unknown
RANGES = {
    'bmi': (13, 69.9), 'birthweight': (227, 8165), 'age': (12, 50), 'height': (30, 78),
    'education': (1, 8), 'prior_living': (0, 30), 'prior_dead': (0, 30), 'smoking': (0, 98),
}

COLUMNS = ['bmi', 'birthweight', 'age', 'height', 'education', 'previous_live_births', 'smoking']
N_PRELIMINARY, N_REFERENCE, N_TRAINING = 20000, 100000, 80000


def parse_record(line: bytes) -> list[float] | None:
    """
    Extracts the variables from one birth record and checks whether the record is eligible.

    Parameters:
    - line: One fixed-width record of the natality file.

    Returns:
    - list | None: Values in the order of COLUMNS, or None if the record is excluded.
    """
    record = {name: line[start - 1:end].decode('ascii').strip() for name, (start, end) in FIELDS.items()}

    # singleton births without imputed plurality or maternal age
    if record['plurality'] != '1' or record['plurality_imputed'] != '' or record['age_imputed'] != '':
        return None

    # the information must be collected in the reporting area
    if any(record[name] != '1' for name in ['education_reporting', 'smoking_reporting', 'height_reporting']):
        return None

    # no missing values or values outside the published ranges
    for name, (lower, upper) in RANGES.items():
        try:
            if not lower <= float(record[name]) <= upper:
                return None
        except ValueError:
            return None

    record['previous_live_births'] = int(record['prior_living']) + int(record['prior_dead'])

    return [float(record[name]) for name in COLUMNS]


@hydra.main(config_path='./config', config_name='natality', version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Prepares the 2023 U.S. natality data: reads the public-use file, selects 200,000 eligible records
    uniformly at random and divides them into samples for preliminary checks, for the reference
    coefficient and for training.

    The file Nat2023us.zip can be downloaded from
    https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/Nat2023us.zip. It is read
    with the unzip command, because Python does not support its compression method.

    Parameters:
        cfg (DictConfig): The Hydra configuration object containing the paths and the random seeds.
    """
    archive_path = to_absolute_path(cfg.archive_path)
    n_selected = N_PRELIMINARY + N_REFERENCE + N_TRAINING

    rng = np.random.default_rng(cfg.sampling_seed)
    selected = np.empty((n_selected, len(COLUMNS)), dtype=np.float32)
    n_records = 0
    n_eligible = 0

    process = subprocess.Popen(['unzip', '-p', archive_path], stdout=subprocess.PIPE)
    for line in process.stdout:
        n_records += 1
        values = parse_record(line)
        if values is None:
            continue
        n_eligible += 1

        # reservoir sampling: every eligible record is selected with the same probability
        slot = n_eligible - 1 if n_eligible <= n_selected else int(rng.integers(n_eligible))
        if slot < n_selected:
            selected[slot] = values

    if process.wait() != 0:
        raise RuntimeError("Reading the archive with unzip failed.")

    # divide the selected records into the three samples
    permutation = np.random.default_rng(cfg.split_seed).permutation(n_selected)
    np.savez_compressed(
        to_absolute_path(cfg.sample_path),
        columns=COLUMNS,
        preliminary=selected[permutation[:N_PRELIMINARY]],
        reference=selected[permutation[N_PRELIMINARY:N_PRELIMINARY + N_REFERENCE]],
        training=selected[permutation[N_PRELIMINARY + N_REFERENCE:]],
    )

    print(f"{n_eligible} of {n_records} records are eligible")
    print(f"File saved successfully: {to_absolute_path(cfg.sample_path)}")


if __name__ == "__main__":
    main()
