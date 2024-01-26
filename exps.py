import os
from datetime import datetime

import pandas as pd

from models.coxph import cox_exp
from models.deephit import deephit_exp
from models.deepsurv import ds_exp
from models.nmtlr import nmtlr_exp
from models.nnet_survival import nnet_exp
from models.pch import pch_exp
from models.rsf import rsf_exp

# Random seeds for reproducibility. We use a single seed for each run.
SEEDS = [1234]

# Dataset to run experiments on. The available datasets are:
# - GBSG2
# - metabric
# - support2
# - flchain
# - Aids2
DATASETS = ["GBSG2"]

# Minimum number of samples per client dataset.
MIN_SAMPLES = 25

# Split types. Each experiment is run on all the split types provided. The available split types are:
# - uniform: splits the dataset uniformly among clients
# - label_skew: splits the dataset among clients so that each client has a different label distribution
SPLITS = [
    ("uniform", {"min_samples": MIN_SAMPLES}),
    ("label_skew", {"min_samples": MIN_SAMPLES, "alpha": 8.0}),
]

# Number of clients in the federation. Each experiment is run on all the number of clients provided.
NUM_CLIENTS = [10]

# Parameters for Random Survival Forests are dataset-specific. 
# The parameters below are the deafult used in the original paper.
RSF_PARAMS = {
    "GBSG2": {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 20, 'n_estimators': 100, 'n_jobs': -1},
    "metabric": {'max_depth': 20, 'max_features': 'log2', 'min_samples_split': 15, 'n_estimators': 200, 'n_jobs': -1},
    "support2": {'max_depth': 1, 'max_features': 'sqrt', 'min_samples_split': 20, 'n_estimators': 500, 'n_jobs': -1},
    "flchain": {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 20, 'n_estimators': 200, 'n_jobs': -1},
    "Aids2": {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 2000, 'n_jobs': -1},
}

# Number of trees to train in the federation. For each dataset, we run experiments on all the number of trees provided.
NUM_FED_TREES = {
    "GBSG2": [100, 200, 300, 400, 500],
    "metabric": [100, 200, 300, 400, 500],
    "support2": [350, 400, 450, 500, 550, 600, 650, 700, 750, 800],
    "flchain": [150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400],
    "Aids2": [500, 1000, 2000],
}

# parameters for CoxPH/DeepSurv experiments
CPH_EPOCHS = 500
CPH_FED_ROUNDS = 500
HIDDEN_FEATURES = 32

# DeepHit loss params
ALPHA = 0.5


def run_exps(log_dir):
    """
    Runs a set of experiments from the product of several lists of global parameters.
    Results are logged as csv in the log_dir directory.
    """
    os.makedirs(log_dir, exist_ok=True)

    dfs = []  # stores exp dataframes

    # repeat over N runs:
    for i, seed in enumerate(SEEDS):
        start_time = datetime.now()

        # repeat over number of clients in the federation
        for num_clients in NUM_CLIENTS:

            # repeat over datasets
            for dataset_name in DATASETS:
                # repeat over split types (uniform, heterogeneous)
                for split_fn, split_args in SPLITS:

                    # run RSF experiments
                    print(f"[{datetime.now()}] RSF with {seed} seed on {dataset_name} with {split_fn} split...")
                    dfs.append(rsf_exp(
                        seed=seed,
                        dataset_name=dataset_name,
                        num_clients=num_clients,
                        split_fn=split_fn,
                        split_args=split_args,
                        rsf_params=RSF_PARAMS,
                        num_fed_trees=NUM_FED_TREES[dataset_name],
                    ))

                    # run CoxPH experiments
                    print(f"[{datetime.now()}] CoxPH with {seed} seed on {dataset_name} with {split_fn} split...")
                    dfs.append(cox_exp(
                        seed=seed,
                        dataset_name=dataset_name,
                        num_clients=num_clients,
                        split_fn=split_fn,
                        split_args=split_args,
                        epochs=CPH_EPOCHS,
                        fed_rounds=CPH_FED_ROUNDS,
                    ))

                    # run DeepSurv experiments
                    print(f"[{datetime.now()}] DeepSurv with {seed} seed on {dataset_name} with {split_fn} split...")
                    dfs.append(ds_exp(
                        seed=seed,
                        dataset_name=dataset_name,
                        num_clients=num_clients,
                        split_fn=split_fn,
                        split_args=split_args,
                        hidden_features=HIDDEN_FEATURES,
                        epochs=CPH_EPOCHS,
                        fed_rounds=CPH_FED_ROUNDS,
                    ))

                    # run Nnet-survival experiments
                    print(f"[{datetime.now()}] Nnet-S with {seed} seed on {dataset_name} with {split_fn} split...")
                    dfs.append(nnet_exp(
                        seed=seed,
                        dataset_name=dataset_name,
                        num_clients=num_clients,
                        split_fn=split_fn,
                        split_args=split_args,
                        hidden_features=HIDDEN_FEATURES,
                        epochs=CPH_EPOCHS,
                        fed_rounds=CPH_FED_ROUNDS,
                    ))

                    # run N-MTLR experiments
                    print(f"[{datetime.now()}] N-MTLR with {seed} seed on {dataset_name} with {split_fn} split...")
                    dfs.append(nmtlr_exp(
                        seed=seed,
                        dataset_name=dataset_name,
                        num_clients=num_clients,
                        split_fn=split_fn,
                        split_args=split_args,
                        hidden_features=HIDDEN_FEATURES,
                        epochs=CPH_EPOCHS,
                        fed_rounds=CPH_FED_ROUNDS,
                    ))

                    # run DeepHit experiments
                    print(f"[{datetime.now()}] DeepHit with {seed} seed on {dataset_name} with {split_fn} split...")
                    dfs.append(deephit_exp(
                        seed=seed,
                        dataset_name=dataset_name,
                        num_clients=num_clients,
                        split_fn=split_fn,
                        split_args=split_args,
                        hidden_features=HIDDEN_FEATURES,
                        alpha=ALPHA,
                        epochs=CPH_EPOCHS,
                        fed_rounds=CPH_FED_ROUNDS,
                    ))

                    # run PC-Hazard experiments
                    print(f"[{datetime.now()}] PC-Hazard with {seed} seed on {dataset_name} with {split_fn} split...")
                    dfs.append(pch_exp(
                        seed=seed,
                        dataset_name=dataset_name,
                        num_clients=num_clients,
                        split_fn=split_fn,
                        split_args=split_args,
                        hidden_features=HIDDEN_FEATURES,
                        epochs=CPH_EPOCHS,
                        fed_rounds=CPH_FED_ROUNDS,
                    ))

                    # log results as csv
                    exp_name = datetime.now().strftime("%y%m%d_%H%M%S_exps.csv")
                    df = pd.concat(dfs)
                    df.to_csv(os.path.join(log_dir, exp_name))

        print(f"[{datetime.now()}] Run #{i+1}/{len(SEEDS)} completed. Took {datetime.now() - start_time}.")


if __name__ == "__main__":
    run_exps(log_dir="./logs")
