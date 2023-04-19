from typing import List

import numpy as np
import pandas as pd
import torch
from sksurv.ensemble import RandomSurvivalForest
from sksurv.functions import avg_fn
from sksurv.metrics import integrated_brier_score
from datetime import datetime

from splits import uniform_split, label_skew_split, quantity_skew_split
from utils import evaluate_surv_fns, prepare_data_splits

SPLIT_FNS = {
    "uniform": uniform_split,
    "label_skew": label_skew_split,
    "quantity_skew": quantity_skew_split,
}


def train_rsf(params, X, y):
    """
    Trains an RSF model.
    """
    rsf = RandomSurvivalForest(**params)
    rsf.fit(X, y)
    return rsf


def fed_train_rsf(num_clients, num_trees, client_data, val_client_data, client_models, uniform_prob):
    """
    Returns a Federated RSF model built on top of several client models.
    """
    tot_samples = sum([len(Xc) for Xc, _ in client_data])
    prob = [len(Xc) / tot_samples for Xc, _ in client_data]
    client_indices = np.random.choice([j for j in range(num_clients)], num_trees, p=prob)
    trees_per_client = np.array([np.sum(client_indices == j) for j in range(num_clients)])

    # error correction if the sampled number of trees exceeds the ones in client's model;
    # basically, if the fed model asks for more trees than the ones in the client's model,
    # the excess trees are sampled at random from other clients with available trees.
    act_trees_per_client = np.array([len(m.estimators_) for m in client_models])
    diff = act_trees_per_client - trees_per_client
    while (diff < 0).any():
        client_neg_idx = np.random.choice(np.where(diff < 0)[0])
        client_pos_idx = np.random.choice(np.where(diff > 0)[0])
        trees_per_client[client_neg_idx] -= 1
        trees_per_client[client_pos_idx] += 1
        diff = act_trees_per_client - trees_per_client

    fed_trees = []

    if uniform_prob:
        for rsf, n in zip(client_models, trees_per_client):
            fed_trees.extend(np.random.choice(rsf.estimators_, n))
    else:
        tree_scores = []
        for rsf, (_, yc_t), (Xc_v, yc_v) in zip(client_models, client_data, val_client_data):
            train_times = np.sort(np.unique(yc_t["time"]))
            val_times = np.sort(np.unique(yc_v["time"]))[:-1]
            max_time = np.min([train_times[-1], val_times[-1]])
            times = np.linspace(np.min(yc_v["time"]), max_time, num=100)
            tree_survs = [t.predict_survival_function(Xc_v) for t in rsf.estimators_]
            tree_preds = [np.asarray([fn(times) for fn in survs]) for survs in tree_survs]
            tree_ibs = [integrated_brier_score(yc_t, yc_v, preds, times) for preds in tree_preds]
            tree_ibs = np.array(tree_ibs)
            sim = 1 / tree_ibs
            sim = (sim - np.min(sim)) / (np.max(sim) - np.min(sim))
            sim /= np.sum(sim)
            tree_scores.append(sim)

        for rsf, n, prob in zip(client_models, trees_per_client, tree_scores):
            fed_trees.extend(np.random.choice(rsf.estimators_, n, p=prob))

    fed_model = RandomSurvivalForest()
    fed_model.estimators_ = fed_trees
    return fed_model


def find_best_fed_result(num_fed_trees, y_train, X_val, y_val, X_test, y_test, client_data, val_client_data, iso_models,
                         uniform_prob):
    best = -1000, None, 0
    for nt in num_fed_trees:
        fed_model = fed_train_rsf(len(client_data), nt, client_data, val_client_data, iso_models, uniform_prob)
        fed_c_idx, fed_c_idx_ipcw, fed_ibs = eval_rsf(fed_model, y_train, y_val, X_val)
        if (fed_c_idx_ipcw - 2 * fed_ibs) > best[0]:
            best = fed_c_idx_ipcw - 2 * fed_ibs, fed_model, nt
    _, best_model, nt = best
    fed_c_idx, fed_c_idx_ipcw, fed_ibs = eval_rsf(best_model, y_train, y_test, X_test)
    return nt, fed_c_idx, fed_c_idx_ipcw, fed_ibs


def eval_rsf(model, y_train, y_test, X_test):
    """
    Evaluates a RSF model returning concordance index, concordance index (IPCW), and brier score.
    """
    tree_survs = [t.predict_survival_function(X_test) for t in model.estimators_]
    survs = [avg_fn([ts[i] for ts in tree_survs]) for i in range(len(X_test))]
    c_idx, c_idx_ipcw, ibs = evaluate_surv_fns(y_train, y_test, survs)
    return c_idx, c_idx_ipcw, ibs


def rsf_exp(
        seed: int,
        dataset_name: str,
        num_clients: int,
        split_fn: str,
        split_args: dict,
        rsf_params: dict,
        num_fed_trees: List[int],
) -> pd.DataFrame:

    # set random seeds
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # obtain train/val/test splits
    print(f"[{datetime.now()}] Preparing data...")
    train_data, val_data, test_data, client_data, val_client_data = \
        prepare_data_splits(dataset_name, num_clients, split_fn, split_args)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # train/test global model
    print(f"[{datetime.now()}] Training global model...")
    glob_model = train_rsf(rsf_params[dataset_name], X_train, y_train)
    glob_c_idx, glob_c_idx_ipcw, glob_ibs = eval_rsf(glob_model, y_train, y_test, X_test)

    # train/test isolated models
    print(f"[{datetime.now()}] Training isolated models...")
    iso_models = [train_rsf(rsf_params[dataset_name], X, y) for X, y in client_data]
    iso_metrics = np.array([eval_rsf(m, y_train, y_test, X_test) for m in iso_models])
    iso_c_idx = np.mean(iso_metrics[:, 0]).item()
    iso_c_idx_ipcw = np.mean(iso_metrics[:, 1]).item()
    iso_ibs = np.mean(iso_metrics[:, 2]).item()

    # train/test federated model
    print(f"[{datetime.now()}] Training federated model...")
    nt, fed_c_idx, fed_c_idx_ipcw, fed_ibs = find_best_fed_result(
        num_fed_trees, y_train, X_val, y_val, X_test, y_test, client_data, val_client_data, iso_models, True)

    print(f"[{datetime.now()}] Training federated model with IBS sampling...")
    nt_ibs, fed_c_idx_ibs, fed_c_idx_ipcw_ibs, fed_ibs_ibs = find_best_fed_result(
        num_fed_trees, y_train, X_val, y_val, X_test, y_test, client_data, val_client_data, iso_models, False)

    # return results as pandas dataframe
    print(f"[{datetime.now()}] Logging...")
    res = {
        "seed": [seed, seed],
        "dataset": [dataset_name, dataset_name],
        "num_clients": [num_clients, num_clients],
        "split": [split_fn, split_fn],
        "split_args": [split_args, split_args],
        "model": ["rsf", "rsf_ibs"],
        "model_params": [{"params": glob_model.get_params(), "num_fed_trees": nt},
                         {"params": glob_model.get_params(), "num_fed_trees": nt_ibs}],
        "glob_c_idx": [glob_c_idx, glob_c_idx],
        "glob_c_idx_ipcw": [glob_c_idx_ipcw, glob_c_idx_ipcw],
        "glob_ibs": [glob_ibs, glob_ibs],
        "iso_c_idx": [iso_c_idx, iso_c_idx],
        "iso_c_idx_ipcw": [iso_c_idx_ipcw, iso_c_idx_ipcw],
        "iso_ibs": [iso_ibs, iso_ibs],
        "fed_c_idx": [fed_c_idx, fed_c_idx_ibs],
        "fed_c_idx_ipcw": [fed_c_idx_ipcw, fed_c_idx_ipcw_ibs],
        "fed_ibs": [fed_ibs, fed_ibs_ibs],
    }
    return pd.DataFrame(res)
