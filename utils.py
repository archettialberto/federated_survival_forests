from collections import OrderedDict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sksurv.metrics import integrated_brier_score, concordance_index_censored, concordance_index_ipcw

from data import get_dataframe, preprocess_dataframe
from splits import uniform_split, label_skew_split, quantity_skew_split

SPLIT_FNS = {
    "uniform": uniform_split,
    "label_skew": label_skew_split,
    "quantity_skew": quantity_skew_split,
}


def to_pycox_labels(y):
    return y["time"].copy(), y["event"].copy()


def unpack_metrics(_, d):
    return d["c_idx_test"], d["c_idx_ipcw_test"], d["ibs_test"]


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.net.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.net.load_state_dict(state_dict, strict=True)


def evaluate_surv_fns(y_train, y_test, survs):
    sorted_train_times = np.sort(np.unique(y_train["time"]))
    sorted_test_times = np.sort(np.unique(y_test["time"]))
    times = np.linspace(
        start=max(sorted_train_times[1], sorted_test_times[1]),
        stop=min(sorted_train_times[-2], sorted_test_times[-2]),
        num=100
    )
    preds = np.array([fn(times) for fn in survs], dtype=np.float32)
    preds = np.nan_to_num(preds, nan=0.5)
    preds[preds < 0.0] = 0.0
    preds[preds > 1.0] = 1.0
    preds = preds * (1.0 - 1e-8) + 1e-8
    risks = -np.log(preds)
    cum_risks = np.sum(risks, axis=1)
    c_index = concordance_index_censored(y_test["event"], y_test["time"], cum_risks)[0]
    c_index_ipcw = concordance_index_ipcw(y_train, y_test, cum_risks)[0]
    ibs = integrated_brier_score(y_train, y_test, preds, times)
    return c_index, c_index_ipcw, ibs


def prepare_data_splits(dataset_name, num_clients, split_fn, split_args):
    X_train, X_test, y_train, y_test = preprocess_dataframe(get_dataframe(dataset_name))
    tot_client_data = SPLIT_FNS[split_fn](num_clients, X_train, y_train, **split_args)

    client_data = []
    val_client_data = []
    for Xc, yc in tot_client_data:
        Xc_t, Xc_v, yc_t, yc_v = train_test_split(Xc, yc, stratify=yc["event"], test_size=0.2)
        client_data.append((Xc_t, yc_t))
        val_client_data.append((Xc_v, yc_v))

    X_val = np.concatenate([Xc for Xc, _ in val_client_data], axis=0)
    y_val = np.concatenate([yc for _, yc in val_client_data], axis=0)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), client_data, val_client_data
