import numpy as np
import pandas as pd
import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
from pycox.models import DeepHitSingle
import torchtuples as tt
from sksurv.functions import StepFunction
from datetime import datetime
import flwr as fl

from utils import to_pycox_labels, get_parameters, set_parameters, evaluate_surv_fns, unpack_metrics, \
    prepare_data_splits


class Net(torch.nn.Module):
    """
    Defines the neural network for DeepHit models.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_features=hidden_features, out_features=hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_features=hidden_features, out_features=out_features)
        )

    def forward(self, x):
        return self.seq(x)


def init_model(in_features, hidden_features, labtrans, alpha):
    """
    Initializes a Pycox DeepHit model.
    """
    return DeepHitSingle(
        Net(in_features, hidden_features, labtrans.out_features),
        torch.optim.Adam,
        alpha=alpha,
        duration_index=labtrans.cuts
    )


def train_deephit(X_train, y_train, X_val, y_val, hidden_features, labtrans, alpha, msg=""):
    """
    Trains a DeepHit model.
    """
    _y_train = to_pycox_labels(y_train)
    __y_train = labtrans.transform(*_y_train)
    _y_val = to_pycox_labels(y_val)
    __y_val = labtrans.transform(*_y_val)
    val_data = (X_val, __y_val)
    model = init_model(X_train.shape[1], hidden_features, labtrans, alpha)
    callbacks = [tt.callbacks.EarlyStopping(patience=100)]
    log = model.fit(X_train, __y_train, epochs=1000, val_data=val_data, verbose=False, callbacks=callbacks)
    # plt.title(f"[{datetime.now()}] {msg}")
    # log.plot()
    return model


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, Xc, yc, hidden_features, labtrans, alpha):
        self.cid = cid
        self.model = init_model(Xc.shape[1], hidden_features, labtrans, alpha)
        self.Xc, self.yc = Xc, labtrans.transform(*to_pycox_labels(yc))

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.fit(self.Xc, self.yc, epochs=1, verbose=False)
        return get_parameters(self.model), len(self.Xc), {}


def get_client_fn(client_data, hidden_features, labtrans, alpha):
    """
    Returns the client initialization function.
    """
    def client_fn(cid) -> FlowerClient:
        X, y = client_data[int(cid)]
        return FlowerClient(cid, X, y, hidden_features, labtrans, alpha)
    return client_fn


def get_evaluate_fn(X_train, y_train, X_val, y_val, X_test, y_test, compute_baseline, hidden_features, labtrans, alpha):
    """
    Returns the flwr evaluation function.
    """
    _y_train = to_pycox_labels(y_train)
    _y_test = to_pycox_labels(y_test)

    def evaluate(server_round, parameters, config):
        model = init_model(X_train.shape[1], hidden_features, labtrans, alpha)
        set_parameters(model, parameters)

        if compute_baseline:
            model.compute_baseline_hazards(X_train, _y_train)

        surv_df_val = model.predict_surv_df(X_val)
        survs_val = [StepFunction(x=np.array(surv_df_val.index), y=surv_df_val[i].to_numpy())
                     for i in range(len(X_val))]
        c_index_val, c_index_ipcw_val, ibs_val = evaluate_surv_fns(y_train, y_val, survs_val)

        surv_df_test = model.predict_surv_df(X_test)
        survs_test = [StepFunction(x=np.array(surv_df_test.index), y=surv_df_test[i].to_numpy())
                      for i in range(len(X_test))]
        c_index_test, c_index_ipcw_test, ibs_test = evaluate_surv_fns(y_train, y_test, survs_test)

        return 0.0, {
            "c_idx_val": c_index_val,
            "c_idx_ipcw_val": c_index_ipcw_val,
            "ibs_val": ibs_val,
            "c_idx_test": c_index_test,
            "c_idx_ipcw_test": c_index_ipcw_test,
            "ibs_test": ibs_test,
        }

    return evaluate


def fed_train_deephit(eval_fn, in_features, hidden_features, labtrans, alpha, client_data, fed_rounds, dataset):
    params = get_parameters(init_model(in_features, hidden_features, labtrans, alpha))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=len(client_data),
        min_available_clients=len(client_data),
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        evaluate_fn=eval_fn,
    )

    fed_logs = fl.simulation.start_simulation(
        client_fn=get_client_fn(client_data, hidden_features, labtrans, alpha),
        num_clients=len(client_data),
        config=fl.server.ServerConfig(num_rounds=fed_rounds),
        strategy=strategy,
    )

    # plot results
    # plt.title(f"[{datetime.now()}] {dataset}, federated")
    # sns.lineplot(x=[i for i in range(fed_rounds)], y=[j[1] for j in fed_logs.metrics_centralized["c_idx_val"]][1:])
    # sns.lineplot(x=[i for i in range(fed_rounds)], y=[j[1] for j in fed_logs.metrics_centralized["c_idx_ipcw_val"]][1:])
    # sns.lineplot(x=[i for i in range(fed_rounds)], y=[j[1] for j in fed_logs.metrics_centralized["ibs_val"]][1:])
    # plt.show()

    # find the best param config on the validation set
    metrics_len = len(fed_logs.metrics_centralized["c_idx_val"])
    c_idx_ipcw_val_list = [fed_logs.metrics_centralized["c_idx_ipcw_val"][i][1] for i in range(metrics_len)]
    ibs_val_list = [fed_logs.metrics_centralized["ibs_val"][i][1] for i in range(metrics_len)]
    cumulative_list = [y - 2 * z for y, z in zip(c_idx_ipcw_val_list, ibs_val_list)]
    max_idx = cumulative_list.index(max(cumulative_list))

    # return the results on the test set
    c_idx_test_list = [fed_logs.metrics_centralized["c_idx_test"][i][1] for i in range(metrics_len)]
    c_idx_ipcw_test_list = [fed_logs.metrics_centralized["c_idx_ipcw_test"][i][1] for i in range(metrics_len)]
    ibs_test_list = [fed_logs.metrics_centralized["ibs_test"][i][1] for i in range(metrics_len)]

    return c_idx_test_list[max_idx], c_idx_ipcw_test_list[max_idx], ibs_test_list[max_idx]


def deephit_exp(
        seed: int,
        dataset_name: str,
        num_clients: int,
        split_fn: str,
        split_args: dict,
        hidden_features: int,
        alpha: float,
        epochs: int,
        fed_rounds: int,
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

    # preprocess labels
    labtrans = DeepHitSingle.label_transform(len(np.unique(y_train["time"])) // 10)
    labtrans.fit(*to_pycox_labels(y_train))

    eval_fn = get_evaluate_fn(
        X_train, y_train, X_val, y_val, X_test, y_test, compute_baseline=False,
        hidden_features=hidden_features, labtrans=labtrans, alpha=alpha
    )

    # train/test global model
    print(f"[{datetime.now()}] Training global model...")
    glob_model = train_deephit(X_train, y_train, X_val, y_val, hidden_features, labtrans, alpha,
                               msg=f"{dataset_name} - global")
    glob_c_idx, glob_c_idx_ipcw, glob_ibs = unpack_metrics(*eval_fn(None, get_parameters(glob_model), None))

    # train/test isolated models
    print(f"[{datetime.now()}] Training isolated models...")
    iso_models = []
    for i, ((Xc_train, yc_train), (Xc_val, yc_val)) in enumerate(zip(client_data, val_client_data)):
        iso_model = train_deephit(Xc_train, yc_train, Xc_val, yc_val, hidden_features, labtrans, alpha,
                                  msg=f"{dataset_name} - client {i}")
        iso_models.append(iso_model)
    iso_metrics = np.array([unpack_metrics(*eval_fn(None, get_parameters(m), None)) for m in iso_models])
    iso_c_idx = np.mean(iso_metrics[:, 0]).item()
    iso_c_idx_ipcw = np.mean(iso_metrics[:, 1]).item()
    iso_ibs = np.mean(iso_metrics[:, 2]).item()

    # train/test federated model
    print(f"[{datetime.now()}] Training federated model...")
    fed_c_idx, fed_c_idx_ipcw, fed_ibs = \
        fed_train_deephit(eval_fn, X_train.shape[1], hidden_features, labtrans, alpha, client_data, fed_rounds, dataset_name)

    # return results as pandas dataframe
    print(f"[{datetime.now()}] Logging...")
    res = {
        "seed": [seed],
        "dataset": [dataset_name],
        "num_clients": [num_clients],
        "split": [split_fn],
        "split_args": [split_args],
        "model": ["deephit"],
        "model_params": [{
            "epochs": epochs,
        }],
        "glob_c_idx": [glob_c_idx],
        "glob_c_idx_ipcw": [glob_c_idx_ipcw],
        "glob_ibs": [glob_ibs],
        "iso_c_idx": [iso_c_idx],
        "iso_c_idx_ipcw": [iso_c_idx_ipcw],
        "iso_ibs": [iso_ibs],
        "fed_c_idx": [fed_c_idx],
        "fed_c_idx_ipcw": [fed_c_idx_ipcw],
        "fed_ibs": [fed_ibs],
    }
    return pd.DataFrame(res)
