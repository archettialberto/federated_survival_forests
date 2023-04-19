import functools
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def legal_split(split_fn):
    @functools.wraps(split_fn)
    def legal_split_fn(
            num_clients: int,
            X: np.ndarray,
            y: np.ndarray,
            min_samples: int = 2,
            **kwargs
    ):

        # preliminary checks
        assert num_clients > 0, f"The number of clients must be greater than zero. Found {num_clients} instead."
        assert num_clients <= len(X), f"The number of clients must be <= the number of samples. " \
                                      f"Found {num_clients}, {len(X)} instead."
        assert len(X) == len(y), f"X and y must have the same length. Found {len(X)} and {len(y)} instead."
        original_size = len(X)
        client_size = 0

        # perform split
        client_data = split_fn(num_clients, X, y, min_samples=min_samples, **kwargs)

        # post-split checks
        for _X, _y in client_data:
            assert len(_X) == len(_y), f"Found a different number of inputs and outputs ({len(_X)}, {len(_y)})."
            assert sum(_y["event"]) > 0, "Each client dataset must contain at least one event."
            assert len(_X) >= min_samples, f"Each client dataset must have at least {min_samples} samples."
            client_size += len(_X)
        assert original_size == client_size, f"The original number of samples ({original_size}) differs from the " \
                                             f"client samples ({client_size})."
        return client_data

    return legal_split_fn


def _extract_min_samples(num_clients, X, y, min_samples):
    assert min_samples >= 2, f"Each dataset should contain at least two samples. Found {min_samples} instead."
    assert len(X) >= num_clients * min_samples, f"Not enough data for the clients."
    client_data = []
    for j in range(num_clients):
        X, Xj, y, yj = train_test_split(X, y, stratify=y["event"], test_size=min_samples)
        client_data.append((Xj, yj))
        assert len(Xj) >= min_samples, f"Each client dataset must contain at least {min_samples} samples."
    return X, y, client_data


@legal_split
def uniform_split(
        num_clients: int,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = 2
) -> List[Tuple[np.ndarray, np.ndarray]]:

    X, y, client_data = _extract_min_samples(num_clients, X, y, min_samples)
    cd_size = [len(X) // num_clients] * num_clients
    while len(X) - sum(cd_size) > 0:
        cd_size[np.random.randint(num_clients)] += 1
    for j in range(num_clients - 1):
        X, _X, y, _y = train_test_split(X, y, test_size=cd_size[j], stratify=y["event"])
        Xj, yj = client_data[j]
        Xj, yj = np.concatenate([Xj, _X]), np.concatenate([yj, _y])
        client_data[j] = (Xj, yj)
    Xj, yj = client_data[-1]
    Xj, yj = np.concatenate([Xj, X]), np.concatenate([yj, y])
    client_data[-1] = (Xj, yj)
    np.random.shuffle(client_data)
    return client_data


@legal_split
def quantity_skew_split(
        num_clients: int,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = 2,
        alpha: float = 1.0
) -> List[Tuple[np.ndarray, np.ndarray]]:

    # extract the minimum number of samples per client
    X, y, client_data = _extract_min_samples(num_clients, X, y, min_samples)

    # Dirichlet distribution evaluation
    d = np.random.dirichlet([alpha] * num_clients, 1)[0]
    d = np.round(d / d.sum() * len(X))

    # rounding error correction
    conv_error = d.sum() - len(X)
    while conv_error > 0:
        rnd_client = np.random.choice(np.where(d > 0)[0])
        d[rnd_client] -= 1
        conv_error -= 1
    while conv_error < 0:
        rnd_client = np.random.randint(num_clients)
        d[rnd_client] += 1
        conv_error += 1
    assert d.sum() == len(X), f"{d.sum()} != {len(X)}"
    d = d.astype(int)

    # build client datasets
    indices = [i for i in range(len(X))]
    for j in range(num_clients):
        idx = list(np.random.choice(indices, size=d[j], replace=False))
        Xj, yj = client_data[j]
        Xj, yj = np.concatenate([Xj, X[idx]]), np.concatenate([yj, y[idx]])
        client_data[j] = (Xj, yj)
    return client_data


@legal_split
def label_skew_split(
        num_clients: int,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = 2,
        alpha: float = 1.0,
        num_bins: int = 10
) -> List[Tuple[np.ndarray, np.ndarray]]:

    assert alpha > 0.0, f"Alpha must be greater than 0. Found {alpha} instead."
    assert num_bins > 0, f"The number of bins must be greater than 0. Found {num_bins} instead."

    # extract a minimum number of samples for each client
    X, y, client_data = _extract_min_samples(num_clients, X, y, min_samples)

    # helper variables
    bins = np.linspace(y["time"].min(), y["time"].max(), num_bins + 1)
    y_classes = np.digitize(y["time"], bins[1:], right=True)
    classes = np.sort(np.unique(y_classes))
    num_samples_per_class = [len(np.where(y_classes == classes[c])[0]) for c in range(len(classes))]

    # Dirichlet distribution evaluation
    d = np.random.dirichlet(np.ones(num_clients) * alpha, len(classes))
    for c in range(len(classes)):
        d[c] = np.round(d[c] / d[c].sum() * num_samples_per_class[c])

        # rounding error correction
        conv_error = d[c].sum() - num_samples_per_class[c]
        while conv_error > 0:
            rnd_client = np.random.choice(np.where(d[c] > 0)[0])
            d[c, rnd_client] -= 1
            conv_error -= 1
        while conv_error < 0:
            rnd_client = np.random.randint(num_clients)
            d[c, rnd_client] += 1
            conv_error += 1

        # check if correction worked
        assert d[c].sum() == num_samples_per_class[c], f"{d[c].sum()} != {num_samples_per_class[c]}"
    d = d.astype(int)

    # build client datasets
    class_idx = [np.where(y_classes == classes[c])[0] for c in range(len(classes))]
    for j in range(num_clients):
        idx = []
        for c in range(len(classes)):
            if d[c, j] > 0:
                idx.extend(list(np.random.choice(class_idx[c], size=d[c, j], replace=False)))
        Xj, yj = client_data[j]
        Xj, yj = np.concatenate([Xj, X[idx]]), np.concatenate([yj, y[idx]])
        client_data[j] = (Xj, yj)
    return client_data
