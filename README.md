# Federated Survival Forests

This repo contains the source code of the experiments presented in the paper "Federated Survival Forests" [1].
Federated Survival Forest (FedSurF) is a novel federated learning algorithm for survival analysis. It relies on 
Random Survival Forests (RSF) [2] to build a federated ensemble of survival trees. Tree selection occurs randomly
with a probability proportional to the local performance of each tree. The algorithm is implemented using a custom 
version of [scikit-survival](https://github.com/archettialberto/scikit-survival) that supports averaging of survival 
functions. Baseline methods relying on neural networks are implemented using [PyCox](https://github.com/havakv/pycox).

Experiments are run on standard dataset for survival analysis with federated splits induced with the algorithms
described in [3], which allow testing on heterogeneous federations with different label splits for each client.

For more information, please take a look at
* [Federated Survival Forests](https://arxiv.org/abs/2302.02807) [1] for the FedSurF algorithm, code, experiments and results description.
* [Heterogeneous Datasets for Federated Survival Analysis Simulation](https://arxiv.org/abs/2301.12166) [3] for the federated dataset generation algorithms.


## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/archettialberto/federated_survival_forests
```
Install the Docker image:
```bash
docker build -t fedsurf/fedsurf:latest .
```

## 🛠️ Usage

Run ```docker compose``` to start the experiments:
```bash
docker compose run fedsurf-experiments
```

The results will be saved in the ```logs``` directory.

NOTE: ```docker compose``` mounts the current directory in the ```/exp``` directory of the container, so you can edit the 
code in ```exps.py``` and run it without rebuilding the image.



## 📕 Bibtex Citation
```
@inproceedings{archetti2023federated,
    author    = {Archetti, Alberto and Matteucci, Matteo},
    title     = {{Federated} {Survival} {Forests}},
    booktitle = {2023 International Joint Conference on Neural Networks (IJCNN2023)},
    year      = {2023},
    publisher = {IEEE (in press)}
}
```

## 📚 References

[1] A. Archetti and M. Matteucci, “Federated Survival Forests.” In press at 2023 International Joint Conference on Neural Networks (IJCNN2023), Feb. 2023.

[2] H. Ishwaran, U. B. Kogalur, E. H. Blackstone, and M. S. Lauer, “Random survival forests,” Annals of Applied Statistics, vol. 2, no. 3, Sep. 2008, doi: 10.1214/08-AOAS169.

[3] A. Archetti, E. Lomurno, F. Lattari, A. Martin, and M. Matteucci, “Heterogeneous Datasets for Federated Survival Analysis Simulation,” in Companion of the 2023 ACM/SPEC International Conference on Performance Engineering, Coimbra Portugal: ACM, Apr. 2023, pp. 173–180. doi: 10.1145/3578245.3584935.


