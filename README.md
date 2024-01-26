# Federated Survival Forests

This repo contains the source code of the experiments presented in the paper "Federated Survival Forests" [1].
Federated Survival Forest (FedSurF) is a novel federated learning algorithm for survival analysis. It relies on 
Random Survival Forests (RSF) [2] to build a federated ensemble of survival trees. Tree selection occurs randomly
with a probability proportional to the local performance of each tree. The algorithm is implemented using a custom 
version of [scikit-survival](https://github.com/archettialberto/scikit-survival) that supports the averaging of survival 
functions. Baseline methods relying on neural networks are implemented using [PyCox](https://github.com/havakv/pycox).

Experiments are run on standard datasets for survival analysis with federated splits induced by the algorithms
described in [3], which allow testing on heterogeneous federations with different label splits for each client.

For more information, please take a look at
* [Federated Survival Forests](https://arxiv.org/abs/2302.02807) [1] for the FedSurF algorithm, code, experiments, and results description.
* [Heterogeneous Datasets for Federated Survival Analysis Simulation](https://arxiv.org/abs/2301.12166) [3] for the federated dataset generation algorithms.


## 🛠️ Usage

Clone the repository:
```bash
git clone https://github.com/archettialberto/federated_survival_forests
```

Run ```docker compose``` to start the experiments:
```bash
docker compose run fedsurf-experiments
```

The results will be saved in the ```logs``` directory. 
Please note that ```docker compose``` mounts the current directory in the ```/exp``` directory of the container, 
so you can edit the code in ```exps.py``` and run without rebuilding the image.

If you want to build a new Docker image, run
```bash
docker build -t aarchetti/fedsurf:latest .
```


## ⚙️ Installation Outside Docker (Not Recommended)

- Install Python 3.10
- Clone the repo with `git clone https://github.com/archettialberto/federated_survival_forests`
- Move to the repo directory with `cd federated_survival_forests`
- Create a new virtual environment with `python -m venv venv`
- Activate the environment with `source venv/bin/activate`
- Install the dependencies with `pip install -r requirements.txt`
- Clone the helper repo with `git clone https://github.com/archettialberto/scikit-survival`
- Move to the helper repo directory with `cd scikit-survival`
- Update git submodules with `git submodule update --init`
- Move back to the `federated_survival_forests` directory with `cd ..`
- Install `scikit-survival` with `pip install ./scikit-survival`
- At this point, you can run `python exps.py` without Docker


## 📕 Bibtex Citation
```
@inproceedings{archetti2023federated,
    author={Archetti, Alberto and Matteucci, Matteo},
    booktitle={2023 International Joint Conference on Neural Networks (IJCNN)}, 
    title={Federated Survival Forests}, 
    year={2023},
    pages={1-9},
    doi={10.1109/IJCNN54540.2023.10190999}
}
```

## 📚 References

[1] A. Archetti and M. Matteucci, “Federated Survival Forests.” In press at 2023 International Joint Conference on Neural Networks (IJCNN2023), Feb. 2023.

[2] H. Ishwaran, U. B. Kogalur, E. H. Blackstone, and M. S. Lauer, “Random survival forests,” Annals of Applied Statistics, vol. 2, no. 3, Sep. 2008, doi: 10.1214/08-AOAS169.

[3] A. Archetti, E. Lomurno, F. Lattari, A. Martin, and M. Matteucci, “Heterogeneous Datasets for Federated Survival Analysis Simulation,” in Companion of the 2023 ACM/SPEC International Conference on Performance Engineering, Coimbra Portugal: ACM, Apr. 2023, pp. 173–180. doi: 10.1145/3578245.3584935.


