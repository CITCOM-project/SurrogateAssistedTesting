# Causal Surrogates Replication Package

### Prerequisites

Python >= 3.9 (Executed using Python 3.9.18)

Anaconda/Miniconda (Python environment management)

GPU CUDA Capabilities (only for Pylot evaluation)

nvidia-docker2 (only for Pylot evaluation)

oref0 v0.7.1 accessible on the command line (https://github.com/openaps/oref0)

It is recommended that this is executed on Ubuntu 22.04

### Installation

```
conda create --name causal-surrogates python=3.9
conda activate causal-surrogates
pip install -r requirements.txt
```

You will then need to create the following empty directories
```
{base_path}/carla/{datasets,outputs,outputs_ensemble,outputs_RS}
{base_path}/apsdigitaltwin/{outputs_3,outputs_2_ensemble,outputs_RS}
```

## Pylot - Proof of Concept

Found in directory `{base_path}/carla`

### Set Up

Please follow the README in the original study (https://github.com/ADS-Testing/SAMOTA/tree/debug) in order to generate the docker container for CARLA. Please ensure the README from the branch `debug` is followed as it is most up to date.

### Evaluation

To execute the evaluation:
```
python generate_data.py     # Generate the pseudo-random datasets

python carla.py             # Evaluate the causal surrogate approach
python carla_ensemble.py    # Evaluate the associative approach
python carla_random.py      # Evaluate random search

python rq1.py               # Analyse the results and generate the figure for RQ1
```

## oref0 - Full evaluation

Found in directory `{base_path}/apsdigitaltwin`

### Data Generation

Data generation is in line with the methodology outlined in https://dx.doi.org/10.2139/ssrn.4732706. Our evaluation used n=183 of the OpenAPS Data Commons (https://openaps.org/outcomes/data-commons/).

Note. This dataset is open source but we cannot distribute it due to a data management agreement with OpenAPS.

### Evaluation

To execute the evaluation:
```
python apsdigitaltwin_hybrid.py     # Evaluate the causal surrogate approach
python apsdigitaltwin_ensemble.py   # Evaluate the associative approach
python apsdigitaltwin_random.py     # Evaluate random search

python rq2.py                       # Analyse the results and generate the figure for RQ2
python rq3.py                       # Analyse the results and generate the figure for RQ3
```