# ThermoMPNN
ThermoMPNN is a graph neural network (GNN) trained using transfer learning to predict changes in stability for protein point mutants.

![ThermoMPNN Scheme](./images/SVG/thermoMPNN_scheme.svg)

For details on ThermoMPNN training and methodology, please see the accompanying [paper](https://www.biorxiv.org/content/10.1101/2023.07.27.550881v1). 

## Colab Implementation
For a user-friendly version of ThermoMPNN requiring no installation, use this [Colab notebook](https://colab.research.google.com/drive/1OcT4eYwzxUFNlHNPk9_5uvxGNMVg3CFA#scrollTo=i06A5VI142NT).

## Installation
To install ThermoMPNN, first clone this repository
```
git clone https://github.com/Kuhlman-Lab/ThermoMPNN.git
```
Then use the file ```environment.yaml``` install the necessary python dependencies (I recommend using mamba for convenience):
```
mamba env create -f environment.yaml
```
This will create a conda environment called ```thermoMPNN```.

## Inference
There are a few different ways to run inference with ThermoMPNN all located in the ```analysis``` directory.

### From a PDB
The simplest way is to use the ```custom_inference.py``` script to pass a custom PDB to ThermoMPNN for site-saturation mutagenesis.

### From a CSV and many PDBs
For larger batches of predictions, it is recommended to set up a **CustomDataset** object by inheriting from the **ddgBenchDataset** class in the ```datasets.py``` file, then add this dataset to the ```SSM.py``` script to get aggregated predictions for the whole dataset.

### For benchmarking purposes
The ```thermompnn_benchmarking.py``` is set up to score different models on a **CustomDataset** object or one of the datasets used in this study. An example inference SLURM script is provided at ```examples/inference.sh```.

## Training
The main training script is ```train_thermompnn.py```. To set up a training run, you must write a ```config.yaml``` file (example provided) to specify model hyperparameters. You also must provide a ```local.yaml``` file to tell ThermoMPNN where to find your data. These files serve as experiment logs as well.

Training ThermoMPNN requires the use of a GPU. On a small dataset (<5000 data points), training takes <30s per epoch, while on a mega-scale dataset (>200,000 data points), it takes 8-12min per epoch (on a single V100 GPU). An example training SLURM script is provided at ```examples/train.sh```.

### Splits and Model Weights
For the purpose of replication and future benchmarking, the dataset splits used in this study are included as ```.pkl``` files under the ```dataset_splits/``` directory.

ThermoMPNN model weights can be found in the ```models/``` directory. The following model weights are provided:
```
- thermoMPNN_default.pt (best ThermoMPNN model trained on Megascale training dataset)
```
