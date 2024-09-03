# Enhancing Algorithm Performance Understanding through tsMorph: Generating Semi-Synthetic Time Series for Robust Forecasting Evaluation

## Overview

This repository is dedicated to implementing experiments related to the paper accepted at AEQUITAS 2024 co-located with ECAI 2024.

## How to reproduce the results

### Conda installation
In this implementation, we use the **conda environment**, an open-source package and environment management system that runs on Windows, macOS, and Linux.
On [this page](https://www.anaconda.com/products/distribution) you can find the Anaconda distribution for your specific OS.

### Setup conda environment
Now with Anaconda installed in your workspace, we will configure the environment with the Python version and respective packages needed for the experiments.

First, with the repository cloned in your workspace, create the conda env:

```console
foo@bar:~$ conda env create -f environment.yml
```

Then activate the conda env:

```console
foo@bar:~$ conda activate tsmorph
```

### Run the code
Your environment is now ready for experiments. We will start by executing the base-level results and then move on to the specific application of the object of study of this work.

Run the code base level (wait for the execution to finish).

```console
(tsmorph)foo@bar:~$ python exec_base_models.py
```

Run the main code

```console
(tsmorph)foo@bar:~$ python main.py
```

## How to access the results

The raw results are in the "./results/" folder.

This folder has two subfolders referring to the two forecasting algorithms: "./results/LSTM" and "./results/DeepAR."

The main subfolders of these experiments are:

- **cor**: correlation matrix between meta-feature and performance
- **img**: all images of meta-features and performance variation for all time series used in the experiment.








