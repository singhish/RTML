# RTML

Keras-based wrapper classes for DNNs geared toward making real-time predictions on non-linear time series data.
Eventually intended to be deployed onto FPGA architectures for real-time state estimation applications.

## Introduction

The aim of this project is to create a library of online machine learning models for making predictions on real-time
time series data streams. Online machine learning is a machine learning approach that involves training on samples
sequentially as they become available. Presently, a working online implementation of a multilayer perception (MLP) is
included in this project; however, the aim is to do same for other machine learning models typically used in time
series analysis.

## Getting Started

First, to install dependencies in a `conda` environment, clone this repository and run
```shell script
conda env create -n rtml -f environment.yml
conda activate rtml
```

To launch a training session, run

```shell script
python benchmark.py
```

By default, this will train a multilayer perceptron neural network online with the following configuration
```
input dimension = 10
hidden layers = 1 with 10 neurons
epochs per new observation = 1
optimizer = adam
loss = mse
```
that predicts 5 timesteps into the future on the experimentally generated vibration signal in the `datasets/` folder.

## Usage

### Command Line Interface

The command line arguments for `benchmark.py` are as follows:
* `--history-length`: the number of past observations to be fed into the model as training data. This will correspond the
`input_dim` of the MLP. Default: `10`
* `--forecast-length`: number of timesteps into the future to predict at. Default: `5`
* `--hidden-layers`: a series of integers specifying the number of neurons in each `Dense` hidden layer of the MLP.
Specifying `-u 30 20 10` would sequentially create three `Dense` hidden layers of sizes 30, 20, and 10. For now, each
hidden layer is hardcoded to use `relu` as its activation function. Default: `10`
* `--epochs-per-sample`: number of epochs to train the model at each timestep. Default: `1`
* `--save`: if specified, saves results of training session to a `.csv` file.

### Output

By default, `benchmark.py` produces lines of output in the following `csv` format
```csv
{history length},{forecast length},{hidden layers},{epochs per sample},{current timestep},{current sample},{timestep being predicted at},{prediction}
```
until the model has fully traversed the training data. The `utils.metrics` module provides a Pandas adapter class called
`MetricsAdapter` that allows analyses of the data outputted by this script. In the future, a Jupyter Notebook will be
included in this repository to demonstrate how to work with this class.

## Acknowledgements

This project is [funded by the National Science Foundation](
https://www.nsf.gov/awardsearch/showAward?AWD_ID=1937535&HistoricalAwards=false) as part of the University of South
Carolina's [Real-Time Machine Learning initiative](
https://www.cse.sc.edu/news/dr-bakos-receives-nsf-grant-award-real-time-machine-learning). It is also a significant
refactor of the now-deprecated [OnlineMLP](https://github.com/singhish/OnlineMLP). 
