# RTML

Keras-based wrapper classes for DNNs geared toward making real-time prediction on non-linear time series data.
Eventually intended to be deployed onto FPGA architectures for real-time state estimation applications.

## Getting Started

First, to install dependencies in a `conda`-like environment, clone this repository and run
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
that predicts 5 timesteps into the future on experimentally generated vibration data.

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

### Output

By default, `benchmark.py` produces lines of output in the following `csv` format
```csv
{history length},{forecast length},{hidden layers},{epochs per sample},{sample rate of dataset},{current time},{current sample},{time being predicted at},{prediction}
```
until the model has fully traversed the training data. The `utils.metrics` module provides a Pandas adapter class called
`MetricsAdapter` that allows analyses of the data outputted by this script.

## Acknowledgements

This project is [funded by the National Science Foundation](
https://www.nsf.gov/awardsearch/showAward?AWD_ID=1937535&HistoricalAwards=false) as part of the University of South
Carolina's [Real-Time Machine Learning initiative](
https://www.cse.sc.edu/news/dr-bakos-receives-nsf-grant-award-real-time-machine-learning). It is also a significant
refactor of the now-deprecated [OnlineMLP](https://github.com/singhish/OnlineMLP). 
