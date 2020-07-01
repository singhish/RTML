# RTML

Keras-based wrapper classes for DNNs geared toward making real-time prediction on non-linear time series data.
Eventually intended to be deployed onto FPGA architectures for real-time state estimation applications.

## Getting Started

To install requirements in a `conda`-like environment, clone this repository and run
```shell script
pip install -r requirements.txt
```

To launch a training session, run

```shell script
python benchmark/online.py
```

By default, this will train a multilayer perceptron neural network online with the following configuration
```
input dimension = 10
hidden layers = 1 with 10 neurons
epochs per new observation = 1
optimizer = adam
loss = mse
```
that predicts 1 timestep into the future on 5 seconds of experimentally generated time series data consisting of 1
sinusoid, a standard deviation of 1, and a resample down to 330 Hz.

## Usage

### Command Line Interface

The command line arguments for `online.py` are as follows:
* `-m`/`--model`: the online model to use. Can be `mlp` or `lstm`. Default: `mlp`
* `-l`/`--history-length`: the number of past observations to be fed into the model as training data. Depending on the
model, this will correspond to either the `input_dim` of the MLP, or the `input_shape` of the LSTM (passed in as
`(history_length, 1)`). Default: `10`
* `-u`/`--units`: a series of integers specifying the number of neurons in each `Dense` hidden layer of the MLP.
Specifying `-u 30 20 10` would sequentially create three `Dense` hidden layers of sizes 30, 20, and 10. For now, each
hidden layer is hardcoded to use `relu` as its activation function. As the LSTM implemented in this project is a simple
LSTM without any stacking, this argument should not be specified if `lstm` is selected for `model`. Default: `10`
* `-e`/`--epochs`: number of epochs to train the model at each timestep. Default: `1`
* `-f`/`--forecast-length`: number of timesteps into the future to predict at. Default: `1`
* `d`/`--delay`: number of timesteps between predictions. A `delay` of `2` means the model lets 2 timesteps pass before
making a prediction. Default: `0`
* `--s`: selects the time series from the `data/` folder composed of `s` sinusoids. Valid values are `1`, `2`, and `3`.
Default: `1`
* `--std`: selects the time series from the `data/` folder with `std` standard deviations. Valid values are `1`, `5`,
and `10`. Default: `1`
* `-r`/`--resample-factor`: fraction of the original sample rate of the training dataset to resample to. Default: `.2`
* `--save`: if specified, saves the model's predictions to a `csv` file.

### Output

By default, `online.py` produces lines of output in the following `csv` format
```csv
{s},{std},{sample rate},{history length},{units},{epochs},{forecast length},{local rmse},{cumulative rmse}
```
until the model has fully traversed the training data. The `units` column is omitted if the LSTM model is used.

### Jupyter Notebook
For a demo on how to use the classes in the `online_models` package in your own project, refer to the [notebook](
/notebooks/Online_Training_Demo.ipynb) in this repository.

## Acknowledgements

This project is [funded by the National Science Foundation](
https://www.nsf.gov/awardsearch/showAward?AWD_ID=1937535&HistoricalAwards=false) as part of the University of South
Carolina's [Real-Time Machine Learning initiative](
https://www.cse.sc.edu/news/dr-bakos-receives-nsf-grant-award-real-time-machine-learning). It is also a significant
refactor of the now-deprecated [OnlineMLP](https://github.com/singhish/OnlineMLP). 
