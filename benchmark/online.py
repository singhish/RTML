import sys; sys.path.append('.'); sys.path.append('..')
import argparse
import os
import pandas as pd
from benchmark.helpers import resample_time_series
from online_models import OnlineMLP, OnlineLSTM

## Constants
FILE = os.path.dirname(__file__)

## Load command line arguments
parser = argparse.ArgumentParser(description='Online Model Benchmark')

# Select model (mlp or lstm)
parser.add_argument('-m', '--model', type=str, default='mlp')

# Select training configuration
parser.add_argument('-l', '--history-length', type=int, default=10)
parser.add_argument('-u', '--units', type=int, default=[10], nargs='*')
parser.add_argument('-e', '--epochs', type=int, default=1)
parser.add_argument('-f', '--forecast-length', type=int, default=1)
parser.add_argument('-r', '--resample-factor', type=float, default=.2)
parser.add_argument('-d', '--delay', type=int, default=0)

# Select dataset configuration
parser.add_argument('--s', type=int, default=1)
parser.add_argument('--std', type=int, default=1)
parser.add_argument('--dataset-size', type=float, default=5.0)
parser.add_argument('--intervals', type=int, default=50)

# Other options
parser.add_argument('--save', action='store_true')

args = parser.parse_args()

## Load and resample dataset
dataset = open(os.path.join(FILE, f'../data/{args.s}S_{args.std}STD.csv'), 'r')
series, sample_rate = resample_time_series(
    pd.read_csv(dataset).query(f'Time <= {args.dataset_size}')[['Time', 'Observation']],
    args.resample_factor
)

## Initialize model
model = None
if args.model == 'mlp':
    model = OnlineMLP(
        args.history_length,
        args.units,
        args.epochs,
        args.forecast_length,
        args.delay,
        args.intervals,
        series.shape[0]
    )
elif args.model == 'lstm':
    model = OnlineLSTM(
        args.history_length,
        args.epochs,
        args.forecast_length,
        args.delay,
        args.intervals,
        series.shape[0]
    )
else:
    raise ValueError(f'Model not found: {args.model}. Must be `mlp` or `lstm`.')

## Execute training
## Log the RMSE at end of each of the number of intervals specified for the specified training configuration
print(f'> Benchmarking online {str.upper(args.model)} with input_size={args.history_length}, '
      f'{f"units={args.units}, " if args.model == "mlp" else ""}'
      f'epochs={args.epochs}, and forecast_length={args.forecast_length}\n'
      f'> at {args.intervals} loss calculation intervals and {args.dataset_size}s of training data\n'
      f'> from {args.s}S_{args.std}STD.csv resampled to {int(sample_rate)} Hz.',
      file=sys.stderr)

for _, _, accel in series.itertuples():
    interval, local_rmse, rmse = model.advance_iteration(accel)
    if rmse:
        if isinstance(model, OnlineMLP):
            print(f'{args.s},{args.std},{int(sample_rate)},{args.history_length},{args.units},{args.epochs},'
                  f'{args.forecast_length},{interval},{local_rmse},{rmse}')
        elif isinstance(model, OnlineLSTM):
            print(f'{args.s},{args.std},{int(sample_rate)},{args.history_length},{args.epochs},{args.forecast_length},'
                  f'{interval},{local_rmse},{rmse}')

if args.save:
    model.to_df().to_csv(f'online-{args.model}-predictions.csv')
