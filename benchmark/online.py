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
parser.add_argument('-d', '--delay', type=int, default=0)

# Select data configuration
parser.add_argument('--s', type=int, default=1)
parser.add_argument('--std', type=int, default=1)
parser.add_argument('-r', '--resample-factor', type=float, default=.2)
parser.add_argument('--rmses-to-log', type=int, default=50)

# Other options
parser.add_argument('--save', action='store_true')

args = parser.parse_args()

## Load and resample dataset
series = open(os.path.join(FILE, f'../data/{args.s}S_{args.std}STD.csv'), 'r')
resampled, sample_rate = resample_time_series(
    pd.read_csv(series)[['Time', 'Observation']],
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
        resampled.shape[0],
        local_rmse_precision=args.rmses_to_log
    )
elif args.model == 'lstm':
    model = OnlineLSTM(
        args.history_length,
        args.epochs,
        args.forecast_length,
        args.delay,
        resampled.shape[0],
        local_rmse_precision=args.rmses_to_log
    )
else:
    raise ValueError(f'Model not found: {args.model}. Must be `mlp` or `lstm`.')

## Execute training
## Log the RMSE at end of each of the number of windows specified for the specified training configuration
print(f'> Benchmarking online {str.upper(args.model)} with history_length={args.history_length}, '
      f'{f"units={args.units}, " if args.model == "mlp" else ""}'
      f'epochs={args.epochs}, and forecast_length={args.forecast_length}\n'
      f'> using the dataset {args.s}S_{args.std}STD.csv resampled to {sample_rate} Hz.',
      file=sys.stderr)

for _, _, obs in resampled.itertuples():
    window, local_rmse, cumul_rmse = model.advance_iteration(obs)
    if local_rmse:
        if isinstance(model, OnlineMLP):
            print(f'{args.s},{args.std},{sample_rate},{args.history_length},{args.units},{args.epochs},'
                  f'{args.forecast_length},{window},{local_rmse},{cumul_rmse}')
        elif isinstance(model, OnlineLSTM):
            print(f'{args.s},{args.std},{sample_rate},{args.history_length},{args.epochs},{args.forecast_length},'
                  f'{window},{local_rmse},{cumul_rmse}')

if args.save:
    model.to_df().to_csv(f'online-{args.model}-predictions.csv')
