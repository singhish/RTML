from argparse import ArgumentParser
import os
from utils.data import read_lvm
import pandas as pd
from models import OnlineMLP


parser = ArgumentParser(description='Specifies configuration for online MLP model.')
parser.add_argument('--history-length', type=int, default=10)
parser.add_argument('--forecast-length', type=int, default=5)
parser.add_argument('--hidden-layers', type=int, default=[10], nargs='*')
parser.add_argument('--epochs-per-sample', type=int, default=1)
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

# load all files from datasets folder (for now only one file in this folder)
#dataset_names = [f for f in os.listdir('datasets') if not f.endswith('.pkl')]
dataset_names = ['Ivol_Acc_Load_data3_w3_w2_50per_STD_downBy256.lvm']
for d in dataset_names:
    dataset = read_lvm(os.path.abspath(f'datasets/{d}'))
    sample_rate = round(dataset.shape[0] / dataset.iloc[-1]['Time'], 2)
    results = pd.DataFrame(columns=[  # maintains model's predictions for export if --save is specified
        'Timestep',
        'Time',
        'Observed',
        'Pred Timestep',
        'Pred Time',
        'Predicted'
    ])

    # initialize online MLP
    model = OnlineMLP(args.history_length, args.forecast_length, args.hidden_layers, args.epochs_per_sample)

    # iterate through each sample in dataset
    for i, time, sample in dataset.itertuples():
        pred_time = time + (args.forecast_length / sample_rate)
        pred = model.update(sample)  # update model with new sample

        # logging
        print(
            f'{", ".join(str(x) for x in vars(args).values() if not type(x) == bool)}, ',  # model configuration
            f'{i}, ',                                                                      # current timestep
            f'{round(time, 6)}, ',                                                         # current time
            f'{sample}, ',                                                                 # current sample
            f'{i + args.forecast_length}, ',                                               # timestep being predicted at
            f'{round(pred_time, 6)}, ',                                                    # time being predicted at
            pred                                                                           # model prediction
        )

        results.loc[results.size + 1] = [i, time, sample, i + args.forecast_length, pred_time, pred]

    if args.save:
        # export results in a 3-column format consisting of a timepoint, the observed value of the time series at the
        # timepoint, and the value predicted by the model at the timepoint
        results = results[['Timestep', 'Time', 'Observed']].merge(
            results[['Pred Timestep', 'Pred Time', 'Predicted']].rename(columns={'Pred Timestep': 'Timestep'}),
            how='outer'
        )
        results['Time'] = results['Time'].fillna(results['Pred Time'])
        results = results[['Time', 'Observed', 'Predicted']]
        results.to_csv(f'results_{d}.csv')
