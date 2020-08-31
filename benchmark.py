from argparse import ArgumentParser
import os
from utils.data import read_lvm
from models import OnlineMLP


parser = ArgumentParser(description='Specifies configuration for online MLP model.')
parser.add_argument('--history-length', type=int, default=10)
parser.add_argument('--forecast-length', type=int, default=5)
parser.add_argument('--hidden-layers', type=int, default=[10], nargs='*')
parser.add_argument('--epochs-per-sample', type=int, default=1)
args = parser.parse_args()

dataset_names = [f for f in os.listdir('data') if not f.endswith('.pkl')]
print(dataset_names)
for d in dataset_names:
    dataset = read_lvm(os.path.abspath(f'data/{d}'))

    sample_rate = round(dataset.shape[0] / dataset.iloc[-1]['Time'], 2)

    model = OnlineMLP(**vars(args))

    for _, time, sample in dataset.itertuples():
        pred_time = time + (args.forecast_length / sample_rate)
        pred = model.update(sample)
        print(
            f'{", ".join(str(x) for x in vars(args).values())}, ',  # model configuration
            f'{round(sample_rate, 2)}, ',                           # sample rate of dataset
            f'{round(time, 6)}, ',                                  # current time
            f'{sample}, ',                                          # current sample
            f'{round(pred_time, 6)}, ',                             # time being predicted at
            pred                                                    # model prediction
        )
