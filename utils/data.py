import lvm_read
import pandas as pd


def read_lvm(path):
    raw_data = lvm_read.read(path)
    dataset = pd.DataFrame(
        raw_data[0]['data'],
        columns=[c.strip() for c in raw_data[0]['Channel names'] if c]
    )
    dataset = dataset[[c for c in dataset.columns if c == 'X_Value' or 'Accel' in c]]
    return dataset.rename({'X_Value': 'Time', dataset.columns[-1]: 'Acceleration'}, axis='columns')
