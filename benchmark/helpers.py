import pandas as pd
from scipy.signal import resample
import numpy as np


def resample_time_series(series: pd.DataFrame, resample_factor: float) -> (pd.DataFrame, float):
    """
    Resamples a Pandas DataFrame containing a time series stored as two columns with labels 'Time' and 'Observation'.

    :param series: Pandas DataFrame representing a time series
    :param resample_factor: the factor of the time series dataset's original sample rate to resample to
    :return: the resampled time series as a DataFrame, and the new sample rate as a float
    """
    n_samples = int(resample_factor * series.shape[0])

    time = series['Time'].values
    obss = series['Observation'].values

    resampled_obss, resampled_time = resample(obss, n_samples, time)

    resampled_series = pd.DataFrame(data=np.array((resampled_time, resampled_obss)).T, columns=['Time', 'Observation'])
    new_sample_rate = n_samples / resampled_series['Time'].values[-1]

    return resampled_series, new_sample_rate
