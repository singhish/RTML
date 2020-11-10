import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


class MetricsAdapter:
    def __init__(self, df: pd.DataFrame):
        assert 'Observed' in df.columns and 'Predicted' in df.columns
        self._df = df
        self._observed = self._df.dropna()['Observed'].values
        self._predicted = self._df.dropna()['Predicted'].values

    def __getattr__(self, item):
        return self._df.item

    def overall_mae(self):
        return mean_absolute_error(self._observed, self._predicted)

    def overall_rmse(self):
        return np.sqrt(mean_squared_error(self._observed, self._predicted))

    def rolling_mae(self, window_size, label):
        return self._rolling_error(window_size, label, strategy=mean_absolute_error)

    def rolling_rmse(self, window_size, label):
        return self._rolling_error(window_size, label, strategy=lambda x, y: np.sqrt(mean_squared_error(x, y)))

    # time response assurance criterion
    # metric source: https://link.springer.com/chapter/10.1007/978-1-4419-9834-7_79
    def trac(self):
        numerator = np.dot(self._observed, self._predicted) ** 2
        denominator = np.dot(self._observed, self._observed) * np.dot(self._predicted, self._predicted)
        return numerator / denominator

    def _rolling_error(self, window_size, label, strategy=mean_absolute_error):
        assert window_size >= 1

        rolled = np.array([])
        for i in range(self._df.dropna().shape[0]):
            if i <= window_size:
                rolled = np.append(rolled, np.nan)
            else:
                rolled = np.append(
                    rolled,
                    strategy(self._observed[(i - window_size):i], self._predicted[(i - window_size):i])
                )

        return pd.DataFrame({'Time': self._df.dropna()['Time'].values, label: rolled})
