from abc import ABC, abstractmethod
import numpy as np


# base class to be used for all online models
class OnlineModelBase(ABC):
    def __init__(self,
                 history_length,
                 forecast_length,
                 verbose=False):
        self._history_length = history_length
        self._forecast_length = forecast_length
        self._verbose = verbose

        self._buffer = np.array([])
        self._samples_processed = 0

    def update(self, sample):
        self._buffer = np.append(self._buffer, sample)

        if self._buffer.size == self._history_length + self._forecast_length:
            pred = self._make_prediction()
            self._buffer = self._buffer[1:]
            ret = pred
        else:
            ret = np.nan

        self._samples_processed += 1
        if self._verbose:
            print(f'\rSamples processed: {self._samples_processed}', end='')

        return ret

    @abstractmethod
    def _make_prediction(self):
        pass
