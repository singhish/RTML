from .base import OnlineBase
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np


class OnlineLSTM(OnlineBase):
    """
    An online LSTM neural network.
    """

    def __init__(self,
                 history_length: int,
                 epochs: int,
                 forecast_length: int,
                 delay: int,
                 intervals: int,
                 timesteps: int,
                 optimizer: str = 'adam'):
        """
        Constructor.

        :param history_length: cell size of LSTM
        :param epochs: number of epochs to train LSTM for
        :param forecast_length: number of timesteps into the future for LSTM to predict at
        :param delay: number of successive timesteps at which a prediction is made
        :param intervals: number of training intervals to divide dataset into while training
        :param timesteps: total number of timesteps to train LSTM for
        :param optimizer: the optimizer used to compile the model
        """

        # Initialize base class
        super(OnlineLSTM, self).__init__(history_length, forecast_length, delay, intervals, timesteps)

        # Initialize epochs
        self._epochs = epochs

        # Initialize model
        self._lstm = Sequential([
            LSTM(self._history_length, input_shape=(self._history_length, 1)),
            Dense(1)
        ])
        self._lstm.compile(optimizer=optimizer, loss='mse')

    def _make_prediction(self) -> float:
        train = np.array(self._buffer[:self._history_length]).reshape((1, self._history_length, 1))
        target = np.array(self._buffer[-1]).reshape((1, 1))
        self._lstm.fit(train, target, epochs=self._epochs, verbose=0)
        return self._lstm.predict(
            np.array(self._buffer[self._forecast_length:]).reshape((1, self._history_length, 1))).item()
