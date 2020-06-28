from .base import OnlineBase
from typing import List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


class OnlineMLP(OnlineBase):
    """
    An online multi-layer perceptron neural network.
    """

    def __init__(self,
                 history_length: int,
                 units: List[int],
                 epochs: int,
                 forecast_length: int,
                 delay: int,
                 intervals: int,
                 timesteps: int,
                 activation: str = 'relu',
                 optimizer: str = 'adam'):
        """
        Constructor.

        :param history_length: number of units in MLP's input layer
        :param units: list describing number of units at each hidden layer of MLP
        :param epochs: number of epochs to train MLP for
        :param forecast_length: number of timesteps into the future for MLP to predict at
        :param delay: number of successive timesteps at which a prediction is made
        :param intervals: number of training intervals to divide dataset into while training
        :param timesteps: total number of timesteps to train MLP for
        :param activation: the activation function each neuron should use
        :param optimizer: the optimizer used to compile the model
        """

        # Initialize base class
        super(OnlineMLP, self).__init__(history_length, forecast_length, delay, intervals, timesteps)

        # Initialize epochs
        self._epochs = epochs

        # Initialize model
        self._mlp = Sequential()

        last_u = history_length
        for u in units:
            self._mlp.add(Dense(u, activation=activation, input_dim=last_u))
            last_u = u

        self._mlp.add(Dense(1))
        self._mlp.compile(optimizer=optimizer, loss='mse')

    def _make_prediction(self) -> float:
        train = np.array(self._buffer[:self._history_length]).reshape((1, self._history_length))
        target = np.array(self._buffer[-1]).reshape((1, 1))
        self._mlp.fit(train, target, epochs=self._epochs, verbose=0)
        return self._mlp.predict(
            np.array(self._buffer[self._forecast_length:]).reshape((1, self._history_length))).item()
