from .base import OnlineModelBase
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class OnlineMLP(OnlineModelBase):
    def __init__(self,
                 history_length,
                 forecast_length,
                 hidden_layers,
                 epochs_per_sample,
                 verbose=False):
        super(OnlineMLP, self).__init__(history_length, forecast_length, verbose)

        self._epochs_per_sample = epochs_per_sample

        self._model = Sequential()
        prev_h = history_length
        for h in hidden_layers:
            self._model.add(Dense(h, activation='relu', input_dim=prev_h))
            prev_h = h
        self._model.add(Dense(1))
        self._model.compile(optimizer='adam', loss='mse')

    def _make_prediction(self):
        inputs = self._buffer[:self._history_length].reshape(1, self._history_length)
        target = self._buffer[-1].reshape(1, 1)
        self._model.fit(inputs, target, epochs=self._epochs_per_sample, verbose=0)
        return self._model.predict(self._buffer[-self._history_length:].reshape(1, self._history_length)).item()
