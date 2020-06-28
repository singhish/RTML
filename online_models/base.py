from abc import ABC, abstractmethod
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error


class OnlineBase(ABC):
    """
    Base class for all online models.
    """

    def __init__(self,
                 history_length: int,
                 forecast_length: int,
                 delay: int,
                 intervals: int,
                 timesteps: int):
        """
        Constructor.

        :param history_length: number of past observations to be fed into the model for training
        :param epochs: number of epochs to train model for
        :param forecast_length: number of timesteps into the future for model to predict at
        :param delay: number of successive timesteps at which a prediction is made
        :param intervals: number of training intervals to divide dataset into while training to compute Local RMSE
        :param timesteps: total number of timesteps to train model for
        """

        # Delay must be less than or equal to forecast length
        if delay > forecast_length:
            raise ValueError('Delay must be less than or equal to the forecast length!')

        # Save training args
        self._history_length = history_length
        self._forecast_length = forecast_length
        self._delay = delay
        self._timesteps = timesteps
        self._intervals = intervals

        # Initialize Pandas DataFrames for loss tracking
        self._ts_label = 'Timestep'
        self._interval_label = 'Interval'
        self._obs_label = 'Observed'
        self._pred_label = 'Predicted'
        self._rmse_label = 'Cumulative RMSE'
        self._local_rmse_label = 'Local RMSE'

        self._obs_df = pd.DataFrame(columns=[self._ts_label, self._obs_label])
        self._pred_df = pd.DataFrame(columns=[self._ts_label, self._pred_label])
        self._loss_df = pd.DataFrame(
            columns=[self._ts_label, self._interval_label, self._rmse_label, self._local_rmse_label])

        # Initialize state
        self._buffer = []
        self._iteration = 0
        self._interval = 0
        self._rmse = 0
        self._local_rmse = 0

    def to_df(self) -> pd.DataFrame:
        """
        Exports the DataFrames keeping track of observations, predictions, and loss values to a Pandas DataFrame.
        :return: a Pandas DataFrame containing observed, predicted, and loss values at each timestep
        """
        return pd.merge_ordered(
            self._obs_df,
            pd.merge_ordered(
                self._pred_df,
                self._loss_df,
                on=self._ts_label,
                how='outer'
            ),
            on=self._ts_label,
            how='outer'
        )

    def advance_iteration(self, obs: float) -> (str, float, float):
        """
        Encapsulates the online training algorithm. Delegates prediction to subclasses via the 'protected'
        `_make_prediction` abstract method.
        :param obs: an observation from a time series obtained from an iteration procedure (e.g. using a for-loop)
        :return: if at the end of a training interval within the dataset, the proportion of the way through the training
            sample and the current rmse; otherwise a None, None tuple
        """
        self._obs_df.loc[len(self._obs_df)] = [self._iteration, obs]
        self._buffer.append(obs)

        if len(self._buffer) == self._history_length + self._forecast_length:
            pred = self._make_prediction()
            self._pred_df.loc[len(self._pred_df)] = [int(self._iteration + self._forecast_length), pred]
            self._update_loss()
            self._buffer = self._buffer[(self._delay + 1):]

        self._iteration += 1
        if self._iteration >= int((self._interval + 1) * (self._timesteps / self._intervals)):
            ret = (
                f'{self._interval / self._intervals}_{(self._interval + 1) / self._intervals}',
                self._local_rmse,
                self._rmse
            )
            self._interval += 1
            return ret

        return None, None, None

    def _update_loss(self):
        """
        Calculates the current RMSE within the current training interval by aligning the observation and prediction
        DataFrames.
        """
        synced_df = pd.merge_ordered(
            self._obs_df,
            self._pred_df,
            on=self._ts_label,
            how='inner'
        )

        inst_synced_df = synced_df.query(
            f'{int((self._interval / self._intervals) * self._timesteps)}'
            f'<= {self._ts_label}'
            f'< {int(((self._interval + 1) / self._intervals) * self._timesteps)}'
        )

        if not (synced_df.empty and inst_synced_df.empty):
            self._rmse = sqrt(
                mean_squared_error(synced_df[self._obs_label].values, synced_df[self._pred_label].values))

            self._local_rmse = sqrt(
                mean_squared_error(inst_synced_df[self._obs_label].values, inst_synced_df[self._pred_label].values))

            self._loss_df.loc[len(self._loss_df)] = [
                self._iteration,
                f'{self._interval / self._intervals}_{(self._interval + 1) / self._intervals}',
                self._rmse,
                self._local_rmse
            ]

    @abstractmethod
    def _make_prediction(self) -> float:
        """
        Makes a prediction using the data currently in the buffer. A 'pure virtual', 'protected' method to be
        implemented by subclasses, as the way this is done depends from model to model.
        :return: the model's prediction using the data in the buffer
        """
        pass
