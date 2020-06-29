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
                 timesteps: int,
                 local_rmse_precision: int,
                 verbose: bool):
        """
        Constructor.

        :param history_length: number of past observations to be fed into the model for training
        :param epochs: number of epochs to train model for at each timestep
        :param forecast_length: number of timesteps into the future for model to predict at
        :param delay: number of timesteps between predictions
        :param local_rmse_precision: parameter used to calculate the number of previous timesteps used in calculating
            Local RMSE
        :param timesteps: total number of timesteps to train model for
        :param verbose: if true, will log current training timestep
        """

        # Delay must be less than or equal to forecast length
        if delay > forecast_length:
            raise ValueError('Delay must be less than or equal to the forecast length!')

        # Save training args
        self._history_length = history_length
        self._forecast_length = forecast_length
        self._delay = delay
        self._timesteps = timesteps
        self._rmse_precision = local_rmse_precision
        self._verbose = verbose

        # Initialize Pandas DataFrames for loss tracking
        self._timestep_label = 'Timestep'
        self._obs_label = 'Observed'
        self._pred_label = 'Predicted'
        self._cumul_rmse_label = 'Cumulative RMSE'
        self._local_rmse_label = 'Local RMSE'

        self._obs_df = pd.DataFrame(columns=[self._timestep_label, self._obs_label])
        self._pred_df = pd.DataFrame(columns=[self._timestep_label, self._pred_label])
        self._loss_df = pd.DataFrame(
            columns=[self._timestep_label, self._cumul_rmse_label, self._local_rmse_label])

        # Initialize state
        self._buffer = []
        self._timestep = 0
        self._rmses_logged = 0
        self._cumul_rmse = 0
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
                on=self._timestep_label,
                how='outer'
            ),
            on=self._timestep_label,
            how='outer'
        )

    def advance_iteration(self, obs: float) -> (str, float, float):
        """
        Encapsulates the online training algorithm. Delegates prediction to subclasses via the 'protected'
        `_make_prediction` abstract method.
        :param obs: an observation from a time series obtained from an iteration procedure (e.g. using a for-loop)
        :return: if at the end of a window, the proportion of the way through the training data and the current rmse
            values; otherwise a None, None, None tuple
        """
        if self._verbose:
            print(f'\rTimestep: {self._timestep + 1}/{self._timesteps}', end='')

        self._obs_df.loc[len(self._obs_df)] = [self._timestep, obs]
        self._buffer.append(obs)

        if len(self._buffer) == self._history_length + self._forecast_length:
            pred = self._make_prediction()
            self._pred_df.loc[len(self._pred_df)] = [int(self._timestep + self._forecast_length), pred]
            self._update_loss()
            self._buffer = self._buffer[(self._delay + 1):]

        self._timestep += 1
        if self._timestep >= int((self._rmses_logged + 1) * (self._timesteps / self._rmse_precision)):
            ret = (
                f'{self._rmses_logged / self._rmse_precision}_{(self._rmses_logged + 1) / self._rmse_precision}',
                self._local_rmse,
                self._cumul_rmse
            )
            self._rmses_logged += 1
            return ret

        return None, None, None

    def _update_loss(self):
        """
        Calculates the current RMSE within the current training window by aligning the observation and prediction
        DataFrames.
        """
        synced_df = pd.merge_ordered(
            self._obs_df,
            self._pred_df,
            on=self._timestep_label,
            how='inner'
        )

        local_synced_df = synced_df.tail(int(self._timesteps / self._rmse_precision))

        if not (synced_df.empty and local_synced_df.empty):
            self._cumul_rmse = sqrt(
                mean_squared_error(synced_df[self._obs_label].values, synced_df[self._pred_label].values))

            self._local_rmse = sqrt(
                mean_squared_error(local_synced_df[self._obs_label].values, local_synced_df[self._pred_label].values))

            self._loss_df.loc[len(self._loss_df)] = [
                self._timestep,
                self._cumul_rmse,
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
