"""Training progress logger for Gensim Word2Vec / Product2Vec."""

from __future__ import annotations

import logging
from collections import deque
from time import perf_counter

from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    """Logs training progress and estimates remaining time during training."""

    def __init__(self, n_latest: int = 5) -> None:
        """Initialize the epoch logger.

        Args:
            n_latest: Number of recent epoch durations to track for averaging.
        """
        self._start_stamp: float = 0.0
        self._current_epoch: int = 1
        self._epoch_duration: deque[float] = deque(maxlen=n_latest)

    @staticmethod
    def format_time(seconds: int) -> str:
        """Convert seconds to HH:MM:SS or MM:SS format."""
        formatted_time: str = ""

        hours: int = seconds // 3600
        if hours > 0:
            formatted_time += f"{hours:02d}:"

        minutes: int = (seconds % 3600) // 60
        formatted_time += f"{minutes:02d}:"

        seconds_left: int = seconds - hours * 3600 - minutes * 60
        formatted_time += f"{seconds_left:02d}"

        return formatted_time

    def on_epoch_begin(self, model: CallbackAny2Vec) -> None:
        """Called at the start of each epoch."""
        self._start_stamp = perf_counter()

        if self._epoch_duration:
            avg_duration: float = sum(self._epoch_duration) / len(self._epoch_duration)
            epochs_remaining: int = model.epochs - (self._current_epoch - 1)
            time_left: float = epochs_remaining * avg_duration
            msg: str = self.format_time(int(time_left))
        else:
            msg = "To be estimated"

        logging.info("Epoch #%s | Estimated time left: %s", self._current_epoch, msg)

    def on_epoch_end(self, _model: CallbackAny2Vec) -> None:
        """Called at the end of each epoch."""
        duration: float = perf_counter() - self._start_stamp
        self._epoch_duration.append(duration)
        self._current_epoch += 1
