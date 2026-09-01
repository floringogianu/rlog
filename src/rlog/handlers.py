"""Extra Handlers that can handle structured LogRecords."""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from .exception_handling import print_fancy_err

__all__ = ["PickleHandler", "TensorboardHandler"]


class PickleHandler(logging.Handler):
    """A Handler that writes `logging.LogRecord`s to pickle files.
    TODO: Check the performance?
    """

    def __init__(self, log_dir, timestamp=None):
        logging.Handler.__init__(self)
        self.log_dir = log_dir
        if timestamp is None:
            self.timestamp = int(datetime.timestamp(datetime.now()))
        else:
            self.timestamp = int(timestamp)

    def emit(self, record: logging.LogRecord) -> None:
        data = self._maybe_load(record.name)
        if hasattr(record, "metrics"):
            self._add_scalars(record, data)
        else:
            # TODO: not sure why is this here
            self._add_text(record, data)
        self._save(record.name, data)

    def _maybe_load(self, logger_name: str) -> dict[str, Any]:
        """For commodity reasons we create a pickle file per logger.
        If a pickle exists it will be loaded in memory.
        """
        file_name = logger_name.replace(".", "_")
        file_path = Path(self.log_dir, f"{self.timestamp}_{file_name}.pkl")

        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def _save(self, logger_name, data):
        file_name = logger_name.replace(".", "_")
        file_path = Path(self.log_dir, f"{self.timestamp}_{file_name}.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def _add_text(self, record, data):
        data.setdefault("text", []).append(record.getMessage())

    def _add_scalars(self, record: logging.LogRecord, data):
        try:
            step = record.metrics["step"]
        except KeyError as err:
            print_fancy_err(
                err,
                issue="PickeHandler expects a LogRecord.metrics with a `step` field",
                fix="Make sure your call to rlog.trace() includes the `step` kw",
            )
            raise

        for k, v in record.metrics.items():
            if k not in ("step", "extra"):
                if isinstance(v, list):
                    step_ = step - len(v)
                    entries = [
                        {"step": step_ + i, "value": v_, "time": record.created}
                        for i, v_ in enumerate(v)
                    ]
                else:
                    entries = [{"step": step, "value": v, "time": record.created}]

                if k in data:
                    data[k].extend(entries)
                else:
                    data[k] = entries


class TensorboardHandler(logging.Handler):
    """A Handler using the Tensorboard SummaryWriter."""

    def __init__(self, log_dir):
        logging.Handler.__init__(self)
        self.log_dir = log_dir
        # Imported here, not at module scope: `import rlog` must not require
        # torch, and must not print anything when torch is absent.
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as err:
            print_fancy_err(
                err,
                issue="TensorboardHandler requires torch and tensorboard",
                fix="pip install 'rlog[tensorboard]'",
            )
            raise

        self.writer = SummaryWriter(log_dir)

    def emit(self, record: logging.LogRecord) -> None:
        if hasattr(record, "metrics"):
            self._add_key_value_items(record)
        else:
            self._add_text(record)

    def _add_text(self, record):
        tag = f"{record.name.replace('.', '/')}/stdout"
        self.writer.add_text(tag, record.getMessage())

    def _add_key_value_items(self, record):
        rec_name = record.name.replace(".", "/")
        try:
            step = record.metrics["step"]
        except KeyError as err:
            print_fancy_err(
                err,
                issue="TensorboardHandler wants LogRecord.metrics with a `step` field",
                fix="Make sure your call to rlog.trace() includes the `step` kw",
            )
            raise

        tb_types = {k: "scalar" for k, v in record.metrics.items() if k != "extra"}
        if "extra" in record.metrics:
            for metric_name, tb_type in record.metrics["extra"]["tb_types"].items():
                tb_types[metric_name] = tb_type

        for metric, value in record.metrics.items():
            if metric not in ("step", "extra"):
                tag = f"{rec_name}/{metric}"

                if tb_types[metric] == "scalar":
                    self._add_scalars(tag, step, value)
                elif tb_types[metric] == "histogram":
                    self._add_histogram(tag, step, value)
                else:
                    raise ValueError("There should be a Tensorboard type.")

    def _add_scalars(self, tag, step, value):
        if isinstance(value, list):
            step = step - len(value)
            for i, v_ in enumerate(value):
                self.writer.add_scalar(tag, v_, global_step=step + i)
        else:
            self.writer.add_scalar(tag, value, global_step=step)

    def _add_histogram(self, tag, step, values):
        try:
            import torch as th
        except ImportError as err:
            print_fancy_err(
                err,
                issue="Histograms require 'torch'",
                fix="pip install 'rlog[tensorboard]'",
            )
            raise
        values = th.tensor(values)
        self.writer.add_histogram(tag, values, global_step=step)

    def close(self):
        self.writer.close()
