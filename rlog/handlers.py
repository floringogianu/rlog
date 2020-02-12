""" Extra Handlers that can handle structured LogRecords.
"""
import logging
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    # TODO: the project itself should use logging without messing things out.
    print(
        "[WARN] rlog: TensorBoard is available in PyTorch-Nightly (~1.1) "
        + "and it should be available in newer version."
    )


__all__ = ["PickleHandler", "TensorboardHandler"]


class PickleHandler(logging.Handler):
    """ A Handler that writes `logging.LogRecord`s to pickle files.
    TODO: Check the performance?
    """

    def __init__(self, log_dir):
        logging.Handler.__init__(self)
        self.log_dir = log_dir
        self.timestamp = int(datetime.timestamp(datetime.now()))

    def emit(self, record):
        data = self._maybe_load(record.name)

        # these functions mutate `data`!
        if isinstance(record.msg, dict) and record.levelname == "TRACE":
            self._add_scalars(record, data)
        else:
            # TODO: need to move this in a formatter!!
            record.msg = str(record.msg)
            self._add_text(record, data)

        self._save(record.name, data)

    def _maybe_load(self, logger_name):
        """ For commodity reasons we create a pickle file per logger.
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
        if "text" in data:
            data["text"].append(record.msg)
        else:
            data["text"] = [record.msg]

    def _add_scalars(self, record, data):
        try:
            step = record.msg["step"]
        except Exception as err:
            raise (
                AttributeError(
                    "PickleHandler expects a LogRecord.msg with a "
                    + "`step` field. Rest of the Exception: "
                    + str(err)
                )
            )

        for k, v in record.msg.items():
            if k not in ("step", "extra"):
                if isinstance(v, list):
                    step = step - len(v)
                    entries = [
                        {"step": step + i, "value": v_, "time": record.created}
                        for i, v_ in enumerate(v)
                    ]
                else:
                    entries = [
                        {"step": step, "value": v, "time": record.created}
                    ]

                if k in data:
                    data[k].extend(entries)
                else:
                    data[k] = entries


class TensorboardHandler(logging.Handler):
    """ A Handler using the Tensorboard SummaryWritter.
    """

    def __init__(self, log_dir):
        logging.Handler.__init__(self)
        self.log_dir = log_dir
        try:
            self.writer = SummaryWriter(log_dir)
        except NameError as err:
            raise NameError(
                "PyTorch >= 1.1.0 is required for logging to Tensorboard."
                + str(err)
            )

    def emit(self, record):
        if isinstance(record.msg, dict) and record.levelname == "TRACE":
            self._add_key_value_items(record)
        else:
            # TODO: need to move this in a formatter!!
            record.msg = str(record.msg)
            self._add_text(record)

    def _add_text(self, record):
        rec_name = record.name.replace(".", "/")
        tag = f"{rec_name}/stdout"
        self.writer.add_text(tag, record.msg)

    def _add_key_value_items(self, record):
        rec_name = record.name.replace(".", "/")
        try:
            step = record.msg["step"]
        except Exception as err:
            raise (
                AttributeError(
                    "TensorboardHandler expects a LogRecord.msg with a "
                    + "`step` field. Rest of the Exception: "
                    + str(err)
                )
            )

        tb_types = {k: "scalar" for k, v in record.msg.items() if k != "extra"}
        if "extra" in record.msg:
            for metric_name, tb_type in record.msg["extra"]["tb_types"].items():
                tb_types[metric_name] = tb_type

        for metric, value in record.msg.items():
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
        if isinstance(values, list):
            flat = np.hstack(values)  # flatten if that's the case
            self.writer.add_histogram(tag, flat, global_step=step)
        else:
            raise ValueError("There should be a list of values...")

    def close(self):
        self.writer.close()
