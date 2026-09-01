"""Extra Handlers that can handle structured LogRecords."""

import logging
import pickle
import time
from pathlib import Path
from typing import Any

from .exception_handling import print_fancy_err

__all__ = ["PickleHandler", "TensorboardHandler"]


class PickleHandler(logging.Handler):
    """A Handler that appends `logging.LogRecord`s to pickle streams.

    One file per logger.
    """

    def __init__(self, log_dir, timestamp=None):
        logging.Handler.__init__(self)
        self.log_dir = Path(log_dir)
        self.timestamp = int(timestamp) if timestamp is not None else int(time.time())
        self._files = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            frame = self._frame(record)
            stream = self._stream(record.name)
            pickle.dump(frame, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def close(self) -> None:
        for stream in self._files.values():
            stream.close()
        self._files.clear()
        logging.Handler.close(self)

    def _frame(self, record: logging.LogRecord) -> dict[str, Any]:
        if not hasattr(record, "metrics"):
            return {
                "kind": "text",
                "time": record.created,
                "level": record.levelname,
                "msg": record.getMessage(),
            }

        if "step" not in record.metrics:
            err = KeyError("step")
            print_fancy_err(
                err,
                issue="PickleHandler expects a LogRecord.metrics with a `step` field",
                fix="Make sure your call to rlog.trace() includes the `step` kw",
            )
            raise err

        return {
            "kind": "metrics",
            "time": record.created,
            **{k: v for k, v in record.metrics.items() if k != "extra"},
        }

    def _stream(self, logger_name):
        stream = self._files.get(logger_name)
        if stream is None:
            lname = logger_name.replace(".", "_")
            fname = f"{self.timestamp}_{lname}.pkl"
            stream = self._files[logger_name] = open(self.log_dir / fname, "ab")
        return stream


class TensorboardHandler(logging.Handler):
    """A Handler that writes records with the Tensorboard SummaryWriter."""

    def __init__(self, log_dir):
        logging.Handler.__init__(self)
        self.log_dir = log_dir
        try:
            import torch
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as err:
            print_fancy_err(
                err,
                issue="TensorboardHandler requires torch and tensorboard",
                fix="pip install 'rlog[tensorboard]'",
            )
            raise

        self._torch = torch
        self.writer = SummaryWriter(log_dir)
        # Learned once from the `extra` payload the Accumulator sends, then
        # kept. Only histograms need an entry; everything else is a scalar.
        self._tb_types: dict[str, str] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            prefix = record.name.replace(".", "/")
            if hasattr(record, "metrics"):
                self._write_metrics(prefix, record.metrics)
            else:
                self.writer.add_text(f"{prefix}/stdout", record.getMessage())
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def close(self) -> None:
        self.writer.close()
        logging.Handler.close(self)

    def _write_metrics(self, prefix: str, metrics: dict[str, Any]) -> None:
        if "step" not in metrics:
            err = KeyError("step")
            print_fancy_err(
                err,
                issue="TensorboardHandler expects a LogRecord.metrics with a "
                "`step` field",
                fix="Make sure your call to rlog.trace() includes the `step` kw",
            )
            raise err

        step = metrics["step"]
        self._tb_types.update(metrics.get("extra", {}).get("tb_types", {}))

        for name, value in metrics.items():
            if name in ("step", "extra"):
                continue
            self._write_one(f"{prefix}/{name}", step, value, self._tb_types.get(name))

    def _write_one(self, tag: str, step: int, value: Any, tb_type: str | None) -> None:
        if tb_type == "histogram":
            self.writer.add_histogram(tag, self._torch.tensor(value), global_step=step)
        elif isinstance(value, list):
            # a cached metric holds one value per step, up to `step`
            for value_step, v in enumerate(value, start=step - len(value)):
                self.writer.add_scalar(tag, v, global_step=value_step)
        else:
            self.writer.add_scalar(tag, value, global_step=step)
