"""RLog definition and configuration."""

import logging
import logging.config
from typing import Literal

from .exception_handling import print_fancy_err
from .filters import MaxLevelFilter
from .formatters import SummaryFormatter
from .handlers import PickleHandler, TensorboardHandler
from .metrics import (
    Accumulator,
    AvgMetric,
    EpisodicMetric,
    EWMAvgMetric,
    FPSMetric,
    MaxMetric,
    SumMetric,
    ValueMetric,
)

__all__ = [
    "getLogger",
    "getRootLogger",
    "init",
    "debug",
    "error",
    "exception",
    "info",
    "trace",
    "warning",
    "addMetrics",
    "put",
    "summarize",
    "traceAndLog",
    "reset",
    "PickleHandler",
    "TensorboardHandler",
    "Accumulator",
    "AvgMetric",
    "EpisodicMetric",
    "EWMAvgMetric",
    "FPSMetric",
    "MaxMetric",
    "SumMetric",
    "ValueMetric",
]


ROOT = None


logging.TRACE = 15
logging.addLevelName(logging.TRACE, "TRACE")


class RLogger(logging.Logger):
    def __init__(self, log_name=None):
        logging.Logger.__init__(self, log_name)

        self.accumulator = None
        self.put, self.reset, self.summarize, self.fmt = None, None, None, None
        self._xtra_kws = ("exc_info", "extra", "stack_info")  # small helper

    def trace(self, *args, **kws):
        # We break with the API for now.
        # And yes, logger takes its '*args' as 'args'.
        if args:
            self._log(logging.TRACE, args[0], args[1:], **kws)
        elif kws:
            _xtra_kws = {k: v for k, v in kws.items() if k in self._xtra_kws}
            self._log(logging.TRACE, kws, args, **_xtra_kws)
        else:
            raise TypeError("Call trace with either a message or a dict-like object.")

    def addMetrics(self, *metrics):
        # TODO: Not really happy about how adding metrics changes the
        # interface of RLogger, need to thing about something else.

        if self.accumulator is None:
            # configure the Accumulator
            self.accumulator = Accumulator(*metrics)
            # and delegate its methods
            self.put = self.accumulator.trace
            self.reset = self.accumulator.reset
            self.summarize = self.accumulator.summarize
        else:
            # just add more metrics
            self.accumulator.add_metrics(*metrics)

    def traceAndLog(self, step, with_reset=True):
        """Calls both trace and summarize on the `Accumulator.summarize()`
        result. Then it calls reset.
        """
        if self.fmt is None:
            self.fmt = SummaryFormatter()

        summary = self.summarize()
        self.info(self.fmt(step=step, **summary))
        self.trace(step=step, **summary)
        if with_reset:
            self.reset()
        return summary


class TimeFilter(logging.Filter):
    """If there is another type of object that processes records, it might be used
    instead of this.
    """

    def __init__(self, datefmt="%H:%M:%S"):
        super().__init__()
        self._datefmt = datefmt

    def filter(self, record):
        total_seconds = record.relativeCreated / 1000.0

        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        record.relative = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return True


LevelValues = Literal[20, 15]


def _build_config(
    name: str,
    path: str | None = None,
    level: LevelValues = logging.INFO,
    pickle: bool = True,
    tensorboard: bool = False,
    relative_time: bool = True,
    datefmt: str = "%H:%M:%S",
    pkl_timestamp_fmt: str | None = None,
    prefix: str | None = None,
):
    ts = "{relative}" if relative_time else "{asctime}"
    fmt = (prefix or "") + ts + " [{levelname[0]}] {name}: {message}"
    time_filter = ["time"] if relative_time else []

    cfg = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"std": {"format": fmt, "datefmt": datefmt, "style": "{"}},
        "filters": {
            "time": {"()": TimeFilter, "datefmt": datefmt},
            "below_warning": {"()": MaxLevelFilter, "level": logging.WARNING},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "level": level,
                "formatter": "std",
                "filters": ["below_warning"] + time_filter,
            },
            "stderr": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "level": max(level, logging.WARNING),
                "formatter": "std",
                "filters": time_filter,
            },
        },
        "loggers": {name: {"level": logging.TRACE, "handlers": ["stdout", "stderr"]}},
    }
    if path:
        cfg["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "filename": f"{path}/log.log",
            "level": level,
            "formatter": "std",
        }
        if pickle:
            cfg["handlers"]["pickle"] = {
                "()": PickleHandler,
                "log_dir": path,
                "timestamp": pkl_timestamp_fmt,
                "level": logging.TRACE,
            }
        if tensorboard:
            cfg["handlers"]["tensorboard"] = {
                "()": TensorboardHandler,
                "log_dir": path,
                "level": logging.TRACE,
            }
        cfg["loggers"][name]["handlers"] = list(cfg["handlers"])
    return cfg


def init(
    name: str,
    path: str | None = None,
    level: LevelValues = logging.INFO,
    pickle: bool = True,
    tensorboard: bool = False,
    relative_time: bool = True,
    datefmt: str = "%H:%M:%S",
    pkl_timestamp_fmt: str | None = None,
    prefix: str | None = None,
):
    global ROOT
    logging.setLoggerClass(RLogger)  # still imperative
    logging.config.dictConfig(
        _build_config(
            name,
            path,
            level,
            pickle,
            tensorboard,
            relative_time,
            datefmt,
            pkl_timestamp_fmt,
            prefix,
        )
    )
    ROOT = logging.getLogger(name)


def getLogger(name):
    return logging.getLogger(name)


def getRootLogger():
    return ROOT


def debug(msg, *args, **kwargs):
    ROOT.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    ROOT.info(msg, *args, **kwargs)


def trace(*args, **kwargs):
    ROOT.trace(*args, **kwargs)


def error(msg, *args, **kwargs):
    ROOT.error(msg, *args, **kwargs)


def exception(msg, *args, exc_info=True, **kwargs):
    ROOT.error(msg, *args, exc_info=exc_info, **kwargs)


def warning(msg, *args, **kwargs):
    ROOT.warning(msg, *args, **kwargs)


def addMetrics(*metrics):
    root = getRootLogger()
    root.addMetrics(*metrics)


def put(**kwargs):
    root = getRootLogger()
    try:
        root.put(**kwargs)
    except AttributeError as err:
        print_fancy_err(
            err,
            issue="RLog has no attribute `put` untill you add a Metric",
            fix="You do so by calling `addMetric(...)` first",
        )
        raise


def summarize():
    root = getRootLogger()
    try:
        return root.summarize()
    except AttributeError as err:
        print_fancy_err(
            err,
            issue="RLog has no attribute `summarize` untill you add a Metric",
            fix="You do so by calling `addMetric(...)` first",
        )
        raise


def traceAndLog(step, with_reset=True):
    root = getRootLogger()
    try:
        return root.traceAndLog(step, with_reset=with_reset)
    except AttributeError as err:
        print_fancy_err(
            err,
            issue="RLog has no attribute `traceAndLog` untill you add a Metric",
            fix="You do so by calling `addMetric(...)` first",
        )
        raise


def reset():
    root = getRootLogger()
    try:
        root.reset()
    except AttributeError as err:
        print_fancy_err(
            err,
            issue="RLog has no attribute `reset` untill you add a Metric",
            fix="You do so by calling `addMetric(...)` first",
        )
        raise


if __name__ == "__main__":
    pass
