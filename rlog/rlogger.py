"""RLog definition and configuration."""

import datetime
import logging
import sys

from .exception_handling import print_fancy_err
from .filters import MaxLevelFilter
from .formatters import SummaryFormatter
from .handlers import PickleHandler, TensorboardHandler
from .metrics import (
    Accumulator,
    AvgMetric,
    BaseMetric,
    EWMAvgMetric,
    FPSMetric,
    MaxMetric,
    SumMetric,
    ValueMetric,
)

__all__ = [
    "getLogger",
    "getRootLogger",
    "info",
    "debug",
    "warning",
    "exception",
    "error",
    "trace",
    "init",
    "addMetrics",
    "put",
    "summarize",
    "traceAndLog",
    "reset",
    "TensorboardHandler",
    "PickleHandler",
    "Accumulator",
    "BaseMetric",
    "SumMetric",
    "AvgMetric",
    "EWMAvgMetric",
    "MaxMetric",
    "ValueMetric",
    "FPSMetric",
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
        duration = datetime.datetime.utcfromtimestamp(record.relativeCreated / 1000.0)
        record.relative = duration.strftime(self._datefmt)
        return True


def init(  # pylint: disable=bad-continuation
    name,
    path=None,
    level=logging.INFO,
    pickle=True,
    tensorboard=False,
    relative_time=False,
    datefmt="%H:%M:%S",
    timestamp=None,
    prefix=None,
):
    """Configures a global RLogger."""
    global ROOT

    logging.setLoggerClass(RLogger)
    ROOT = logging.getLogger(name)
    ROOT.setLevel(logging.TRACE)

    if relative_time:
        fmt = "{relative} [{levelname[0]}] {name}: {message}"
    else:
        fmt = "{asctime} [{levelname[0]}] {name}: {message}"

    if prefix:
        fmt = prefix + fmt

    formatter = logging.Formatter(
        fmt=fmt,
        datefmt=datefmt,
        style="{",
    )

    stdout_ch = logging.StreamHandler(sys.stdout)
    stderr_ch = logging.StreamHandler(sys.stderr)

    # add relative time filter
    if relative_time:
        stdout_ch.addFilter(TimeFilter(datefmt=datefmt))
        stderr_ch.addFilter(TimeFilter(datefmt=datefmt))

    # add levels
    stdout_ch.addFilter(MaxLevelFilter(logging.WARNING))
    stdout_ch.setLevel(level)
    stderr_ch.setLevel(max(level, logging.WARNING))

    # set the formatter
    stdout_ch.setFormatter(formatter)
    stderr_ch.setFormatter(formatter)

    ROOT.handlers.clear()

    ROOT.addHandler(stdout_ch)
    ROOT.addHandler(stderr_ch)

    if path:
        fh = logging.FileHandler(f"{path}/log.log")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        ROOT.addHandler(fh)

        if pickle:
            ph = PickleHandler(path, timestamp=timestamp)
            ph.setLevel(logging.TRACE)
            ROOT.addHandler(ph)

        if tensorboard:
            swh = TensorboardHandler(path)
            swh.setLevel(logging.TRACE)
            ROOT.addHandler(swh)


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
