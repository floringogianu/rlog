import logging
from typing import NamedTuple

from .handlers import TensorboardHandler, PickleHandler
from .metrics import (
    Accumulator,
    BaseMetric,
    SumMetric,
    AvgMetric,
    MaxMetric,
    FPSMetric,
)


__all__ = [
    "getLogger",
    "info",
    "trace",
    "init",
    "TensorboardHandler",
    "PickleHandler",
    "BaseMetric",
    "SumMetric",
    "AvgMetric",
    "MaxMetric",
    # "ValueMetric",
    "Accumulator",
    "FPSMetric",
]


ROOT = None


logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")


class RLogger(logging.Logger):
    def __init__(self, log_name=None):
        logging.Logger.__init__(self, log_name)

        self.accumulator = None
        self._log_kws = ("exc_info", "extra", "stack_info")  # small helper

    def trace(self, *args, **kws):
        # We break with the API for now.
        # And yes, logger takes its '*args' as 'args'.
        if args:
            self._log(logging.TRACE, args[0], args[1:], **kws)
        elif kws:
            _log_kws = {k: v for k, v in kws.items() if k in self._log_kws}
            self._log(logging.TRACE, kws, args, **_log_kws)
        else:
            raise ("Call trace with either a message or a dict-like object.")

    def addMetrics(self, metrics):
        # TODO: Not really happy about how adding metrics changes the
        # interface of RLogger, need to thing about something else.

        if self.accumulator:
            raise ("Metrics already set.")
        else:
            self.accumulator = Accumulator(metrics)

        self.put = self.accumulator.trace
        self.reset = self.accumulator.reset
        self.summarize = self.accumulator.summarize


def init(name, path=None, lvl=logging.TRACE, pickle=True, tensorboard=False):
    """ Configures a global RLogger.
    """
    global ROOT

    logging.setLoggerClass(RLogger)
    ROOT = logging.getLogger(name)
    ROOT.setLevel(lvl)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)5s][%(levelname)7s]: %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(lvl)
    ROOT.addHandler(ch)

    if path:
        fh = logging.FileHandler(f"{path}/log.log")
        fh.setFormatter(formatter)
        fh.setLevel(lvl)
        ROOT.addHandler(fh)

        if pickle:
            ph = PickleHandler(path)
            ph.setLevel(lvl)
            ROOT.addHandler(ph)

        if tensorboard:
            swh = TensorboardHandler(path)
            swh.setLevel(lvl)
            ROOT.addHandler(swh)


def getLogger(name):
    return logging.getLogger(name)


def debug(msg, *args, **kwargs):
    ROOT.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    ROOT.info(msg, *args, **kwargs)


def trace(msg, *args, **kwargs):
    ROOT.info(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    ROOT.error(msg, *args, **kwargs)


def exception(msg, *args, exc_info=True, **kwargs):
    ROOT.error(msg, *args, exc_info=exc_info, **kwargs)


def warning(msg, *args, **kwargs):
    ROOT.warning(msg, *args, **kwargs)


if __name__ == "__main__":
    pass
