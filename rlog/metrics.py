import math
import re
import time

__all__ = [
    "Accumulator",
    "AvgMetric",
    "BaseMetric",
    "EpisodicMetric",
    "EWMAvgMetric",
    "FPSMetric",
    "MaxMetric",
    "SumMetric",
    "ValueMetric",
]


"""TODO: needs refactoring, class hierarchy makes no sense:
    - why reset does not reset _val in BaseMetric?
"""


class BaseMetric:
    def __init__(
        self, name, resetable=True, emph=False, metargs=None, tb_type="scalar"
    ):
        self._name = name
        self._val = 0
        self._resetable = resetable
        self._emph = emph
        self._metargs = metargs
        self._tb_type = tb_type
        self._updated = False

    @property
    def value(self):
        return self._val

    def accumulate(self, val, *args):
        raise NotImplementedError

    def reset(self):
        if self._resetable:
            self._updated = False

    @property
    def name(self):
        return self._name

    @property
    def updated(self):
        return self._updated

    @property
    def emph(self):
        return self._emph

    @property
    def metargs(self):
        return self._metargs

    @property
    def tb_type(self):
        return self._tb_type

    def __repr__(self):
        return "{:s}::{:s}".format(self.__class__.__name__, self._name)


class ValueMetric(BaseMetric):
    def __init__(
        self, name, resetable=True, emph=False, metargs=None, tb_type="scalar"
    ):
        BaseMetric.__init__(self, name, resetable, emph, metargs=metargs)
        self._val = []
        self._tb_type = tb_type

    def accumulate(self, val):
        self._val.append(val)
        self._updated = True

    def reset(self):
        super().reset()
        if self._resetable:
            self._val = []


class MaxMetric(BaseMetric):
    def __init__(self, name, resetable=True, emph=False, metargs=None):
        BaseMetric.__init__(self, name, resetable, emph, metargs=metargs)
        self._val = -math.inf

    def accumulate(self, val):
        self._val = max(self._val, val)
        self._updated = True

    def reset(self):
        super().reset()
        if self._resetable:
            self._val = -math.inf


class SumMetric(BaseMetric):
    def __init__(self, name, resetable=True, emph=False, metargs=None):
        BaseMetric.__init__(self, name, resetable, emph, metargs=metargs)

    def accumulate(self, val):
        self._val += val
        self._updated = True

    def reset(self):
        super().reset()
        if self._resetable:
            self._val = 0


class AvgMetric(BaseMetric):
    def __init__(self, name, resetable=True, emph=False, metargs=None):
        BaseMetric.__init__(self, name, resetable, emph, metargs=metargs)
        self._counter = 0

    @property
    def value(self):
        if self._counter == 0:
            return self._val
        return self._val / self._counter

    def accumulate(self, val, n):
        self._val += val
        self._counter += n
        self._updated = True

    def reset(self):
        super().reset()
        if self._resetable:
            self._val = 0
            self._counter = 0


class EWMAvgMetric(BaseMetric):
    """An exponential weighted moving average of the AverageMetric. Instead of
    directly returning the current value of the AverageMetric it computes a
    smoothed version of the previous values.
    """

    def __init__(self, name, emph=False, metargs=None, beta=0.1):
        BaseMetric.__init__(self, name, False, emph, metargs=metargs)
        assert 0 < beta < 1, "β has to be between 0 and 1."
        self._beta = beta
        self._val = None
        self._avg = AvgMetric(name, True, emph, metargs)

    @property
    def value(self):
        β = self._beta
        # TODO: this might be buggy
        # only compute the EWM if there are averages to report.
        if self._avg.updated:
            val = self._avg.value
            self._avg.reset()
            self._val = val if self._val is None else self._val * β + val * (1 - β)
        return self._val

    def accumulate(self, val, n=1):
        self._avg.accumulate(val, n)
        self._updated = True

    def reset(self):
        super().reset()
        self._avg.reset()


class EpisodicMetric(BaseMetric):
    def __init__(self, name, resetable=True, emph=False, metargs=None):
        BaseMetric.__init__(self, name, resetable, emph, metargs=metargs)
        self.counter = 0
        self.partial_val = 0

    @property
    def value(self):
        if self.counter == 0:
            return self._val
        return self._val / self.counter

    def accumulate(self, val, n=1):
        if n == 0:
            self.partial_val += val
        else:
            self._val += self.partial_val + val
            self.counter += n
            self.partial_val = 0
        self._updated = True

    def reset(self):
        super().reset()
        if self._resetable:
            self._val = 0
            self.counter = 0
            self.partial_val = 0


class FPSMetric(BaseMetric):
    def __init__(self, name, resetable=True, emph=False, metargs=None):
        BaseMetric.__init__(self, name, resetable, emph, metargs=metargs)
        self._start = time.time()

    @property
    def value(self):
        return self._val / (time.time() - self._start)

    def accumulate(self, val, *args):
        self._val += val
        self._updated = True

    def reset(self):
        super().reset()
        if self._resetable:
            self._val = 0
            self._start = time.time()


def clip(x):
    return max(min(1, x), -1)


FNS = {"clip": clip, "int": int}


class Accumulator:
    def __init__(self, *metrics, console_options=None):
        self.metrics = {}
        self.add_metrics(*metrics)
        self.console_options = console_options

    def add_metrics(self, *metrics):
        """Add metrics to the Accumulator."""
        self.metrics.update({m.name: m for m in metrics})

    def summarize(self):
        # check wether the metric has been updated between two resets.
        updated_metrics = [m for m in self.metrics.values() if m.updated]
        # and get the return values of each metric
        payload = {m.name: m.value for m in updated_metrics}
        # add the tensorboard types
        payload["extra"] = {
            "tb_types": {m.name: m.tb_type for m in self.metrics.values()}
        }
        return payload

    def accumulate(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self.metrics, (
                f"The metric you are trying to accumulate is not in {self}."
            )
            if isinstance(v, list | tuple):
                self.metrics[k].accumulate(*v)
            else:
                self.metrics[k].accumulate(v)

    def trace(self, **kwargs):
        for metric in self._updatable_metrics(kwargs):
            args = self._process(metric.metargs, kwargs)
            metric.accumulate(*args)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def _updatable_metrics(self, kwargs):
        """Return the metrics that have metargs appearing in kwargs"""
        metrics = []
        for metarg in kwargs:
            for metric in self.metrics.values():
                if metarg in metric.metargs:
                    metrics.append(metric)
        return set(metrics)

    def _process(self, metargs, kwargs):
        args = []
        for metarg in metargs:
            if isinstance(metarg, int | float):
                # metarg is a number
                args.append(metarg)
            elif "(" in metarg:
                # metarg is a function such as "clip(reward)"
                fn_name = metarg.split("(")[0]  # get the 'f' in 'f(x)'
                # get 'x' in 'f(x)'
                fn_arg = re.search(r"\((.*?)\)", metarg).group(1)
                args.append(FNS[fn_name](kwargs[fn_arg]))
            else:
                # metarg is a string we are tracing such as "lorem ipsum"
                args.append(kwargs[metarg])
        return args

    def __repr__(self):
        return f"Accumulator[{', '.join([str(m) for m in self.metrics.values()])}]"


def main():
    N = 1000

    group = Accumulator(
        AvgMetric("R_per_ep"),
        EpisodicMetric("episodicR"),
        AvgMetric("rw_per_ep"),
        SumMetric("ep_cnt", resetable=False),
        FPSMetric("train_fps"),
    )
    print(group)

    ep, control = 0, []
    for step in range(1, N):
        done = random.random() < 0.01
        r = (random.random() - 0.5) * step

        group.accumulate(
            R_per_ep=[r, int(done)],
            episodicR=[r, int(done)],
            ep_cnt=int(done),
            rw_per_ep=[clip(r), int(done)],
            train_fps=2,
        )

        control.append(r)
        ep += int(done)

    for k, v in group.summarize().items():
        if k != "extra":
            print(f"{k}:\t {v:>9,.1f}")

    print("-------")
    print(f"avg_returns:\t {(sum(control) / ep):>9,.1f}")
    print(f"avg_clipped:\t {(sum([clip(x) for x in control]) / ep):>9,.1f}")
    # print(", ".join([f"{i:2.3f}" for i in control]))


def fancy():
    N = 1000

    ep, control = 0, []
    # define the metrics you want to log and the arguments these metrics
    # should trace. It evens supports functions, such as "clip(x)".
    log = Accumulator(
        SumMetric("ep_cnt", resetable=False, metargs=["done"]),
        AvgMetric("R_per_ep", metargs=["reward", "done"]),
        AvgMetric("R_per_step", metargs=["reward", 1]),
        AvgMetric("rw_per_ep", metargs=["clip(reward)", "done"]),
        FPSMetric("train_fps", metargs=["frame_no"]),
    )

    # start a game
    for step in range(1, N):
        done = random.random() < 0.01
        reward = (random.random() - 0.5) * step

        # and simply trace all the values you passed as `metargs` above.
        # the logger will know how to dispatch each argument.
        log.trace(reward=reward, done=done, frame_no=1)

        # control
        control.append(reward)
        ep += done

    for k, v in log.summarize().items():
        if k != "extra":
            print(f"{k}:\t {v:>9,.1f}")

    print("-------")
    print(f"ep_cnt:\t {ep:>9,.1f}")
    print(f"R_per_ep:\t {(sum(control) / ep):>9,.1f}")
    print(f"R_per_step:\t {(sum(control) / step):>9,.1f}")
    print(f"re_per_ep:\t {(sum([clip(x) for x in control]) / ep):>9,.1f}")
    # print(", ".join([f"{i:2.3f}" for i in control]))


if __name__ == "__main__":
    import random

    print("\nMAIN:")
    main()
    # print("\nFANCY:")
    # fancy()
