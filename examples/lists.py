"""Simple examples."""

import os
import random
from datetime import datetime

import torch as th

import rlog


def get_experiment_path():
    timestamp = "{:%Y%b%d-%H%M%S}".format(datetime.now())
    experiment_path = f"./sota_results/{timestamp}/"
    os.makedirs(experiment_path)
    return experiment_path


def main():
    experiment_path = get_experiment_path()
    rlog.init("dqn", path=experiment_path)
    rlog.info("logging in %s", experiment_path)

    rlog.addMetrics(
        rlog.ValueMetric("norms", metargs=["layer_norms"]),
        rlog.ValueMetric("cached", metargs=["to_cache"]),
    )

    mean = 0
    for step in range(1, 11):
        norms = th.tensor([random.gauss(mean, 0.1) for _ in range(3)])
        rlog.put(to_cache=step, layer_norms=norms)

    rlog.traceAndLog(step)
    rlog.info("done")
    rlog.close()


if __name__ == "__main__":
    main()
