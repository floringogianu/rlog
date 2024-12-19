""" Simple examples.
"""
import math
import random
import os
from datetime import datetime

import rlog


def tanh(x):
    ex = math.exp(x)
    _ex = math.exp(-x)
    return (ex - _ex) / (ex + _ex)


def reward_following_policy(step, knee=40_000):
    # simulates an agent that improves with each step
    done = random.random() > tanh(random.gauss(step / knee, 0.5))
    if step % 500 == 0:
        return 1, True
    return 1, done


def get_experiment_path():
    timestamp = "{:%Y%b%d-%H%M%S}".format(datetime.now())
    experiment_path = f"./sota_results/{timestamp}/"
    os.makedirs(experiment_path)
    return experiment_path


def main():
    # get the root logger, preconfigured to log to the console,
    # to a text file, a pickle and a tensorboard protobuf.
    experiment_path = get_experiment_path()
    rlog.init("dqn", path=experiment_path, tensorboard=False)
    rlog.info("Logging application level stuff.")
    rlog.info("Log artifacts will be saved in %s", experiment_path)

    rlog.addMetrics(
        # counts each time it receives a `done=True`, aka counts episodes
        rlog.SumMetric("ep_cnt", resetable=False, metargs=["done"]),
        # sums up all the `reward=value` it receives and divides it
        # by the number of `done=True`, aka mean reward per episode
        rlog.AvgMetric("R/ep", metargs=["reward", "done"]),
    )

    for step in range(5):
        # probably not a good idea to call this every step if it is a hot loop?
        # also this will not be logged to the console or to the text file
        # since the default log-level for these two is INFO.
        rlog.trace(step=step, aux_loss=7.23 - step)

    # but we can register metrics that will accumulate traced events
    # and summarize them. Each Metric accepts a name and some metargs
    # that tells it which arguments received by the `put` call bellow
    # to accumulate and summarize.
    rlog.addMetrics(
        # counts each time it receives a `done=True`, aka counts episodes
        rlog.SumMetric("ep_cnt", resetable=False, metargs=["done"]),
        # sums up all the `reward=value` it receives and divides it
        # by the number of `done=True`, aka mean reward per episode
        rlog.AvgMetric("R/ep", metargs=["reward", "done"]),
        # same but keeps a running average instead (experimental).
        rlog.EWMAvgMetric("ewm R/ep", metargs=["reward", "done"], beta=0.6),
        # same as above but now we divide by the number of rewards
        rlog.AvgMetric("R/step", metargs=["reward", 1]),
        # same but with clipped rewards (to +- 1)
        rlog.AvgMetric("clip R/ep", metargs=["clip(reward)", "done"]),
        # computes the no of frames per second
        rlog.FPSMetric("train_fps", metargs=["frame_no"]),
        # caches all the values it receives and inserts them into a
        # tensorboad.summary.histogram every time you call `log.trace`
        rlog.ValueMetric("gaussians", metargs=["sample"], tb_type="histogram"),
    )

    mean = 0
    for step in range(1, 300_001):

        # make a step in the "environment"
        reward, done = reward_following_policy(step)

        # sample from a gaussian for showcasing the histogram
        sample = random.gauss(mean, 0.1)

        # simply trace all the values you passed as `metargs` above.
        # the logger will know how to dispatch each argument.
        rlog.put(reward=reward, done=done, frame_no=1, sample=sample)

        if step % 10_000 == 0:
            # this is the call that dumps everything to the logger.

            # summary = rlog.summarize()
            # rlog.trace(step=step, **summary)
            # rlog.info(
            #     "{0:6d}, ep {ep_cnt:3d}, RunR/ep{RunR:8.2f}  |  rw/ep{R_per_ep:8.2f}.".format(
            #         step, **summary
            #     )
            # )
            rlog.traceAndLog(step)
            mean += 1

    rlog.trace("But we can continue tracing stuff manually...")
    # inlcuding structured stuff as long as we provide a `step` keyarg
    rlog.trace(step=step, aux_loss=0.23)

    rlog.info("Run `tensorboard --logdir sota_results` to see the results.")


if __name__ == "__main__":
    main()
