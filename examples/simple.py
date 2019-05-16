""" Simple examples.
"""
import random
import rlog


def main():
    # get the root logger, preconfigured to log to the console,
    # to a text file and to a pickle.
    rlog.init("dqn", path="./sota_results/", tensorboard=True)
    rlog.info("Logging application level stuff.")

    # create a new logger that will log training events
    train_log = rlog.getLogger("dqn.train")
    train_log.info("Starting training... ")

    for step in range(5):
        # probably not a good idea to call this every step if it is a hot loop?
        train_log.trace(step=step, loss=-0.23, target_val=2.38)

    # but we can register metrics that will accumulate traced events
    # and summarize them. Each Metric accepts a name and some metargs
    # that tells it which arguments received by the `put` call bellow
    # to accumulate and summarize.
    train_log.addMetrics([
        rlog.SumMetric("ep_cnt", resetable=False, metargs=["done"]),
        rlog.AvgMetric("R_per_ep", metargs=["reward", "done"]),
        rlog.AvgMetric("R_per_step", metargs=["reward", 1]),
        rlog.AvgMetric("rw_per_ep", metargs=["clip(reward)", "done"]),
        rlog.FPSMetric("train_fps", metargs=["frame_no"]),
    ])

    for step in range(5, 1001):
        done = random.random() < 0.01
        reward = (random.random() - 0.5) * step

        # simply trace all the values you passed as `metargs` above.
        # the logger will know how to dispatch each argument.
        train_log.put(reward=reward, done=done, frame_no=1)

        if step % 250 == 0:
            # this is the call that dumps everything to the logger.
            train_log.trace(step=step, **train_log.summarize())
            train_log.reset()

    train_log.trace("But we can continue tracing stuff manually...")
    # inlcuding structured stuff as long as we provide a `step` keyarg
    train_log.trace(step=step, aux_loss=-0.48, target_val=2.99)

    # We can also configure an evaluation logger.
    eval_log = rlog.getLogger("dqn.eval")
    eval_log.info("Starting evaluation... ")


if __name__ == "__main__":
    main()
