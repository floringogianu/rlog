# RLog

I got tired of configuring half-baked loggers in my Reinforcement Learning
experimets so I wrote `RLog`, a preconfigured logger using Python's `logging`.

In addition `RLog` makes it easy to log experiment stats in a RL algorithm and
it can write this structured information to `tensorboard` event files and
`pickles`. Additional handlers can be easily added.

## Demo

With `RLog` you can start logging right away, no need to fiddle with settings.

```python
import rlog

rlog.init("dqn")
rlog.info("Logging application level stuff.")
```

will output

```sh
11:52:53 [  dqn][   INFO]: Logging application level stuff.
```

This will not write to disk though. If we pass a folder path to `rlog.init()` a
`rlog` will write to a `log.log`, to a `<logger_name>.pkl` and optionaly to a
tensorboard event file.

```python
rlog.init("dqn", path="./sota_results/", tensorboard=True)
rlog.info("Logging application level stuff.")
```

We can also create a new logger that will inherit the settings of the root one.

```python
train_log = rlog.getLogger("dqn.train")
train_log.info("Starting training... ")
```

`RLog` provides a `RLogger.trace()` method that allows logging structured info
like this:

```python
train_log.trace(step=1, loss=-0.23, target_val=2.38)
train_log.trace(step=2, loss=-0.21, target_val=1.87)
...
train_log.trace(step=5, loss=-0.18, target_val=1.33)
```

Although it probably not a good idea to trace every step all the time.
For experiments running for millions of steps we have another mechanism that
accumulates values at every step and summarizes them, logging only these
summaries.

Each `Metric` accepts a name and some `metargs` that tells it which arguments
received by the `put` call bellow to accumulate and summarize.

```python
train_log.addMetrics(
    rlog.SumMetric("ep_cnt", resetable=False, metargs=["done"]),
    rlog.AvgMetric("R_per_ep", metargs=["reward", "done"]),
    rlog.AvgMetric("R_per_step", metargs=["reward", 1]),
    rlog.AvgMetric("rw_per_ep", metargs=["clip(reward)", "done"]),
    rlog.FPSMetric("learning_fps", metargs=["frame_no"]),
)
```

And somehwere in your training loop you can do:

```python
for step in range(1, training_steps):

    # simply trace all the values you passed as `metargs` above.
    # the logger will know how to dispatch each argument.
    train_log.put(reward=reward, done=done, frame_no=32)

    if step % 250 == 0:
        # this is the call that dumps everything to the logger.
        train_log.trace(step=step, **train_log.summarize())
        # and here we reset the metrics so that we the statistics
        # are not a running average but are windowed.
        train_log.reset()
```

And ofcourse we can configure a different logger for evaluating the
agent we are training:

```python
eval_log = rlog.getLogger("dqn.eval")
eval_log.info("Starting evaluation... ")
```

### How about tracing values at every step?

`RLog` supports a form of caching events traced at every step through the
`rlog.ValueMetric`. You can configure it like this:

```python
    ...
    rlog.ValueMetric("mse_err", metargs=["err"]),
    ...
```

and in your hot loop do:

```python
    # do this at every step, when you want to log `err`.
    train_log.put(err=err)

    if is logging_time:
        rlog.trace(step=policy.steps, **rlog.summarize())
```

## Structured data

By default, `rlog` writes to:

- `stdout`/`stderr` according to the logging levels
- every text record or above `level` also goes to a log.log
- recorder metrics go to `.pkl` files, one per logger
- same for tensorboard, if `rlog.init(...,tensorboard=True)`

## Logging Levels

The logging levels are:

| Level    | Numeric value |
| -------- | :-----------: |
| CRITICAL |      50       |
| ERROR    |      40       |
| WARNING  |      30       |
| **INFO** |    **20**     |
| DEBUG    |      10       |
| NOTSET   |       0       |

Levels control the console and the text file.
They do not control the structured output.
A metrics record reaches the pickle file and the tensorboard file whatever level you pass, because those two handlers select on record type.
Use `debug()` for text you want in the pickle file but not on screen.
