import logging

import rlog

rlog.init("dqn", level=logging.DEBUG)
rlog.debug("debug now prints")  # was silent at every level

rlog.init("dqn", level=logging.WARNING, path="/tmp")
rlog.addMetrics(rlog.AvgMetric("R", metargs=["reward", 1]))
rlog.put(reward=1.0)
rlog.traceAndLog(step=1)  # console quiet, pickle still written
