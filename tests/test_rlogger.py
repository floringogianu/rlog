import pytest

import rlog


class TestRLogger:
    def test_init_basic(self):
        rlog.init("test_logger")
        logger = rlog.getRootLogger()
        assert logger is not None
        assert logger.name == "test_logger"

    def test_init_with_relative_time(self):
        rlog.init("test_relative", relative_time=True)
        logger = rlog.getRootLogger()
        assert logger is not None
        assert logger.name == "test_relative"

    def test_init_without_relative_time(self):
        rlog.init("test_absolute", relative_time=False)
        logger = rlog.getRootLogger()
        assert logger is not None
        assert logger.name == "test_absolute"

    def test_get_logger(self):
        rlog.init("test_root")
        child_logger = rlog.getLogger("test_root.child")
        assert child_logger is not None
        assert child_logger.name == "test_root.child"

    def test_trace_method(self):
        rlog.init("test_trace")
        logger = rlog.getLogger("test_trace")
        logger.trace(step=1, value=2.5)

    def test_trace_with_string(self):
        rlog.init("test_trace_str")
        logger = rlog.getLogger("test_trace_str")
        logger.trace("Test message")

    def test_trace_with_dict(self):
        rlog.init("test_trace_dict")
        logger = rlog.getLogger("test_trace_dict")
        logger.trace({"key": "value", "number": 42})

    def test_logging_levels(self):
        rlog.init("test_levels")
        logger = rlog.getLogger("test_levels")

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.trace("Trace message")

    def test_global_functions(self):
        rlog.init("test_global")

        rlog.debug("Debug")
        rlog.info("Info")
        rlog.warning("Warning")
        rlog.error("Error")
        rlog.trace("Trace")

    def test_add_metrics(self):
        rlog.init("test_metrics")
        logger = rlog.getLogger("test_metrics")

        logger.addMetrics(rlog.ValueMetric("test_val", metargs=["val"]))
        logger.put(val=1.0)

    def test_trace_and_log_without_metrics(self):
        rlog.init("test_trace_log")
        logger = rlog.getLogger("test_trace_log")

        with pytest.raises(TypeError):
            logger.traceAndLog(step=1)
