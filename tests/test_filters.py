import logging

from rlog.filters import MaxLevelFilter


class TestMaxLevelFilter:
    def test_filter_below_level(self):
        filter = MaxLevelFilter(logging.WARNING)
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})

        assert filter.filter(record) is True

    def test_filter_at_level(self):
        filter = MaxLevelFilter(logging.WARNING)
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.WARNING})

        assert filter.filter(record) is False

    def test_filter_above_level(self):
        filter = MaxLevelFilter(logging.WARNING)
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.ERROR})

        assert filter.filter(record) is False

    def test_filter_with_debug_level(self):
        filter = MaxLevelFilter(logging.DEBUG)
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.DEBUG})

        assert filter.filter(record) is False

    def test_filter_with_info_level(self):
        filter = MaxLevelFilter(logging.INFO)
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.DEBUG})

        assert filter.filter(record) is True

    def test_filter_with_trace_level(self):
        filter = MaxLevelFilter(logging.INFO)
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.TRACE})

        assert filter.filter(record) is True
