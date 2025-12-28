import logging

from rlog.rlogger import TimeFilter


class TestTimeFilter:
    def test_basic_formatting(self):
        filter = TimeFilter()
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})
        record.relativeCreated = 90_000

        assert filter.filter(record) is True
        assert record.relative == "00:01:30"

    def test_zero_seconds(self):
        filter = TimeFilter()
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})
        record.relativeCreated = 0

        assert filter.filter(record) is True
        assert record.relative == "00:00:00"

    def test_exactly_24_hours(self):
        filter = TimeFilter()
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})
        record.relativeCreated = 86_400_000

        assert filter.filter(record) is True
        assert record.relative == "24:00:00"

    def test_over_24_hours(self):
        filter = TimeFilter()
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})
        record.relativeCreated = 93_600_000

        assert filter.filter(record) is True
        assert record.relative == "26:00:00"

    def test_multiple_days(self):
        filter = TimeFilter()
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})
        record.relativeCreated = 180_000_000

        assert filter.filter(record) is True
        assert record.relative == "50:00:00"

    def test_minutes_and_seconds_only(self):
        filter = TimeFilter()
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})
        record.relativeCreated = 366_000

        assert filter.filter(record) is True
        assert record.relative == "00:06:06"

    def test_partial_seconds(self):
        filter = TimeFilter()
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})
        record.relativeCreated = 1_500

        assert filter.filter(record) is True
        assert record.relative == "00:00:01"

    def test_large_hours_value(self):
        filter = TimeFilter()
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})
        record.relativeCreated = 3_600_000_000

        assert filter.filter(record) is True
        assert record.relative == "1000:00:00"

    def test_zero_padding(self):
        filter = TimeFilter()
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})
        record.relativeCreated = 3_661_000

        assert filter.filter(record) is True
        assert record.relative == "01:01:01"

    def test_custom_datefmt_ignored(self):
        filter = TimeFilter(datefmt="%H-%M-%S")
        record = logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})
        record.relativeCreated = 366_000

        assert filter.filter(record) is True
        assert record.relative == "00:06:06"
        assert "-" not in record.relative
