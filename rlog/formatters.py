""" This is not respecting the logging.Formatter module (if such thing exists
    :) ). It is just an helper class that maintains a format string to be used
    with rlog.info().

    Special about it is that it checks for the values that needs to be displayed
    and formats the strings accordingly. For example if one of the metrics
    report no value, that metric will not be displayed.
"""


class SummaryFormatter:
    def __init__(self):
        pass

    def _compute_string_format(self, summary):
        to_string = {
            k: v
            for k, v in summary.items()
            if (v is not None) and (type(v) not in (list, tuple, dict))
        }
        to_string.pop("step")
        fmt = ", ".join([f"{k}={{{k}:2.2f}}" for k in to_string.keys()])
        return "[{step:06d}]   " + fmt

    def __call__(self, **summary):
        fmt = self._compute_string_format(summary)
        return fmt.format(**summary)
