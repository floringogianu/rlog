""" Logger filters.
"""
import logging


class MaxLevelFilter(logging.Filter):
    """ Filters (lets through) all messages with level < LEVEL

    https://stackoverflow.com/a/24956305/1493507
    """

    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno < self.level
