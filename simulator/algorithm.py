"""
This module provides algorithms for mining difficulty adjustment.

Fanghao Yang
03/15/2022
"""
from enum import Enum
from pandas import Timestamp
from copy import copy
import numpy as np


class TimeUnit(Enum):
    """Time interval unit between each difficulty adjustment."""
    Second = 1
    Minute = 60
    Hour = 3600
    Day = 3600 * 24
    Week = 3600 * 24 * 7


class TimeInterval:
    def __init__(self, interval: int, unit: TimeUnit):
        self._interval = interval
        self._unit = unit

    def seconds(self):
        return self._interval * self._unit.value


BITCOIN_INTERVAL = TimeInterval(2, TimeUnit.Week)
BITCOIN_BLOCK_TIME = TimeInterval(10, TimeUnit.Minute)


class SimpleAvgAdjust:
    INIT_DIFFICULTY = 1.0

    def __init__(self,
                 adjust_interval: TimeInterval = BITCOIN_INTERVAL,
                 target: TimeInterval = BITCOIN_BLOCK_TIME):
        # all time value are in seconds.
        self._adjust_time_interval = adjust_interval.seconds()
        self._block_time_target = target.seconds()
        self._block_count_target = self._adjust_time_interval / target.seconds()
        self._block_time = copy(self._block_time_target)
        self._prev_timestamp = None
        self._total_block_count = 0
        self._difficulty = self.INIT_DIFFICULTY

    def __call__(self, blk_cnt: int, timestamp: Timestamp) -> float:
        self._total_block_count += blk_cnt
        if self._prev_timestamp is None:
            self._prev_timestamp = timestamp
        elif self._total_block_count > self._block_count_target:
            # reach block adjustment count.
            actual_interval_sec = (timestamp - self._prev_timestamp) / np.timedelta64(1, 's')
            # compare between actual interval with targeted interval
            excess_block_count = self._total_block_count - self._block_count_target
            # update actual block time.
            self._block_time = (actual_interval_sec - excess_block_count * self._block_time) / self._block_count_target
            # adjust difficulty
            self._difficulty *= self._block_time_target / self._block_time
            # update block count.
            self._total_block_count -= self._block_count_target
            self._prev_timestamp = timestamp
        return self._difficulty
