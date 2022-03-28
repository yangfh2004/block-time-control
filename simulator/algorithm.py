"""
This module provides algorithms for mining difficulty adjustment.

Fanghao Yang
03/15/2022
"""
from enum import Enum
from pandas import Timestamp, Timedelta
from copy import copy
from abc import ABC, abstractmethod
from math import log2
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


class AdjustAlgorithm(ABC):
    def __init__(self,
                 adjust_interval: TimeInterval,
                 target: TimeInterval):
        # all time value are in seconds.
        self._adjust_time_interval = adjust_interval.seconds()
        self._block_time_target = target.seconds()
        self._block_count_target = self._adjust_time_interval // target.seconds()
        # this is the measured average block time in an adjustment time interval (NOT current block time!).
        self._measured_block_time = copy(self.block_time_target)
        # the timestamp of last difficulty adjustment.
        self._adjust_timestamp = None
        self.total_block_count = 0
        self._difficulty = self.INIT_DIFFICULTY

    def __call__(self, blk_cnt: int, timestamp: Timestamp, excess_seconds=0):
        self.total_block_count += blk_cnt
        if self._adjust_timestamp is None:
            self._adjust_timestamp = timestamp
        # check if reach block adjustment count.
        elif self.total_block_count >= self._block_count_target:
            assert (excess_seconds >= 0)
            actual_interval_sec = (timestamp - self._adjust_timestamp) / np.timedelta64(1, 's') - excess_seconds
            # update actual average block time.
            self._measured_block_time = actual_interval_sec / self._block_count_target
            self._adjust_difficulty()
            # update block count.
            self.total_block_count -= self._block_count_target
            self._adjust_timestamp = timestamp - Timedelta(seconds=excess_seconds)
        return self._difficulty

    @property
    def adjust_time_interval(self):
        return self._adjust_time_interval

    @property
    def block_time_target(self):
        return self._block_time_target

    @property
    def block_count_target(self):
        return self._block_count_target

    @property
    @abstractmethod
    def INIT_DIFFICULTY(self):
        pass

    @abstractmethod
    def _adjust_difficulty(self):
        pass


class SimpleAvgAdjust(AdjustAlgorithm):
    INIT_DIFFICULTY = 1.0

    def __init__(self, adjust_interval: TimeInterval = BITCOIN_INTERVAL, target: TimeInterval = BITCOIN_BLOCK_TIME):
        # all time value are in seconds.
        super().__init__(adjust_interval, target)

    def _adjust_difficulty(self):
        # adjust difficulty
        self._difficulty *= self.block_time_target / self._measured_block_time
        # the minimum difficulty is 1.0 for Bitcoin
        self._difficulty = max(1.0, self._difficulty)


class SimpleBitAdjust(AdjustAlgorithm):
    INIT_DIFFICULTY = 59

    def __init__(self, adjust_interval: TimeInterval = BITCOIN_INTERVAL,
                 target: TimeInterval = BITCOIN_BLOCK_TIME,
                 adjust_range: int = 3,):
        # all time value are in seconds.
        super().__init__(adjust_interval, target)
        self._adjust_bit_range = adjust_range

    def _adjust_difficulty(self):
        # adjust difficulty
        adjustment = max(min(round(log2((self.block_time_target / self._measured_block_time)**2)),
                             self._adjust_bit_range),
                         -self._adjust_bit_range)
        self._difficulty += adjustment


class PIDBitAdjust(AdjustAlgorithm):
    INIT_DIFFICULTY = 59

    def __init__(self, adjust_interval: TimeInterval = BITCOIN_INTERVAL,
                 target: TimeInterval = BITCOIN_BLOCK_TIME,
                 adjust_range: int = 3,
                 p: float = 1.63, i: float = 0.002, d: float = 0.03):
        # all time value are in seconds.
        super().__init__(adjust_interval, target)
        self._adjust_bit_range = adjust_range
        self._proportional_gain = p
        self._integral_gain = i
        self._derivative_gain = d
        self._prev_error = 0.0
        self._accumulated_error = 0.0

    def _adjust_difficulty(self):
        error = self.block_time_target - self._measured_block_time
        p_term = self._proportional_gain * error
        i_term = self._integral_gain * self._accumulated_error
        d_term = self._derivative_gain * (error - self._prev_error)
        # only use P and D term
        total_gain = p_term + d_term + i_term
        adjustment = round(log2((1 + total_gain/self.block_time_target)**2))
        adjustment = max(min(adjustment, self._adjust_bit_range), -self._adjust_bit_range)
        self._difficulty += adjustment
        self._prev_error = error
        self._accumulated_error += error
