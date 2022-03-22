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
import simulator.miner as miner
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
        self.block_time_target = target.seconds()
        self.block_count_target = self._adjust_time_interval / target.seconds()
        self._block_time = copy(self.block_time_target)
        self._prev_timestamp = None
        self.total_block_count = 0
        self._difficulty = self.INIT_DIFFICULTY

    @abstractmethod
    def __call__(self, blk_cnt: int, timestamp: Timestamp, excess_seconds=0):
        pass

    @property
    @abstractmethod
    def INIT_DIFFICULTY(self):
        pass

    def _update_block_time(self, excess_seconds: int, timestamp: Timestamp) -> int:
        # reach block adjustment count.
        actual_interval_sec = (timestamp - self._prev_timestamp) / np.timedelta64(1, 's')
        # compare between actual interval with targeted interval
        # excess_block_count = self.total_block_count - self.block_count_target
        # one_day = TimeInterval(1, TimeUnit.Day)
        # excess_seconds = excess_block_count * one_day.seconds() // daily_blk
        # update actual block time.
        new_block_time = (actual_interval_sec - excess_seconds) \
            / self.block_count_target
        assert (excess_seconds >= 0)
        self._block_time = new_block_time

        return excess_seconds


class SimpleAvgAdjust(AdjustAlgorithm):
    INIT_DIFFICULTY = miner.BitcoinMiner.INIT_DIFFICULTY

    def __init__(self, adjust_interval: TimeInterval = BITCOIN_INTERVAL, target: TimeInterval = BITCOIN_BLOCK_TIME):
        # all time value are in seconds.
        super().__init__(adjust_interval, target)

    def __call__(self, blk_cnt: int, timestamp: Timestamp, excess_seconds=0):
        self.total_block_count += blk_cnt
        if self._prev_timestamp is None:
            self._prev_timestamp = timestamp
        elif self.total_block_count >= self.block_count_target:
            time_delta = self._update_block_time(excess_seconds, timestamp)
            # adjust difficulty
            self._difficulty *= self.block_time_target / self._block_time
            # the minimum difficulty is 1.0 for Bitcoin
            self._difficulty = max(1.0, self._difficulty)
            # update block count.
            self.total_block_count -= self.block_count_target
            self._prev_timestamp = timestamp - Timedelta(seconds=time_delta)
        else:
            one_day = TimeInterval(1, TimeUnit.Day)
            self._block_time = one_day.seconds() / blk_cnt
        return self._difficulty


class SimpleBitAdjust(AdjustAlgorithm):
    INIT_DIFFICULTY = miner.CapsuleMiner.INIT_DIFFICULTY

    def __init__(self, adjust_interval: TimeInterval = BITCOIN_INTERVAL, adjust_range: int = 3,
                 target: TimeInterval = BITCOIN_BLOCK_TIME):
        # all time value are in seconds.
        super().__init__(adjust_interval, target)
        self._adjust_bit_range = adjust_range

    def __call__(self, blk_cnt: int, timestamp: Timestamp, excess_seconds=0):
        self.total_block_count += blk_cnt
        if self._prev_timestamp is None:
            self._prev_timestamp = timestamp
        elif self.total_block_count >= self.block_count_target:
            time_delta = self._update_block_time(blk_cnt, timestamp)
            # adjust difficulty
            adjustment = max(min(round(log2(self.block_time_target / self._block_time)), self._adjust_bit_range),
                             -self._adjust_bit_range)
            self._difficulty += adjustment
            # update block count.
            self.total_block_count -= self.block_count_target
            self._prev_timestamp = timestamp - Timedelta(seconds=time_delta)
        return self._difficulty
