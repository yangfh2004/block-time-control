from abc import ABC, abstractmethod
from copy import copy
from enum import Enum
from math import isqrt
from typing import Optional
from pandas import Timestamp, Timedelta, to_datetime
import numpy as np
import simulator.algorithm as sim_algo


class HashRateUnit(Enum):
    G = 10 ** 9
    T = 10 ** 12


class Miner(ABC):
    """Abstract miner class."""

    def __init__(self, algo, unit=HashRateUnit.T):
        self._hash_rate: float = self.INIT_HASH_RATE
        # block time in seconds
        self._predicted_block_time: float = 0.0
        self._unit = unit
        self._difficulty = self.INIT_DIFFICULTY
        self._algorithm = algo
        self._timestamp: Optional[Timestamp] = None

    def _is_diff_adjustment(self, blk_cnt) -> bool:
        return blk_cnt + self._algorithm.total_block_count > self._algorithm.block_count_target

    def generate_blocks(self, hash_rate: float, timestamp: Timestamp) -> int:
        excess_blk_time = 0
        # update hash rate.
        self.hash_rate = hash_rate
        # get the block time (sec) to mine a new block.
        prev_timestamp = timestamp - Timedelta(days=1) if self._timestamp is None else self._timestamp
        self._timestamp = timestamp
        time_interval = int((self._timestamp - prev_timestamp) / np.timedelta64(1, 's'))
        predicted_blk_cnt = time_interval // int(self.block_time)
        if self._is_diff_adjustment(predicted_blk_cnt):
            # get the residual block count before the difficulty adjustment.
            residual_blk_cnt = self._algorithm.block_count_target - self._algorithm.total_block_count
            residual_blk_time = residual_blk_cnt * int(self.block_time)
            excess_blk_time = time_interval - residual_blk_time
            # excess_blk_time = (predicted_blk_cnt - residual_blk_cnt) * self._day.seconds() // predicted_blk_cnt
            # try to adjust miner difficulty.
            temp_algo = copy(self._algorithm)
            self.difficulty = temp_algo(predicted_blk_cnt, self._timestamp, excess_blk_time)
            # predict block count after difficulty adjustment.
            new_blk_cnt = excess_blk_time // int(self.block_time)
            predicted_blk_cnt = new_blk_cnt + residual_blk_cnt
            assert (predicted_blk_cnt > 0)

        self.difficulty = self._algorithm(predicted_blk_cnt, self._timestamp, excess_blk_time)
        return predicted_blk_cnt

    @abstractmethod
    def _update_block_time(self):
        """Calculate block time in seconds."""
        pass

    @property
    def block_time_target(self) -> int:
        return self._algorithm.block_time_target

    @property
    def hash_rate(self) -> float:
        return self._hash_rate

    @hash_rate.setter
    def hash_rate(self, val: float):
        self._hash_rate = val
        self._update_block_time()

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @difficulty.setter
    def difficulty(self, val):
        self._difficulty = val
        self._update_block_time()

    @property
    @abstractmethod
    def INIT_DIFFICULTY(self):
        pass

    @property
    @abstractmethod
    def INIT_HASH_RATE(self):
        pass

    @property
    def block_time(self) -> float:
        return self._predicted_block_time


class BitcoinMiner(Miner):
    INIT_DIFFICULTY = sim_algo.SimpleAvgAdjust.INIT_DIFFICULTY
    INIT_HASH_RATE = 9.44495E-07

    def _update_block_time(self):
        self._predicted_block_time = self.difficulty * 2 ** 32 / self._unit.value / self.hash_rate


class CapsuleMiner(Miner):
    """
    Capsule miner use hash power to find the solution of asymmetric encryption system with Floyd's cycle finding
    algorithm.
    """
    INIT_DIFFICULTY = sim_algo.SimpleBitAdjust.INIT_DIFFICULTY
    INIT_HASH_RATE = 9.44495E-07
    # Data source:
    # Chapter 2.1.3
    # Shi Bai and Richard P. Brent. 2008. On the efficiency of Pollard's rho method for discrete logarithms.
    # In Proceedings of the fourteenth symposium on Computing: the Australasian theory - Volume 77 (CATS '08).
    # Australian Computer Society, Inc., AUS, 125–131.
    mining_factor = 1.03 + 3.09

    def _update_block_time(self):
        difficulty = int(self.difficulty)
        g = 2 ** (difficulty - 1) + 2 ** (difficulty - 2)
        self._predicted_block_time = self.mining_factor * isqrt(g) / self._unit.value / self.hash_rate
