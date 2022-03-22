from abc import ABC, abstractmethod
from enum import Enum
from math import isqrt


class HashRateUnit(Enum):
    G = 10 ** 9
    T = 10 ** 12


class Miner(ABC):
    """Abstract miner class."""

    def __init__(self, unit=HashRateUnit.T):
        self._hash_rate: float = self.INIT_HASH_RATE
        # block time in seconds
        self._block_time: float = 0.0
        # previous block time for regression.
        self.prev_block_time: float = 0.0
        self._unit = unit
        self._difficulty = self.INIT_DIFFICULTY

    @abstractmethod
    def _update_block_time(self):
        """Calculate block time in seconds."""
        pass

    @property
    def hash_rate(self) -> float:
        return self._hash_rate

    @hash_rate.setter
    def hash_rate(self, val: float):
        self._hash_rate = val
        self.prev_block_time = self._block_time
        self._update_block_time()

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @difficulty.setter
    def difficulty(self, val):
        self._difficulty = val
        self.prev_block_time = self._block_time
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
        return self._block_time


class BitcoinMiner(Miner):
    INIT_DIFFICULTY = 1.0
    INIT_HASH_RATE = 9.44495E-07

    def _update_block_time(self):
        self._block_time = self.difficulty * 2 ** 32 / self._unit.value / self.hash_rate


class CapsuleMiner(Miner):
    """
    Capsule miner use hash power to find the solution of asymmetric encryption system with Floyd's cycle finding
    algorithm.
    """
    INIT_DIFFICULTY = 59
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
        self._block_time = self.mining_factor * isqrt(g) / self._unit.value / self.hash_rate
