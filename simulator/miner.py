from abc import ABC, abstractmethod
from enum import Enum


class HashRateUnit(Enum):
    G = 10 ** 9
    T = 10 ** 12


class Miner(ABC):
    """Abstract miner class."""

    def __init__(self, unit=HashRateUnit.T):
        self.hash_rate = self.init_hash_rate
        self._unit = unit
        self.difficulty: float = self.init_difficulty

    @property
    @abstractmethod
    def init_difficulty(self):
        pass

    @property
    @abstractmethod
    def init_hash_rate(self):
        pass

    @abstractmethod
    def block_time(self) -> float:
        pass


class BitcoinMiner(Miner):
    init_difficulty = 1.0
    init_hash_rate = 9.44495E-07

    def block_time(self) -> float:
        """Calculate block time in seconds.

        Returns:
            Mining block time in seconds.
        """
        return self.difficulty * 2 ** 32 / self._unit.value / self.hash_rate
