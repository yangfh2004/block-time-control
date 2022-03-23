from simulator import MiningSimulator, BitcoinMiner, CapsuleMiner
from simulator.algorithm import SimpleAvgAdjust, SimpleBitAdjust, TimeInterval, TimeUnit
from pathlib import Path

if __name__ == '__main__':
    csv_path = Path("test/data/bitcoin_hash_difficulty.csv")
    adjust_time_interval = TimeInterval(2, TimeUnit.Week)
    algo = SimpleAvgAdjust(adjust_interval=adjust_time_interval)
    miner = BitcoinMiner(algo)
    sim = MiningSimulator(hash_data=csv_path, miner=miner)
    sim.calibrate()

