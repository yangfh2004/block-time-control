from simulator import MiningSimulator, BitcoinMiner
from simulator.algorithm import SimpleAvgAdjust
from pathlib import Path

if __name__ == '__main__':
    algo = SimpleAvgAdjust()
    csv_path = Path("test/data/bitcoin_hash_difficulty.csv")
    miner = BitcoinMiner()
    sim = MiningSimulator(algo=algo, hash_data=csv_path, miner=miner)
    sim.run()

