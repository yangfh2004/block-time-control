import click
from simulator import MiningSimulator, BitcoinMiner, CapsuleMiner
from simulator.algorithm import SimpleAvgAdjust, SimpleBitAdjust, TimeInterval, TimeUnit
from pathlib import Path


@click.group()
def cli():
    pass


@cli.command()
def calibrate():
    csv_path = Path("test/data/bitcoin_hash_difficulty.csv")
    print(f"The tool runs against Bitcoin historical data '{csv_path}'"
          " to calibrate the simulation and control algorithm.")
    algo = SimpleAvgAdjust()
    miner = BitcoinMiner(algo)
    sim = MiningSimulator(hash_data=csv_path, miner=miner)
    sim.calibrate()


@cli.command()
@click.option('--data', type=str, default="test/data/bitcoin_hash_difficulty.csv",
              help=".csv data file contains hash data and timestamps.")
@click.option('--start', type=str, help='Start date (d/m/y) of evaluation of simulation.')
@click.option('--end', type=str, help='End date (d/m/y) of evaluation of simulation.')
@click.option('--interval', type=int, default=14, help='Time interval (days) between each difficulty adjustment.')
@click.option('--time', type=int, default=10, help='Block time (minutes) between each block generation.')
@click.option('--algo', type=str, default='SimpleAvgAdjust')
@click.option('--miner', type=str, default='BitcoinMiner')
def simulate(data, start, end, interval, time, algo, miner):
    csv_path = Path(data)
    print(f"The simulation uses hash data and timestamps from '{csv_path}'")
    adjust_time_interval = TimeInterval(interval, TimeUnit.Day)
    blk_time_target = TimeInterval(time, TimeUnit.Minute)
    if algo == 'SimpleAvgAdjust':
        sim_algo = SimpleAvgAdjust(adjust_interval=adjust_time_interval, target=blk_time_target)
    elif algo == 'SimpleBitAdjust':
        sim_algo = SimpleBitAdjust(adjust_interval=adjust_time_interval, target=blk_time_target, adjust_range=3)
    else:
        raise "User defined algorithm is not supported!"
    if miner == 'BitcoinMiner':
        sim_miner = BitcoinMiner(algo=sim_algo)
    elif miner == 'CapsuleMiner':
        sim_miner = CapsuleMiner(algo=sim_algo)
    else:
        raise "User defined miner is not supported!"
    print(f"Start simulation with algorithm: {algo} and miner: {miner}")
    sim = MiningSimulator(hash_data=csv_path, miner=sim_miner)
    sim.simulate(start=start, end=end)


if __name__ == '__main__':
    cli()
