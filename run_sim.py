import click
from simulator import MiningSimulator, BitcoinMiner, CapsuleMiner
from simulator.algorithm import SimpleAvgAdjust, SimpleBitAdjust, PIDBitAdjust, TimeInterval, TimeUnit
from pathlib import Path
from numpy import arange
from tqdm import tqdm
from math import sqrt
from pandas import DataFrame


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
    elif algo == 'PIDBitAdjust':
        sim_algo = PIDBitAdjust(adjust_interval=adjust_time_interval, target=blk_time_target, adjust_range=3)
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


@cli.command()
@click.option('--data', type=str, default="test/data/bitcoin_hash_difficulty.csv",
              help=".csv data file contains hash data and timestamps.")
@click.option('--interval', type=int, default=14, help='Time interval (days) between each difficulty adjustment.')
@click.option('--time', type=int, default=10, help='Block time (minutes) between each block generation.')
@click.option('--algo', type=str, default='SimpleAvgAdjust')
@click.option('--miner', type=str, default='BitcoinMiner')
@click.option('--p', type=str, help='The start, end and step length of proportional gain, e.g. "0.5_3.0_0.1"')
@click.option('--i', type=str, default='0.0_1.0_1.0',
              help='The start, end and step length of integral gain, e.g. "0.5_3.0_0.1"')
@click.option('--d', type=str, default='0.0_1.0_1.0',
              help='The start, end and step length of derivative gain, e.g. "0.5_3.0_0.1"')
def tuning(data, interval, time, algo: str, miner, p, i, d):
    if not algo.startswith('PID'):
        raise "User defined algorithm does not support auto tuning!"
    csv_path = Path(data)
    print(f"The simulation uses hash data and timestamps from '{csv_path}'")
    adjust_time_interval = TimeInterval(interval, TimeUnit.Day)
    blk_time_target = TimeInterval(time, TimeUnit.Minute)
    p_start, p_end, p_step = tuple(map(float, p.split('_'))) if '_' in p else (float(p), float(p)+1.0, 1.0)
    assert p_start <= p_end, "Proportional gain parameters are not valid!"
    i_start, i_end, i_step = tuple(map(float, i.split('_'))) if '_' in i else (float(i), float(i)+1.0, 1.0)
    assert i_start <= i_end, "Integral gain parameters are not valid!"
    d_start, d_end, d_step = tuple(map(float, d.split('_'))) if '_' in d else (float(d), float(d)+1.0, 1.0)
    assert d_start <= d_end, "Derivative gain parameters are not valid!"
    param_list = []
    for pp in arange(p_start, p_end, p_step):
        for ii in arange(i_start, i_end, i_step):
            for dd in arange(d_start, d_end, d_step):
                param_list.append((pp, ii, dd))
    sim_res = []
    for pp, ii, dd in tqdm(param_list):
        if algo == 'PIDBitAdjust':
            sim_algo = PIDBitAdjust(adjust_interval=adjust_time_interval,
                                    target=blk_time_target,
                                    adjust_range=3,
                                    p=pp,
                                    i=ii,
                                    d=dd)
        else:
            raise "User defined algorithm is not supported!"
        if miner == 'BitcoinMiner':
            sim_miner = BitcoinMiner(algo=sim_algo)
        elif miner == 'CapsuleMiner':
            sim_miner = CapsuleMiner(algo=sim_algo)
        else:
            raise "User defined miner is not supported!"
        sim = MiningSimulator(hash_data=csv_path, miner=sim_miner)
        sim.run()
        sim.data['BlockTimeShift'] = (sim.data['SimBlockTime'] - sim.miner.block_time_target)
        block_time_mean = sim.data['SimBlockTime'].mean()
        blk_time_se = sim.data['BlockTimeShift'] ** 2
        block_time_mse = sqrt(blk_time_se.mean())
        sim_res.append((block_time_mean, block_time_mse, pp, ii, dd))
    sim_df = DataFrame(sim_res, columns=['AvgBlkTime', 'MSE', 'p', 'i', 'd'])
    print(f"Parameter tuning of algorithm: {algo} is finished!")
    print(sim_df)


if __name__ == '__main__':
    cli()
