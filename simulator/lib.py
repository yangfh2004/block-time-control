from pathlib import Path
from typing import Optional
from simulator.miner import Miner
from simulator.algorithm import AdjustAlgorithm, TimeInterval, TimeUnit
from pandas import Timestamp, to_datetime
from copy import copy
import pandas
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting._matplotlib.style import get_standard_colors


class MiningSimulator:
    """This simulator uses DAILY history stats to test and calibrate different mining system and difficulty
    adjustment algorithms. """

    def __init__(self, algo: AdjustAlgorithm, hash_data: Path, miner: Miner):
        self.algorithm = algo
        ext = hash_data.suffix
        if ext == '.csv':
            self.data = pandas.read_csv(hash_data)
        else:
            raise FileExistsError("Data file must be csv files!")
        self.miner = miner
        self._timestamp: Optional[Timestamp] = None

    def _plot_multi(self, cols=None, spacing=.1, **kwargs):
        # Get default color style from pandas - can be changed to any other color list
        if cols is None:
            cols = self.data.columns
        if len(cols) == 0:
            return
        colors = get_standard_colors(num_colors=len(cols))
        # First axis
        ax = self.data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
        ax.set_ylabel(ylabel=cols[0])
        lines, labels = ax.get_legend_handles_labels()

        for n in range(1, len(cols)):
            # Multiple y-axes
            ax_new = ax.twinx()
            ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
            self.data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
            ax_new.set_ylabel(ylabel=cols[n])
            # Proper legend position
            line, label = ax_new.get_legend_handles_labels()
            lines += line
            labels += label
        ax.legend(lines, labels, loc='upper center')
        return ax

    def run(self):
        sim_blk_cnt = []
        sim_difficulty = []
        sim_block_time = []
        day = TimeInterval(1, TimeUnit.Day)

        for _, data in self.data.iterrows():
            timestamp = data['time']
            hash_rate = data['HashRate']
            excess_blk_time = 0
            if self._timestamp is None:
                self._timestamp = to_datetime(timestamp)
                blk_cnt = data['BlkCnt']
            else:
                # update hash rate.
                self.miner.hash_rate = hash_rate
                # get the block time (sec) to mine a new block.
                prev_timestamp = self._timestamp
                self._timestamp = to_datetime(timestamp)
                time_interval = int((self._timestamp - prev_timestamp) / np.timedelta64(1, 's'))
                blk_cnt = time_interval // int(self.miner.block_time)
                # check if there is difficulty adjustment today.
                if blk_cnt + self.algorithm.total_block_count > self.algorithm.block_count_target:
                    # if so correct the block count by linear regression.
                    # get the block count before the difficulty adjustment.
                    residual_blk_cnt = self.algorithm.block_count_target - self.algorithm.total_block_count
                    residual_blk_time = residual_blk_cnt * int(self.miner.block_time)
                    # try to adjust miner difficulty.
                    temp_algo = copy(self.algorithm)
                    excess_blk_time = day.seconds() - residual_blk_time
                    self.miner.difficulty = temp_algo(blk_cnt, self._timestamp, excess_blk_time)
                    new_blk_cnt = excess_blk_time // int(self.miner.block_time)
                    blk_cnt = new_blk_cnt + residual_blk_cnt
                    assert (blk_cnt > 0)
            self.miner.difficulty = self.algorithm(blk_cnt, self._timestamp, excess_blk_time)
            sim_blk_cnt.append(blk_cnt)
            sim_block_time.append(self.miner.block_time)
            sim_difficulty.append(self.miner.difficulty)
        self.data['SimBlkCnt'] = sim_blk_cnt
        self.data['DiffSim'] = sim_difficulty
        self.data['SimBlockTime'] = sim_block_time

    def calibrate(self, compute_error=True):
        """Calibrate mining simulator with daily data."""
        self.run()
        plt.figure()
        self.data.plot(x='time', y=['SimBlkCnt', 'BlkCnt'])
        plt.figure()
        self.data.plot(x='time', y=['DiffSim', 'DiffLast'])
        if compute_error:
            self.data['BlkCntError'] = (self.data['SimBlkCnt'] - self.data['BlkCnt']) / self.data['BlkCnt']
            print(f"The mean error of block count is {self.data['BlkCntError'].mean()}")
            plt.figure()
            self.data.plot(x='time', y=['BlkCntError'])
            self.data['DiffError'] = (self.data['DiffSim'] - self.data['DiffLast']) / self.data['DiffLast']
            print(f"The mean error of difficulty is {self.data['DiffError'].mean()}")
            plt.figure()
            self.data.plot(x='time', y=['DiffError'])
        plt.show()

    def simulate(self):
        """Simulate mining daily dynamics."""
        self.run()
        plt.figure()
        self._plot_multi(cols=['SimBlockTime', 'HashRate'], title='Simulated Block Time and Real Hash Rate', logy=True)
        plt.figure()
        self.data.plot(x='time', y='DiffSim', title='Simulated Mining Difficulty')
        day = TimeInterval(1, TimeUnit.Day)
        target_daily_blk_cnt = day.seconds() // self.algorithm.block_time_target
        self.data['BlkCntShift'] = self.data['SimBlkCnt'] - target_daily_blk_cnt
        plt.figure()
        self.data.plot(x='time', y='BlkCntShift', title='Daily Block Generation Shift')
        self.data['BlockTimeShift'] = (self.data['SimBlockTime'] - self.algorithm.block_time_target) \
            / self.algorithm.block_time_target
        self.data.plot(x='time', y='BlockTimeShift', ylim=(-1.0, 1.0), title='Daily Average Block Time Shift')
        print(f"Average daily block time error is "
              f"{self.data['BlockTimeShift'].abs().mean() * self.algorithm.block_time_target} seconds")
        plt.show()
