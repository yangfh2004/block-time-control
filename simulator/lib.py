from pathlib import Path
from typing import Callable, Optional
from .miner import Miner
from pandas import Timestamp, to_datetime
import pandas
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting._matplotlib.style import get_standard_colors


class MiningSimulator:
    def __init__(self, algo: Callable[[int, Timestamp], float], hash_data: Path, miner: Miner):
        self.algorithm = algo
        ext = hash_data.suffix
        if ext == '.csv':
            self.data = pandas.read_csv(hash_data)
        else:
            raise FileExistsError("Data file must be csv files!")
        self.miner = miner
        self._timestamp: Optional[Timestamp] = None

    def plot_multi(self, cols=None, spacing=.1, **kwargs):
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
        ax.legend(lines, labels, loc=0)
        return ax

    def run(self):
        """Run mining simulator with data."""
        sim_blk_cnt = []
        sim_difficulty = []
        for _, data in self.data.iterrows():
            timestamp = data['time']
            hash_rate = data['HashRate']
            if self._timestamp is None:
                self._timestamp = to_datetime(timestamp)
                blk_cnt = 0
            else:
                # update hash rate.
                self.miner.hash_rate = hash_rate
                # get the block time to mine a new block.
                block_time = self.miner.block_time()
                prev_timestamp = self._timestamp
                self._timestamp = to_datetime(timestamp)
                time_interval = int((self._timestamp - prev_timestamp) / np.timedelta64(1, 's'))
                blk_cnt = time_interval // int(block_time)
            # adjust miner difficulty.
            sim_blk_cnt.append(blk_cnt)
            self.miner.difficulty = self.algorithm(blk_cnt, self._timestamp)
            sim_difficulty.append(self.miner.difficulty)
        self.data['SimBlkCnt'] = sim_blk_cnt
        self.data['DiffSim'] = sim_difficulty
        plt.figure()
        self.data.plot(x='time', y=['SimBlkCnt', 'BlkCnt'])
        plt.figure()
        self.data.plot(x='time', y=['DiffSim', 'DiffLast'])
        plt.show()
