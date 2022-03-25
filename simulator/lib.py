from pathlib import Path
from pandas import read_csv
from simulator.miner import Miner
from simulator.algorithm import TimeInterval, TimeUnit
from math import sqrt
import matplotlib.pyplot as plt
from pandas.plotting._matplotlib.style import get_standard_colors


class MiningSimulator:
    """This simulator uses DAILY history stats to test and calibrate different mining system and difficulty
    adjustment algorithms. """

    def __init__(self,hash_data: Path, miner: Miner):
        ext = hash_data.suffix
        if ext == '.csv':
            self.data = read_csv(hash_data, parse_dates=True, index_col=0)
        else:
            raise FileExistsError("Data file must be csv files!")
        self.miner = miner

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
        """Run simulation thru all mining hash rate data"""
        sim_blk_cnt = []
        sim_difficulty = []
        sim_block_time = []
        for timestamp, data in self.data.iterrows():
            hash_rate = data['HashRate']
            blk_cnt = self.miner.generate_blocks(hash_rate, timestamp)
            sim_blk_cnt.append(blk_cnt)
            sim_block_time.append(self.miner.block_time)
            sim_difficulty.append(self.miner.difficulty)
        self.data['SimBlkCnt'] = sim_blk_cnt
        self.data['DiffSim'] = sim_difficulty
        self.data['SimBlockTime'] = sim_block_time

    def calibrate(self, compute_error=True):
        """Calibrate mining simulator with historical mining data."""
        self.run()
        plt.figure()
        self.data.plot(y=['SimBlkCnt', 'BlkCnt'],
                       title="Actual & Simulated Daily Block Count")
        plt.figure()
        self.data.plot(y=['DiffSim', 'DiffLast'],
                       title="Actual & Simulated Daily Average Mining Difficulty", logy=True)
        if compute_error:
            self.data['BlkCntError'] = (self.data['SimBlkCnt'] - self.data['BlkCnt'])
            blk_se = self.data['BlkCntError']**2
            print(f"The MSE of daily block generation rate is {sqrt(blk_se.mean()):.2f} blocks.")
            plt.figure()
            self.data.plot(y=['BlkCntError'])
            self.data['DiffError'] = (self.data['DiffSim'] - self.data['DiffLast']) / self.data['DiffLast']
            diff_se = self.data['DiffError']**2
            print(f"The MSE of difficulty is {sqrt(diff_se.mean()) * 100:.2f}%")
            plt.figure()
            self.data.plot(y=['DiffError'])
        plt.show()

    def simulate(self, start=None, end=None):
        """Simulate mining daily dynamics."""
        self.run()
        if start:
            if end:
                self.data = self.data[start:end]
            else:
                self.data = self.data[start:]
        plt.figure()
        self._plot_multi(cols=['SimBlockTime', 'HashRate'], title='Simulated Block Time and Real Hash Rate', logy=True)
        plt.figure()
        self.data.plot(y='DiffSim', title='Simulated Mining Difficulty')
        day = TimeInterval(1, TimeUnit.Day)
        target_daily_blk_cnt = day.seconds() // self.miner.block_time_target
        self.data['BlkCntShift'] = self.data['SimBlkCnt'] - target_daily_blk_cnt
        plt.figure()
        self.data.plot(y='BlkCntShift', title='Daily Block Generation Shift')
        self.data['BlockTimeShift'] = (self.data['SimBlockTime'] - self.miner.block_time_target)
        self.data.plot(y='BlockTimeShift', title='Daily Average Block Time Shift')
        blk_time_se = self.data['BlockTimeShift']**2
        print(f"The average simulated block time is {self.data['SimBlockTime'].mean():.2f} seconds. ")
        window_size = self.miner.block_count_target // target_daily_blk_cnt
        roll_mean_bt = self.data['BlockTimeShift'].rolling(window_size).mean()
        print(f"The rolling average of {window_size} day window has "
              f"{roll_mean_bt.mean():.2f} seconds of shift.")
        print(f"The MSE of against targeted block time is {sqrt(blk_time_se.mean()):.2f} seconds")
        plt.show()
