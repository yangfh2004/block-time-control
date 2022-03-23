from pathlib import Path
from pandas import to_datetime, read_csv
from simulator.miner import Miner
from simulator.algorithm import TimeInterval, TimeUnit
import matplotlib.pyplot as plt
from pandas.plotting._matplotlib.style import get_standard_colors


class MiningSimulator:
    """This simulator uses DAILY history stats to test and calibrate different mining system and difficulty
    adjustment algorithms. """

    def __init__(self,hash_data: Path, miner: Miner):
        ext = hash_data.suffix
        if ext == '.csv':
            self.data = read_csv(hash_data)
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
        sim_blk_cnt = []
        sim_difficulty = []
        sim_block_time = []
        for _, data in self.data.iterrows():
            timestamp = to_datetime(data['time'])
            hash_rate = data['HashRate']
            blk_cnt = self.miner.generate_blocks(hash_rate, timestamp)
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
        self.data.plot(x='time', y=['SimBlkCnt', 'BlkCnt'],
                       title="Actual & Simulated Daily Block Count")
        plt.figure()
        self.data.plot(x='time', y=['DiffSim', 'DiffLast'],
                       title="Actual & Simulated Daily Average Mining Difficulty", logy=True)
        if compute_error:
            self.data['BlkCntError'] = (self.data['SimBlkCnt'] - self.data['BlkCnt']) / self.data['BlkCnt']
            print(f"The mean error of block count is {self.data['BlkCntError'].abs().mean()}")
            plt.figure()
            self.data.plot(x='time', y=['BlkCntError'])
            self.data['DiffError'] = (self.data['DiffSim'] - self.data['DiffLast']) / self.data['DiffLast']
            print(f"The mean error of difficulty is {self.data['DiffError'].abs().mean()}")
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
        target_daily_blk_cnt = day.seconds() // self.miner.block_time_target
        self.data['BlkCntShift'] = self.data['SimBlkCnt'] - target_daily_blk_cnt
        plt.figure()
        self.data.plot(x='time', y='BlkCntShift', title='Daily Block Generation Shift')
        self.data['BlockTimeShift'] = (self.data['SimBlockTime'] - self.miner.block_time_target) \
            / self.miner.block_time_target
        self.data.plot(x='time', y='BlockTimeShift', ylim=(-1.0, 1.0), title='Daily Average Block Time Shift')
        print(f"Average daily block time error is "
              f"{self.data['BlockTimeShift'].abs().mean() * self.miner.block_time_target} seconds")
        plt.show()
