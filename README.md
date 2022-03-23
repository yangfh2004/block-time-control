# Block Time Simulator
A framework to simulate and test control algorithms for PoW block time.

## Introduction
This tool can be used to accurately back-test and predict block time or block generation rate
only using hash rate data and its corresponding timestamp (e.g. daily average hash rate).

## Bitcoin Test Results
Using this framework to simulate the daily block generation and daily average mining difficulty against 
the actual historical data as shown below.

The mean error of daily block count is 0.83%.
![plot](./charts/bitcoin_blk_sim.png)

The mean error of daily average difficulty is 0.97%.
![plot](./charts/bitcoin_diff_sim.png)

## Data Source
The Bitcoin hash rate and difficulty data is from:
https://coinmetrics.io