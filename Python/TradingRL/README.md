# TradingRL: an RL framework for trading experiments

## 1) Description

Trading experiments using RL.

## Trading Environment Class (StockTradingEnv)

- `observation_space` : input variables that the agent see before considering making an action. 
The action space can be a collection of OHLC stock data points for a time window, as well as other data 
like its account balance, current stock positions, and current profit.

- `action_space` : once the agent has perceived the environment, it takes actions. They will consist of three possibilities: buy a stock, sell a stock, or do nothing.

- `reward` : We want to incentivize profits that is sustained over long periods of time. 
At each step, we will set the reward to the account balance multiplied by some fraction of the number of time steps so far.

## Trading Environment Class (BitcoinTradingEnv)

Goal: optimize the Net Worth

- `observation_space` : The action space in this case is OHLC bitcoin data for a time window, as well as the volume traded,
BTC bought/sold and total amount USD spent/received from those BTC.

- `action_space` : once the agent has perceived the environment, it takes actions. They will consist of three possibilities: buy a stock, sell a stock, or do nothing.

- `reward` : We want to reward profits that are sustained over long periods of time.
At each step, we will set the reward to the account balance multiplied by some fraction of the number of time steps so far.


## Trading Environment Class (OptBitcoinTradingEnv)

- `observation_space` : 
- `action_space` : 
- `reward` : 

The reward function in `BitcoinTradingEnv` (i.e. rewarding incremental net worth gains) is not the best we can do. 
To improve on this, we are going to need to consider other metrics to reward, besides simply unrealized profit.

A simple improvement to this strategy, is to not only reward profits from holding BTC while it is increasing in price, 
but also reward profits from not holding BTC while it is decreasing in price. For example, we could reward our agent for any 
incremental increase in net worth while it is holding a BTC/USD position, and again for the incremental decrease in value of BTC/USD 
while it is not holding any positions.

While this strategy is great at rewarding increased returns, it fails to take into account the risk of producing those high returns. 
Investors have long since discovered this flaw with simple profit measures, and have traditionally turned to risk-adjusted return metrics to account for it.

###### Volatility-Based Metrics to measure reward 

 + Sharpe ratio:  

The most common risk-adjusted return metric is the **Sharpe ratio**. This is a simple ratio of a portfolio's excess
returns to volatility, measured over a specific period of time. To maintain a high Sharpe ratio, an investment must 
have both high returns and low volatility (i.e. risk). The Sharpe ratio is defined as follows:

![Sharpe ratio](./imgs/sharpe.png)

+ Sortino ratio

This metric is broadly used across. However it not very good for our purposes, as it penalizes upside volatility. 
For Bitcoin, this can be problematic as upside volatility (wild upwards price movement) can often be quite profitable to be a part of. 

For that reason we will be using the **Sortino ratio** which is very similar to the Sharpe ratio, except it only considers 
downside volatility as risk, rather than overall volatility. As a result, this ratio does not penalize upside volatility:
 
![Sortino ratio](./imgs/sortino.png)

+ Calmar ratio

This metric encourage strategies that actively prevent large [drawdowns](https://www.investopedia.com/terms/d/drawdown.asp).
This ratio is identical to the Sharpe ratio, except that it uses maximum drawdown in place of the portfolio value's standard deviation.
 
![Calmar ratio](./imgs/calmar.png)

+ Omega ratio

The Omega ratio should be better than both the Sortino and Calmar ratios at measuring risk vs. return. 
It is able to account for the entirety of the risk over return distribution in a single metric. 
To find it, we need to calculate the probability distributions of a portfolio moving above or below a specific benchmark, 
and then take the ratio of the two. The higher the ratio, the higher the probability of upside potential over downside potential.

![Omega ratio](./imgs/omega.png)


###### Implementation

To implement the previous reward metrics we use the [Quantopian's](https://www.quantopian.com/) library [empyrical](https://github.com/quantopian/empyrical).
This library includes the three rewards metrics we’ve defined above. Getting one of the previous ratios at each time step is as simple 
as providing the list of asset and benchmark returns for a time period to the corresponding metric function.


###### Optuna 

The objective function to optimize consists of training and testing our PPO2 model on our Bitcoin trading environment. 
The cost we return from our function is the average reward over the testing period, negated (Optuna interprets lower return value as better trials). 
The optimize function provides a trial object to our objective function, which we then use to specify each variable to optimize.


## Reinforcement Learning Algorithms

- To implement the RL algorithm we use [Stable Baselines](https://github.com/hill-a/stable-baselines) which is an improvement over [Beselines](https://github.com/openai/baselines).

- A description of Stable Baselines can be found [here](https://towardsdatascience.com/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82)

- The user is also encouraged to use Baselines algorithms for comparison.

### RL algorithms and deep/shallow policies 

Stable-baselines provides a set of default policies, that can be used with most action spaces. 

To customize the default policies, you can specify the `policy_kwargs` parameter to the model class you use. 

Those kwargs are then passed to the policy on instantiation (see [Custom Policy Network](https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html#custom-policy) for an example). 

If you need more control on the policy architecture, you can also create a custom policy (see [Custom Policy Network](https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html#custom-policy)).

- CnnPolicies are for images only. 
- MlpPolicies are made for other type of features
 
 Actor critics


https://stable-baselines.readthedocs.io/en/master/modules/policies.html

- We use a Multi-Layer Perceptron (MLP) network for our experiments in the Stock and Bitcoin Trading Envs
- We use a Long Short-Term Memory (LSTM) network as policy.


















## Installation of (Stable) Baselines

##### Install TensorFlow (GPU/CPU)

Currently baselines support Tensorflow 1.14

```bash
pip install tensorflow-gpu==1.14 # if you have a CUDA-compatible gpu and proper drivers
```

or

```bash
pip install tensorflow==1.14
```

##### Installation (Stable Baselines)

- To build `mpi4py` in Stable Baselines we need system packages CMake, OpenMPI and zlib.

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

- Install the Stable Baselines package with optional dependency on MPI, enabling algorithms DDPG, GAIL, PPO1 and TRPO. 

```bash
pip install stable-baselines[mpi]
```


##### Installation (Baselines)

- Clone the repo and cd into it:

```bash
git clone https://github.com/openai/baselines.git
cd baselines
```

- Install baselines package

```bash
pip install -e .
```






## Data

- We use data from the [IEX](https://iextrading.com). To download the data we use the library `iexfinance`.

- Bitcoin Historical Data from [Kaggle](https://www.kaggle.com/mczielinski/bitcoin-historical-data)

## Features

TA-Lib

https://github.com/mrjbq7/ta-lib


## References

#### Trading RL series

- [Create custom gym environments from scratch — A stock market example](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)

- [Rendering elegant stock trading agents using Matplotlib and Gym](https://towardsdatascience.com/visualizing-stock-trading-agents-using-matplotlib-and-gym-584c992bc6d4)

- [Creating Bitcoin trading bots don’t lose money](https://towardsdatascience.com/creating-bitcoin-trading-bots-that-dont-lose-money-2e7165fb0b29)

- [Optimizing deep learning trading bots using state-of-the-art techniques](https://towardsdatascience.com/using-reinforcement-learning-to-trade-bitcoin-for-massive-profit-b69d0e8f583b)

#### Intraday data

- [How to download all historic intraday OHCL data from IEX: with Python, asynchronously, via API & for free.](https://towardsdatascience.com/how-to-download-all-historic-intraday-ohcl-data-from-iex-with-python-asynchronously-via-api-b5b04a31b187)

- [Alpaca: hack financial systems](https://alpaca.markets/)

- Library [iexfinance](https://addisonlynch.github.io/iexfinance/stable/#)


#### OpenAI

- [OpenAI Baselines](https://github.com/openai/baselines)
- [PPO explained](https://www.youtube.com/watch?v=5P7I-xPq8u8)
- [PPO Arxiv paper](https://arxiv.org/abs/1707.06347)