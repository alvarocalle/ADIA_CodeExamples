from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C
from env.OptBitcoinTradingEnv import OptBitcoinTradingEnv
from env.BitcoinTradingEnv import BitcoinTradingEnv

import pandas as pd
import sys

COMMISSION = 0.00075*1.
INVESTMENT = 10000
#REWARD = 'sortino'
#REWARD = 'calmar'
REWARD = 'omega'

# bitcoin data
btc_df = pd.read_csv('./data/kaggle/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv', sep=',', header=0,
                     names=['Timestamp', 'Open', 'High', 'Low', 'Close',
                            'Volume_BTC', 'Volume_Currency', 'Weighted_Price'])
btc_df = btc_df.fillna(method='bfill').reset_index(drop=True)
feat_df = pd.read_csv('./data/processed/features.csv', sep=',')
#####data = pd.concat([btc_df, feat_df], axis=1)
data = btc_df

# train/test split
train_size = int(len(data) * 0.7)
train = data[:train_size]
test = data[train_size:]
print('Total number of observations: {}'.format(len(data)))
print('Training Observations: {:.0f}%'.format(len(train)/len(data)*100))
print('Testing Observations: {:.0f}%'.format(len(test)/len(data)*100))

# The algorithms require a vectorized environment to run
train_env = DummyVecEnv([lambda: OptBitcoinTradingEnv(train,
                                                      commission=COMMISSION,
                                                      reward_type=REWARD,
                                                      initial_balance=INVESTMENT,
                                                      serial=False)])
test_env = DummyVecEnv([lambda: OptBitcoinTradingEnv(test,
                                                     commission=COMMISSION,
                                                     reward_type=REWARD,
                                                     initial_balance=INVESTMENT,
                                                     serial=True)])
#######train_env = DummyVecEnv([lambda: BitcoinTradingEnv(train, commission=COMMISSION, serial=False)])
#######test_env = DummyVecEnv([lambda: BitcoinTradingEnv(test, commission=COMMISSION, serial=True)])

#####model = PPO2(MlpPolicy,
#####model = A2C(MlpPolicy,
model = A2C(MlpLstmPolicy,
            train_env,
            verbose=1,
            tensorboard_log="./tensorboard/")

model.learn(total_timesteps=200)
#sys.exit()

# save fitted model
#https://stable-baselines.readthedocs.io/en/master/guide/save_format.html

env = test_env
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='live', title='BTC')

