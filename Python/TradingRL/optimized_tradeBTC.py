from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C
from env.OptBitcoinTradingEnv import OptBitcoinTradingEnv
from env.BitcoinTradingEnv import BitcoinTradingEnv

import pandas as pd
import optuna
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

# get best optimized parameters
file = 'sqlite:///params.db'
study = optuna.load_study(study_name='optimize_profit', storage=file)
params = study.best_trial.params
env_params = {
    'reward_len': int(params['reward_len']),
    'window_size': int(params['window_size'])
    #,'confidence_interval': params['confidence_interval']
}

# The algorithms require a vectorized environment to run
train_env = DummyVecEnv([lambda: OptBitcoinTradingEnv(train,
                                                      commission=COMMISSION,
                                                      reward_type=REWARD,
                                                      initial_balance=INVESTMENT,
                                                      serial=False,
                                                      reward_len=env_params['reward_len'],
                                                      window_size=env_params['window_size']
                                                      #,confidence_interval=env_params['confidence_interval']
                                                    )])

test_env = DummyVecEnv([lambda: OptBitcoinTradingEnv(test,
                                                     commission=COMMISSION,
                                                     reward_type=REWARD,
                                                     initial_balance=INVESTMENT,
                                                     serial=True,
                                                     reward_len=env_params['reward_len'],
                                                     window_size=env_params['window_size']
                                                     #,confidence_interval=env_params['confidence_interval']
                                                    )])

model_params = {
    'gamma': params['gamma'],
    'n_steps': int(params['n_steps']),
    'ent_coef': params['ent_coef'],
    'learning_rate': params['learning_rate']
}

model = A2C(policy=MlpLstmPolicy,
            env=train_env,
            **model_params)

env = test_env
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='live', title='BTC')

