from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C
from env.OptBitcoinTradingEnv import OptBitcoinTradingEnv

import pandas as pd
import numpy as np
import optuna
import sys

COMMISSION = 0.00075*1.
INVESTMENT = 10000
#REWARD = 'sortino'
#REWARD = 'calmar'
#REWARD = 'omega'
REWARD = 'simple'


# bitcoin data
btc_df = pd.read_csv('./data/kaggle/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv', sep=',', header=0,
                     names=['Timestamp', 'Open', 'High', 'Low', 'Close',
                            'Volume_BTC', 'Volume_Currency', 'Weighted_Price'])
btc_df = btc_df.fillna(method='bfill').reset_index(drop=True)
feat_df = pd.read_csv('./data/processed/features.csv', sep=',')
data = pd.concat([btc_df, feat_df], axis=1)
#####data = btc_df

# train/test split
train_size = int(len(data) * 0.8)
train = data[:train_size]
test = data[train_size:]


def optimize_a2c(trial):
    return {
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        #'vf_coef': trial.suggest_loguniform('vf_coef', 1e-1, 7e-1),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-4, 1e-1),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    }


def optimize_ppo2(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }


def optimize_envs(trial):
    return {
        'reward_len': int(trial.suggest_loguniform('reward_len', 1, 200)),
        'window_size': int(trial.suggest_loguniform('window_size', 1, 100))
        #,'confidence_interval': trial.suggest_uniform('confidence_interval', 0.7, 0.99),
    }


def initialize_envs(env_params):
    """
    :param env_params: environment parameters
    :return: train and test environments
    """

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
    return train_env, test_env


def objective_fn(trial):

    env_params = optimize_envs(trial)
    agent_params = optimize_ppo2(trial)
    ####agent_params = optimize_a2c(trial)

    train_env, validation_env = initialize_envs(env_params)

    ####model = A2C(policy=MlpLstmPolicy,
    model = PPO2(policy=MlpLstmPolicy, nminibatches=1,
                env=train_env,
                verbose=1,
                tensorboard_log="./tensorboard/",
                **agent_params)

    model.learn(total_timesteps=10)
    print('Training finished ...')

    rewards, done = [], False
    obs = validation_env.reset()
    for i in range(10):
        action, _ = model.predict(obs)
        obs, reward, done, _ = validation_env.step(action)
        rewards = np.append(rewards, reward)

    print(-np.mean(rewards))
    return -np.mean(rewards)


def optimize(n_trials=100, n_jobs=4):
    study = optuna.create_study(study_name='optimize_profit', storage='sqlite:///params.db', load_if_exists=True)
    study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs)
    print(study.best_params)
    return 0


def main():

    print('Total number of observations: {}'.format(len(data)))
    print('Training Observations: {:.0f}%'.format(len(train) / len(data) * 100))
    print('Testing Observations: {:.0f}%'.format(len(test) / len(data) * 100))

    # run optimization:
    optimize(n_trials=4, n_jobs=4)
    print('Optuna optimization finished:')


if __name__ == "__main__":
    main()
