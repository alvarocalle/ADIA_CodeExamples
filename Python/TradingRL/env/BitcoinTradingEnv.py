import numpy as np
import gym
from gym import spaces
from sklearn.preprocessing import MinMaxScaler
from render.BitcoinTradingGraph import BitcoinTradingGraph


MAX_TRADING_SESSION = 100000  # ~2 months


class BitcoinTradingEnv(gym.Env):
    """
    Bitcoin trading environment for OpenAI gym

    Parameters
    ----------
    df : pandas.DataFrame with OHLCV data - ['Close', 'High', 'Low', 'Close', 'Volume']
    window_size : number of timestamps to look back in historical prices
    commission : commission per trade in %
    initial_balance : initial BTCs in our account
    serial : False/True to transverse the data frame in random slices
    partitions : discrete set of amounts that are traded (e.g. buy, sell or hold 1/10, 2/10, ...)
    btc_held : number of BTCs held
    """
    metadata = {'render.modes': ['live', 'file', 'none']}
    scaler = MinMaxScaler()

    def __init__(self, df,
                 window_size=50,
                 commission=0.00075,
                 initial_balance=10000,
                 serial=False,
                 partitions=10,
                 btc_held=0):

        super(BitcoinTradingEnv, self).__init__()

        # initial data without NA
        self.df = df.dropna().reset_index(drop=True)

        # set initial observation
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.commission = commission
        self.serial = serial
        self.partitions = partitions
        self.btc_held = btc_held
        self.current_step = 0

        # trades: will store trade history of agent as
        # 'step': current step of the trade,
        # 'amount': btc_sold or btc_bought,
        # 'total': sales or costs,
        # 'type': sell or buy
        self.trades = []

        # account history: OHLCV + net_worth + BTC bought/sold + USD spent/received
        self.account_history = np.repeat([
            [initial_balance],
            [0],
            [0],
            [0],
            [0]
        ], window_size + 1, axis=1)

        # action stace is multi-discrete meaning that we have three actions: [Buy, Sell, Hold]
        # and at each one of these three actions we can select x possible amounts (1/10, 2/10, ...)
        # action_space = [action_type, partitions]
        #   - action_type = 0 (buy) / 1 (sell) / hold (2)
        #   - partitions = number in range (0, partitions)
        self.action_space = spaces.MultiDiscrete([3, self.partitions])

        # observation_space : for window_size + today
        #   + OHCL prices
        #   + V traded
        #   + Net worth
        #   + BTC bought/sold
        #   + Total amount USD spent/received from those BTC
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(10, self.window_size + 1), dtype=np.float16)

        # initialize viewer:
        self.viewer = None

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.trades = []

        self.account_history = np.repeat([
            [self.initial_balance],
            [0],
            [0],
            [0],
            [0]
        ], self.window_size + 1, axis=1)

        self._reset_session()

        return self._next_observation()

    def _reset_session(self):
        """
        Reset the trading session of time MAX_TRADING_SESSION and limit the amount
        of continuous time frames in self.df that our agent will see in a row.
        """

        self.current_step = 0

        if self.serial:
            self.steps_left = len(self.df) - self.window_size - 1
            self.frame_start = self.window_size
        else:
            self.steps_left = np.random.randint(1, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(
                self.window_size, len(self.df) - self.steps_left)

        self.active_df = self.df[self.frame_start - self.window_size:self.frame_start + self.steps_left]

    def _next_observation(self):
        """
        Get a new observation of the environment
        Scale observed history to values in range (0, 1)
        Append the observation to the history
        :return: new observation
        """

        end = self.current_step + self.window_size + 1

        # get ohlcv observation
        obs = np.array([
            self.active_df['Open'].values[self.current_step:end],
            self.active_df['High'].values[self.current_step:end],
            self.active_df['Low'].values[self.current_step:end],
            self.active_df['Close'].values[self.current_step:end],
            self.active_df['Volume_BTC'].values[self.current_step:end],
        ])

        # scale the history that the agent as already seen (check)
        scaled_history = self.scaler.fit_transform(self.account_history)

        # append new observation to the scaled history
        obs = np.append(obs, scaled_history[:, -(self.window_size + 1):], axis=0)

        return obs

    def step(self, action):
        """
        Execute one time step within the environment:
        - At each step we take an action (chosen by our model),
          calculate the reward, and return the next observation.
        - The action can be buy or sell a specified amount of BTCs
        - The reward function: net worth
        - At the end of the trading session we sell any BTC we hold and _reset_session().
        :param action:
        :return:
        """

        current_price = self._get_current_price() + 0.001
        self._take_action(action, current_price)
        self.current_step += 1
        self.steps_left -= 1

        if self.steps_left == 0:
            self.balance += self.btc_held * current_price
            self.btc_held = 0
            self._reset_session()

        obs = self._next_observation()

        # ----------------------------------------
        # define reward
        reward = self.net_worth
        #reward = self.net_worth - prev_net_worth
        # ----------------------------------------

        done = self.net_worth <= 0

        return obs, reward, done, {}

    def _get_current_price(self):
        """
        Get a fraction of the DataFrame for the close price for the current time window
        :return:
        """
        return self.df['Close'].values[self.frame_start + self.current_step]

    def _take_action(self, action, current_price):
        """
        Take the action provided by the model and store trading history
        :param action:
            - action[0] = buy/sell
            - action[1] = 1,2,...,10 parts
        :param current_price:

        :return:
        """

        action_type = action[0]
        amount = action[1] / 10

        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        # buy action
        if action_type == 0:

            btc_bought = self.balance / current_price * amount
            cost = btc_bought * current_price * (1 + self.commission)
            self.btc_held += btc_bought
            self.balance -= cost

        # sell action
        elif action_type == 1:

            btc_sold = self.btc_held * amount
            sales = btc_sold * current_price * (1 - self.commission)
            self.btc_held -= btc_sold
            self.balance += sales

        # store trades, net worth ans accounting history:
        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({
                'step': self.frame_start + self.current_step,
                'amount': btc_sold if btc_sold > 0 else btc_bought,
                'total': sales if btc_sold > 0 else cost,
                'type': "sell" if btc_sold > 0 else "buy"
            })

        self.net_worth = self.balance + self.btc_held * current_price
        self.account_history = np.append(self.account_history, [
            [self.net_worth],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)


    def render(self, mode='live', title=None, **kwargs):
        """
        Render the environment to the screen
        :param mode: live or file
        :param title: title of the graph
        :param kwargs:
        :return:
        """
        if mode == 'live':
            if self.viewer == None:
                self.viewer = BitcoinTradingGraph(self.df, 'title')

        self.viewer.render(self.frame_start + self.current_step,
                           self.net_worth,
                           self.trades,
                           window_size=self.window_size)
