import pandas as pd
import numpy as np
import time
import torch
from gym_trading_env.environments import TradingEnv
import gymnasium as gym


def read_data(data_path="Data/BTC_USD-Hourly.csv"):
    df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7 * 24).max()
    df.dropna(inplace=True)
    df.drop("symbol", axis=1, inplace=True)
    df = (df - df.mean()) / df.std()
    return df


class TradingEnv:
    def __init__(
        self,
        no_grad=True,
        logdir=None,
        episode_length=50,
        actions=[-1, -0.5, 0, 0.5, 1, 1.5, 2],
    ):  # From -1 (=SHORT), to +1 (=LONG)
        self.episode_length = episode_length
        self.act_space = actions
        self.num_observations = 7
        self.num_actions = len(actions)
        self.num_environments = 1
        # self.obs_space = 0

        self.env = gym.make(
            "TradingEnv",
            name="BTCUSD",
            df=read_data(),
            windows=1,
            positions=self.act_space,
            initial_position="random",  # Initial position
            trading_fees=0.01 / 100,  # 0.01% per stock buy / sell
            borrow_interest_rate=0.0003 / 100,  # per timestep (= 1h)
            reward_function=self.reward_function,
            portfolio_initial_value=1000,  # in USD
            max_episode_duration=500,
        )

    def reward_function(self, history):
        return np.log(
            history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
        )  # log (p_t / p_t-1 )

    def step(self, action):
        # print("*********actions ", action)
        max_value, max_index = torch.max(action, dim=-1)
        # print("*********action ",max_index.item())
        action_idx = max_index.item()

        self.action_penalty = 0.0
        observation, rew, done, truncated, info = self.env.step(action_idx)
        act_penalty = torch.sum(action**2, dim=-1) * self.action_penalty
        reward = -rew + act_penalty
        reward = reward.squeeze()

        info["termination"] = torch.tensor(done)
        info["truncation"] = torch.tensor(truncated)
        # info["contact_forces"] =  torch.rand (1,8)
        # info["accelerations"] =  torch.rand(1,6,8)
        info["contact_forces"] = torch.tensor(
            [[0.1234, 0.5678, 0.9876, 0.5432, 0.2468, 0.1357, 0.7890, 0.4321]]
        )
        info["accelerations"] = torch.tensor(
            [
                [
                    [0.1234, 0.5678, 0.9876, 0.5432, 0.2468, 0.1357, 0.7890, 0.4321],
                    [0.1111, 0.2222, 0.3333, 0.4444, 0.5555, 0.6666, 0.7777, 0.8888],
                    [0.9999, 0.8888, 0.7777, 0.6666, 0.5555, 0.4444, 0.3333, 0.2222],
                    [0.9876, 0.8765, 0.7654, 0.6543, 0.5432, 0.4321, 0.3210, 0.2109],
                    [0.1234, 0.2345, 0.3456, 0.4567, 0.5678, 0.6789, 0.7890, 0.8901],
                    [0.1111, 0.2222, 0.3333, 0.4444, 0.5555, 0.6666, 0.7777, 0.8888],
                ]
            ]
        )
        self.observation = torch.tensor(observation, dtype=torch.float32)
        info["obs_before_reset"] = self.observation

        return self.observation, reward, done, info

    def reset(self):
        observation, info = self.env.reset()
        self.observation = torch.tensor(np.expand_dims(observation, axis=0))
        return self.observation

    def clear_grad(self):
        pass

    def initialize_trajectory(self):
        self.clear_grad()

        return self.observation

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations
