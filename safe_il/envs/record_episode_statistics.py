from collections import deque
from copy import deepcopy
import time

import gym
import numpy as np


class RecordEpisodeStatistics(gym.Wrapper):
    """ Keep track of episode length and returns per instantiated env 
        adapted from https://github.com/openai/gym/blob/38a1f630dc9815a567aaf299ae5844c8f8b9a6fa/gym/wrappers/record_episode_statistics.py
    """

    def __init__(self, env, deque_size=None, **kwargs):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.deque_size = deque_size

        self.t0 = time.time()
        self.episode_return = 0.0
        self.episode_length = 0
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

        # other tracked stats
        self.episode_stats = {}
        self.accumulated_stats = {}
        self.queued_stats = {}

    def add_tracker(self, name, init_value, mode="accumulate"):
        """Adds a specific stat to be tracked (accumulate|queue).
        
        Different modes to track stats
        - accumulate: rolling sum, e.g. total # of constraint violations during training.
        - queue: finite, individual storage, e.g. returns, lengths, constraint costs.
        """
        self.episode_stats[name] = init_value
        if mode == "accumulate":
            self.accumulated_stats[name] = init_value
        elif mode == "queue":
            self.queued_stats[name] = deque(maxlen=self.deque_size)
        else:
            raise Exception("Tracker mode not implemented.")

    def reset(self, **kwargs):
        self.episode_return = 0.0
        self.episode_length = 0
        # reset other tracked stats
        for key in self.episode_stats:
            self.episode_stats[key] *= 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.episode_return += reward
        self.episode_length += 1
        # add other tracked stats
        for key in self.episode_stats:
            if key in info:
                self.episode_stats[key] += info[key]

        if done:
            info['episode'] = {'r': self.episode_return, 'l': self.episode_length, 't': round(time.time() - self.t0, 6)}
            self.return_queue.append(self.episode_return)
            self.length_queue.append(self.episode_length)
            self.episode_return = 0.0
            self.episode_length = 0
            # other tracked stats
            for key in self.episode_stats:
                info['episode'][key] = deepcopy(self.episode_stats[key])
                if key in self.accumulated_stats:
                    self.accumulated_stats[key] += deepcopy(self.episode_stats[key])
                elif key in self.queued_stats:
                    self.queued_stats[key].append(deepcopy(self.episode_stats[key]))
                self.episode_stats[key] *= 0

        return observation, reward, done, info


